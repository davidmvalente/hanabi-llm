"""Client for OpenRouter's rerank endpoint.

A rerank model does not generate text or call tools. It takes a ``query`` and a list of
``documents`` and returns each document's relevance to the query; here the documents are the
candidate actions and the best-scored one is played.

    POST https://openrouter.ai/api/v1/rerank
    {"model": "...", "query": "...", "documents": ["...", ...], "top_n": N}
    -> {"results": [{"index": 3, "relevance_score": 0.91, ...}, ...], "usage": {...}}

The request fields and the error codes follow OpenRouter's documentation. The per-result field
names (``index``, ``relevance_score``) are as described for this endpoint and are parsed
defensively: a result without ``index`` is matched by its document text when that is unambiguous,
and ``score`` is accepted for ``relevance_score``. Anything unusable raises ``LLMError``, which the
player turns into an infrastructure fault (a provider problem, never the model's mistake).
"""

from __future__ import annotations

import asyncio
import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .llm import LLMError

OPENROUTER_RERANK_URL = "https://openrouter.ai/api/v1/rerank"
#: Transient: worth retrying with backoff (429 rate limit, 502 upstream, 524 timeout, 529 overloaded, ...).
_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504, 524, 529})
#: Retrying cannot help: out of credits, unknown model, payload too large.
_FINAL_STATUS = {402: "insufficient credits", 404: "model or endpoint not found", 413: "payload too large"}


@dataclass(frozen=True, slots=True)
class Ranked:
    index: int    # position in the documents list that was sent
    score: float


@dataclass(frozen=True, slots=True)
class Ranking:
    ranked: tuple[Ranked, ...]  # as returned by the provider
    usage: dict[str, Any]

    @property
    def tokens(self) -> int:
        """Total tokens if the provider reports them, else 0."""
        for key in ("total_tokens", "prompt_tokens", "input_tokens"):
            value = self.usage.get(key)
            if isinstance(value, int):
                return value
        return 0


class RerankClient(Protocol):
    model: str

    async def rank(self, query: str, documents: Sequence[str]) -> Ranking: ...


class OpenRouterRerankClient:
    """``RerankClient`` backed by OpenRouter. Retries transient failures with exponential backoff."""

    def __init__(self, model: str, api_key: str, http: httpx.AsyncClient, *, url: str = OPENROUTER_RERANK_URL, max_attempts: int = 4) -> None:
        self.model = model
        self._api_key = api_key
        self._http = http
        self._url = url
        self._max_attempts = max_attempts

    async def rank(self, query: str, documents: Sequence[str]) -> Ranking:
        body = {"model": self.model, "query": query, "documents": list(documents), "top_n": len(documents)}
        headers = {"Authorization": f"Bearer {self._api_key}", "X-Title": "hanabi-llm"}
        last_error = "unknown error"
        for attempt in range(self._max_attempts):
            try:
                resp = await self._http.post(self._url, json=body, headers=headers)
            except httpx.TransportError as exc:
                last_error = f"transport error: {exc}"
            else:
                if resp.status_code in _FINAL_STATUS:
                    raise LLMError(f"HTTP {resp.status_code} ({_FINAL_STATUS[resp.status_code]}) from the rerank endpoint: {resp.text[:200]}")
                if resp.status_code in _RETRY_STATUS:
                    last_error = f"HTTP {resp.status_code}"
                elif resp.status_code != 200:
                    raise LLMError(f"HTTP {resp.status_code} from the rerank endpoint: {resp.text[:300]}")
                else:
                    return self._parse(resp.json(), documents)
            await asyncio.sleep(min(2.0**attempt, 20.0))
        raise LLMError(f"gave up after {self._max_attempts} attempts: {last_error}")

    @staticmethod
    def _parse(data: Any, documents: Sequence[str]) -> Ranking:
        if not isinstance(data, dict):
            raise LLMError(f"malformed rerank response: {str(data)[:200]}")
        if "error" in data:
            raise LLMError(f"provider error: {data['error']}")
        results = data.get("results")
        if not isinstance(results, list) or not results:
            raise LLMError(f"malformed rerank response (no results): {str(data)[:200]}")
        seen: set[int] = set()
        ranked: list[Ranked] = []
        for item in results:
            if not isinstance(item, dict):
                raise LLMError(f"malformed rerank result: {str(item)[:100]}")
            index = item.get("index")
            if index is None:  # match by the echoed document text, if that is unambiguous
                doc = item.get("document")
                text = doc.get("text") if isinstance(doc, dict) else doc
                matches = [i for i, d in enumerate(documents) if d == text]
                index = matches[0] if len(matches) == 1 else None
            score = item.get("relevance_score", item.get("score"))
            if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(documents) or index in seen:
                raise LLMError(f"unusable rerank index {index!r} (documents: {len(documents)})")
            if isinstance(score, bool) or not isinstance(score, int | float) or not math.isfinite(score):
                raise LLMError(f"unusable rerank score {score!r}")
            seen.add(index)
            ranked.append(Ranked(index, float(score)))
        usage = data.get("usage")
        return Ranking(tuple(ranked), usage if isinstance(usage, dict) else {})
