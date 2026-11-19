"""Minimal async chat-completions client for OpenRouter (OpenAI-compatible).

Kept deliberately small: one method, ``complete``, that sends messages and tool
definitions and returns the assistant's text and tool calls. Anything that
satisfies the ``ChatClient`` protocol can stand in (see tests).
"""

from __future__ import annotations

import asyncio
import time
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

from .clock import now_iso

JSON = dict[str, Any]

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class LLMError(Exception):
    """The model provider failed or returned something unusable.

    When the failure came from a call that was timed, ``attempts``, ``latency_s`` and ``retries``
    say how long it took and what was retried, so a failed call is as visible in the trace as a good one.
    """

    def __init__(self, message: str, *, attempts: int = 0, latency_s: float = 0.0, retries: Sequence[JSON] = (), started: str | None = None) -> None:
        super().__init__(message)
        self.attempts, self.latency_s, self.retries, self.started = attempts, latency_s, tuple(retries), started

    def call_info(self) -> JSON | None:
        """What to record in the trace about the failed call, or ``None`` if it was not timed."""
        if not self.attempts:
            return None
        return {"started": self.started, "latency_s": round(self.latency_s, 3), "attempts": self.attempts,
                "retries": list(self.retries), "error": str(self)}


@dataclass(frozen=True, slots=True)
class CallMeta:
    """How one model call went: when, how long, how many tries, and who actually answered."""

    started: str                # UTC ISO time the call began
    latency_s: float            # wall time of the whole call, retries and backoff included
    attempts: int               # HTTP attempts (1 = no retry)
    retries: tuple[JSON, ...]   # one entry per failed attempt: {"status" | "error", "waited_s"}
    http_status: int
    finish_reason: str | None   # "stop", "tool_calls", "length" (cut off by max_tokens) ...
    provider: str | None        # the upstream provider that served it (OpenRouter reports this)
    response_model: str | None  # the model actually used, which can differ from the one requested
    request_id: str | None
    usage: JSON                 # the provider's usage block, as received (tokens, cost, reasoning tokens ...)

    def to_dict(self) -> JSON:
        return {"started": self.started, "latency_s": round(self.latency_s, 3), "attempts": self.attempts,
                "retries": list(self.retries), "http_status": self.http_status, "finish_reason": self.finish_reason,
                "provider": self.provider, "response_model": self.response_model, "request_id": self.request_id,
                "usage": self.usage}


@dataclass(frozen=True, slots=True)
class ToolCall:
    id: str
    name: str
    arguments: str  # raw JSON string, exactly as the model produced it


@dataclass(frozen=True, slots=True)
class Completion:
    content: str | None
    tool_calls: tuple[ToolCall, ...]
    prompt_tokens: int = 0
    completion_tokens: int = 0
    meta: CallMeta | None = None

    def to_message(self) -> JSON:
        """The assistant message as it must be echoed back into the history."""
        msg: JSON = {"role": "assistant", "content": self.content or ""}
        if self.tool_calls:
            msg["tool_calls"] = [
                {"id": c.id, "type": "function", "function": {"name": c.name, "arguments": c.arguments}}
                for c in self.tool_calls
            ]
        return msg


class ChatClient(Protocol):
    model: str

    async def complete(self, messages: list[JSON], tools: list[JSON]) -> Completion: ...


class OpenRouterClient:
    """``ChatClient`` backed by OpenRouter. Retries transient HTTP failures."""

    def __init__(
        self,
        model: str,
        api_key: str,
        http: httpx.AsyncClient,
        *,
        temperature: float = 0.3,
        max_tokens: int = 4096,
        max_attempts: int = 4,
        url: str = OPENROUTER_URL,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._http = http
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._max_attempts = max_attempts
        self._url = url

    async def complete(self, messages: list[JSON], tools: list[JSON]) -> Completion:
        body: JSON = {
            "model": self.model,
            "messages": messages,
            "tools": tools,
            "tool_choice": "auto",
            "temperature": self._temperature,
            "max_tokens": self._max_tokens,
        }
        headers = {"Authorization": f"Bearer {self._api_key}", "X-Title": "hanabi-llm"}
        last_error = "unknown error"
        started, t0 = now_iso(), time.perf_counter()
        retries: list[JSON] = []

        def timing() -> JSON:
            return {"attempts": len(retries) + 1, "latency_s": time.perf_counter() - t0, "retries": retries, "started": started}

        for attempt in range(self._max_attempts):
            try:
                resp = await self._http.post(self._url, json=body, headers=headers)
            except httpx.TransportError as exc:
                last_error = f"transport error: {exc}"
                retries.append({"error": str(exc)[:120]})
            else:
                if resp.status_code in _RETRY_STATUS:
                    last_error = f"HTTP {resp.status_code}"
                    retries.append({"status": resp.status_code})
                elif resp.status_code != 200:
                    raise LLMError(f"HTTP {resp.status_code} from OpenRouter: {resp.text[:300]}", **timing())
                else:
                    data = resp.json()
                    try:
                        return self._parse(data, started, time.perf_counter() - t0, retries)
                    except LLMError as exc:
                        raise LLMError(str(exc), **timing()) from exc
            if attempt < self._max_attempts - 1:  # no point waiting after the last attempt
                wait = min(2.0**attempt, 20.0)
                retries[-1]["waited_s"] = wait
                await asyncio.sleep(wait)
        raise LLMError(f"gave up after {self._max_attempts} attempts: {last_error}", **timing() | {"attempts": self._max_attempts})

    @staticmethod
    def _parse(data: JSON, started: str = "", latency_s: float = 0.0, retries: Sequence[JSON] = ()) -> Completion:
        if "error" in data:
            raise LLMError(f"provider error: {data['error']}")
        try:
            message = data["choices"][0]["message"]
        except (KeyError, IndexError, TypeError) as exc:
            raise LLMError(f"malformed response: {str(data)[:300]}") from exc
        calls = tuple(
            ToolCall(
                id=str(c.get("id", f"call_{i}")),
                name=str(c["function"]["name"]),
                arguments=c["function"].get("arguments") or "{}",
            )
            for i, c in enumerate(message.get("tool_calls") or [])
        )
        usage = data.get("usage") or {}
        return Completion(
            content=message.get("content"),
            tool_calls=calls,
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            meta=CallMeta(
                started=started, latency_s=latency_s, attempts=len(retries) + 1, retries=tuple(retries), http_status=200,
                finish_reason=data["choices"][0].get("finish_reason"), provider=data.get("provider"),
                response_model=data.get("model"), request_id=data.get("id"), usage=usage if isinstance(usage, dict) else {},
            ),
        )
