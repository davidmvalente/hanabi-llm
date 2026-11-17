"""Minimal async chat-completions client for OpenRouter (OpenAI-compatible).

Kept deliberately small: one method, ``complete``, that sends messages and tool
definitions and returns the assistant's text and tool calls. Anything that
satisfies the ``ChatClient`` protocol can stand in (see tests).
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass
from typing import Any, Protocol

import httpx

JSON = dict[str, Any]

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
_RETRY_STATUS = frozenset({408, 429, 500, 502, 503, 504})


class LLMError(Exception):
    """The model provider failed or returned something unusable."""


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
        for attempt in range(self._max_attempts):
            try:
                resp = await self._http.post(self._url, json=body, headers=headers)
            except httpx.TransportError as exc:
                last_error = f"transport error: {exc}"
            else:
                if resp.status_code in _RETRY_STATUS:
                    last_error = f"HTTP {resp.status_code}"
                elif resp.status_code != 200:
                    raise LLMError(f"HTTP {resp.status_code} from OpenRouter: {resp.text[:300]}")
                else:
                    return self._parse(resp.json())
            await asyncio.sleep(min(2.0**attempt, 20.0))
        raise LLMError(f"gave up after {self._max_attempts} attempts: {last_error}")

    @staticmethod
    def _parse(data: JSON) -> Completion:
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
        )
