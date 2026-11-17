"""Timestamps and per-call tracing: when things happened, how long each model call took, who served it."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import httpx
import pytest

import hanabi_llm.llm as llm
from hanabi_llm.clock import now_iso, parse_iso
from hanabi_llm.llm import LLMError, OpenRouterClient
from hanabi_llm.trace_report import format_report, summarise_all



def test_timestamps_are_utc_iso_milliseconds_and_parse_back() -> None:
    ts = now_iso()
    assert ts.endswith("Z") and len(ts) == len("2026-01-01T00:00:00.000Z")
    assert parse_iso(ts).utcoffset().total_seconds() == 0  # type: ignore[union-attr]


async def test_a_call_reports_latency_retries_finish_reason_and_who_served_it(monkeypatch: pytest.MonkeyPatch) -> None:
    sent = {"n": 0}

    def handler(_: httpx.Request) -> httpx.Response:
        sent["n"] += 1
        if sent["n"] == 1:
            return httpx.Response(429)
        return httpx.Response(200, json={
            "id": "gen-1", "model": "x/y-served", "provider": "SomeHost",
            "choices": [{"finish_reason": "length", "message": {"content": "cut off", "tool_calls": None}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "completion_tokens_details": {"reasoning_tokens": 2}},
        })

    real_sleep = asyncio.sleep
    monkeypatch.setattr(llm.asyncio, "sleep", lambda *_: real_sleep(0))
    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
        c = await OpenRouterClient("x/y", "k", http).complete([], [])
    meta = c.meta
    assert meta is not None and meta.attempts == 2 and meta.finish_reason == "length"
    assert meta.provider == "SomeHost" and meta.response_model == "x/y-served" and meta.request_id == "gen-1"
    assert meta.retries == ({"status": 429, "waited_s": 1.0},) and meta.latency_s >= 0 and meta.started.endswith("Z")
    assert json.loads(json.dumps(meta.to_dict()))["usage"]["completion_tokens_details"]["reasoning_tokens"] == 2


async def test_a_failed_call_is_timed_too(monkeypatch: pytest.MonkeyPatch) -> None:
    real_sleep = asyncio.sleep
    monkeypatch.setattr(llm.asyncio, "sleep", lambda *_: real_sleep(0))
    async with httpx.AsyncClient(transport=httpx.MockTransport(lambda r: httpx.Response(503))) as http:
        with pytest.raises(LLMError) as info:
            await OpenRouterClient("x/y", "k", http, max_attempts=3).complete([], [])
    call = info.value.call_info()
    assert call is not None and call["attempts"] == 3 and [r["status"] for r in call["retries"]] == [503, 503, 503]
    assert "waited_s" not in call["retries"][-1]       # no waiting after the last attempt


def call(latency: float, **extra: Any) -> dict[str, Any]:
    return {"call": {"started": "2026-01-01T00:00:00.000Z", "latency_s": latency, "attempts": 1, "retries": [], **extra}}


def test_the_report_shows_where_the_time_went() -> None:
    rows = [
        {"turn": 1, "player": "Alice", "action": None, "tokens": [1, 1], "duration_s": 30.0, "messages": [
            call(10.0, provider="P1", response_model="m/a", finish_reason="tool_calls"),
            call(20.0, provider="P2", response_model="m/a", finish_reason="length", retries=[{"status": 429, "waited_s": 4.0}])]},
    ]
    stats, _, slow = summarise_all(rows)
    a = stats["Alice"]
    assert a.latencies == [10.0, 20.0] and a.turn_time_s == 30.0 and a.cut == 1 and (a.retries, a.retry_wait_s) == (1, 4.0)
    assert slow[0].latency_s == 20.0 or max(s.latency_s for s in slow) == 20.0
    report = format_report(rows)
    assert "m/a via P1 x1" in report and "m/a via P2 x1" in report
    assert "1 reply(ies) cut off by the token limit" in report and "Slowest calls:" in report
    assert "20.0 s  T1 Alice" in report
