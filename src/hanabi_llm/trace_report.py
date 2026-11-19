"""Summarise a trace file: who called what, and exactly what was rejected.

    uv run hanabi-trace trace.jsonl

Answers "what went wrong with this model?" and "why is it slow?" without reading raw JSON: per
player, how many model calls and tool calls, how long they took (total, average, slowest), how many
were retried or cut off by the token limit, who actually served them, how many replies had no tool
call at all, how many actions were rejected, and then every rejected attempt in words (the tool, its
arguments, and the error the model was given). A turn that ended the game as a fault is included
and marked.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .ontology import load_ontology

Row = dict[str, Any]


def _brief(text: str, limit: int = 160) -> str:
    flat = " ".join(str(text).split())
    return flat if len(flat) <= limit else flat[: limit - 1] + "…"


@dataclass(slots=True)
class PlayerStats:
    turns: int = 0
    model_calls: int = 0
    tools: Counter[str] = field(default_factory=Counter)
    no_tool_replies: int = 0
    rejected_actions: int = 0
    read_errors: int = 0
    tokens_in: int = 0
    tokens_out: int = 0
    faults: list[str] = field(default_factory=list)
    latencies: list[float] = field(default_factory=list)   # seconds per model call, retries and backoff included
    turn_time_s: float = 0.0                                # seconds per turn, summed
    retries: int = 0
    retry_wait_s: float = 0.0
    cut: int = 0                                            # replies cut off by the token limit (finish_reason "length")
    reasoning_tokens: int = 0
    served_by: Counter[str] = field(default_factory=Counter)  # "model via provider" -> calls


@dataclass(frozen=True, slots=True)
class SlowCall:
    latency_s: float
    turn: int
    player: str
    detail: str


@dataclass(frozen=True, slots=True)
class Incident:
    turn: int
    player: str
    text: str


def read_trace(path: Path) -> list[Row]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def _record_call(st: PlayerStats, slow: list[SlowCall], call: Row, turn: int, who: str) -> None:
    """Fold one ``{"call": {...}}`` trace entry (a good call or a failed one) into the statistics."""
    latency = call.get("latency_s")
    retries = call.get("retries") or []
    st.retries += len(retries)
    st.retry_wait_s += sum(r.get("waited_s", 0.0) for r in retries)
    if call.get("finish_reason") == "length":
        st.cut += 1
    usage = call.get("usage") or {}
    st.reasoning_tokens += int((usage.get("completion_tokens_details") or {}).get("reasoning_tokens") or 0)
    if call.get("provider") or call.get("response_model"):
        st.served_by[f"{call.get('response_model') or '?'} via {call.get('provider') or '?'}"] += 1
    if isinstance(latency, int | float):
        st.latencies.append(float(latency))
        bits = [f"{len(retries) + 1} attempt(s)"] + ([f"finish={call['finish_reason']}"] if call.get("finish_reason") else []) \
            + ([f"via {call['provider']}"] if call.get("provider") else []) + (["FAILED"] if call.get("error") else [])
        slow.append(SlowCall(float(latency), turn, who, ", ".join(bits)))


def summarise(rows: list[Row]) -> tuple[dict[str, PlayerStats], list[Incident]]:
    """Per-player counts, and every rejected attempt or reply without a tool call, in order."""
    stats, incidents, _ = summarise_all(rows)
    return stats, incidents


def summarise_all(rows: list[Row]) -> tuple[dict[str, PlayerStats], list[Incident], list[SlowCall]]:
    """Like ``summarise``, plus every timed call (for the slowest-calls list)."""
    action_tools = {n for n, a in load_ontology().affordances.items() if a.ends_turn}
    stats: dict[str, PlayerStats] = {}
    incidents: list[Incident] = []
    slow: list[SlowCall] = []
    for row in rows:
        who, turn = row["player"], row["turn"]
        st = stats.setdefault(who, PlayerStats())
        st.turns += 1
        st.turn_time_s += float(row.get("duration_s") or 0.0)
        if row.get("tokens"):
            st.tokens_in += row["tokens"][0]
            st.tokens_out += row["tokens"][1]
        calls: dict[str, tuple[str, str]] = {}  # tool_call_id -> (tool name, arguments)
        for msg in row.get("messages", []):
            if isinstance(msg.get("call"), dict):
                _record_call(st, slow, msg["call"], turn, who)
            if msg.get("role") == "assistant":
                st.model_calls += 1
                tool_calls = msg.get("tool_calls") or []
                if not tool_calls:
                    st.no_tool_replies += 1
                    said = _brief(msg.get("content") or "", 100)
                    incidents.append(Incident(turn, who, f'no tool call, it said: "{said}"' if said else "no tool call and an empty reply"))
                for tc in tool_calls:
                    name, args = tc["function"]["name"], tc["function"].get("arguments", "")
                    calls[tc["id"]] = (name, args)
                    st.tools[name] += 1
            elif msg.get("role") == "tool" and str(msg.get("content", "")).startswith("Error:"):
                name, args = calls.get(msg.get("tool_call_id", ""), ("?", ""))
                if name in action_tools or name == "?":
                    st.rejected_actions += 1
                    incidents.append(Incident(turn, who, f"{name}({_brief(args, 80)}) -> {_brief(msg['content'])}"))
                else:
                    st.read_errors += 1
        if row.get("fault"):
            detail = row["fault"]["detail"]
            headline = detail.split(": ", 1)[0] if "attempts in one turn: " in detail else detail   # the attempts are listed above
            st.faults.append(f"T{turn} {row['fault']['kind']}: {headline}")
    return stats, incidents, slow


def _secs(x: float) -> str:
    return f"{x:.1f}" if x else "-"


def format_report(rows: list[Row]) -> str:
    stats, incidents, slow = summarise_all(rows)
    lines = [f"{'player':<8} {'turns':>5} {'calls':>5} {'time s':>7} {'avg s':>6} {'max s':>6} {'retry':>5} {'cut':>3} "
             f"{'no-tool':>7} {'rejected':>8} {'read-err':>8} {'tokens in/out':>14}   tools"]
    for who, s_ in stats.items():
        used = ", ".join(f"{n} x{c}" for n, c in s_.tools.most_common())
        total = s_.turn_time_s or sum(s_.latencies)
        avg = sum(s_.latencies) / len(s_.latencies) if s_.latencies else 0.0
        lines.append(f"{who:<8} {s_.turns:>5} {s_.model_calls:>5} {_secs(total):>7} {_secs(avg):>6} {_secs(max(s_.latencies, default=0.0)):>6} "
                     f"{s_.retries:>5} {s_.cut:>3} {s_.no_tool_replies:>7} {s_.rejected_actions:>8} {s_.read_errors:>8} "
                     f"{f'{s_.tokens_in}/{s_.tokens_out}':>14}   {used}")
    lines.append("")
    served = [(who, k, n) for who, s_ in stats.items() for k, n in s_.served_by.most_common()]
    if served:
        lines.append("Served by:")
        lines += [f"  {who}: {k} x{n}" for who, k, n in served]
    extras = [(who, s_) for who, s_ in stats.items() if s_.retries or s_.cut or s_.reasoning_tokens]
    for who, s_ in extras:
        notes = ([f"{s_.retries} retried attempt(s), {s_.retry_wait_s:.1f} s spent waiting to retry"] if s_.retries else []) \
            + ([f"{s_.cut} reply(ies) cut off by the token limit"] if s_.cut else []) \
            + ([f"{s_.reasoning_tokens} reasoning tokens"] if s_.reasoning_tokens else [])
        lines.append(f"  {who}: " + "; ".join(notes))
    if slow:
        lines.append("")
        lines.append("Slowest calls:")
        lines += [f"  {c.latency_s:>6.1f} s  T{c.turn} {c.player}: {c.detail}" for c in sorted(slow, key=lambda c: -c.latency_s)[:5]]
    lines.append("")
    if incidents:
        lines.append("Rejected attempts and replies without a tool call:")
        lines += [f"  T{i.turn} {i.player}: {i.text}" for i in incidents]
    else:
        lines.append("No rejected attempts and no replies without a tool call.")
    for who, s_ in stats.items():
        for fault in s_.faults:
            lines.append(f"FAULT  {who}  {fault}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarise a hanabi-llm trace file.")
    ap.add_argument("trace", type=Path, help="the JSONL written by `hanabi-llm --trace`")
    args = ap.parse_args()
    print(format_report(read_trace(args.trace)))


if __name__ == "__main__":
    main()
