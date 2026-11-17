"""Summarise a trace file: who called what, and exactly what was rejected.

    uv run hanabi-trace trace.jsonl

Answers "what went wrong with this model?" without reading raw JSON: per player, how many model
calls and tool calls, how many replies had no tool call at all, how many actions were rejected,
and then every rejected attempt in words (the tool, its arguments, and the error the model was
given). A turn that ended the game as a fault is included and marked.
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


@dataclass(frozen=True, slots=True)
class Incident:
    turn: int
    player: str
    text: str


def read_trace(path: Path) -> list[Row]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


def summarise(rows: list[Row]) -> tuple[dict[str, PlayerStats], list[Incident]]:
    """Per-player counts, and every rejected attempt or reply without a tool call, in order."""
    action_tools = {n for n, a in load_ontology().affordances.items() if a.ends_turn}
    stats: dict[str, PlayerStats] = {}
    incidents: list[Incident] = []
    for row in rows:
        who, turn = row["player"], row["turn"]
        st = stats.setdefault(who, PlayerStats())
        st.turns += 1
        if row.get("tokens"):
            st.tokens_in += row["tokens"][0]
            st.tokens_out += row["tokens"][1]
        calls: dict[str, tuple[str, str]] = {}  # tool_call_id -> (tool name, arguments)
        for msg in row.get("messages", []):
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
    return stats, incidents


def format_report(rows: list[Row]) -> str:
    stats, incidents = summarise(rows)
    lines = [f"{'player':<8} {'turns':>5} {'calls':>5} {'no-tool':>7} {'rejected':>8} {'read-err':>8} {'tokens in/out':>14}   tools"]
    for who, s in stats.items():
        used = ", ".join(f"{n} x{c}" for n, c in s.tools.most_common())
        lines.append(f"{who:<8} {s.turns:>5} {s.model_calls:>5} {s.no_tool_replies:>7} {s.rejected_actions:>8} {s.read_errors:>8} "
                     f"{f'{s.tokens_in}/{s.tokens_out}':>14}   {used}")
    lines.append("")
    if incidents:
        lines.append("Rejected attempts and replies without a tool call:")
        lines += [f"  T{i.turn} {i.player}: {i.text}" for i in incidents]
    else:
        lines.append("No rejected attempts and no replies without a tool call.")
    for who, s in stats.items():
        for fault in s.faults:
            lines.append(f"FAULT  {who}  {fault}")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description="Summarise a hanabi-llm trace file.")
    ap.add_argument("trace", type=Path, help="the JSONL written by `hanabi-llm --trace`")
    args = ap.parse_args()
    print(format_report(read_trace(args.trace)))


if __name__ == "__main__":
    main()
