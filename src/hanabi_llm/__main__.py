"""CLI: ``uv run hanabi-llm --models modelA,modelB`` (use ``random`` for a bot).

Two optional output files, both JSONL and flushed line by line:

  --log   the game FEED: header with the initial deal, one record per event,
          a final result. This is what the viewer follows (``hanabi-view``).
  --trace the model conversations (messages and token counts), one line per turn.

Exit codes: 0 = completed, 2 = aborted by infrastructure fault (re-run it),
3 = aborted by a model fault (that seat is to blame).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any

import httpx

from .engine import Game
from .faults import GameStatus
from .feed import FeedWriter, action_record, event_record, header_record, result_record
from .llm import OPENROUTER_URL, OpenRouterClient
from .ontology import available_knowledge, load_ontology
from .players import DEFAULT_MAX_FAILED_ACTIONS, LLMPlayer, Player, RandomPlayer
from .prompt import load_prompt
from .runner import GameResult, TurnTrace, play_game
from .views import narrate_event

#: Neutral seat names. NEVER derive them from model ids: a model that can see
#: "claude" or "gpt" in a teammate's name could be biased by it.
SEAT_NAMES = ("Alice", "Bob", "Carol", "Dave", "Eve")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Play Hanabi between language models via OpenRouter.")
    ap.add_argument("--models", required=True, help="comma-separated OpenRouter model ids, one per seat (2-5); 'random' = bot")
    ap.add_argument("--seed", type=int, default=None, help="deal seed; a random one is chosen (and logged) if omitted")
    ap.add_argument("--log", type=Path, default=None, help="write the game feed (JSONL) here; follow it with hanabi-view")
    ap.add_argument("--trace", type=Path, default=None, help="write the model conversations (JSONL) here")
    ap.add_argument("--max-calls", type=int, default=8, help="model calls allowed per turn")
    ap.add_argument("--max-failed-actions", type=int, default=DEFAULT_MAX_FAILED_ACTIONS, help="rejected actions per turn before a model fault")
    ap.add_argument("--knowledge", default="", help=f"comma-separated knowledge graphs to offer ({', '.join(available_knowledge())}); default: none, the baseline")
    ap.add_argument("--api-url", default=OPENROUTER_URL, help="chat-completions endpoint (OpenRouter by default; any OpenAI-compatible server works)")
    ap.add_argument("--prompt", default=None, help="system prompt version, e.g. v1 (default: latest)")
    ap.add_argument("--audit", action="store_true", help="SHACL-validate the truth and every view after each turn (slow; needs pyshacl)")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--turn-timeout", type=float, default=300.0, help="safety net in seconds; trips as an INFRA fault")
    return ap


async def run(args: argparse.Namespace) -> GameResult:
    models: list[str] = [m.strip() for m in args.models.split(",") if m.strip()]
    if not 2 <= len(models) <= len(SEAT_NAMES):
        sys.exit(f"need between 2 and {len(SEAT_NAMES)} models")
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and any(m != "random" for m in models):
        sys.exit("Set OPENROUTER_API_KEY to use real models (or pass only 'random').")

    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    names = list(SEAT_NAMES[: len(models)])
    onto = load_ontology([k.strip() for k in args.knowledge.split(",") if k.strip()])
    game = Game(names, seed=seed, ontology=onto)
    prompt = load_prompt(args.prompt)
    feed = FeedWriter(args.log) if args.log else None
    trace = FeedWriter(args.trace) if args.trace else None

    config = {
        "temperature": args.temperature, "max_tokens": args.max_tokens, "max_calls": args.max_calls,
        "max_failed_actions": args.max_failed_actions, "knowledge": list(onto.knowledge), "prompt": {"version": prompt.version, "sha256": prompt.sha256},
    }
    if feed:  # the header carries the initial deal, so it is written before anything is played
        feed.write(header_record(game, dict(zip(names, models, strict=True)), config))

    if args.audit:
        from .validate import audit  # needs pyshacl; imported only when asked for

        audit(game)  # the freshly dealt game must already conform

    def on_turn(t: TurnTrace) -> None:
        if args.audit:
            audit(game)  # a violation is a bug in this code, never a model's fault: fail loudly
        for ev in t.events:
            print(narrate_event(ev, names))
        if feed:
            for ev in t.events:
                feed.write(event_record(ev))
        if trace:
            trace.write({"turn": t.turn, "player": t.player, "action": action_record(t.action),
                         "messages": t.messages, "tokens": [t.prompt_tokens, t.completion_tokens]})

    try:
        async with httpx.AsyncClient(timeout=120.0) as http:
            players: list[Player] = []
            for seat, (name, model) in enumerate(zip(names, models, strict=True)):
                if model == "random":
                    players.append(RandomPlayer(name, seed=seed + seat))
                else:
                    client = OpenRouterClient(model, api_key, http, temperature=args.temperature, max_tokens=args.max_tokens, url=args.api_url)
                    players.append(LLMPlayer(name, client, game, seat, prompt=prompt, max_calls=args.max_calls,
                                             max_failed_actions=args.max_failed_actions))
            result = await play_game(game, players, on_turn=on_turn, turn_timeout=args.turn_timeout)
        if trace and result.fault is not None:
            # The turn that failed is the one worth reading, so it goes in the trace like any other turn.
            fault = result.fault
            trace.write({"turn": fault.turn, "player": fault.player, "action": None, "fault": {"kind": fault.kind.value, "detail": fault.detail},
                         "messages": list(fault.messages), "tokens": None})
        if feed:
            feed.write(result_record(result))
        return result
    finally:
        for f in (feed, trace):
            if f:
                f.close()


def main() -> None:
    args = build_parser().parse_args()
    result = asyncio.run(run(args))

    print(f"\nStatus: {result.status.value}. {result.reason}. Seed {result.seed}.")
    if result.status is GameStatus.COMPLETED:
        print(f"Stack score {result.score}/25, official score {result.official_score}, {result.turns} turns.")
        sys.exit(0)
    assert result.fault is not None
    blame = "MODEL fault" if result.status is GameStatus.ABORTED_MODEL_FAULT else "infrastructure fault (re-run)"
    print(f"{blame}: {result.fault.player} - {result.fault.kind.value}: {result.fault.detail}")
    print("This game does not count for ranking.")
    sys.exit(3 if result.status is GameStatus.ABORTED_MODEL_FAULT else 2)


if __name__ == "__main__":
    main()
