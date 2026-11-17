"""CLI: ``uv run hanabi-llm --models modelA,modelB`` (use ``random`` for a bot).

Exit codes: 0 = completed, 2 = aborted by infrastructure fault (re-run it),
3 = aborted by a model fault (that seat is to blame).
"""

from __future__ import annotations

import argparse
import asyncio
import dataclasses
import json
import os
import secrets
import sys
from pathlib import Path
from typing import Any, TextIO

import httpx

from .engine import Game
from .faults import GameStatus
from .llm import OpenRouterClient
from .players import LLMPlayer, Player, RandomPlayer
from .prompt import load_prompt
from .runner import GameResult, TurnTrace, play_game

#: Neutral seat names. NEVER derive them from model ids: a model that can see
#: "claude" or "gpt" in a teammate's name could be biased by it.
SEAT_NAMES = ("Alice", "Bob", "Carol", "Dave", "Eve")


def _jsonable(obj: Any) -> Any:
    """Make dataclasses/enums serialisable for the JSONL log."""
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        return {k: _jsonable(v) for k, v in dataclasses.asdict(obj).items()}
    if isinstance(obj, tuple | list):
        return [_jsonable(v) for v in obj]
    return obj if isinstance(obj, str | int | float | bool | type(None) | dict) else str(obj)


def _write(log: TextIO | None, record: dict[str, Any]) -> None:
    if log is not None:
        log.write(json.dumps(_jsonable(record), default=str) + "\n")
        log.flush()  # a crash must not lose the transcript


async def run(args: argparse.Namespace) -> GameResult:
    models: list[str] = [m.strip() for m in args.models.split(",") if m.strip()]
    if not 2 <= len(models) <= len(SEAT_NAMES):
        sys.exit(f"need between 2 and {len(SEAT_NAMES)} models")
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and any(m != "random" for m in models):
        sys.exit("Set OPENROUTER_API_KEY to use real models (or pass only 'random').")

    seed = args.seed if args.seed is not None else secrets.randbelow(2**31)
    names = list(SEAT_NAMES[: len(models)])
    game = Game(names, seed=seed)
    prompt = load_prompt(args.prompt)
    log = args.log.open("w") if args.log else None

    config = {
        "seed": seed, "seats": dict(zip(names, models, strict=True)), "temperature": args.temperature,
        "max_tokens": args.max_tokens, "prompt": {"version": prompt.version, "sha256": prompt.sha256}, "max_calls": args.max_calls, "max_failed_actions": args.max_failed_actions,
    }
    _write(log, {"type": "header", **config})

    def on_turn(t: TurnTrace) -> None:
        print(f"turn {t.turn:>3}  {t.player:<6} {t.action.affordance} {dataclasses.asdict(t.action)}")
        _write(log, {"type": "turn", "turn": t.turn, "player": t.player, "action": t.action, "events": t.events,
                     "messages": t.messages, "tokens": [t.prompt_tokens, t.completion_tokens]})

    async with httpx.AsyncClient(timeout=120.0) as http:
        players: list[Player] = []
        for seat, (name, model) in enumerate(zip(names, models, strict=True)):
            if model == "random":
                players.append(RandomPlayer(name, seed=seed + seat))
            else:
                client = OpenRouterClient(model, api_key, http, temperature=args.temperature, max_tokens=args.max_tokens)
                players.append(LLMPlayer(name, client, game, seat, prompt=prompt, max_calls=args.max_calls,
                                         max_failed_actions=args.max_failed_actions))
        result = await play_game(game, players, on_turn=on_turn, turn_timeout=args.turn_timeout)
    _write(log, {"type": "result", "status": result.status, "score": result.score, "official_score": result.official_score,
                 "reason": result.reason, "counts_for_ranking": result.counts_for_ranking, "fault": result.fault})
    if log is not None:
        log.close()
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Play Hanabi between language models via OpenRouter.")
    ap.add_argument("--models", required=True, help="comma-separated OpenRouter model ids, one per seat (2-5); 'random' = bot")
    ap.add_argument("--seed", type=int, default=None, help="deal seed; a random one is chosen (and logged) if omitted")
    ap.add_argument("--log", type=Path, default=None, help="write a JSONL transcript here")
    ap.add_argument("--max-calls", type=int, default=8, help="model calls allowed per turn")
    ap.add_argument("--max-failed-actions", type=int, default=3, help="rejected actions per turn before a model fault")
    ap.add_argument("--prompt", default=None, help="system prompt version, e.g. v1 (default: latest)")
    ap.add_argument("--temperature", type=float, default=0.3)
    ap.add_argument("--max-tokens", type=int, default=4096)
    ap.add_argument("--turn-timeout", type=float, default=300.0, help="safety net in seconds; trips as an INFRA fault")
    args = ap.parse_args()
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
