"""Measure the size of the candidate-action set per player count (the table in docs/RERANK.md).

    uv run python scripts/action_space.py [games-per-count]

Prints, per number of players: the formulas (every receiver x value combination, and the most hints
that can touch a card), and what real games show: the number of legal actions per turn and the
number of turns per game (one rerank call per turn). Games are played by the heuristic demo bot,
which runs through the whole deck, so hands and token counts are realistic.
"""

from __future__ import annotations

import asyncio
import statistics
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from demo_game import Heuristic  # noqa: E402

from hanabi_llm.actions import action_space  # noqa: E402
from hanabi_llm.engine import Game  # noqa: E402
from hanabi_llm.ontology import load_ontology  # noqa: E402
from hanabi_llm.runner import play_game  # noqa: E402


async def measure(players: int, games: int) -> tuple[list[int], list[int]]:
    per_turn: list[int] = []
    lengths: list[int] = []
    for seed in range(1, games + 1):
        names = ["Alice", "Bob", "Carol", "Dave", "Eve"][:players]
        game = Game(names, seed=seed)
        bots = [Heuristic(n) for n in names]

        class Counting:
            def __init__(self, inner: Heuristic) -> None:
                self.name, self._inner = inner.name, inner

            async def take_turn(self, ctx):  # type: ignore[no-untyped-def]
                per_turn.append(len(ctx.legal))
                return await self._inner.take_turn(ctx)

        result = await play_game(game, [Counting(b) for b in bots])  # type: ignore[misc]
        lengths.append(result.turns)
    return per_turn, lengths


def main() -> None:
    games = int(sys.argv[1]) if len(sys.argv) > 1 else 5
    rules = load_ontology().rules
    print(f"| Players | Hand | Candidates, every combination | Max legal | Legal per turn (mean, min-max) | Turns per game (= rerank calls) |")
    print("|---|---|---|---|---|---|")
    for n in range(2, 6):
        space = action_space(rules, n)
        per_turn, lengths = asyncio.run(measure(n, games))
        print(f"| {n} | {space.hand_size} | {space.total_all} | {space.total_max_legal} | {statistics.mean(per_turn):.1f} ({min(per_turn)}-{max(per_turn)}) "
              f"| {statistics.mean(lengths):.0f} ({min(lengths)}-{max(lengths)}) |")


if __name__ == "__main__":
    main()
