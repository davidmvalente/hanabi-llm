"""The state tool: six sections that partition what a player may see."""

from __future__ import annotations

import pytest

from conftest import B, G, R, Y, rig
from hanabi_llm.engine import Game
from hanabi_llm.model import DiscardCard, GiveHint, IllegalAction, PlayCard
from hanabi_llm.players import RandomPlayer
from hanabi_llm.runner import play_game
from hanabi_llm.states import SECTIONS, fuse_pool, hint_pool, state_text
from hanabi_llm.views import build_view


def text(game: Game, seat: int = 0, section: str | None = None) -> str:
    return state_text(build_view(game, seat), game.rules, section)


def test_sections_partition_the_state(game2: Game) -> None:
    """The full state is exactly the sections in order: nothing missing, nothing twice."""
    parts = [text(game2, 0, s) for s in SECTIONS]
    assert text(game2, 0) == "\n\n".join(parts)
    assert all(p.startswith(f"## {s}\n") for p, s in zip(parts, SECTIONS, strict=True))
    assert text(game2, 0).count("## ") == len(SECTIONS)


def test_unknown_section_is_an_error(game2: Game) -> None:
    with pytest.raises(IllegalAction, match="Sections"):
        text(game2, 0, "everything")


def test_token_pools_use_one_vocabulary(game2: Game) -> None:
    rig(game2, [[(R, 1)] * 5, [(B, 1)] * 5], deck=[(Y, 2)] * 10)
    start = text(game2, 0, "tokens")
    assert "Hint tokens: available 8, spent 0, capacity 8." in start
    assert "Fuse tokens: available 3, spent 0, capacity 3." in start
    game2.apply(0, GiveHint(1, color=B))
    assert "Hint tokens: available 7, spent 1, capacity 8." in text(game2, 0, "tokens")
    game2.apply(1, DiscardCard(0))   # a refund just raises `available`; nothing else to track
    assert "Hint tokens: available 8, spent 0, capacity 8." in text(game2, 0, "tokens")
    game2.apply(0, PlayCard(0))      # red 1 on an empty red stack is a success: no fuse lost
    assert "Fuse tokens: available 3, spent 0, capacity 3." in text(game2, 0, "tokens")


def test_a_misfire_spends_a_fuse(game2: Game) -> None:
    rig(game2, [[(R, 3)] * 5, [(B, 1)] * 5], deck=[(Y, 2)] * 10)
    game2.apply(0, PlayCard(0))
    assert "Fuse tokens: available 2, spent 1, capacity 3." in text(game2, 1, "tokens")


@pytest.mark.parametrize("seed", range(5))
async def test_pools_always_match_the_engine(seed: int) -> None:
    """available + spent == capacity, and the view agrees with the engine, on every turn."""
    names = ["a", "b", "c"]
    game = Game(names, seed=seed)

    def check(_trace: object) -> None:
        for seat in range(3):
            view = build_view(game, seat)
            for pool, truth in ((hint_pool(view, game.rules), game.state.hints), (fuse_pool(view, game.rules), game.state.fuses)):
                assert pool.available == truth and pool.available + pool.spent == pool.capacity

    await play_game(game, [RandomPlayer(n, seed + i) for i, n in enumerate(names)], on_turn=check)


def test_turn_section_uses_the_existing_turn_state(game2: Game) -> None:
    assert "it is YOUR turn" in text(game2, 0, "turn")
    assert "it is Ann's turn" in text(game2, 1, "turn")
    assert "Cards left in the deck: 40." in text(game2, 0, "turn")


def test_final_round_and_game_over_appear_in_turn(game2: Game) -> None:
    rig(game2, [[(R, 1)] * 5, [(B, 1)] * 5], deck=[(Y, 1)])
    game2.apply(0, DiscardCard(0))
    assert "FINAL ROUND" in text(game2, 1, "turn")
    game2.apply(1, DiscardCard(0))
    game2.apply(0, DiscardCard(0))
    assert "GAME OVER" in text(game2, 0, "turn")


def test_fireworks_show_what_is_needed_next(game2: Game) -> None:
    game2.state.stacks[R], game2.state.stacks[G] = 2, 5
    out = text(game2, 0, "fireworks")
    assert "red: 2, next needed: red 3" in out and "green: 5, complete" in out and "Score 7/25." in out


def test_hands_show_teammates_but_never_your_own(game2: Game) -> None:
    rig(game2, [[(R, 1), (B, 2)], [(G, 3), (Y, 4)]])
    out = text(game2, 0, "hands")
    assert "green 3" in out and "yellow 4" in out
    assert "red 1" not in out and "blue 2" not in out


def test_discard_and_history(game2: Game) -> None:
    rig(game2, [[(R, 3)] * 5, [(B, 1)] * 5], deck=[(Y, 2)] * 10)
    assert "empty" in text(game2, 0, "discard")
    game2.apply(0, PlayCard(0))  # misfire: the red 3 is discarded for everyone to see
    assert "red 3" in text(game2, 1, "discard")
    assert "MISFIRE" in text(game2, 1, "history")


def test_history_keeps_only_the_most_recent_events() -> None:
    game = Game(["Ann", "Bob"], seed=0)
    rig(game, [[(R, 1)] * 5, [(B, 1)] * 5], deck=[(Y, 2)] * 40)
    for i in range(30):
        game.apply(i % 2, DiscardCard(0))
    assert text(game, 0, "history").startswith("## history\nLast 20 event(s):")
