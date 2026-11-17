"""Candidate actions as strings, and how many there are as the player count grows."""

from __future__ import annotations

import re

import pytest

from conftest import B, G, R, W, Y, rig
from hanabi_llm.actions import action_space, action_text, candidate_actions
from hanabi_llm.engine import Game
from hanabi_llm.model import DiscardCard, GiveHint, PlayCard
from hanabi_llm.ontology import load_ontology
from hanabi_llm.views import build_view

NAMES = ["Alice", "Bob", "Carol"]
COLORS = [B, G, R, W, Y]  # the order the ontology lists them in


def candidates(game: Game, which: str = "legal", seat: int = 0):  # type: ignore[no-untyped-def]
    return candidate_actions(build_view(game, seat), game.rules, game.legal_actions(seat), which)


def test_the_wording() -> None:
    assert action_text(PlayCard(2), NAMES) == "play the card at position 2"
    assert action_text(DiscardCard(0), NAMES) == "discard the card at position 0"
    assert action_text(GiveHint(1, G), NAMES) == "hint Bob: green"
    assert action_text(GiveHint(2, 3), NAMES) == "hint Carol: rank 3"


@pytest.mark.parametrize("seed", range(6))
def test_the_legal_set_is_exactly_the_engines_legal_actions(seed: int) -> None:
    game = Game(NAMES, seed=seed)
    got = candidates(game)
    assert set(got) == set(game.legal_actions(0)) and len(got) == len(game.legal_actions(0))
    kinds = [type(a).__name__ for a in got]
    assert kinds == sorted(kinds, key=["PlayCard", "DiscardCard", "GiveHint"].index)      # canonical: plays, discards, hints
    assert [a for a in got if isinstance(a, GiveHint)] == sorted((a for a in got if isinstance(a, GiveHint)), key=lambda a: a.receiver)


def test_every_candidate_has_its_own_text_that_maps_back() -> None:
    for which in ("legal", "all"):
        game = Game(NAMES, seed=2)
        acts = candidates(game, which)
        texts = [action_text(a, NAMES) for a in acts]
        assert len(set(texts)) == len(texts)
        assert {t: a for t, a in zip(texts, acts, strict=True)}[texts[-1]] == acts[-1]


def test_the_all_set_is_every_receiver_and_value_combination() -> None:
    game = Game(NAMES, seed=2)
    acts = candidates(game, "all")
    hints = [a for a in acts if isinstance(a, GiveHint)]
    assert len(hints) == 2 * 10                                              # 2 receivers x (5 colours + 5 ranks)
    assert len(acts) == action_space(game.rules, 3).total_all == 30
    legal = set(game.legal_actions(0))
    assert legal < set(acts) and len(legal) < len(acts)                       # some combinations are impossible right now


def test_the_all_set_keeps_hints_when_no_token_is_left_the_legal_set_does_not() -> None:
    game = Game(NAMES, seed=2)
    game.rig(hints=0)
    assert not [a for a in candidates(game, "legal") if isinstance(a, GiveHint)]
    assert len([a for a in candidates(game, "all") if isinstance(a, GiveHint)]) == 20


def test_the_text_does_not_reveal_which_hints_are_legal() -> None:
    """A possible and an impossible hint are worded identically, in both candidate sets."""
    game = Game(NAMES, seed=2)
    legal = {action_text(a, NAMES) for a in candidates(game, "legal") if isinstance(a, GiveHint)}
    impossible = {action_text(a, NAMES) for a in candidates(game, "all") if isinstance(a, GiveHint)} - legal
    assert legal and impossible
    shape = re.compile(r"^hint (Bob|Carol): (blue|green|red|white|yellow|rank [1-5])$")
    assert all(shape.match(t) for t in legal | impossible)


@pytest.mark.parametrize("players", [2, 3, 4, 5])
def test_the_complexity_formulas_match_brute_force(players: int) -> None:
    rules = load_ontology().rules
    space = action_space(rules, players)
    names = [f"P{i}" for i in range(players)]
    h = space.hand_size
    game = Game(names, seed=1)
    assert len(candidates(game, "all")) == space.total_all                            # every combination, counted by building them
    # the worst case for legal actions: every hand shows as many distinct colours AND ranks as it can
    hand = [(COLORS[i % 5], i % 5 + 1) for i in range(h)]
    rig(game, [list(hand) for _ in range(players)])
    legal = game.legal_actions(0)
    assert len(legal) == space.total_max_legal
    assert space.plays_and_discards == 2 * h and space.total_all == space.plays_and_discards + space.hints_all


def test_the_documented_numbers() -> None:
    """The table in docs/RERANK.md."""
    rules = load_ontology().rules
    assert [(action_space(rules, n).hand_size, action_space(rules, n).total_all, action_space(rules, n).total_max_legal) for n in (2, 3, 4, 5)] == [
        (5, 20, 20), (5, 30, 30), (4, 38, 32), (4, 48, 40)]


def test_real_play_never_exceeds_the_bounds() -> None:
    import asyncio

    from hanabi_llm.players import RandomPlayer, TurnContext, TurnResult
    from hanabi_llm.runner import play_game

    seen: list[int] = []

    class Counting(RandomPlayer):
        async def take_turn(self, ctx: TurnContext) -> TurnResult:
            seen.append(len(ctx.legal))
            return await super().take_turn(ctx)

    for seed in range(5):
        game = Game(NAMES, seed=seed)
        asyncio.run(play_game(game, [Counting(n, seed + i) for i, n in enumerate(NAMES)]))
    space = action_space(load_ontology().rules, 3)
    assert seen and space.plays_and_discards <= min(seen) and max(seen) <= space.total_max_legal


@pytest.mark.parametrize("players", [2, 3, 4, 5])
def test_documents_stay_short(players: int) -> None:
    """docs/RERANK.md promises no document longer than 30 characters."""
    names = [f"P{i}" for i in range(players)]
    game = Game(names, seed=1)
    docs = [action_text(a, names) for a in candidate_actions(build_view(game, 0), game.rules, game.legal_actions(0), "all")]
    assert max(map(len, docs)) <= 30
