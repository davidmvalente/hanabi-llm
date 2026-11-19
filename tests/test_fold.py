"""Graphs as a fold: the feed rebuilds the game, and projection commutes with the fold.

    fold(feed[:t])                  == the engine's true state after turn t
    fold(project(feed[:t], player)) == that player's view after turn t
"""

from __future__ import annotations

import json
from typing import Any

import pytest

from hanabi_llm.engine import Game
from hanabi_llm.feed import FeedWriter, event_record, header_record, project_record, read_feed
from hanabi_llm.fold import FoldState, fold, initial_state, step
from hanabi_llm.model import PlayCard
from hanabi_llm.players import RandomPlayer, TurnContext, TurnResult
from hanabi_llm.runner import play_game
from hanabi_llm.views import PlayerView, build_view

Comparable = dict[str, Any]


def from_fold(s: FoldState) -> Comparable:
    return {
        "hands": [[(c.color, c.rank, c.known_color, c.known_rank, sorted(c.not_colors), sorted(c.not_ranks)) for c in hand]
                  for hand in s.hands],
        "stacks": dict(s.stacks), "discard": [(c["color"], c["rank"]) for c in s.discard],
        "hints": s.hints, "fuses": s.fuses, "deck_size": s.deck_size, "final_turns_left": s.final_turns_left,
        "over_reason": s.over_reason, "current": s.current, "turn": s.turn,
    }


def from_view(v: PlayerView) -> Comparable:
    return {
        "hands": [[(cv.card.color.value if cv.card else None, cv.card.rank if cv.card else None,
                    cv.knowledge.color.value if cv.knowledge.color else None, cv.knowledge.rank,
                    sorted(c.value for c in cv.knowledge.not_colors), sorted(cv.knowledge.not_ranks)) for cv in hand]
                  for hand in v.hands],
        "stacks": {c.value: h for c, h in v.stacks.items()}, "discard": [(c.color.value, c.rank) for c in v.discard],
        "hints": v.hints, "fuses": v.fuses, "deck_size": v.deck_size, "final_turns_left": v.final_turns_left,
        "over_reason": v.over_reason, "current": v.current, "turn": v.turn,
    }


def from_truth(game: Game) -> Comparable:
    """The engine's full state, in the same shape (nobody is blinded)."""
    st = game.state
    return {
        "hands": [[(c.color.value, c.rank, k.color.value if k.color else None, k.rank, sorted(x.value for x in k.not_colors),
                    sorted(k.not_ranks)) for c, k in zip(hand, st.knowledge[seat], strict=True)] for seat, hand in enumerate(st.hands)],
        "stacks": {c.value: h for c, h in st.stacks.items()}, "discard": [(c.color.value, c.rank) for c in st.discard],
        "hints": st.hints, "fuses": st.fuses, "deck_size": len(st.deck), "final_turns_left": st.final_turns_left,
        "over_reason": st.over_reason, "current": st.current, "turn": st.turn,
    }


class Cautious(RandomPlayer):
    """Never plays, so no fuse is ever lost: the game runs until the deck is exhausted and the final round."""

    async def take_turn(self, ctx: TurnContext) -> TurnResult:
        safe = tuple(a for a in ctx.legal if not isinstance(a, PlayCard))
        return await super().take_turn(TurnContext(ctx.seat, ctx.view, ctx.toolbox, safe))


async def check_whole_game(seed: int, n: int, player: type[RandomPlayer]) -> Game:
    names = [f"p{i}" for i in range(n)]
    game = Game(names, seed=seed)
    regime = game.ontology.regime
    feed = [header_record(game, {}, {})]

    def check(_trace: object = None) -> None:
        assert from_fold(fold(feed)) == from_truth(game)                                     # omniscient
        for seat in range(n):
            observed = fold(project_record(r, seat, regime) for r in feed)                     # fold of the projection ...
            assert from_fold(observed) == from_view(build_view(game, seat))                    # ... is the view

    def on_turn(trace: Any) -> None:
        feed.extend(event_record(e) for e in trace.events)
        check()

    check()
    await play_game(game, [player(nm, seed=seed + i) for i, nm in enumerate(names)], on_turn=on_turn)
    assert game.is_over
    return game


@pytest.mark.parametrize(("seed", "n"), [(1, 2), (2, 3), (3, 4), (4, 5), (5, 2), (6, 3), (7, 4), (8, 5)])
async def test_the_fold_equals_the_truth_and_projection_commutes_with_it(seed: int, n: int) -> None:
    """After every turn of whole games (they end by misfires), for every player count."""
    await check_whole_game(seed, n, RandomPlayer)


@pytest.mark.parametrize(("seed", "n"), [(1, 2), (2, 3), (3, 4), (4, 5)])
async def test_the_same_holds_through_the_final_round(seed: int, n: int) -> None:
    """Games that exhaust the deck, so the final-round clock and the draws of the last cards are covered."""
    game = await check_whole_game(seed, n, Cautious)
    assert "final round" in (game.state.over_reason or ""), "the game must reach the end of the deck"
    assert game.state.final_turns_left == 0


def test_projection_hides_only_what_the_regime_hides() -> None:
    game = Game(["Ann", "Bob"], seed=2)
    regime = game.ontology.regime
    header = header_record(game, {}, {})
    seen = project_record(header, 0, regime)
    assert seen["deal"]["hands"][0] == [None] * 5                                                  # my own cards: there, identity hidden
    assert seen["deal"]["hands"][1] == header["deal"]["hands"][1]                                  # my teammate's
    draw = {"type": "event", "kind": "draw", "turn": 1, "actor": 0, "card": {"color": "red", "rank": 3}}
    assert project_record(draw, 0, regime)["card"] is None                                         # I do not see my draw
    assert project_record(draw, 1, regime) == draw                                                 # my teammate does
    hint = {"type": "event", "kind": "hint", "turn": 1, "actor": 1, "receiver": 0, "attribute": "color", "value": "red", "targets": [0]}
    assert project_record(hint, 0, regime) == hint                                                 # public


def test_a_feed_that_contradicts_itself_is_refused() -> None:
    game = Game(["Ann", "Bob"], seed=2)
    header = header_record(game, {}, {})
    state = initial_state(header)
    real = header["deal"]["hands"][0][0]
    other = {"color": "red" if real["color"] != "red" else "blue", "rank": real["rank"]}      # not the card that is in that slot
    wrong = {"type": "event", "kind": "discard", "turn": 1, "actor": 0, "slot": 0, "card": other}
    with pytest.raises(ValueError, match="inconsistent"):
        step(state, wrong)


def test_a_hidden_card_is_not_checked_against_the_event_that_reveals_it() -> None:
    """In a projected feed a player's own cards are unknown; playing one reveals it, and that is accepted."""
    game = Game(["Ann", "Bob"], seed=2)
    mine = fold([project_record(header_record(game, {}, {}), 0, game.ontology.regime)])
    assert all(c.color is None for c in mine.hands[0])
    step(mine, {"type": "event", "kind": "discard", "turn": 1, "actor": 0, "slot": 0, "card": {"color": "red", "rank": 1}})
    assert mine.discard == [{"color": "red", "rank": 1}] and len(mine.hands[0]) == 4


def test_an_older_feed_that_still_carries_card_ids_folds_the_same() -> None:
    """Schema 1 wrote {"uid", "color", "rank"}; the ids are ignored, so those feeds still play back."""
    game = Game(["Ann", "Bob"], seed=2)
    new = header_record(game, {}, {})
    old = {**new, "schema": 1, "deal": {**new["deal"], "hands": [[{"uid": 100 + i, **c} for i, c in enumerate(h)] for h in new["deal"]["hands"]]}}
    assert from_fold(fold([old])) == from_fold(fold([new]))
    drawn = {"type": "event", "kind": "draw", "turn": 1, "actor": 0, "card": {"uid": 7, "color": "red", "rank": 2}}
    assert from_fold(fold([old, drawn])) == from_fold(fold([new, {**drawn, "card": {"color": "red", "rank": 2}}]))


def test_the_feed_is_plain_jsonl_that_survives_a_file(tmp_path: Any) -> None:
    game = Game(["Ann", "Bob"], seed=2)
    path = tmp_path / "g.jsonl"
    writer = FeedWriter(path)
    records = [header_record(game, {"Ann": "m1", "Bob": "m2"}, {"temperature": 0.3})]
    for r in records:
        writer.write(r)
    assert json.loads(path.read_text().splitlines()[0])["names"] == ["Ann", "Bob"]  # flushed before close
    writer.close()
    assert [{k: v for k, v in r.items() if k != "ts"} for r in read_feed(path)] == records
    assert all(r["ts"].endswith("Z") for r in read_feed(path))  # each line is stamped with its UTC write time
