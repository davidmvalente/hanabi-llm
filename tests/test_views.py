from __future__ import annotations

import pytest

from conftest import B, G, R, W, Y, rig
from hanabi_llm.engine import Game
from hanabi_llm.model import DiscardCard, GiveHint, PlayCard
from hanabi_llm.states import state_text
from hanabi_llm.views import build_view, view_graph

Q_IDENTITIES = "SELECT ?c WHERE { ?c hn:heldBy ?p ; hn:hasColor ?col }"


def _held_by(graph, seat: int) -> set[str]:  # type: ignore[no-untyped-def]
    q = f"SELECT ?c WHERE {{ ?c hn:heldBy ?p ; hn:hasColor ?col . ?p hn:seatIndex {seat} }}"
    return {str(r[0]) for r in graph.query(q)}


def test_own_identities_hidden_others_visible(game2: Game) -> None:
    rig(game2, [[(R, 1), (B, 2)], [(G, 3), (W, 4)]])
    v0 = build_view(game2, 0)
    g0 = view_graph(game2, v0)
    assert _held_by(g0, 0) == set()          # I cannot see my own cards
    assert len(_held_by(g0, 1)) == 2         # I can see Bob's
    g1 = view_graph(game2, build_view(game2, 1))
    assert _held_by(g1, 1) == set() and len(_held_by(g1, 0)) == 2


def test_hint_knowledge_is_visible_to_holder(game2: Game) -> None:
    rig(game2, [[(R, 1)] * 5, [(B, 1), (R, 2), (B, 3), (G, 4), (W, 5)]], deck=[(Y, 1)] * 10)
    game2.apply(0, GiveHint(1, B))
    text = state_text(build_view(game2, 1), game2.rules)
    assert "colour=blue" in text
    g = view_graph(game2, build_view(game2, 1))
    rows = list(g.query("SELECT ?c WHERE { ?c hn:knownColor hn:Blue }"))
    assert len(rows) == 2


def test_drawn_card_hidden_from_drawer_only(game2: Game) -> None:
    rig(game2, [[(R, 1)] * 5, [(B, 1)] * 5], deck=[(Y, 5)])
    game2.apply(0, PlayCard(0))
    v0, v1 = build_view(game2, 0), build_view(game2, 1)
    assert "yellow 5" not in state_text(v0, game2.rules)
    assert "yellow 5" in state_text(v1, game2.rules)  # Bob sees what Ann drew


def test_affordance_pointers_only_for_active_player(game2: Game) -> None:
    q = "SELECT ?f WHERE { ?p hn:seatIndex ?s ; hn:canInvoke ?a . ?a hn:functionName ?f . FILTER(?s = 0) }"
    active = {str(r[0]) for r in view_graph(game2, build_view(game2, 0)).query(q)}
    assert {"play_card", "give_hint", "discard_card"} <= active
    q1 = q.replace("?s = 0", "?s = 1")
    waiting = {str(r[0]) for r in view_graph(game2, build_view(game2, 1)).query(q1)}
    assert "play_card" not in waiting and "get_state" in waiting


def test_there_are_no_card_ids_and_the_deck_is_shuffled() -> None:
    """Regression: ids once ran in colour/rank order and leaked every hidden card. Now there are no ids at all."""
    import dataclasses

    from hanabi_llm.model import Card
    from hanabi_llm.views import CardView

    assert [f.name for f in dataclasses.fields(Card)] == ["color", "rank"]                    # a card is a (colour, rank) value
    assert [f.name for f in dataclasses.fields(CardView)] == ["slot", "card", "knowledge"]  # a view names a slot, not a card
    orders = set()
    for seed in range(10):
        deck = list(Game(["Ann", "Bob"], seed=seed).state.deck)
        identities = [(c.color.value, c.rank) for c in deck]
        assert identities != sorted(identities)                       # shuffled, not in colour/rank order
        orders.add(tuple(identities))
    assert len(orders) == 10                                          # and different for every deal


def test_cards_are_values_two_copies_are_equal_and_a_card_is_validated() -> None:
    from hanabi_llm.model import Card

    assert Card(B, 1) == Card(B, 1) and Card(B, 1) != Card(B, 2) and len({Card(B, 1), Card(B, 1), Card(R, 1)}) == 2
    assert str(Card(B, 3)) == "blue 3"
    for bad in ((B, 0), (B, -1), (B, True), ("blue", 1), (None, 1)):
        with pytest.raises(ValueError):
            Card(*bad)  # type: ignore[arg-type]


def test_no_graph_node_is_named_by_a_card_id() -> None:
    import re

    game = Game(["Ann", "Bob"], seed=2)
    game.apply(0, DiscardCard(0))
    names = {str(s).rsplit("#", 1)[1] for s in view_graph(game, build_view(game, 0)).subjects() if "#" in str(s)}
    assert not [n for n in names if re.fullmatch(r"card\d+", n)]
    assert {"hand0_0", "hand1_4", "discard_0", "event0_card"} <= names   # cards are named by where they are
