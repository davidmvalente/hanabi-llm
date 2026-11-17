from __future__ import annotations

from conftest import B, G, R, W, Y, rig
from hanabi_llm.engine import Game
from hanabi_llm.model import GiveHint, PlayCard
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
    game2.apply(0, GiveHint(1, color=B))
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


def test_card_ids_do_not_encode_identity() -> None:
    """Regression: ids once ran in colour/rank order, leaking every hidden card."""
    sequences = set()
    for seed in range(10):
        g = Game(["Ann", "Bob"], seed=seed)
        cards = sorted((c for h in g.state.hands for c in h), key=lambda c: c.uid)
        identities = [(c.color.value, c.rank) for c in cards]
        assert identities != sorted(identities)          # not ordered by identity
        sequences.add(tuple(identities))
    assert len(sequences) == 10                          # and different for every deal
    ids_by_seed = {seed: Game(["Ann", "Bob"], seed=seed).state.hands[0][0].uid for seed in range(10)}
    first_cards = {str(Game(["Ann", "Bob"], seed=s).state.hands[0][0]) for s in range(10)}
    assert len(first_cards) > 3                          # card #uid is not a fixed identity
    assert ids_by_seed                                   # (ids exist; they just carry no information)
