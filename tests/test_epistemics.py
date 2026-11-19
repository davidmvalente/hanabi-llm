"""The declared information regime: loaded from the ontology, interpreted by the views."""

from __future__ import annotations

from dataclasses import replace

import pytest
from rdflib import Graph

from conftest import B, R, Y, rig
from hanabi_llm.engine import Game
from hanabi_llm.model import DiscardCard, DrawEvent, PlayCard
from hanabi_llm.ontology import HN, Regime, Visibility, _load_regime, load_ontology
from hanabi_llm.states import state_text
from hanabi_llm.views import build_view, graph_namespace, view_graph


def open_hand_game() -> Game:
    """A variant in which every hand is public, to prove the regime is interpreted and not hard-coded."""
    onto = load_ontology()
    onto = replace(onto, regime=Regime(Visibility.PUBLIC, Visibility.PUBLIC, Visibility.PUBLIC, Visibility.HIDDEN_FROM_ALL))
    return Game(["Ann", "Bob"], seed=1, ontology=onto)


def test_the_standard_regime_is_declared_in_the_ontology() -> None:
    regime = load_ontology().regime
    assert (regime.hand, regime.discard, regime.stack, regime.deck) == (
        Visibility.HIDDEN_FROM_HOLDER, Visibility.PUBLIC, Visibility.PUBLIC, Visibility.HIDDEN_FROM_ALL)
    assert regime.hand_identity_visible(viewer=0, holder=1) and not regime.hand_identity_visible(viewer=1, holder=1)


@pytest.mark.parametrize(("zone", "wrong"), [
    (HN.Hand, HN.HiddenFromAll), (HN.DiscardPile, HN.HiddenFromAll), (HN.Stack, HN.HiddenFromHolder), (HN.Deck, HN.Public)])
def test_regimes_the_engine_cannot_honour_are_refused(zone, wrong) -> None:  # type: ignore[no-untyped-def]
    g = Graph()
    for t in load_ontology().graph:
        g.add(t)
    g.set((zone, HN.identityVisibility, wrong))
    with pytest.raises(ValueError, match="unsupported regime"):
        _load_regime(g)


def test_a_missing_declaration_is_refused() -> None:
    g = Graph()
    for t in load_ontology().graph:
        g.add(t)
    g.remove((HN.Deck, HN.identityVisibility, None))
    with pytest.raises(ValueError, match="no hn:identityVisibility"):
        _load_regime(g)


def test_the_view_applies_the_regime_to_cards_text_and_graph() -> None:
    game = open_hand_game()
    view = build_view(game, 0)
    assert all(cv.card is not None for hand in view.hands for cv in hand)           # own cards are visible
    assert "You can see your cards" in state_text(view, game.rules, "hands")
    g = view_graph(game, view)
    assert (graph_namespace()["hand0_0"], HN.hasColor, None) in g


def test_a_drawn_card_is_hidden_from_the_drawer_only_because_of_the_hand_regime() -> None:
    standard = Game(["Ann", "Bob"], seed=2)
    rig(standard, [[(R, 1)] * 5, [(B, 1)] * 5], deck=[(Y, 5)])
    standard.apply(0, PlayCard(0))
    draw = lambda game, seat: next(e for e in build_view(game, seat).events if isinstance(e, DrawEvent))  # noqa: E731
    assert draw(standard, 0).card is None and draw(standard, 1).card.rank == 5

    variant = open_hand_game()
    rig(variant, [[(R, 1)] * 5, [(B, 1)] * 5], deck=[(Y, 5)])
    variant.apply(0, DiscardCard(0))
    assert draw(variant, 0).card is not None and draw(variant, 0).card.rank == 5     # open hands: the drawer sees it too


def test_events_point_at_the_affordance_they_exercise(game2: Game) -> None:
    rig(game2, [[(R, 1)] * 5, [(B, 1)] * 5], deck=[(Y, 2)] * 10)
    game2.apply(0, PlayCard(0))
    game2.apply(1, DiscardCard(0))
    g = view_graph(game2, build_view(game2, 0))
    rows = {(str(t).rsplit("#", 1)[1], str(g.value(a, HN.functionName)))
            for t, a in g.query("SELECT ?t ?a WHERE { ?e a ?t ; hn:exercises ?a }", initNs={"hn": HN})}
    assert rows == {("PlayEvent", "play_card"), ("DiscardEvent", "discard_card")}
