"""SHACL audit: the shapes accept games in play and reject every kind of corruption we know."""

from __future__ import annotations

import pytest
from rdflib import RDF, Graph

from conftest import R, rig
from hanabi_llm.engine import Game
from hanabi_llm.model import CardKnowledge, Color, DiscardCard, GiveHint, PlayCard
from hanabi_llm.ontology import HN
from hanabi_llm.players import RandomPlayer
from hanabi_llm.runner import play_game
from hanabi_llm.validate import AuditError, audit, check_truth, check_view, truth_graph
from hanabi_llm.views import build_view, graph_namespace, view_graph

HG = graph_namespace()


def test_a_fresh_game_conforms(game2: Game) -> None:
    audit(game2)


async def test_whole_random_games_conform_at_sampled_turns() -> None:
    """Truth and every view conform at the start, every 6th turn, and at the end."""
    names = ["a", "b"]
    game = Game(names, seed=7)
    audit(game)
    seen = 0

    def sample(trace: object) -> None:
        nonlocal seen
        seen += 1
        if seen % 6 == 0:
            audit(game)

    await play_game(game, [RandomPlayer(n, seed=i) for i, n in enumerate(names)], on_turn=sample)
    audit(game)  # the final state, including the end of the game


def test_audit_follows_hints_plays_and_discards() -> None:
    """A naturally dealt game (not rigged: a rigged deal rightly fails conservation) stays conformant."""
    game = Game(["Ann", "Bob"], seed=4)
    hint = next(a for a in game.legal_actions(0) if isinstance(a, GiveHint))
    game.apply(0, hint)                       # hint knowledge appears in the truth graph
    game.apply(1, PlayCard(0))                # a play (success or misfire) is tracked
    game.apply(0, DiscardCard(0))
    audit(game)
    played_or_discarded = len(game.state.played) + len(game.state.discard)
    assert played_or_discarded == 2


def test_the_shapes_reject_a_rigged_deal() -> None:
    """Conservation is real: invented cards make the whole deck inconsistent."""
    game = rig(Game(["Ann", "Bob"], seed=4), [[(R, 1)] * 5, [(R, 1)] * 5])
    with pytest.raises(AuditError, match="copies per colour"):
        check_truth(truth_graph(game))


def corrupt(label: str, game: Game, mutate: object, check: object, message: str) -> None:
    """Apply ``mutate`` to a graph and expect the shapes to reject it with ``message``."""
    graph = truth_graph(game) if check is check_truth else view_graph(game, build_view(game, 0))
    mutate(graph)  # type: ignore[operator]
    with pytest.raises(AuditError, match=message):
        check(graph)  # type: ignore[operator]


@pytest.fixture
def g3() -> Game:
    return Game(["Alice", "Bob", "Carol"], seed=3)


def test_truth_rejects_a_card_with_two_colours(g3: Game) -> None:
    corrupt("two colours", g3, lambda g: (g.add((HG["card0"], HN.hasColor, HN.Red)), g.add((HG["card0"], HN.hasColor, HN.Blue))),
            check_truth, "exactly one colour")


def test_truth_rejects_a_duplicated_card() -> None:
    game = Game(["Alice", "Bob"], seed=1)
    deck = list(game.state.deck)
    deck[0] = deck[1]
    game.rig(deck=deck)
    with pytest.raises(AuditError, match="copies per colour"):
        check_truth(truth_graph(game))


def test_truth_rejects_a_stack_taller_than_its_cards() -> None:
    game = Game(["Alice", "Bob"], seed=1)
    game.rig(stacks={Color.RED: 2})
    with pytest.raises(AuditError, match="missing a rank"):
        check_truth(truth_graph(game))


def test_truth_rejects_tokens_beyond_capacity() -> None:
    game = Game(["Alice", "Bob"], seed=1)
    game.rig(hints=9)
    with pytest.raises(AuditError, match="0 <= available <= capacity"):
        check_truth(truth_graph(game))


def test_truth_rejects_a_hint_that_lies() -> None:
    game = Game(["Alice", "Bob"], seed=1)
    card = game.state.hands[1][0]
    game.rig(knowledge={card.uid: CardKnowledge(color=Color.BLUE if card.color is not Color.BLUE else Color.RED)})
    with pytest.raises(AuditError, match="hint told the holder"):
        check_truth(truth_graph(game))


def test_view_rejects_a_leaked_own_card(g3: Game) -> None:
    own = g3.state.hands[0][0]
    corrupt("leak", g3, lambda g: g.add((HG[f"card{own.uid}"], HN.hasColor, HN.Red)), check_view, "LEAK: the viewer can see")


def test_view_rejects_a_missing_teammate_identity(g3: Game) -> None:
    other = g3.state.hands[1][0]
    corrupt("hidden teammate", g3, lambda g: g.remove((HG[f"card{other.uid}"], HN.hasRank, None)), check_view, "cannot see the identity")


def test_view_rejects_a_deck_card_and_a_deck_zone(g3: Game) -> None:
    corrupt("deck card", g3, lambda g: g.add((HG["card999"], RDF.type, HN.Card)), check_view, "deck")
    corrupt("deck zone", g3, lambda g: g.add((HG["deck"], RDF.type, HN.Deck)), check_view, "deck zone")


def test_truth_and_view_graphs_differ_in_one_marker(g3: Game) -> None:
    """Only a view graph names a viewer; the truth graph never does."""
    truth = truth_graph(g3)
    view = view_graph(g3, build_view(g3, 1))
    assert not list(truth.triples((None, HN.viewer, None)))
    assert list(view.triples((HG["state"], HN.viewer, HG["player1"])))
    assert len(list(truth.subjects(RDF.type, HN.Card))) == 50      # the truth has every card ...
    assert len(list(view.subjects(RDF.type, HN.Card))) < 50        # ... a view only the ones it may see
