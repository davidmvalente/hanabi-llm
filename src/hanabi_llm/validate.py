"""Ground truth and SHACL audit of a game in play.

``truth_graph`` is the whole game as RDF, hidden cards included. It is NEVER
shown to a model; it exists so that two things can be checked mechanically:

  * the truth obeys the structure and conservation laws (``shapes/truth.ttl``),
  * every player's view obeys the declared epistemic regime (``shapes/view.ttl``).

``audit(game)`` runs both for all seats. Tests call it on whole games, and the
CLI's ``--audit`` flag calls it after every turn. A violation is a bug in this
code, not a fault of any model, so it raises ``AuditError`` loudly.

Needs the optional dependency ``pyshacl`` (``pip install hanabi-llm[audit]``).
"""

from __future__ import annotations

from functools import cache
from importlib import resources

from rdflib import RDF, Graph

from .engine import Game
from .ontology import HN
from .views import (
    add_identity,
    add_knowledge,
    add_public_state,
    build_view,
    graph_namespace,
    iris_for,
    lit,
    new_graph,
    view_graph,
)


class AuditError(AssertionError):
    """A graph violates the declared structure or epistemic regime."""


@cache
def _shapes(name: str) -> Graph:
    text = resources.files("hanabi_llm").joinpath("shapes", f"{name}.ttl").read_text("utf-8")
    return Graph().parse(data=text, format="turtle")


def _conform(data: Graph, shapes: str, what: str) -> None:
    try:
        import pyshacl
    except ImportError as exc:  # pragma: no cover - depends on the install
        raise RuntimeError("SHACL audit needs pyshacl: pip install 'hanabi-llm[audit]'") from exc
    conforms, _, report = pyshacl.validate(data, shacl_graph=_shapes(shapes), inference="none")
    if not conforms:
        raise AuditError(f"{what} violates its shapes:\n{report}")


def truth_graph(game: Game, game_id: str = "g") -> Graph:
    """The whole game as RDF: every card with its identity, in its real zone. Not for models."""
    st = game.state
    hg = graph_namespace(game_id)
    iris = iris_for(game)
    g = new_graph(game, hg)
    add_public_state(g, hg, game, iris, names=st.names, current=st.current, turn=st.turn,
                     deck_size=len(st.deck), hints=st.hints, fuses=st.fuses, stacks=st.stacks)
    # No hn:viewer: this is how a truth graph differs from a view graph.

    def place(name: str, zone: object) -> object:
        """A card node, named by the place it occupies (a hand slot, a deck position, ...)."""
        node = hg[name]
        g.add((node, RDF.type, HN.Card))
        g.add((node, HN.inZone, zone))  # type: ignore[arg-type]
        return node

    for seat, hand in enumerate(st.hands):
        zone = hg[f"hand{seat}"]
        g.add((zone, RDF.type, HN.Hand))
        for i, card in enumerate(hand):
            node = place(f"hand{seat}_{i}", zone)
            g.add((node, HN.heldBy, hg[f"player{seat}"]))  # type: ignore[arg-type]
            g.add((node, HN.position, lit(i)))  # type: ignore[arg-type]
            add_identity(g, node, card, iris)  # type: ignore[arg-type]
            add_knowledge(g, node, st.knowledge[seat][i], iris)  # type: ignore[arg-type]
    deck = hg["deck"]
    g.add((deck, RDF.type, HN.Deck))
    for i, card in enumerate(st.deck):
        node = place(f"deck_{i}", deck)
        g.add((node, HN.position, lit(i)))  # type: ignore[arg-type]
        add_identity(g, node, card, iris)  # type: ignore[arg-type]
    discard = hg["discard"]
    g.add((discard, RDF.type, HN.DiscardPile))
    for i, card in enumerate(st.discard):
        node = place(f"discard_{i}", discard)
        g.add((node, HN.discarded, lit(True)))  # type: ignore[arg-type]
        add_identity(g, node, card, iris)  # type: ignore[arg-type]
    for card in st.played:
        add_identity(g, place(f"played_{card.color.value}_{card.rank}", hg[f"stack_{card.color.value}"]), card, iris)  # type: ignore[arg-type]
    return g


def check_truth(graph: Graph) -> None:
    """Validate a ground-truth graph. Raises ``AuditError``."""
    _conform(graph, "truth", "the ground truth")


def check_view(graph: Graph) -> None:
    """Validate one player's view graph. Raises ``AuditError``."""
    _conform(graph, "view", "a player's view")


def validate_truth(game: Game) -> None:
    check_truth(truth_graph(game))


def validate_view(game: Game, seat: int) -> None:
    check_view(view_graph(game, build_view(game, seat)))


def audit(game: Game) -> None:
    """Validate the ground truth and every seat's view."""
    validate_truth(game)
    for seat in range(len(game.state.names)):
        validate_view(game, seat)
