"""Per-player projections of the game state.

Design: there is ONE authoritative ``GameState`` (in the engine). A player
never touches it. Instead ``build_view(game, viewer)`` produces a
``PlayerView``: a redacted snapshot containing exactly what that seat is
allowed to know. From a view we derive

  * an RDF graph (``view_graph``), used for validation and analysis, and
  * text sections (``states.state_text``) shown to the model.

The "synchronised common part" is therefore not replicated state but the
public portion of every projection, computed from the same source, so it
cannot diverge. Hidden information is removed at projection time, in one
place, and that is the only place to audit for leaks.

What is hidden from seat S:
  * the identity (colour, rank) of cards in S's own hand,
  * the identity of a card S just drew (the DrawEvent is redacted),
  * the deck order.
Everything else (other hands, discard pile, stacks, tokens, hint history,
per-card hint knowledge) is visible.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from rdflib import RDF, RDFS, XSD, Graph, Literal, Namespace, URIRef

from .engine import Game
from .model import (
    Card,
    CardKnowledge,
    Color,
    DiscardEvent,
    DrawEvent,
    EndEvent,
    Event,
    HintEvent,
    PlayEvent,
    Seat,
)
from .feed import event_record
from .narrate import segments, to_text
from .ontology import HN, Regime


@dataclass(frozen=True, slots=True)
class CardView:
    """One hand slot as seen by the viewer. ``card`` is None when its identity is hidden."""

    position: int
    card: Card | None
    knowledge: CardKnowledge


@dataclass(frozen=True, slots=True)
class PlayerView:
    viewer: Seat
    names: tuple[str, ...]
    current: Seat
    turn: int
    hints: int
    fuses: int
    deck_size: int
    final_turns_left: int | None
    stacks: dict[Color, int]
    discard: tuple[Card, ...]
    hands: tuple[tuple[CardView, ...], ...]  # indexed by seat
    events: tuple[Event, ...]  # already redacted for the viewer
    over_reason: str | None
    score: int


def _redact(event: Event, viewer: Seat, regime: Regime) -> Event:
    """Hide the identity of a freshly drawn card where the regime says it is hidden.

    A drawn card goes into the actor's hand, so it is visible to ``viewer``
    exactly when hand cards held by the actor are. ``DrawEvent.card`` is
    replaced by a placeholder with rank 0; renderers show "a new card".
    """
    if isinstance(event, DrawEvent) and not regime.hand_identity_visible(viewer, event.actor):
        return DrawEvent(event.turn, event.actor, None)
    return event


def build_view(game: Game, viewer: Seat) -> PlayerView:
    """Project the authoritative state down to what ``viewer`` may know.

    The policy (who may see which cards) is declared in the ontology's
    epistemics section and read here as ``regime``; this function applies it.
    """
    st = game.state
    regime = game.ontology.regime
    hands: list[tuple[CardView, ...]] = []
    for seat, hand in enumerate(st.hands):
        hands.append(
            tuple(
                CardView(
                    position=i,
                    card=c if regime.hand_identity_visible(viewer, seat) else None,
                    knowledge=st.knowledge[seat][i],
                )
                for i, c in enumerate(hand)
            )
        )
    return PlayerView(
        viewer=viewer,
        names=tuple(st.names),
        current=st.current,
        turn=st.turn,
        hints=st.hints,
        fuses=st.fuses,
        deck_size=len(st.deck),
        final_turns_left=st.final_turns_left,
        stacks=dict(st.stacks),
        discard=tuple(st.discard),
        hands=tuple(hands),
        events=tuple(_redact(e, viewer, regime) for e in st.events),
        over_reason=st.over_reason,
        score=st.score,
    )


# ---------------------------------------------------------------------------
# RDF projection
# ---------------------------------------------------------------------------


def graph_namespace(game_id: str = "g") -> Namespace:
    """Namespace of the instance nodes of one game (cards, players, zones, events)."""
    return Namespace(f"https://example.org/hanabi/game/{game_id}#")


def lit(value: int | bool) -> Literal:
    return Literal(value, datatype=XSD.boolean if isinstance(value, bool) else XSD.integer)


@dataclass(frozen=True, slots=True)
class Iris:
    """Ontology individuals for colours and ranks."""

    color: dict[Color, URIRef]
    rank: dict[int, URIRef]


def iris_for(game: Game) -> Iris:
    return Iris(
        color={c: HN[c.value.capitalize()] for c in game.rules.colors},
        rank={r: HN[f"Rank{r}"] for r in game.rules.copies_by_rank},
    )


def new_graph(game: Game, hg: Namespace) -> Graph:
    """An empty game graph with the ontology merged in (so queries can join the two)."""
    g = Graph()
    g.bind("hn", HN)
    g.bind("hg", hg)
    for triple in game.ontology.graph:
        g.add(triple)
    return g


def add_public_state(
    g: Graph, hg: Namespace, game: Game, iris: Iris, *, names: Sequence[str], current: Seat, turn: int,
    deck_size: int, hints: int, fuses: int, stacks: Mapping[Color, int],
) -> None:
    """The part of the state that is identical in every graph: game node, token pools, stacks, players."""
    state_node = hg["state"]
    g.add((state_node, RDF.type, HN.Game))
    g.add((state_node, HN.deckSize, lit(deck_size)))
    g.add((state_node, HN.turnNumber, lit(turn)))
    # Token pools: available + spent = capacity.
    for node_name, cls, available, capacity in (
        ("hintState", HN.HintState, hints, game.rules.max_hints),
        ("fuseState", HN.FuseState, fuses, game.rules.initial_fuses),
    ):
        pool = hg[node_name]
        g.add((pool, RDF.type, cls))
        g.add((pool, HN.available, lit(available)))
        g.add((pool, HN.capacity, lit(capacity)))
        g.add((pool, HN.spent, lit(capacity - available)))
    for color, height in stacks.items():
        stack = hg[f"stack_{color.value}"]
        g.add((stack, RDF.type, HN.Stack))
        g.add((stack, HN.stackOf, iris.color[color]))
        g.add((stack, HN.stackHeight, lit(height)))
    for seat, name in enumerate(names):
        player = hg[f"player{seat}"]
        g.add((player, RDF.type, HN.Player))
        g.add((player, HN.seatIndex, lit(seat)))
        g.add((player, RDFS.label, Literal(name)))
        g.add((player, HN.isActivePlayer, lit(seat == current)))


def add_identity(g: Graph, node: URIRef, card: Card, iris: Iris) -> None:
    """State a card's identity. Call this ONLY where the regime lets the viewer see it."""
    g.add((node, HN.hasColor, iris.color[card.color]))
    g.add((node, HN.hasRank, iris.rank[card.rank]))


def add_knowledge(g: Graph, node: URIRef, k: CardKnowledge, iris: Iris) -> None:
    """What hints have told the holder about a card (common knowledge)."""
    if k.color is not None:
        g.add((node, HN.knownColor, iris.color[k.color]))
    if k.rank is not None:
        g.add((node, HN.knownRank, iris.rank[k.rank]))
    for c in k.not_colors:
        g.add((node, HN.excludedColor, iris.color[c]))
    for r in k.not_ranks:
        g.add((node, HN.excludedRank, iris.rank[r]))


def _event_card(g: Graph, event: URIRef, node: URIRef, card: Card, iris: Iris) -> None:
    """The card an event revealed, as a node of its own (there are no card ids to share one)."""
    g.add((node, RDF.type, HN.Card))
    add_identity(g, node, card, iris)
    g.add((event, HN.card, node))


def view_graph(game: Game, view: PlayerView, game_id: str = "g") -> Graph:
    """RDF graph of one seat's view: the ontology plus what the seat may see.

    Used for validation (``validate.py``) and analysis; it is not a model-facing
    tool. It is a read model: a pure function of ``(game, view)``, rebuilt on
    demand and never written back.
    """
    hg = graph_namespace(game_id)
    iris = iris_for(game)
    g = new_graph(game, hg)
    add_public_state(g, hg, game, iris, names=view.names, current=view.current, turn=view.turn,
                     deck_size=view.deck_size, hints=view.hints, fuses=view.fuses, stacks=view.stacks)
    me = hg[f"player{view.viewer}"]
    g.add((hg["state"], HN.viewer, me))

    # Discard pile: identities are public (see the regime in the ontology).
    discard = hg["discard"]
    g.add((discard, RDF.type, HN.DiscardPile))
    for i, card in enumerate(view.discard):
        node = hg[f"discard_{i}"]  # a card is named by where it is, never by an id
        g.add((node, RDF.type, HN.Card))
        g.add((node, HN.inZone, discard))
        g.add((node, HN.discarded, lit(True)))
        add_identity(g, node, card, iris)

    # Hands. ``cv.card`` is None exactly where the regime hides identity from this viewer.
    for seat, hand in enumerate(view.hands):
        zone = hg[f"hand{seat}"]
        g.add((zone, RDF.type, HN.Hand))
        for cv in hand:
            node = hg[f"hand{seat}_{cv.position}"]
            g.add((node, RDF.type, HN.Card))
            g.add((node, HN.heldBy, hg[f"player{seat}"]))
            g.add((node, HN.inZone, zone))
            g.add((node, HN.position, lit(cv.position)))
            if cv.card is not None:
                add_identity(g, node, cv.card, iris)
            add_knowledge(g, node, cv.knowledge, iris)

    # Callable pointers: affordances currently open to the viewer.
    if view.viewer == view.current and view.over_reason is None:
        for spec in game.ontology.affordances.values():
            if spec.ends_turn and spec.function_name == "give_hint" and view.hints <= 0:
                continue
            g.add((me, HN.canInvoke, spec.iri))
    for spec in game.ontology.affordances.values():
        if not spec.ends_turn:  # read tools are always open
            g.add((me, HN.canInvoke, spec.iri))

    # Event log (already redacted). Each action event points at the affordance it exercises.
    exercised = {PlayEvent: "play_card", DiscardEvent: "discard_card", HintEvent: "give_hint"}
    for i, ev in enumerate(view.events):
        node = hg[f"event{i}"]
        g.add((node, HN.turnNumber, lit(ev.turn)))
        if type(ev) in exercised:
            g.add((node, HN.exercises, game.ontology.affordances[exercised[type(ev)]].iri))
        if isinstance(ev, PlayEvent):
            g.add((node, RDF.type, HN.PlayEvent))
            g.add((node, HN.actor, hg[f"player{ev.actor}"]))
            _event_card(g, node, hg[f"event{i}_card"], ev.card, iris)  # a played card is revealed to all
            g.add((node, HN.succeeded, lit(ev.success)))
        elif isinstance(ev, DiscardEvent):
            g.add((node, RDF.type, HN.DiscardEvent))
            g.add((node, HN.actor, hg[f"player{ev.actor}"]))
            _event_card(g, node, hg[f"event{i}_card"], ev.card, iris)  # a discarded card is revealed to all
        elif isinstance(ev, HintEvent):
            g.add((node, RDF.type, HN.HintEvent))
            g.add((node, HN.actor, hg[f"player{ev.actor}"]))
            g.add((node, HN.receiver, hg[f"player{ev.receiver}"]))
            if isinstance(ev.value, Color):  # the predicate says which attribute, the object its value
                g.add((node, HN.hintedColor, iris.color[ev.value]))
            else:
                g.add((node, HN.hintedRank, iris.rank[ev.value]))
            for pos in ev.targets:
                g.add((node, HN.targetedPosition, lit(pos)))
        elif isinstance(ev, DrawEvent):
            g.add((node, RDF.type, HN.DrawEvent))
            g.add((node, HN.actor, hg[f"player{ev.actor}"]))
        elif isinstance(ev, EndEvent):
            g.add((node, RDF.type, HN.EndEvent))
    return g


# ---------------------------------------------------------------------------
# Text describers (used by states.py to render sections)
# ---------------------------------------------------------------------------


def describe_knowledge(k: CardKnowledge) -> str:
    parts: list[str] = []
    if k.color is not None:
        parts.append(f"colour={k.color.value}")
    elif k.not_colors:
        parts.append("not " + "/".join(sorted(c.value for c in k.not_colors)))
    if k.rank is not None:
        parts.append(f"rank={k.rank}")
    elif k.not_ranks:
        parts.append("not rank " + "/".join(str(r) for r in sorted(k.not_ranks)))
    return ", ".join(parts) if parts else "nothing known"


def describe_event(ev: Event, view: PlayerView) -> str:
    return narrate_event(ev, view.names, you=view.viewer)


def narrate_event(ev: Event, names: Sequence[str], you: Seat | None = None) -> str:
    """One event as a line of text (the wording lives in ``narrate.py``)."""
    return to_text(segments(event_record(ev)), names, you)
