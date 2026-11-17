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

from dataclasses import dataclass

from rdflib import RDF, RDFS, XSD, Graph, Literal, Namespace

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
from .ontology import HN


@dataclass(frozen=True, slots=True)
class CardView:
    """One card as seen by the viewer. ``card`` is None when it is hidden."""

    uid: int
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


def _redact(event: Event, viewer: Seat) -> Event:
    """Hide the identity of a card from the player who just drew it.

    ``DrawEvent.card`` is replaced by a placeholder with rank 0; renderers
    show it as "a new card".
    """
    if isinstance(event, DrawEvent) and event.actor == viewer:
        return DrawEvent(event.turn, event.actor, Card(event.card.uid, Color.RED, 0))
    return event


def build_view(game: Game, viewer: Seat) -> PlayerView:
    """Project the authoritative state down to what ``viewer`` may know."""
    st = game.state
    hands: list[tuple[CardView, ...]] = []
    for seat, hand in enumerate(st.hands):
        hands.append(
            tuple(
                CardView(
                    uid=c.uid,
                    position=i,
                    card=None if seat == viewer else c,
                    knowledge=st.knowledge[c.uid],
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
        events=tuple(_redact(e, viewer) for e in st.events),
        over_reason=st.over_reason,
        score=st.score,
    )


# ---------------------------------------------------------------------------
# RDF projection
# ---------------------------------------------------------------------------


def _hg(game_id: str) -> Namespace:
    return Namespace(f"https://example.org/hanabi/game/{game_id}#")


def view_graph(game: Game, view: PlayerView, game_id: str = "g") -> Graph:
    """Knowledge graph for one seat: ontology + the seat's visible state.

    The ontology triples are merged in so SPARQL can join state with
    affordances (and their ``hn:functionName`` callable pointers).
    """
    hg = _hg(game_id)
    g = Graph()
    g.bind("hn", HN)
    g.bind("hg", hg)
    for triple in game.ontology.graph:
        g.add(triple)

    color_iri = {c: HN[c.value.capitalize()] for c in game.rules.colors}
    rank_iri = {r: HN[f"Rank{r}"] for r in game.rules.copies_by_rank}

    def lit(value: int | bool) -> Literal:
        return Literal(value, datatype=XSD.boolean if isinstance(value, bool) else XSD.integer)

    # Turn state (the game node) and the two token pools: available + spent = capacity.
    state_node = hg["state"]
    g.add((state_node, RDF.type, HN.Game))
    g.add((state_node, HN.deckSize, lit(view.deck_size)))
    g.add((state_node, HN.turnNumber, lit(view.turn)))
    for node_name, cls, available, capacity in (
        ("hintState", HN.HintState, view.hints, game.rules.max_hints),
        ("fuseState", HN.FuseState, view.fuses, game.rules.initial_fuses),
    ):
        pool = hg[node_name]
        g.add((pool, RDF.type, cls))
        g.add((pool, HN.available, lit(available)))
        g.add((pool, HN.capacity, lit(capacity)))
        g.add((pool, HN.spent, lit(capacity - available)))

    # Stacks.
    for color, height in view.stacks.items():
        stack = hg[f"stack_{color.value}"]
        g.add((stack, RDF.type, HN.Stack))
        g.add((stack, HN.stackOf, color_iri[color]))
        g.add((stack, HN.stackHeight, lit(height)))

    # Players.
    for seat, name in enumerate(view.names):
        player = hg[f"player{seat}"]
        g.add((player, RDF.type, HN.Player))
        g.add((player, HN.seatIndex, lit(seat)))
        g.add((player, RDFS.label, Literal(name)))
        g.add((player, HN.isActivePlayer, lit(seat == view.current)))

    # Discard pile: public, identities visible.
    discard = hg["discard"]
    g.add((discard, RDF.type, HN.DiscardPile))
    for card in view.discard:
        node = hg[f"card{card.uid}"]
        g.add((node, RDF.type, HN.Card))
        g.add((node, HN.inZone, discard))
        g.add((node, HN.discarded, lit(True)))
        g.add((node, HN.hasColor, color_iri[card.color]))
        g.add((node, HN.hasRank, rank_iri[card.rank]))

    # Hands. The one visibility rule lives here: identity only if card is set.
    for seat, hand in enumerate(view.hands):
        player = hg[f"player{seat}"]
        zone = hg[f"hand{seat}"]
        g.add((zone, RDF.type, HN.Hand))
        for cv in hand:
            node = hg[f"card{cv.uid}"]
            g.add((node, RDF.type, HN.Card))
            g.add((node, HN.heldBy, player))
            g.add((node, HN.inZone, zone))
            g.add((node, HN.position, lit(cv.position)))
            if cv.card is not None:  # <- hidden for the viewer's own hand
                g.add((node, HN.hasColor, color_iri[cv.card.color]))
                g.add((node, HN.hasRank, rank_iri[cv.card.rank]))
            k = cv.knowledge
            if k.color is not None:
                g.add((node, HN.knownColor, color_iri[k.color]))
            if k.rank is not None:
                g.add((node, HN.knownRank, rank_iri[k.rank]))
            for c in k.not_colors:
                g.add((node, HN.excludedColor, color_iri[c]))
            for r in k.not_ranks:
                g.add((node, HN.excludedRank, rank_iri[r]))

    # Callable pointers: affordances currently open to the viewer.
    me = hg[f"player{view.viewer}"]
    if view.viewer == view.current and view.over_reason is None:
        for spec in game.ontology.affordances.values():
            if spec.ends_turn and spec.function_name == "give_hint" and view.hints <= 0:
                continue
            g.add((me, HN.canInvoke, spec.iri))
    for spec in game.ontology.affordances.values():
        if not spec.ends_turn:  # read tools are always open
            g.add((me, HN.canInvoke, spec.iri))

    # Event log (already redacted).
    for i, ev in enumerate(view.events):
        node = hg[f"event{i}"]
        g.add((node, HN.turnNumber, lit(ev.turn)))
        if isinstance(ev, PlayEvent):
            g.add((node, RDF.type, HN.PlayEvent))
            g.add((node, HN.actor, hg[f"player{ev.actor}"]))
            g.add((node, HN.card, hg[f"card{ev.card.uid}"]))
            g.add((hg[f"card{ev.card.uid}"], HN.hasColor, color_iri[ev.card.color]))  # revealed on play
            g.add((hg[f"card{ev.card.uid}"], HN.hasRank, rank_iri[ev.card.rank]))
            g.add((node, HN.succeeded, lit(ev.success)))
        elif isinstance(ev, DiscardEvent):
            g.add((node, RDF.type, HN.DiscardEvent))
            g.add((node, HN.actor, hg[f"player{ev.actor}"]))
            g.add((node, HN.card, hg[f"card{ev.card.uid}"]))
        elif isinstance(ev, HintEvent):
            g.add((node, RDF.type, HN.HintEvent))
            g.add((node, HN.actor, hg[f"player{ev.actor}"]))
            g.add((node, HN.target, hg[f"player{ev.target}"]))
            if ev.color is not None:
                g.add((node, HN.hintedColor, color_iri[ev.color]))
            if ev.rank is not None:
                g.add((node, HN.hintedRank, rank_iri[ev.rank]))
            for pos in ev.touched:
                g.add((node, HN.touchedPositions, lit(pos)))
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
    n = view.names
    me = view.viewer

    def who(seat: Seat) -> str:
        return f"{n[seat]} (you)" if seat == me else n[seat]

    if isinstance(ev, HintEvent):
        what = ev.color.value if ev.color is not None else f"rank {ev.rank}"
        slots = ", ".join(str(p) for p in ev.touched)
        return f"T{ev.turn} {who(ev.actor)} hinted {who(ev.target)}: {what} cards are at positions [{slots}]"
    if isinstance(ev, PlayEvent):
        outcome = "SUCCESS" if ev.success else "MISFIRE (fuse lost)"
        return f"T{ev.turn} {who(ev.actor)} played position {ev.position}: {ev.card} -> {outcome}"
    if isinstance(ev, DiscardEvent):
        return f"T{ev.turn} {who(ev.actor)} discarded position {ev.position}: {ev.card}"
    if isinstance(ev, DrawEvent):
        card = "a new card" if ev.card.rank == 0 else f"{ev.card}"
        return f"T{ev.turn} {who(ev.actor)} drew {card} into position (last slot)"
    return f"T{ev.turn} GAME OVER: {ev.reason}. Score {ev.score}."
