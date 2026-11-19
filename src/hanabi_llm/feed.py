"""The game feed: a JSONL event stream that fully describes one game.

One JSON object per line, flushed as it is written, so a viewer can follow a
game live and a crash loses nothing:

  header   once, first: seed, seat names, rules, and the INITIAL DEAL (every
           card in every starting hand) so the game can be rebuilt from here
  event    one per event: hint | play | discard | draw | end
  result   once, last: status, scores, and the fault if the game was aborted

The feed is omniscient (a spectator's view). ``project_record`` removes what a
given player may not know, according to the declared regime; folding the
projected feed gives exactly that player's view (see ``fold.py`` and
docs/EPISTEMICS.md, "Graphs as a fold").

Model conversations are NOT in the feed (they are large and not game state);
the CLI writes them to a separate trace file.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING, Any, TextIO

from .clock import now_iso
from .model import Action, Card, Color, DiscardEvent, DrawEvent, EndEvent, Event, GiveHint, HintEvent, PlayEvent
from .ontology import Regime

if TYPE_CHECKING:
    from .engine import Game
    from .runner import GameResult

SCHEMA = 3  # 3: a hand index is a `slot` and a draw says which slot it refills; 2: cards are {color, rank}, hidden = null
Record = dict[str, Any]


def event_slot(rec: Record) -> int:
    """The slot a play or discard event is about (feeds before schema 3 called it ``position``)."""
    return int(rec["slot"] if "slot" in rec else rec["position"])


def card_record(card: Card | None) -> Record | None:
    """A card is ``{"color", "rank"}`` and nothing else; ``None`` is a card whose identity is hidden."""
    return None if card is None else {"color": card.color.value, "rank": card.rank}


def event_record(ev: Event) -> Record:
    """One event as a feed record (omniscient: identities included)."""
    base: Record = {"type": "event", "turn": ev.turn}
    if isinstance(ev, HintEvent):
        value = ev.value.value if isinstance(ev.value, Color) else ev.value
        return base | {"kind": "hint", "actor": ev.actor, "receiver": ev.receiver, "attribute": ev.attribute,
                       "value": value, "targets": list(ev.targets)}
    if isinstance(ev, PlayEvent):
        return base | {"kind": "play", "actor": ev.actor, "slot": ev.slot,
                       "card": card_record(ev.card), "success": ev.success}
    if isinstance(ev, DiscardEvent):
        return base | {"kind": "discard", "actor": ev.actor, "slot": ev.slot, "card": card_record(ev.card)}
    if isinstance(ev, DrawEvent):
        return base | {"kind": "draw", "actor": ev.actor, "card": card_record(ev.card), "slot": ev.slot}
    assert isinstance(ev, EndEvent)
    return base | {"kind": "end", "reason": ev.reason, "score": ev.score}


def action_record(action: Action) -> Record:
    """A chosen action as a plain record (for the trace)."""
    if isinstance(action, GiveHint):
        value = action.value.value if isinstance(action.value, Color) else action.value
        return {"tool": action.affordance, "receiver": action.receiver, "attribute": action.attribute, "value": value}
    return {"tool": action.affordance, "slot": action.slot}


def header_record(game: Game, seats: dict[str, str], config: dict[str, Any]) -> Record:
    """Everything needed to start a fold: rules and the initial deal. Write it BEFORE any action."""
    r = game.rules
    return {
        "type": "header", "schema": SCHEMA, "seed": game.seed,
        "names": list(game.state.names), "seats": seats,
        "rules": {"colors": [c.value for c in r.colors], "top_rank": r.top_rank, "max_hints": r.max_hints,
                  "initial_hints": r.initial_hints, "fuses": r.initial_fuses},
        "deal": {"hands": [[card_record(c) for c in hand] for hand in game.state.hands], "deck_size": len(game.state.deck)},
        "config": config,
    }


def result_record(result: GameResult) -> Record:
    f = result.fault
    return {
        "type": "result", "status": result.status.value, "reason": result.reason, "seed": result.seed,
        "score": result.score, "official_score": result.official_score, "counts_for_ranking": result.counts_for_ranking,
        "fault": None if f is None else {"seat": f.seat, "player": f.player, "kind": f.kind.value, "detail": f.detail, "turn": f.turn},
    }


class FeedWriter:
    """Appends records to a JSONL file, flushing every line.

    Every record gets a ``ts`` (UTC ISO 8601, millisecond precision) taken when it is written. The
    events of one turn are written together when the turn is over, so the gap between the timestamps
    of consecutive turns is how long the turn took.
    """

    def __init__(self, path: Path, clock: Callable[[], str] = now_iso) -> None:
        self._fh: TextIO = path.open("w", encoding="utf-8")
        self._clock = clock

    def write(self, record: Record) -> None:
        stamped = record if "ts" in record else {"ts": self._clock(), **record}
        self._fh.write(json.dumps(stamped, ensure_ascii=False) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()


def read_feed(path: Path) -> list[Record]:
    return [json.loads(line) for line in path.read_text("utf-8").splitlines() if line.strip()]


# --- the observer projection: what ``viewer`` may know --------------------------


def project_record(rec: Record, viewer: int, regime: Regime) -> Record:
    """Remove from one record whatever the regime hides from ``viewer``.

    Only two things carry hidden identities: the initial deal, and a draw into
    a hand the viewer may not see. Everything else (hints, plays, discards,
    the end) is public and passes through unchanged.
    """
    if rec["type"] == "header":
        hands = [[c if regime.hand_identity_visible(viewer, seat) else None for c in hand]
                 for seat, hand in enumerate(rec["deal"]["hands"])]
        return {**rec, "deal": {**rec["deal"], "hands": hands}}
    if rec["type"] == "event" and rec["kind"] == "draw" and not regime.hand_identity_visible(viewer, rec["actor"]):
        return {**rec, "card": None}
    return rec
