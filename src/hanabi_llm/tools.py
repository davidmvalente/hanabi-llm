"""Turns ontology affordances into model-callable tools, and executes calls.

Two kinds of tool, told apart by ``hn:endsTurn`` in the ontology:

  * ACTION tools (play_card, discard_card, give_hint): validated here with
    ``Game.check`` but NOT applied; the referee applies them. The first valid
    one ends the turn.
  * READ tools (get_state, get_rules, get_concepts, estimate_card,
    query_knowledge, take_notes): may be called any number of times (bounded
    by the per-turn call budget). They never change the game.
"""

from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import Awaitable, Callable
from dataclasses import dataclass
from typing import Any

from rdflib import Graph
from rdflib.plugins.sparql import prepareQuery

from .engine import Game
from .estimates import estimate_card
from .model import Action, Color, DiscardCard, GiveHint, IllegalAction, PlayCard, Seat
from .ontology import AffordanceSpec, Ontology, ParamSpec, RuleSet
from .registry import bind
from .views import PlayerView, render_view, view_graph

JSON = dict[str, Any]

MAX_SPARQL_CHARS = 1500
MAX_SPARQL_ROWS = 40
MAX_NOTES_CHARS = 2000
#: A pathological query must not stall the whole tournament. rdflib is
#: synchronous, so we run it in a worker thread and stop waiting after this.
SPARQL_TIMEOUT_S = 5.0


# ---------------------------------------------------------------------------
# Read tools, bound to the function names the ontology declares.
# ---------------------------------------------------------------------------


@bind("run_sparql_select")
def run_sparql_select(graph: Graph, query: str) -> str:
    """Execute a read-only SELECT over a player's own view graph."""
    if len(query) > MAX_SPARQL_CHARS:
        raise IllegalAction(f"Query too long (max {MAX_SPARQL_CHARS} characters).")
    if "service" in query.lower():
        raise IllegalAction("SERVICE (federated queries) is not allowed.")
    try:
        prepared = prepareQuery(query, initNs={"hn": "https://example.org/hanabi#"})
    except Exception as exc:  # rdflib raises many parse error types
        raise IllegalAction(f"SPARQL parse error: {exc}") from exc
    if prepared.algebra.name != "SelectQuery":
        raise IllegalAction("Only SELECT queries are allowed.")
    result = graph.query(prepared)
    variables = [str(v) for v in result.vars or []]
    rows: list[str] = []
    for i, row in enumerate(result):
        if i >= MAX_SPARQL_ROWS:
            rows.append(f"... truncated at {MAX_SPARQL_ROWS} rows")
            break
        cells = [("" if v is None else v.n3(graph.namespace_manager)) for v in row]  # type: ignore[union-attr]
        rows.append(" | ".join(cells))
    if not rows:
        return "(no results)"
    return " | ".join(variables) + "\n" + "\n".join(rows)


@bind("store_notes")
def store_notes(store: dict[Seat, str], seat: Seat, text: str) -> str:
    """Save a private note for one seat, replacing the previous one."""
    store[seat] = text[:MAX_NOTES_CHARS]
    return f"Notes saved ({len(store[seat])} chars)."


@bind("describe_rules")
def describe_rules(rules: RuleSet) -> str:
    dist = ", ".join(f"{n}x rank {r}" for r, n in sorted(rules.copies_by_rank.items()))
    lines = [
        f"Colours: {', '.join(c.value for c in rules.colors)}. Each colour has: {dist}. Deck: {rules.deck_size} cards.",
        f"Hand size by player count: {dict(sorted(rules.hand_size_by_players.items()))}.",
        f"Hint tokens: start {rules.initial_hints}, max {rules.max_hints}. Fuse tokens: start {rules.initial_fuses}.",
        "Rules:",
        *(f"- {t}" for t in rules.rule_texts),
    ]
    return "\n".join(lines)


@bind("describe_state")
def describe_state(view: PlayerView) -> str:
    return render_view(view)


@bind("describe_concepts")
def describe_concepts(onto: Ontology, category: str | None, name: str | None) -> str:
    concepts = onto.concepts
    if name:
        key = name.strip().lower()
        if key not in concepts:
            raise IllegalAction(f"Unknown concept {name!r}. Known: {sorted(concepts)}")
        c = concepts[key]
        return f"[{c.category}] {c.name}\n{c.summary}\nGuidance: {c.guidance}"
    wanted = category.strip().lower() if category else None
    if wanted and wanted not in {"mechanic", "dynamic", "kinetic"}:
        raise IllegalAction("category must be one of: mechanic, dynamic, kinetic.")
    lines: list[str] = []
    for c in sorted(concepts.values(), key=lambda c: (c.category, c.name)):
        if wanted is None or c.category == wanted:
            lines.append(f"[{c.category}] {c.name}: {c.summary}")
    lines.append("Call get_concepts with `name` to read the guidance for one concept.")
    return "\n".join(lines)


@bind("estimate_card_probabilities")
def estimate_card_probabilities(view: PlayerView, rules: RuleSet, position: int) -> str:
    return estimate_card(view, rules, position)


# ---------------------------------------------------------------------------
# Schema generation
# ---------------------------------------------------------------------------


def _param_schema(p: ParamSpec, rules: RuleSet, other_names: list[str]) -> JSON:
    base: JSON
    match p.type:
        case "integer":
            base = {"type": "integer", "minimum": 0}
        case "color":
            base = {"type": "string", "enum": [c.value for c in rules.colors]}
        case "rank":
            base = {"type": "integer", "enum": sorted(rules.copies_by_rank)}
        case "player":
            base = {"type": "string", "enum": other_names}
        case "string":
            base = {"type": "string"}
        case other:  # pragma: no cover - guarded by the ontology loader
            raise ValueError(f"unknown parameter type {other}")
    base["description"] = p.description
    return base


def tool_schema(spec: AffordanceSpec, rules: RuleSet, other_names: list[str]) -> JSON:
    """OpenAI-style function tool definition derived from one affordance."""
    return {
        "type": "function",
        "function": {
            "name": spec.function_name,
            "description": spec.description,
            "parameters": {
                "type": "object",
                "properties": {p.name: _param_schema(p, rules, other_names) for p in spec.params},
                "required": [p.name for p in spec.params if not p.optional],
                "additionalProperties": False,
            },
        },
    }


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class ToolOutcome:
    """Result of one tool call, ready to send back to the model."""

    text: str
    #: Set only when an action tool was called AND passed validation.
    action: Action | None = None
    is_error: bool = False
    #: True when the model attempted a turn-ending action and it was rejected.
    #: Only these count against the model's invalid-action budget; a mistake
    #: in a read-only query is free (it is bounded by the call budget instead).
    failed_action: bool = False


def _as_int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise IllegalAction(f"`{what}` must be an integer.")
    try:
        return int(value)
    except ValueError:
        raise IllegalAction(f"`{what}` must be an integer, got {value!r}.") from None


Reader = Callable[[JSON], Awaitable[str]]


async def _call_with_timeout(fn: Callable[..., str], *args: Any, timeout: float) -> str:
    """Run a blocking function in a DAEMON thread and stop waiting after ``timeout``.

    ``asyncio.to_thread`` would join its worker at interpreter exit, so one
    runaway query could delay shutdown. A daemon thread cannot. Python cannot
    kill a thread, so a runaway query keeps using CPU until the process ends;
    the real fix is a subprocess sandbox (see docs/AUDIT.md, known limitations).
    """
    loop = asyncio.get_running_loop()
    future: asyncio.Future[str] = loop.create_future()

    def deliver(setter: Callable[[Any], None], value: Any) -> None:
        if not future.done():  # the waiter may already have timed out
            setter(value)

    def work() -> None:
        try:
            result = fn(*args)
        except BaseException as exc:  # noqa: BLE001 - forwarded to the awaiting task
            outcome: tuple[Callable[[Any], None], Any] = (future.set_exception, exc)
        else:
            outcome = (future.set_result, result)
        try:
            loop.call_soon_threadsafe(deliver, *outcome)
        except RuntimeError:  # event loop already closed
            pass

    threading.Thread(target=work, daemon=True).start()
    return await asyncio.wait_for(future, timeout)


class ToolBox:
    """Tools available to one seat during one turn."""

    def __init__(self, game: Game, seat: Seat, view: PlayerView, notes: dict[Seat, str]) -> None:
        self._game = game
        self._seat = seat
        self._view = view
        self._notes = notes
        self._graph: Graph | None = None  # built lazily; many turns never query it
        self._others = [n for i, n in enumerate(game.state.names) if i != seat]
        self._readers: dict[str, Reader] = {
            "query_knowledge": self._query,
            "take_notes": self._take_notes,
            "get_rules": self._get_rules,
            "get_state": self._get_state,
            "get_concepts": self._get_concepts,
            "estimate_card": self._estimate,
        }
        declared = {n for n, s in game.ontology.affordances.items() if not s.ends_turn}
        if declared != self._readers.keys():
            raise RuntimeError(f"read-only affordances {sorted(declared)} do not match handlers {sorted(self._readers)}")

    def schemas(self) -> list[JSON]:
        specs = sorted(self._game.ontology.affordances.values(), key=lambda s: (not s.ends_turn, s.function_name))
        return [tool_schema(s, self._game.rules, self._others) for s in specs]

    async def execute(self, name: str, raw_args: str | JSON) -> ToolOutcome:
        """Run one tool call. Never raises: errors become ``is_error`` outcomes."""
        spec = self._game.ontology.affordances.get(name)
        is_action = spec is not None and spec.ends_turn
        try:
            args = self._parse_args(raw_args)
            if spec is None:
                raise IllegalAction(f"Unknown tool {name!r}. Available: {sorted(self._game.ontology.affordances)}")
            if spec.ends_turn:
                action = self._build_action(name, args)
                self._game.check(self._seat, action)  # may raise IllegalAction
                return ToolOutcome(f"Accepted: {name}. Your turn is over.", action=action)
            return ToolOutcome(await self._readers[name](args))
        except IllegalAction as exc:
            # An unknown tool name counts as a failed action: it is a model mistake.
            return ToolOutcome(f"Error: {exc}", is_error=True, failed_action=is_action or spec is None)

    # -- read-only handlers ---------------------------------------------------

    @property
    def graph(self) -> Graph:
        if self._graph is None:
            self._graph = view_graph(self._game, self._view)
        return self._graph

    async def _query(self, args: JSON) -> str:
        query = str(args.get("sparql", ""))
        try:
            return await _call_with_timeout(run_sparql_select, self.graph, query, timeout=SPARQL_TIMEOUT_S)
        except TimeoutError:
            raise IllegalAction(f"Query took longer than {SPARQL_TIMEOUT_S:.0f}s; write a simpler query.") from None

    async def _take_notes(self, args: JSON) -> str:
        return store_notes(self._notes, self._seat, str(args.get("notes", "")))

    async def _get_rules(self, args: JSON) -> str:
        return describe_rules(self._game.rules)

    async def _get_state(self, args: JSON) -> str:
        return describe_state(self._view)

    async def _get_concepts(self, args: JSON) -> str:
        cat, name = args.get("category"), args.get("name")
        return describe_concepts(self._game.ontology, None if cat is None else str(cat), None if name is None else str(name))

    async def _estimate(self, args: JSON) -> str:
        return estimate_card_probabilities(self._view, self._game.rules, _as_int(args.get("position"), "position"))

    # -- helpers --------------------------------------------------------------

    @staticmethod
    def _parse_args(raw: str | JSON) -> JSON:
        if isinstance(raw, dict):
            return raw
        if not raw.strip():
            return {}
        try:
            parsed = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise IllegalAction(f"Tool arguments are not valid JSON: {exc}") from exc
        if not isinstance(parsed, dict):
            raise IllegalAction("Tool arguments must be a JSON object.")
        return parsed

    def _build_action(self, name: str, args: JSON) -> Action:
        match name:
            case "play_card":
                return PlayCard(_as_int(args.get("position"), "position"))
            case "discard_card":
                return DiscardCard(_as_int(args.get("position"), "position"))
            case "give_hint":
                target_name = args.get("target")
                names = self._game.state.names
                if target_name not in names:
                    raise IllegalAction(f"Unknown target {target_name!r}. Players: {self._others}.")
                color: Color | None = None
                if args.get("color") is not None:
                    try:
                        color = Color(str(args["color"]).lower())
                    except ValueError:
                        raise IllegalAction(f"Unknown colour {args['color']!r}.") from None
                rank = _as_int(args["rank"], "rank") if args.get("rank") is not None else None
                return GiveHint(names.index(str(target_name)), color=color, rank=rank)
        raise IllegalAction(f"Tool {name!r} is not an action.")
