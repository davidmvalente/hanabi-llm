"""The tools a model can call, built from the ontology's affordances.

Three families, mutually exclusive and together exhaustive, plus an optional fourth:

  action   play_card, discard_card, give_hint   change the game; the first
                                                accepted one ends the turn
  state    get_state                            read what is true now
  rules    explain_rules                        read how the game works
  knowledge get_concepts                        only with a knowledge graph loaded; never in the baseline

Action tools are validated here with ``Game.check`` but NOT applied: the
referee applies them, so a model has no path to mutate the game except
``Game.apply``. Read tools are dispatched through the same function registry
as the engine (``hn:effectFunction`` -> ``@bind``), so the ontology and the
code cannot drift apart.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .engine import Game
from .explain import explain, explain_concepts, topics
from .model import Action, Color, DiscardCard, GiveHint, HintValue, IllegalAction, PlayCard, Seat
from .ontology import AffordanceSpec, ParamSpec, RuleSet
from .registry import FUNCTIONS, bind
from .states import SECTIONS, state_text
from .views import PlayerView

JSON = dict[str, Any]


@dataclass(frozen=True, slots=True)
class ToolContext:
    """Everything a read tool may look at. A view, never the hidden state."""

    game: Game
    seat: Seat
    view: PlayerView


# ---------------------------------------------------------------------------
# Read tools (bound to the effect function names declared in the ontology)
# ---------------------------------------------------------------------------


@bind("tool_get_state")
def _tool_get_state(ctx: ToolContext, args: JSON) -> str:
    section = args.get("section")
    return state_text(ctx.view, ctx.game.rules, None if section is None else str(section))


@bind("tool_explain_rules")
def _tool_explain_rules(ctx: ToolContext, args: JSON) -> str:
    topic = args.get("topic")
    return explain(ctx.game.ontology, None if topic is None else str(topic))


@bind("tool_get_concepts")
def _tool_get_concepts(ctx: ToolContext, args: JSON) -> str:
    """Only offered when a knowledge graph is loaded (the baseline has no such tool)."""
    cat, name = args.get("category"), args.get("name")
    return explain_concepts(ctx.game.ontology, None if cat is None else str(cat), None if name is None else str(name))


# ---------------------------------------------------------------------------
# Schema generation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SchemaContext:
    """Game-specific values that fill in the parameter enums."""

    rules: RuleSet
    others: list[str]   # the other players' names (never offers yourself)
    topics: list[str]   # actions and rules that explain_rules accepts


def _param_schema(p: ParamSpec, ctx: SchemaContext) -> JSON:
    base: JSON
    match p.type:
        case "integer":
            base = {"type": "integer", "minimum": 0}
        case "string":
            base = {"type": "string"}
        case "hint":  # a colour or a rank; one field, so "both" or "neither" cannot be written
            base = {"type": "string", "enum": [c.value for c in ctx.rules.colors] + [str(r) for r in sorted(ctx.rules.copies_by_rank)]}
        case "player":
            base = {"type": "string", "enum": ctx.others}
        case "choice":
            base = {"type": "string", "enum": list(p.choices)}
        case "topic":
            base = {"type": "string", "enum": ctx.topics}
        case other:  # pragma: no cover - guarded by the ontology loader
            raise ValueError(f"unknown parameter type {other}")
    base["description"] = p.description
    return base


def tool_schema(spec: AffordanceSpec, ctx: SchemaContext) -> JSON:
    """OpenAI-style function tool definition derived from one affordance."""
    return {
        "type": "function",
        "function": {
            "name": spec.function_name,
            "description": spec.description,
            "parameters": {
                "type": "object",
                "properties": {p.name: _param_schema(p, ctx) for p in spec.params},
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
    #: in a read tool is free (bounded by the per-turn call budget instead).
    failed_action: bool = False


def _as_int(value: object, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | str):
        raise IllegalAction(f"`{what}` must be an integer.")
    try:
        return int(value)
    except ValueError:
        raise IllegalAction(f"`{what}` must be an integer, got {value!r}.") from None


def parse_hint(raw: object, rules: RuleSet) -> HintValue:
    """A hint is a colour name or a rank. Ranks may arrive as 3 or "3"; the attribute follows from the value."""
    if isinstance(raw, bool) or raw is None:
        raise IllegalAction("`hint` is required: a colour or a rank.")
    text = str(raw).strip().lower()
    if text in {c.value for c in rules.colors}:
        return Color(text)
    if text.isdigit():
        return int(text)
    raise IllegalAction(f"Unknown hint {raw!r}: name a colour {[c.value for c in rules.colors]} or a rank {sorted(rules.copies_by_rank)}.")


def build_action(game: Game, name: str, args: JSON) -> Action:
    """Turn tool arguments into an ``Action``. Raises ``IllegalAction`` if malformed."""
    match name:
        case "play_card":
            return PlayCard(_as_int(args.get("position"), "position"))
        case "discard_card":
            return DiscardCard(_as_int(args.get("position"), "position"))
        case "give_hint":
            names = game.state.names
            player = args.get("player")
            if player not in names:
                raise IllegalAction(f"Unknown player {player!r}. Players: {names}.")
            return GiveHint(names.index(str(player)), parse_hint(args.get("hint"), game.rules))
    raise IllegalAction(f"Tool {name!r} is not an action.")


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


class ToolBox:
    """The tools available to one seat during one turn."""

    def __init__(self, game: Game, seat: Seat, view: PlayerView) -> None:
        self._game = game
        self._ctx = ToolContext(game, seat, view)
        others = [n for i, n in enumerate(game.state.names) if i != seat]
        self._schema_ctx = SchemaContext(game.rules, others, topics(game.ontology))
        self._check_wiring()

    def _check_wiring(self) -> None:
        """Fail early if the ontology and the code disagree about the tools."""
        affs = self._game.ontology.affordances
        for spec in affs.values():
            if not spec.ends_turn and spec.effect_function not in FUNCTIONS:
                raise RuntimeError(f"read tool {spec.function_name!r} has no bound function {spec.effect_function!r}")
        declared = affs["get_state"].params[0].choices
        if declared != SECTIONS:
            raise RuntimeError(f"get_state sections in the ontology {declared} differ from the code {SECTIONS}")

    _FAMILY_ORDER = ("action", "state", "rules", "knowledge")

    def schemas(self) -> list[JSON]:
        specs = sorted(self._game.ontology.affordances.values(),
                       key=lambda s: (self._FAMILY_ORDER.index(s.family), s.function_name))
        return [tool_schema(s, self._schema_ctx) for s in specs]

    @staticmethod
    def _check_arguments(spec: AffordanceSpec, args: JSON) -> None:
        """Say exactly which arguments are wrong and what the tool takes, so a model can fix the call."""
        allowed = {p.name for p in spec.params}
        extra = sorted(set(args) - allowed)
        missing = sorted(p.name for p in spec.params if not p.optional and p.name not in args)
        if not extra and not missing:
            return
        problems = ([f"unexpected argument(s) {extra}"] if extra else []) + ([f"missing required argument(s) {missing}"] if missing else [])
        takes = ", ".join(f"{p.name} ({'optional' if p.optional else 'required'})" for p in spec.params) or "no arguments"
        raise IllegalAction(f"{spec.function_name}: {' and '.join(problems)}. It takes: {takes}.")

    def execute(self, name: str, raw_args: str | JSON) -> ToolOutcome:
        """Run one tool call. Never raises: errors become ``is_error`` outcomes."""
        spec = self._game.ontology.affordances.get(name)
        is_action = spec is not None and spec.ends_turn
        try:
            args = _parse_args(raw_args)
            if spec is None:
                raise IllegalAction(f"Unknown tool {name!r}. Available: {sorted(self._game.ontology.affordances)}")
            self._check_arguments(spec, args)
            if spec.ends_turn:
                action = build_action(self._game, name, args)
                self._game.check(self._ctx.seat, action)  # may raise IllegalAction
                return ToolOutcome(f"Accepted: {name}. Your turn is over.", action=action)
            return ToolOutcome(FUNCTIONS[spec.effect_function](self._ctx, args))
        except IllegalAction as exc:
            # An unknown tool name counts as a failed action: it is a model mistake.
            return ToolOutcome(f"Error: {exc}", is_error=True, failed_action=is_action or spec is None)
