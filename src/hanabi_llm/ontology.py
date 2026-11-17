"""Loads the Turtle world model and exposes it as typed Python objects.

The ontology is the source of truth for game constants, the card
distribution and the affordances. Python code *reads* it rather than
duplicating it, and ``verify_bindings`` fails loudly if the ontology names a
function that the engine has not implemented.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib import resources
from typing import Any, Final

from rdflib import RDF, Graph, Literal, Namespace, URIRef
from rdflib.term import Node

from .model import Color

HN: Final = Namespace("https://example.org/hanabi#")

#: Parameter types the ontology may use, mapped to JSON-schema in tools.py.
PARAM_TYPES: Final = frozenset({"integer", "string", "color", "rank", "player", "choice", "topic"})


@dataclass(frozen=True, slots=True)
class ParamSpec:
    name: str
    type: str
    description: str
    optional: bool
    choices: tuple[str, ...] = ()  # allowed values, for type "choice"


@dataclass(frozen=True, slots=True)
class AffordanceSpec:
    """One callable capability, as declared in the ontology."""

    iri: URIRef
    function_name: str
    description: str
    params: tuple[ParamSpec, ...]
    ends_turn: bool
    family: str  # action | state | rules
    required_checks: tuple[str, ...]  # checkFunction names run before applying
    effect_function: str
    precondition_rules: tuple[str, ...]  # rule names that must hold first
    effect_rules: tuple[str, ...]  # rule names applied when it resolves


@dataclass(frozen=True, slots=True)
class RuleSet:
    """Constants and the card distribution, read from the ontology."""

    colors: tuple[Color, ...]
    copies_by_rank: dict[int, int]
    hand_size_by_players: dict[int, int]
    max_hints: int
    initial_hints: int
    initial_fuses: int
    rule_texts: tuple[str, ...]

    @property
    def top_rank(self) -> int:
        return max(self.copies_by_rank)

    @property
    def deck_size(self) -> int:
        return len(self.colors) * sum(self.copies_by_rank.values())


@dataclass(frozen=True, slots=True)
class RuleDoc:
    """A rule as explained to a player (the `explain_rule` tool)."""

    name: str
    text: str
    rationale: str
    example: str
    kind: str  # "precondition" | "effect" | "game rule"
    applies_to: tuple[str, ...]  # tool names of the actions it concerns


@dataclass(frozen=True, slots=True)
class Concept:
    """A piece of game knowledge: a mechanic, a dynamic or a kinetic."""

    name: str
    category: str  # "mechanic" | "dynamic" | "kinetic"
    summary: str
    guidance: str


@dataclass(frozen=True, slots=True)
class Ontology:
    graph: Graph
    rules: RuleSet
    affordances: dict[str, AffordanceSpec]  # keyed by function_name
    concepts: dict[str, Concept]  # keyed by concept name
    rule_docs: dict[str, RuleDoc]  # keyed by rule name

    def referenced_functions(self) -> set[str]:
        """Every Python function name the ontology expects to exist."""
        names: set[str] = set()
        for _, _, o in self.graph.triples((None, HN.checkFunction, None)):
            names.add(str(o))
        for _, _, o in self.graph.triples((None, HN.effectFunction, None)):
            names.add(str(o))
        return names


def _int(g: Graph, s: Node, p: URIRef) -> int:
    value = g.value(s, p)
    if not isinstance(value, Literal):
        raise ValueError(f"ontology: {s} is missing integer property {p}")
    return int(value.toPython())


def _str(g: Graph, s: Node, p: URIRef, default: str | None = None) -> str:
    value = g.value(s, p)
    if value is None:
        if default is None:
            raise ValueError(f"ontology: {s} is missing string property {p}")
        return default
    return str(value)


def _bool(g: Graph, s: Node, p: URIRef, default: bool) -> bool:
    value = g.value(s, p)
    return bool(value.toPython()) if isinstance(value, Literal) else default


def _load_rules(g: Graph) -> RuleSet:
    colors = tuple(
        Color(str(g.value(c, HN.colorName)))
        for c in sorted(g.subjects(RDF.type, HN.Color), key=str)
    )
    copies = {
        _int(g, r, HN.rankValue): _int(g, r, HN.copiesPerColor)
        for r in g.subjects(RDF.type, HN.Rank)
    }
    hand_sizes = {
        _int(g, s, HN.playerCount): _int(g, s, HN.handSize)
        for s in g.subjects(RDF.type, HN.PlayerCountSetup)
    }
    texts = tuple(
        str(o)
        for _, o in sorted(
            ((s, o) for s, o in g.subject_objects(HN.ruleText)), key=lambda t: str(t[0])
        )
    )
    rules = RuleSet(
        colors=colors,
        copies_by_rank=copies,
        hand_size_by_players=hand_sizes,
        max_hints=_int(g, HN.Setup, HN.maxHintTokens),
        initial_hints=_int(g, HN.Setup, HN.initialHintTokens),
        initial_fuses=_int(g, HN.Setup, HN.initialFuseTokens),
        rule_texts=texts,
    )
    declared_total = _int(g, HN.Setup, HN.deckTotalCards)
    if rules.deck_size != declared_total:
        raise ValueError(
            f"ontology: card distribution gives {rules.deck_size} cards "
            f"but hn:deckTotalCards says {declared_total}"
        )
    return rules


def _load_affordances(g: Graph) -> dict[str, AffordanceSpec]:
    specs: dict[str, AffordanceSpec] = {}
    for aff in g.subjects(RDF.type, HN.Affordance):
        assert isinstance(aff, URIRef)
        params = []
        for p in g.objects(aff, HN.hasParameter):
            assert isinstance(p, URIRef)
            ptype = _str(g, p, HN.paramType)
            if ptype not in PARAM_TYPES:
                raise ValueError(f"ontology: unknown parameter type {ptype!r} on {p}")
            params.append(
                ParamSpec(
                    name=_str(g, p, HN.paramName),
                    type=ptype,
                    description=_str(g, p, HN.paramDescription, ""),
                    optional=_bool(g, p, HN.optional, False),
                    choices=tuple(c.strip() for c in _str(g, p, HN.choices, "").split(",") if c.strip()),
                )
            )
        # Stable ordering: required parameters first, then by name.
        params.sort(key=lambda s: (s.optional, s.name))
        checks = tuple(
            sorted(
                _str(g, rule, HN.checkFunction)
                for rule in g.objects(aff, HN.requires)
            )
        )
        name = _str(g, aff, HN.functionName)
        pre_rules = tuple(sorted(_str(g, r, HN.ruleName) for r in g.objects(aff, HN.requires)))
        eff_rules = tuple(sorted(_str(g, r, HN.ruleName) for r in g.objects(aff, HN.appliesRule)))
        specs[name] = AffordanceSpec(
            iri=aff,
            function_name=name,
            description=_str(g, aff, HN.description),
            params=tuple(params),
            ends_turn=_bool(g, aff, HN.endsTurn, False),
            family=_str(g, aff, HN.family),
            required_checks=checks,
            effect_function=_str(g, aff, HN.effectFunction),
            precondition_rules=pre_rules,
            effect_rules=eff_rules,
        )
    return specs


def _load_rule_docs(g: Graph) -> dict[str, RuleDoc]:
    docs: dict[str, RuleDoc] = {}
    for rule in g.subjects(RDF.type, HN.Rule):
        name = _str(g, rule, HN.ruleName)
        required_by = {_str(g, a, HN.functionName) for a in g.subjects(HN.requires, rule)}
        applied_by = {_str(g, a, HN.functionName) for a in g.subjects(HN.appliesRule, rule)}
        kind = "precondition" if required_by else "effect" if applied_by else "game rule"
        docs[name] = RuleDoc(
            name=name,
            text=_str(g, rule, HN.ruleText),
            rationale=_str(g, rule, HN.rationale, ""),
            example=_str(g, rule, HN.example, ""),
            kind=kind,
            applies_to=tuple(sorted(required_by | applied_by)),
        )
    return docs


_CATEGORIES = {HN.Mechanic: "mechanic", HN.Dynamic: "dynamic", HN.Kinetic: "kinetic"}


def _load_concepts(g: Graph) -> dict[str, Concept]:
    concepts: dict[str, Concept] = {}
    for class_iri, category in _CATEGORIES.items():
        for node in g.subjects(RDF.type, class_iri):
            name = _str(g, node, HN.conceptName)
            concepts[name] = Concept(
                name=name,
                category=category,
                summary=_str(g, node, HN.summary),
                guidance=_str(g, node, HN.guidance),
            )
    return concepts


def load_ontology() -> Ontology:
    """Parse the bundled Turtle file."""
    g = Graph()
    g.bind("hn", HN)
    text = resources.files("hanabi_llm").joinpath("hanabi.ttl").read_text("utf-8")
    g.parse(data=text, format="turtle")
    return Ontology(
        graph=g,
        rules=_load_rules(g),
        affordances=_load_affordances(g),
        concepts=_load_concepts(g),
        rule_docs=_load_rule_docs(g),
    )


def verify_bindings(onto: Ontology, registry: dict[str, Any]) -> None:
    """Fail if the ontology references a function the engine has not bound."""
    missing = sorted(onto.referenced_functions() - registry.keys())
    if missing:
        raise RuntimeError(f"ontology references unbound functions: {missing}")
