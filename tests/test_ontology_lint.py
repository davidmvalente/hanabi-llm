"""Mechanical lint of the ontology and of every graph the code emits.

These checks would have caught the first draft's undeclared ``hn:Game``, an
effect function on the wrong domain, untyped data, and dead vocabulary.
"""

from __future__ import annotations

import asyncio
import re
from pathlib import Path

import pytest
from rdflib import RDF, RDFS, Graph, Literal, URIRef

import hanabi_llm.engine  # noqa: F401  (registers the bound functions)
import hanabi_llm.tools  # noqa: F401
from hanabi_llm.engine import Game
from hanabi_llm.ontology import HN, available_knowledge, load_ontology
from hanabi_llm.players import RandomPlayer
from hanabi_llm.registry import FUNCTIONS
from hanabi_llm.runner import play_game
from hanabi_llm.validate import truth_graph
from hanabi_llm.views import build_view, view_graph

HN_NS = str(HN)
SRC = Path(__file__).parent.parent / "src" / "hanabi_llm"


def local(term: object) -> str:
    return str(term).removeprefix(HN_NS)


@pytest.fixture(scope="module", params=[(), ("mdk",)], ids=["baseline", "with-mdk"])
def onto(request: pytest.FixtureRequest) -> Graph:
    """The ontology as a game sees it: the baseline, and with each optional knowledge graph loaded."""
    return load_ontology(request.param).graph


@pytest.fixture(scope="module")
def everything() -> Graph:
    """The core plus every knowledge graph: every term any part of the code may reference."""
    return load_ontology(available_knowledge()).graph


@pytest.fixture(scope="module")
def game_graphs() -> Graph:
    """Every triple the code emits over two whole games: truth and each seat's view, at every turn."""
    union = Graph()

    async def play(seed: int, names: list[str]) -> None:
        game = Game(names, seed=seed)

        def collect(_t: object = None) -> None:
            for t in truth_graph(game):
                union.add(t)
            for seat in range(len(game.state.names)):
                for t in view_graph(game, build_view(game, seat)):
                    union.add(t)

        collect()
        await play_game(game, [RandomPlayer(n, seed=seed + i) for i, n in enumerate(names)], on_turn=collect)
        collect()

    async def both() -> None:
        await play(3, ["a", "b", "c"])
        await play(5, ["a", "b"])

    asyncio.run(both())
    return union


def hn_individuals(g: Graph) -> set[URIRef]:
    return {s for s in g.subjects() if isinstance(s, URIRef) and str(s).startswith(HN_NS)}


def declared(g: Graph, kind: URIRef) -> set[URIRef]:
    return {s for s in g.subjects(RDF.type, kind) if str(s).startswith(HN_NS)}


# --- declarations ------------------------------------------------------------


def test_every_type_used_is_declared(onto: Graph) -> None:
    used = {o for _, o in onto.subject_objects(RDF.type) if str(o).startswith(HN_NS)}
    assert {local(t) for t in used - declared(onto, RDFS.Class)} == set()


def test_every_term_the_code_references_is_declared(everything: Graph) -> None:
    """``HN.Something`` in the source must exist in the ontology (this is the hn:Game bug)."""
    referenced = {m for f in SRC.glob("*.py") for m in re.findall(r"\bHN\.([A-Za-z]+)\b", f.read_text())}
    known = {local(s) for s in hn_individuals(everything)}
    assert referenced - known == set()


def test_no_untyped_data_individuals(onto: Graph) -> None:
    meta = declared(onto, RDFS.Class) | declared(onto, RDF.Property)
    untyped = {local(s) for s in hn_individuals(onto) - meta
               if not list(onto.objects(s, RDF.type)) and any(p not in (RDFS.comment, RDFS.label, RDFS.subClassOf)
                                                              for p in onto.predicates(s))}
    assert untyped == set()


# --- usage -------------------------------------------------------------------


def violations(g: Graph) -> list[str]:
    """Domain, class-range and datatype-range violations, over the whole graph."""
    from rdflib import XSD  # imported here: a module-level rdflib namespace makes pytest warn

    out: list[str] = []
    for prop, dom in g.subject_objects(RDFS.domain):
        if not str(prop).startswith(HN_NS):
            continue
        allowed = set(g.transitive_subjects(RDFS.subClassOf, dom))
        for s in set(g.subjects(prop, None)):
            ok = (s, RDF.type, RDFS.Class) in g if dom == RDFS.Class else bool(set(g.objects(s, RDF.type)) & allowed)
            if not ok:
                out.append(f"domain: {local(s)} uses {local(prop)} but is not a {local(dom)}")
    for prop, rng in g.subject_objects(RDFS.range):
        if not str(prop).startswith(HN_NS):
            continue
        for o in set(g.objects(None, prop)):
            if str(rng).startswith(str(XSD)):
                if not isinstance(o, Literal) or (o.datatype or XSD.string) != rng:
                    out.append(f"range: {local(prop)} has {o!r}, expected {local(rng)}")
            elif isinstance(o, URIRef) and not (set(g.objects(o, RDF.type)) & set(g.transitive_subjects(RDFS.subClassOf, rng))):
                out.append(f"range: {local(prop)} points at {local(o)}, which is not a {local(rng)}")
    return sorted(set(out))


def test_the_ontology_respects_its_own_domains_and_ranges(onto: Graph) -> None:
    assert violations(onto) == []


def test_every_graph_the_code_emits_respects_domains_and_ranges(game_graphs: Graph) -> None:
    assert violations(game_graphs) == []


def test_no_dead_vocabulary(onto: Graph, game_graphs: Graph) -> None:
    """Every declared class is instantiated (or an abstract parent, or states a regime); every property is used."""
    everything = onto + game_graphs
    instantiated = {o for _, o in everything.subject_objects(RDF.type)}
    parents = set(onto.objects(None, RDFS.subClassOf))
    regime_bearing = set(onto.subjects(HN.identityVisibility, None))
    dead_classes = declared(onto, RDFS.Class) - instantiated - parents - regime_bearing
    used_props = {p for _, p, _ in everything}
    dead_props = declared(onto, RDF.Property) - used_props
    assert (sorted(map(local, dead_classes)), sorted(map(local, dead_props))) == ([], [])


# --- the ontology and the code agree both ways ---------------------------------


def test_every_bound_function_is_named_in_the_ontology_and_vice_versa() -> None:
    assert set(FUNCTIONS) == load_ontology(available_knowledge()).referenced_functions()


def test_measured_by_points_at_declared_terms(onto: Graph) -> None:
    subjects = {s for s in onto.subjects() if isinstance(s, URIRef)}
    assert {local(o) for o in onto.objects(None, HN.measuredBy) if o not in subjects} == set()


def test_documentation_is_complete(onto: Graph) -> None:
    for rule in declared(onto, HN.Rule):
        for p in (HN.ruleName, HN.ruleText, HN.rationale, HN.example):
            assert str(onto.value(rule, p) or "").strip(), f"{local(rule)} lacks {local(p)}"
    for aff in declared(onto, HN.Affordance):
        for p in (HN.functionName, HN.description, HN.family, HN.effectFunction, HN.endsTurn):
            assert onto.value(aff, p) is not None, f"{local(aff)} lacks {local(p)}"
    for param in declared(onto, HN.Parameter):
        for p in (HN.paramName, HN.paramType, HN.paramDescription):
            assert str(onto.value(param, p) or "").strip(), f"{local(param)} lacks {local(p)}"


def test_every_concept_is_complete(onto: Graph) -> None:
    for concept in declared(onto, HN.Concept):
        for p in (HN.conceptName, HN.category, HN.summary, HN.guidance):
            assert str(onto.value(concept, p) or "").strip(), f"{local(concept)} lacks {local(p)}"
