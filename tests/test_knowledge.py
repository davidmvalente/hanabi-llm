"""Knowledge graphs are optional arms. The baseline contains none, anywhere a model can reach."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from hanabi_llm.__main__ import build_parser, run
from hanabi_llm.engine import Game
from hanabi_llm.explain import topics
from hanabi_llm.feed import read_feed
from hanabi_llm.model import IllegalAction
from hanabi_llm.ontology import available_knowledge, load_ontology
from hanabi_llm.players import LLMPlayer
from hanabi_llm.prompt import load_prompt
from hanabi_llm.states import SECTIONS
from hanabi_llm.tools import ToolBox
from hanabi_llm.views import build_view

NAMES = ["Alice", "Bob", "Carol"]


def toolbox(game: Game, seat: int = 0) -> ToolBox:
    return ToolBox(game, seat, build_view(game, seat))


def test_the_available_graphs_exclude_the_shared_machinery() -> None:
    assert available_knowledge() == ["mdk"]


def test_the_baseline_has_no_concepts_and_no_knowledge_tool() -> None:
    onto = load_ontology()
    assert onto.concepts == {} and onto.knowledge == ()
    assert "get_concepts" not in onto.affordances
    names = [s["function"]["name"] for s in toolbox(Game(NAMES, seed=1)).schemas()]
    assert names == ["discard_card", "give_hint", "play_card", "get_state", "explain_rules"]


def test_no_strategy_text_reaches_a_baseline_model() -> None:
    """Everything a model can receive in the baseline (prompt, tool schemas, every tool answer) is free of the notes."""
    notes = load_ontology(["mdk"]).concepts.values()
    assert len(notes) > 10
    game = Game(NAMES, seed=1)
    tb = toolbox(game)
    seen = [LLMPlayer("Alice", None, game, 0).prompt.render(name="Alice", players="Alice, Bob, Carol", max_calls=8, max_failed_actions=3),  # type: ignore[arg-type]
            json.dumps(tb.schemas())]
    seen += [tb.execute("get_state", {"section": s}).text for s in SECTIONS]
    seen += [tb.execute("explain_rules", {}).text] + [tb.execute("explain_rules", {"topic": t}).text for t in topics(game.ontology)]
    haystack = "\n".join(seen).lower()
    for note in notes:
        assert note.guidance.lower() not in haystack and note.summary.lower() not in haystack


def test_with_a_knowledge_graph_the_tool_appears_and_serves_its_notes() -> None:
    game = Game(NAMES, seed=1, ontology=load_ontology(["mdk"]))
    schemas = {s["function"]["name"]: s for s in toolbox(game).schemas()}
    assert list(schemas)[-1] == "get_concepts"                                   # the knowledge family comes last
    assert game.ontology.affordances["get_concepts"].family == "knowledge"
    tb = toolbox(game)
    overview = tb.execute("get_concepts", {}).text
    for category in ("[mechanic]", "[dynamic]", "[kinetic]"):
        assert category in overview
    assert "card_mass" in tb.execute("get_concepts", {"category": "kinetic"}).text
    assert "hand_shift" not in tb.execute("get_concepts", {"category": "kinetic"}).text
    one = tb.execute("get_concepts", {"name": "Activation-Energy"}).text           # forgiving names
    assert one.startswith("[kinetic] activation_energy") and "Guidance:" in one


def test_note_mistakes_are_free_like_every_read_tool() -> None:
    tb = toolbox(Game(NAMES, seed=1, ontology=load_ontology(["mdk"])))
    for args in ({"name": "nope"}, {"category": "nope"}):
        out = tb.execute("get_concepts", args)
        assert out.is_error and not out.failed_action


def test_unknown_graphs_are_refused_and_duplicates_ignored() -> None:
    with pytest.raises(ValueError, match="unknown knowledge graph.*mdk"):
        load_ontology(["strategy-x"])
    assert load_ontology(["mdk", "mdk"]).knowledge == ("mdk",)
    with pytest.raises(ValueError):
        load_ontology(["_concepts"])                                              # infrastructure, not selectable


def test_note_names_never_collide_with_rule_or_action_names() -> None:
    """One name, one meaning: `get_concepts name=x` and `explain_rules topic=x` must never both answer."""
    onto = load_ontology(available_knowledge())
    assert set(onto.concepts) & (set(onto.rule_docs) | set(onto.affordances)) == set()


def test_the_notes_never_mention_tools_that_do_not_exist() -> None:
    game = Game(NAMES, seed=1, ontology=load_ontology(["mdk"]))
    text = " ".join(c.summary + " " + c.guidance for c in game.ontology.concepts.values()).lower()
    for gone in ("estimate_card", "query_knowledge", "take_notes", "get_card_census", "sparql"):
        assert gone not in text


def test_the_run_records_which_knowledge_it_used(tmp_path: Path) -> None:
    def header(*extra: str) -> dict:  # type: ignore[type-arg]
        log = tmp_path / f"{len(extra)}.jsonl"
        asyncio.run(run(build_parser().parse_args(["--models", "random,random,random", "--seed", "2", "--log", str(log), *extra])))
        return read_feed(log)[0]

    assert header()["config"]["knowledge"] == []                                  # the baseline says so explicitly
    assert header("--knowledge", "mdk")["config"]["knowledge"] == ["mdk"]
