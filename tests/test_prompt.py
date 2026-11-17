"""The system prompt: a versioned YAML file with a few run-setting variables, nothing from the graph."""

from __future__ import annotations

import re
from importlib import resources

import pytest

from hanabi_llm.engine import Game
from hanabi_llm.prompt import available_versions, load_prompt, parse_prompt

GOOD = b"version: 3\ndescription: d\nvariables: [name]\nsystem: |\n  Hello ${name}.\n"


def test_loads_the_latest_version_by_default() -> None:
    assert available_versions() == [1]
    assert load_prompt().version == 1
    assert load_prompt("v1") == load_prompt(1) == load_prompt("1")


def test_unknown_version_is_an_error() -> None:
    with pytest.raises(ValueError, match="no system prompt version 9"):
        load_prompt(9)


def test_hash_identifies_the_exact_file() -> None:
    import hashlib

    raw = resources.files("hanabi_llm").joinpath("prompts", "system.v1.yaml").read_bytes()
    assert load_prompt(1).sha256 == hashlib.sha256(raw).hexdigest()


def test_render_fills_exactly_the_declared_variables() -> None:
    spec = parse_prompt(GOOD, 3)
    assert spec.render(name="Ann") == "Hello Ann.\n"
    for bad in ({}, {"name": "Ann", "extra": 1}):
        with pytest.raises(ValueError, match="needs exactly"):
            spec.render(**bad)


@pytest.mark.parametrize(("raw", "needle"), [
    (b"- not a mapping", "mapping"),
    (b"version: 3\nsystem: x\n", "missing key"),
    (GOOD.replace(b"version: 3", b"version: 4"), "declares version"),
    (GOOD.replace(b"[name]", b"[name, unused]"), "declared variables"),
    (GOOD.replace(b"${name}", b"${name} ${typo}"), "declared variables"),
])
def test_malformed_files_fail_at_load_time(raw: bytes, needle: str) -> None:
    with pytest.raises(ValueError, match=needle):
        parse_prompt(raw, 3)


def test_rendered_prompt_has_the_run_settings_and_no_leftover_placeholders() -> None:
    game = Game(["Ann", "Bob", "Cat"], seed=1)
    text = load_prompt().render(name="Ann", players=", ".join(game.state.names), max_calls=8, max_failed_actions=3)
    assert "You are Ann." in text and "Ann, Bob, Cat" in text and "8 model calls" in text and "3 rejected actions" in text
    assert "${" not in text


def test_nothing_is_read_from_the_graph() -> None:
    """Rules, colours and constants are learned through the tools, never stated in the prompt."""
    game = Game(["Ann", "Bob"], seed=1)
    text = load_prompt().template.lower()
    for doc in game.ontology.rule_docs.values():
        assert doc.text.lower() not in text
    for color in game.rules.colors:
        assert color.value not in text
    assert "capacity" not in text and "fuse" not in text


def test_every_tool_name_in_the_prompt_exists() -> None:
    """Static text must not drift from the ontology's tools."""
    game = Game(["Ann", "Bob"], seed=1)
    text = re.sub(r"\$\{[^}]*\}", "", load_prompt().template)   # ignore placeholders
    mentioned = set(re.findall(r"\b[a-z]+_[a-z]+\b", text))
    assert mentioned and mentioned <= set(game.ontology.affordances)
