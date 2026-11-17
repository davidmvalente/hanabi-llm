"""The tool layer: three families (action, state, rules), and its error accounting."""

from __future__ import annotations

import pytest

import hanabi_llm.tools as tools
from conftest import B, R, Y, rig
from hanabi_llm.engine import Game
from hanabi_llm.explain import topics
from hanabi_llm.model import DiscardCard, GiveHint, PlayCard
from hanabi_llm.states import SECTIONS
from hanabi_llm.tools import ToolBox
from hanabi_llm.views import build_view


def toolbox(game: Game, seat: int = 0) -> ToolBox:
    return ToolBox(game, seat, build_view(game, seat))


def rigged() -> Game:
    return rig(Game(["Ann", "Bob"], seed=0), [[(R, 1)] * 5, [(B, 1)] * 5], deck=[(Y, 2)] * 10)


def test_tool_set_is_minimal_and_partitioned_into_three_families(game2: Game) -> None:
    affs = game2.ontology.affordances
    by_family = {f: {n for n, a in affs.items() if a.family == f} for f in ("action", "state", "rules")}
    assert by_family == {"action": {"play_card", "discard_card", "give_hint"},
                         "state": {"get_state"}, "rules": {"explain_rules"}}
    assert sum(len(v) for v in by_family.values()) == len(affs)           # no tool outside a family
    assert {n for n, a in affs.items() if a.ends_turn} == by_family["action"]  # only actions end the turn


def test_schemas(game2: Game) -> None:
    schemas = {s["function"]["name"]: s["function"]["parameters"] for s in toolbox(game2).schemas()}
    assert list(schemas) == ["discard_card", "give_hint", "play_card", "get_state", "explain_rules"]  # action, state, rules
    assert schemas["give_hint"]["required"] == ["hint", "player"]
    assert schemas["give_hint"]["properties"]["player"]["enum"] == ["Bob"]          # never offers yourself
    assert schemas["give_hint"]["properties"]["hint"]["enum"] == ["blue", "green", "red", "white", "yellow", "1", "2", "3", "4", "5"]
    assert schemas["get_state"]["properties"]["section"]["enum"] == list(SECTIONS)
    assert schemas["get_state"]["required"] == [] and schemas["explain_rules"]["required"] == []
    assert schemas["explain_rules"]["properties"]["topic"]["enum"] == topics(game2.ontology)


def test_ontology_and_code_must_agree_on_sections(game2: Game, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tools, "SECTIONS", ("turn",))
    with pytest.raises(RuntimeError, match="sections"):
        toolbox(game2)


def test_actions_are_validated_not_applied() -> None:
    game = rigged()
    tb = toolbox(game)
    before = (game.state.turn, len(game.state.events), game.state.hints)
    assert tb.execute("play_card", {"slot": 0}).action == PlayCard(0)
    assert tb.execute("discard_card", {"slot": "2"}).action == DiscardCard(2)   # numeric strings tolerated
    assert tb.execute("give_hint", {"player": "Bob", "hint": "blue"}).action == GiveHint(1, B)
    assert tb.execute("give_hint", {"player": "Bob", "hint": "1"}).action == GiveHint(1, 1)    # the attribute follows from the value
    assert tb.execute("give_hint", {"player": "Bob", "hint": 1}).action == GiveHint(1, 1)      # a bare integer is tolerated
    assert tb.execute("give_hint", {"player": "Bob", "hint": "Blue"}).action == GiveHint(1, B)
    assert (game.state.turn, len(game.state.events), game.state.hints) == before


def test_rejected_actions_count_but_read_mistakes_are_free(game2: Game) -> None:
    tb = toolbox(game2)
    for call in (("play_card", "not json"), ("play_card", {"slot": 99}), ("give_hint", {"player": "Nobody", "hint": "red"}),
                 ("give_hint", {"player": "Ann", "hint": "red"}), ("give_hint", {"player": "Bob"}),
                 ("give_hint", {"player": "Bob", "hint": "purple"}), ("give_hint", {"player": "Bob", "hint": True}),
                 ("nonsense", {})):
        out = tb.execute(*call)
        assert out.is_error and out.failed_action, call
    for call in (("get_state", {"section": "everything"}), ("explain_rules", {"topic": "nope"})):
        out = tb.execute(*call)
        assert out.is_error and not out.failed_action, call


def test_get_state_tool_returns_sections(game2: Game) -> None:
    tb = toolbox(game2)
    assert tb.execute("get_state", {}).text.count("## ") == len(SECTIONS)
    only = tb.execute("get_state", {"section": "tokens"}).text
    assert only.startswith("## tokens") and "## turn" not in only


def test_explain_overview_lists_actions_and_every_rule(game2: Game) -> None:
    out = toolbox(game2).execute("explain_rules", {}).text
    for name in ("play_card", "discard_card", "give_hint", *game2.ontology.rule_docs):
        assert name in out
    assert "Spend one hint token" in out and "ACTION (ends your turn)" not in out   # real descriptions, no marker


def test_explain_a_rule_gives_text_reason_and_example(game2: Game) -> None:
    out = toolbox(game2).execute("explain_rules", {"topic": "hint_token_available"}).text
    assert "at least one hint token" in out and "Why:" in out and "Example:" in out and "give_hint" in out
    assert toolbox(game2).execute("explain_rules", {"topic": "Hint-Token-Available"}).text == out   # forgiving names


def test_explain_an_action_gives_parameters_and_its_rules(game2: Game) -> None:
    out = toolbox(game2).execute("explain_rules", {"topic": "give_hint"}).text
    assert "player (required)" in out and "hint (required)" in out and "color (optional)" not in out
    for rule in ("hint_token_available", "hint_receiver_is_other", "hint_targets_cards", "hint_one_attribute", "hint_is_complete"):
        assert rule in out
    assert "EVERY card" in toolbox(game2).execute("explain_rules", {"topic": "hint_is_complete"}).text
    assert "misfire_costs_fuse" in toolbox(game2).execute("explain_rules", {"topic": "play_card"}).text


def test_rule_kinds_come_from_the_ontology(game2: Game) -> None:
    docs = game2.ontology.rule_docs
    assert docs["hint_token_available"].kind == "precondition" and docs["hint_refund"].kind == "effect"
    assert docs["game_ends"].kind == "game rule" and docs["card_in_hand"].applies_to == ("discard_card", "play_card")


def test_hidden_cards_never_reach_the_model_through_any_read_tool() -> None:
    """Two games differing only in Ann's own cards and the deck: every read tool answers identically."""
    from conftest import G

    def answers(hand: list, deck: list) -> list[str]:  # type: ignore[type-arg]
        game = rig(Game(["Ann", "Bob"], seed=0), [hand, [(B, 1)] * 5], deck=deck)
        tb = toolbox(game)
        calls = [("get_state", {}), *[("get_state", {"section": s}) for s in SECTIONS],
                 ("explain_rules", {}), *[("explain_rules", {"topic": t}) for t in topics(game.ontology)]]
        return [tb.execute(n, a).text for n, a in calls]

    assert answers([(R, 1)] * 5, [(Y, 2)] * 10) == answers([(G, 4), (G, 5), (Y, 3), (B, 2), (R, 5)], [(Y, 5)] * 10)


def test_wrong_argument_names_get_an_error_that_says_what_the_tool_takes(game2: Game) -> None:
    """A model that mixes up argument names (say, from an older tool shape) is told how to fix the call."""
    tb = toolbox(game2)
    old_shape = tb.execute("give_hint", {"target": "Bob", "color": "red"})
    assert old_shape.is_error and old_shape.failed_action
    assert "unexpected argument(s) ['color', 'target']" in old_shape.text and "missing required argument(s) ['hint', 'player']" in old_shape.text
    assert "It takes: hint (required), player (required)." in old_shape.text
    half = tb.execute("give_hint", {"player": "Bob"})
    assert "missing required argument(s) ['hint']" in half.text and "unexpected" not in half.text
    assert "play_card: missing required argument(s) ['slot']" in tb.execute("play_card", {}).text
    assert "It takes: slot (required)." in tb.execute("discard_card", {"pos": 1}).text


def test_argument_mistakes_in_read_tools_are_free_and_just_as_clear(game2: Game) -> None:
    tb = toolbox(game2)
    out = tb.execute("get_state", {"part": "tokens"})
    assert out.is_error and not out.failed_action
    assert "unexpected argument(s) ['part']" in out.text and "It takes: section (optional)." in out.text
    assert not tb.execute("get_state", {}).is_error and not tb.execute("explain_rules", {}).is_error     # all-optional tools still work bare
    assert tb.execute("play_card", {"slot": 0, "extra": 1}).failed_action                            # extras are refused, not ignored
