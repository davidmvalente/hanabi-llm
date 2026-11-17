"""Rule and action explanations, generated from the ontology.

Nothing here is hand-written prose about the rules: the text, rationale and
examples live in ``hanabi.ttl`` and are only formatted here, so explanations
cannot drift from what the engine enforces.
"""

from __future__ import annotations

from .model import IllegalAction
from .ontology import AffordanceSpec, Ontology, RuleDoc

_PREFIX = "ACTION (ends your turn). "  # every action description starts with this marker


def _what_it_does(spec: AffordanceSpec) -> str:
    """The action's description without the marker that tells the model it ends the turn."""
    return spec.description.removeprefix(_PREFIX)


_WHEN = {
    "precondition": "Checked BEFORE the action; if it does not hold the action is rejected.",
    "effect": "Applied WHEN the action resolves.",
    "game rule": "A rule about the game as a whole.",
}


def topics(onto: Ontology) -> list[str]:
    """Everything `explain_rules` accepts: the turn-ending actions and all rule names."""
    actions = sorted(n for n, a in onto.affordances.items() if a.ends_turn)
    return actions + sorted(onto.rule_docs)


def _rule(doc: RuleDoc) -> str:
    lines = [f"{doc.name} ({doc.kind})", f"  Rule: {doc.text}", f"  When: {_WHEN[doc.kind]}"]
    if doc.rationale:
        lines.append(f"  Why: {doc.rationale}")
    if doc.example:
        lines.append(f"  Example: {doc.example}")
    if doc.applies_to:
        lines.append(f"  Concerns: {', '.join(doc.applies_to)}")
    return "\n".join(lines)


def _action(onto: Ontology, spec: AffordanceSpec) -> str:
    lines = [f"{spec.function_name} (action, ends your turn): {_what_it_does(spec)}"]
    if spec.params:
        lines.append("  Parameters:")
        lines += [f"    {p.name} ({'optional' if p.optional else 'required'}): {p.description}" for p in spec.params]
    if spec.precondition_rules:
        lines.append("  Must hold before it is accepted:")
        lines += [f"    - {n}: {onto.rule_docs[n].text}" for n in spec.precondition_rules]
    if spec.effect_rules:
        lines.append("  Applied when it resolves:")
        lines += [f"    - {n}: {onto.rule_docs[n].text}" for n in spec.effect_rules]
    return "\n".join(lines)


def _overview(onto: Ontology) -> str:
    lines = ["Actions (one per turn; call explain_rules with a name for details):"]
    for name in sorted(n for n, a in onto.affordances.items() if a.ends_turn):
        first_sentence = _what_it_does(onto.affordances[name]).split(". ")[0].rstrip(".")
        lines.append(f"  {name}: {first_sentence}.")
    lines.append("Rules (call explain_rules with a name for the reasoning and an example):")
    for kind in ("precondition", "effect", "game rule"):
        for doc in sorted(onto.rule_docs.values(), key=lambda d: d.name):
            if doc.kind == kind:
                lines.append(f"  [{kind}] {doc.name}: {doc.text}")
    return "\n".join(lines)


def explain(onto: Ontology, topic: str | None) -> str:
    if not topic:
        return _overview(onto)
    key = topic.strip().lower().replace("-", "_").replace(" ", "_")
    spec = onto.affordances.get(key)
    if spec is not None and spec.ends_turn:
        return _action(onto, spec)
    if key in onto.rule_docs:
        return _rule(onto.rule_docs[key])
    raise IllegalAction(f"Unknown topic {topic!r}. Topics: {topics(onto)}")
