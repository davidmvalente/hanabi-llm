# Knowledge graphs: optional arms

The **baseline** game offers models three families of tool and nothing else: actions, the
state (`get_state`) and the rules (`explain_rules`). It contains no strategy knowledge
anywhere a model can reach. A **knowledge graph** is an optional extra that adds notes a
model may consult, so that "with" and "without" can be compared as separate experimental arms.

```bash
uv run hanabi-llm --models a,b,c                    # the baseline: no knowledge graph
uv run hanabi-llm --models a,b,c --knowledge mdk    # an arm: the mechanics/dynamics/kinetics notes
```

The selection is written to the feed header (`config.knowledge`, `[]` for the baseline), so every
logged game says which arm it belongs to.

## What a knowledge graph is

A Turtle file in `src/hanabi_llm/knowledge/`, made of `hn:Concept` nodes:

```turtle
hn:StableSlots a hn:Concept ; hn:category "mechanic" ; hn:conceptName "stable_slots" ;
    hn:summary  "Slots are 0-based and stay put: a played card is replaced in the same slot." ;
    hn:guidance "What a hint told you about slot 2 is still true of slot 2 until you play or discard it." .
```

* `hn:category` is free text, used only to group the notes in the overview.
* `hn:measuredBy` (optional) points at an ontology term the note is about.
* Loading any graph also loads the shared machinery (`_concepts.ttl`: the vocabulary and the
  `get_concepts` tool). Files starting with `_` are infrastructure and cannot be selected.
* Several graphs can be combined: `--knowledge mdk,another`.

Shipped: **`mdk`**, the mechanics, dynamics and kinetics notes (causal potentials: mass,
inertia, torque, activation energy, collision rate, ...).

## Guarantees (tested)

* The baseline's prompt, tool schemas and every tool answer contain no note text
  (`test_no_strategy_text_reaches_a_baseline_model`).
* A note's name never equals a rule or action name, so `get_concepts` and `explain_rules` never
  answer the same name differently.
* Notes never mention tools that do not exist.
* Every concept has a name, category, summary and guidance; the lint runs on the baseline and
  with each graph loaded.

## Adding one

Copy `mdk.ttl`, change the concepts, and it appears in `available_knowledge()` and in
`--knowledge`. Keep concept names unique across graphs and distinct from rule and action names.
