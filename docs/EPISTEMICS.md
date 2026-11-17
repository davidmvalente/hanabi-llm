# Epistemics: who may know what

Hanabi is a game of incomplete information, so *who may know what* is the
heart of the model. It is defined in **one declaration**, interpreted in
**one function**, and enforced by **three independent checks**.

| | Where |
|---|---|
| Declared | `src/hanabi_llm/hanabi.ttl`, section 7 (`hn:identityVisibility`, `hn:visibility`) |
| Interpreted | `views.build_view` (and `states._hands`, which renders what the view contains) |
| Enforced | SHACL shapes (`shapes/view.ttl`, `shapes/truth.ttl`), the non-interference tests, the `--audit` flag |

## The four layers

| Layer | What it is | Lives in |
|---|---|---|
| 1. Ground truth | Every card, hidden ones included, and the true order of the deck. Written only by `Game.apply`. | `GameState` (`engine.py`). As RDF: `validate.truth_graph`, used only for validation, never shown to a model. |
| 2. Information regime | The policy: who may see the *identity* (colour and rank) of the cards in each kind of zone. A card's existence and position are always public. | Ontology section 7, loaded as `Regime` (`ontology.py`). |
| 3. Common knowledge | What hints have told each holder about their cards (including negative information). Everyone saw the hints, so it is public. | `CardKnowledge` (`model.py`), updated in `engine._apply_give_hint`; declared public with `hn:visibility`. |
| 4. A player's view | The truth projected through the regime for one seat. | `PlayerView` (`views.build_view`), rendered as text sections (`states.py`) and, for validation, as an RDF graph (`views.view_graph`). |

Private memory would be a fifth layer. Experiment 1 has none: turns are stateless.

## The regime

```turtle
hn:Hand        hn:identityVisibility hn:HiddenFromHolder .   # you see everyone's cards but your own
hn:DiscardPile hn:identityVisibility hn:Public .
hn:Stack       hn:identityVisibility hn:Public .
hn:Deck        hn:identityVisibility hn:HiddenFromAll .      # not even the order
```

* The hand rule is genuinely interpreted: `Regime.hand_identity_visible(viewer, holder)`
  decides which cards `build_view` reveals, and the redaction of a freshly drawn
  card follows from it (a drawn card enters the drawer's hand). Switching the hand to
  `Public` gives an open-hand variant end to end (`test_the_view_applies_the_regime_...`).
* Combinations the engine cannot honour are refused when the ontology loads, so a
  declaration can never be silently ignored.
* The view shapes encode the *standard* regime. An open-hand variant would need its
  own shapes file.

## Enforcement

1. **SHACL on the view graph** (`shapes/view.ttl`): the viewer cannot see the identity of
   a card they hold; can see everyone else's; no deck card or deck zone appears; a draw
   event does not name the card.
2. **SHACL on the ground truth** (`shapes/truth.ttl`): every card is in exactly one zone
   with one identity; stacks hold exactly ranks 1 to their height of their colour; every
   colour/rank exists exactly as often as declared; token pools satisfy
   `0 <= available <= capacity` and `spent = capacity - available`; **hints never lie**
   (what a holder was told equals the truth).
3. **Non-interference tests**: two games that differ only in a player's hidden cards and
   the deck must give that player's model byte-identical input, across every read tool
   and the whole conversation. Breaking the redaction makes five tests fail.

Run the SHACL audit on a real game with `uv run hanabi-llm ... --audit`: it validates the
truth and every seat's view after each turn (about 1.3 s per turn for three seats), and a
violation aborts the run loudly, because it is a bug in this code, not a model's fault.

## What is still open

How state should *live*. Today the truth is a Python object, and graphs are read models
rebuilt on demand (option A in the earlier discussion, plus `truth_graph` for validation).
Option B, an event log as the canonical persisted record with graphs as folds over it,
would give the diachronic per-turn view, crash resume and replay from one mechanism, and
is still my recommendation. It is not built, and nothing here prevents it: the epistemic
layers above would simply become folds over `(events up to t, observer)`.
