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
| 2. Information regime | The policy: who may see the *identity* (colour and rank) of the cards in each kind of zone. A card's existence and slot are always public. | Ontology section 7, loaded as `Regime` (`ontology.py`). |
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

## Who makes, shuffles and serves the deck

* **Made and shuffled** in `Game.__init__` (`engine.py`): the identities are generated from the
  ontology's distribution (every colour times its copies per rank), shuffled with
  `random.Random(seed)`. There are no card ids: a card is a `(colour, rank)` value, and which copy is meant is
  always said by where it is (a hand slot, a place in the discard pile). Hands
  are dealt by popping from the end of the deck. The seed makes a deal reproducible, and the
  feed's header records the initial deal.
* **Never served.** The only code that moves a card out of the deck is `_draw`, which pops the
  top card into a hand. A view carries only `deck_size`; the regime declares the deck
  `HiddenFromAll`; the view shapes fail if a deck zone, or a card that is not in a hand, the
  discard pile or an event, appears; and a draw event is redacted for the player who drew.

## Graphs as a fold

The game is an event log; every state, and every graph, is derived from it:

```
state(t)    = fold(step, deal, events[:t])           step is a pure transition on events
view(o, t)  = fold(step, project(o, deal), [project(o, e) for e in events[:t]])
graph(o, t) = graph_of(view(o, t))                   a read model, rebuilt on demand
```

* `deal` is the initial hands (the header of the feed). `project(o, .)` removes exactly what the
  regime hides from player `o`: the deal of `o`'s own hand and `o`'s own draws. Hints, plays,
  discards and the end are public, so they pass through unchanged.
* **The law**: viewing the truth equals folding the projected log, `view(o, state(t)) =
  fold(project(o, events[:t]))`. A player can therefore keep their own state from their own
  stream, never seeing the truth, and the "common part" of everyone's graph cannot diverge: it
  comes from the same events through the same `step`.
* **Diachronic graph** = the graph after each prefix, or the triples each event adds and
  removes, stacked in order. It needs no new machinery, only folding prefixes.
* **Implemented**: `feed.py` (schema, `project_record`), `fold.py` (the reference `step` and
  `fold`) and `tests/test_fold.py`, which checks both equalities after every turn of whole games,
  for 2 to 5 players, including games that run through the final round. The viewer is served
  folded states, so it holds no game logic.
* **Option B or C?** B (this): the events are authoritative and graphs are derived. C would make
  an RDF store authoritative, with `step` expressed as triples added and removed; the same fold,
  but the rules would live in SPARQL or SHACL rather than typed Python. B keeps the engine and
  can still produce the graph when needed (`validate.py`).
* **Unified**: the engine decides and `fold.step` applies. The engine validates an action, decides
  what happens (a play's outcome, the cards a hint covers, the draw from the secret deck), turns
  that into events and folds them; it never writes the state itself, and a test guards that. The
  state the views and the SHACL truth graph read is a read-only snapshot of the fold, so the
  fold that rebuilds states from the feed is the same code that runs the game.
