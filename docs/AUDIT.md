# Design audit: fairness and robustness

Review of the first draft's decisions. Each finding says what was wrong, what
changed, and how it is tested. "Open" items need a decision or belong to the
tournament layer.

## Findings fixed in this revision

| # | Finding | Severity | Fix | Test |
|---|---|---|---|---|
| 1 | **Card ids encoded identity.** Ids were assigned before the shuffle, so they ascended with colour and rank. `hg:card4` in a model's own hand narrowed down what it was; SPARQL exposes ids. | High (information leak) | Shuffle first, then number cards in shuffled order. | `test_card_ids_do_not_encode_identity` |
| 2 | **Leak tests were too weak.** They checked colour/rank triples only, so #1 slipped through. | High (process) | Added a non-interference test: two games differing only in the viewer's hidden cards and the deck must give byte-identical input to the model across every read tool. Mutation-checked: breaking the redaction fails it. | `test_noninterference_hidden_cards_never_reach_the_model` |
| 3 | **Forfeit = silent substitute move.** After failures the referee discarded position 0, possibly a 5. That changes the game the *teammates* play, makes scores incomparable, and can reward or punish a seat arbitrarily. | High (fairness) | Removed. A player returns a valid action or raises `PlayerFault`; the game is aborted and never ranked. | `test_faults_abort_the_game_and_never_change_it` |
| 4 | **Blame was undifferentiated.** Provider outages, rate limits and slow networks were handled like model mistakes. | High (fairness) | `FaultKind` splits attributable model faults (`invalid_actions`, `no_action`) from non-attributable infra faults (`provider_error`, `timeout`). Infra faults void the game for everyone and say it should be re-run. | parametrized fault tests |
| 5 | **Wall-clock timeout punished slow models.** Latency is not skill. | Medium | Timeout is only a generous safety net (300 s) and is classed as infra. Per-turn budgets are counted in model calls and rejected actions, not seconds. | `test_timeout_is_infra_not_blame` |
| 6 | **Seat names were built from model ids** (`P0_claude-sonnet...`). A model could see which teammate is which model, and be biased by it. | Medium (bias) | Neutral names (Alice, Bob...). The seat to model map lives only in the log header. | `test_seat_names_are_neutral` |
| 7 | **Sync SPARQL could stall everyone.** rdflib blocks the event loop; `asyncio.timeout` cannot interrupt it. | High (robustness) | Superseded: the generic SPARQL tool was removed altogether (see "Experiment 1: MECE and minimal tools"), so the hazard no longer exists. | n/a |
| 8 | **Mistakes in free tools counted against the model.** | Medium | Only rejected *action* attempts count toward the budget. Read-tool errors are free (bounded by the call budget). | `test_rejected_actions_count_but_read_mistakes_are_free` |
| 9 | **Games were not reproducible.** Seed optional, not logged, no action log. | Medium | Seed always chosen and logged; `GameResult.action_log` + seed replays a game exactly. | `test_action_log_and_seed_replay_exactly` |
| 10 | **Sampling limits uncontrolled.** Temperature and output length were implicit. | Low | `--temperature`, `--max-tokens`, budgets are identical for all seats and written to the log header. | n/a |
| 11 | **Transcript could be lost on a crash.** | Low | Log is flushed after every record; a final `result` record carries status, blame and `counts_for_ranking`. | n/a |
| 12 | **The system prompt restated the rules** (read from the ontology), duplicating `explain_rules` and putting the same information through two channels. | Low | The prompt is a versioned YAML with objective and turn protocol only; rules are learned through the tools. Version and hash are logged. | `test_nothing_is_read_from_the_graph`, `test_every_tool_name_in_the_prompt_exists` |

## Fairness policy as implemented

* Every seat gets the same system prompt (apart from its own name; a versioned
  YAML whose version and hash are logged), the same
  tools, the same call budget and the same sampling limits.
* A game ends in exactly one of: `completed`, `aborted_model_fault`,
  `aborted_infra_fault`. Only `completed` has an official score.
* A model that makes 3 rejected action attempts in one turn, or never acts
  within 8 model calls, is at fault. The default budgets are visible to the
  model in its prompt.
* When one model faults in a cooperative game, the teammates' game is void
  too. Whether that is a loss for the faulting model only, and how many
  voided games disqualify it from the championship, is a tournament-layer
  rule (see Open items).
* If a message contains several tool calls, they are processed in order and
  the first accepted action ends the turn; later calls are ignored.

## Known limitations

* **Truncated replies.** A reply cut off by `max_tokens` currently looks like
  "no tool call" and counts against the model. It should be detected through
  `finish_reason == "length"` and retried or classed as configuration, not blame.
* **Provider routing.** OpenRouter may serve the same model id from different
  providers or quantizations. For ranking runs, pin the provider and disable
  fallbacks, and record which provider answered.
* **Tool-calling support differs by model.** A model with no tool support will
  fault on every game. A capability pre-flight before a tournament is advisable.

## Open items (need your decision)

1. **Helper tools (decided for experiment 1).** Odds, census, strategy notes and
   private notes are not offered; see the next section. Revisit for later
   experiments, as a separate arm so their effect can be measured.
2. **Fault consequences.** Options: (a) the faulting model scores 0 and the
   teammates' game is re-run; (b) the game is simply void and re-run, with
   repeated faults disqualifying the model; (c) a fixed penalty. The code
   already provides what each needs (`Fault.seat`, `kind.attributable`).
3. **Seat and pairing design.** Seat 0 moves first. A fair championship should
   rotate seats and reuse the same deal seeds across every pairing, then
   compare over many seeds. That belongs in a runner above `play_game`.

## Experiment 1: MECE and minimal tools

The tool set is exactly three families, with no overlap and nothing missing:

| Family | Tools | Job |
|---|---|---|
| action | `play_card`, `discard_card`, `give_hint` | change the game; only these end the turn |
| state | `get_state(section?)` | read what is true now |
| rules | `explain_rules(topic?)` | read how the game works, and why |

* **State is a partition.** The six sections (turn, tokens, fireworks, discard,
  hands, history) cover every field of a player's view exactly once, and the
  full state is the sections in order (`test_sections_partition_the_state`).
  The first message of each turn is the full state; `get_state` re-reads a part.
* **Tokens have one vocabulary.** Hint and fuse tokens are pools with
  `available`, `spent` and `capacity` (`spent = capacity - available`). This
  removed the event-replay code that tracked refunds. The turn state already
  existed (`turn`, `current`, `final_turns_left`, `deck_size`) and is exposed as
  the `turn` section.
* **Rules are explained from the ontology.** Each rule has a name, text,
  rationale and example in `hanabi.ttl`; `explain_rules` only formats them.
* **Removed, because each changes what is measured:** SPARQL (the model had to
  write queries instead of using game vocabulary), `estimate_card` and the card
  census (they do the counting for the model), strategy concepts, private notes
  (turns are stateless anyway) and dry-run helpers. The concept notes remain in
  the ontology as data.
* **Wiring is checked at start-up.** Read tools are dispatched through the same
  `@bind` registry as the engine, and `ToolBox` verifies that the sections the
  ontology declares match the ones the code renders.

## Ontology and epistemics cleanup

* **Defects fixed:** `hn:Game` is declared; `hn:Setup` is typed (`hn:GameSetup`); an
  effect function on a rule no longer breaks a domain (`hn:Executable` is the common
  parent of rules and affordances); the lint then found two more, `hn:turnNumber`
  (used on both events and the game) and `hn:identityVisibility` (stated on classes).
* **Dead and confusing vocabulary removed:** `hn:GameObject`, the `hn:Action` family and
  `hn:affordanceAction` (they duplicated the affordance and the event), and an unused
  fuse constant. Events now point at the affordance they exercise (`hn:exercises`).
* **Lint** (`tests/test_ontology_lint.py`): types and referenced terms are declared, no
  untyped data, domains and ranges hold on the ontology *and on every graph the code
  emits over whole games*, no dead classes or properties, the documentation is
  complete, and the bound functions equal the functions the ontology names, in both
  directions. Each of six re-introduced defects is caught.
* **Epistemics** is declared in one place and enforced three ways; see
  `docs/EPISTEMICS.md`. SHACL catches nine deliberate corruptions (`tests/test_validate.py`).
* **Not done:** the event-sourced state with diachronic per-turn graphs (option B). The
  audit costs about 1.3 s per turn for three seats, so it is opt-in (`--audit`).

## Feed, fold and viewer

* **The old log was not a log.** Events were written as Python `repr` strings, unusable as a
  stream. The feed (`feed.py`) is plain JSON with the initial deal in the header, so a game can be
  rebuilt from it alone.
* **The fold is tested against the engine** after every turn of whole games, for 2 to 5 players,
  both for the truth and for every player's projected view (`tests/test_fold.py`). A first version
  of that test was vacuous: random bots lose long before the deck runs out, so the final-round
  clock was never exercised and a deliberately broken clock went unnoticed. Players that never play
  now drive games through the final round, and the same mutation fails four tests.
* **Hint encoding.** `GiveHint(target, color, rank)` with two nullable fields could represent
  nonsense and called the player the "target". A hint is now `GiveHint(receiver, value)`; the
  subset of the hand it covers is `targets`; rules were renamed to match (`hint_receiver_is_other`,
  `hint_targets_cards`) and `hint_is_complete` documents that every matching card is covered.
* **The viewer holds no game logic.** The server folds the feed and streams states; the page only
  draws them. Server tests check each streamed state equals the fold, that a half-written line
  waits, and that nothing external is requested; a headless-browser run checks the blinding and
  the scrubbing.

## v1.1 plan: token parsimony and caching

**Is the conversation cached?** Partly, and not as a "thread". Each API call is
stateless. Providers can cache a repeated *prefix* of the prompt, which lowers
price and latency but not the tokens counted against the context window. Per
the [OpenRouter prompt-caching docs](https://openrouter.ai/docs/guides/best-practices/prompt-caching),
some providers cache implicitly and automatically; Anthropic and Google use
`cache_control` markers; and OpenRouter tries to keep follow-up requests on the
same provider ("sticky routing", sped up by passing a `session_id`). Cache
lifetimes are short for some providers (a 5-minute write TTL is documented for
Anthropic), so a slow tournament can miss.

**What that means for this design.** Today each turn is stateless: the system
prompt is an identical prefix (cacheable), but the user message holds the whole
state and changes every turn, so only the prefix benefits.

**Plan:**
1. Make each seat's conversation append-only. Turn 1 sends the full snapshot;
   later turns append only a delta (events since that seat's last turn, newly
   visible cards, changed hint knowledge, token counts) and keep earlier
   assistant and tool messages verbatim, so the entire history is a cache hit.
2. Keep `get_state` as the resync tool, and rebuild a fresh snapshot every N
   turns (this costs one cold cache but bounds context growth).
3. Pass a stable `session_id` per seat and per game for sticky routing, and mark
   an explicit `cache_control` breakpoint for providers that need one.
4. Never depend on the cache for correctness. The delta must be generated by one
   deterministic code path, and a cache miss must only cost money.
5. Measure first: log cached vs uncached prompt tokens per turn (the usage
   block already reports prompt tokens) before optimising further.

Trade-off: an append-only thread grows linearly per turn (quadratic total
tokens if the cache misses), whereas the stateless design is linear in total
but pays the full state each turn. The crossover is worth measuring on real
models.
