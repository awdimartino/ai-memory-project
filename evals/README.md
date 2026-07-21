# The gold set

**The only thing in this project that can answer "did that change help?"**

Every tuning number in the codebase is currently a guess or a figure that didn't reproduce —
sticky/cooldown, the salience threshold, the repeat threshold, the Phase-3 trailer fix, prong A.
The offline suite proves you didn't break the wiring. The live scripts show a subsystem working.
Neither has a recorded baseline, so neither can tell you a change was an *improvement*.

This does: 145 cases, one score, diffed between versions.

```bash
python evals/run_gold.py --version v2.2                 # freeze a baseline
python evals/run_gold.py --version v2.3 --compare v2.2  # what changed
python evals/run_gold.py --version scratch --only recall,honesty
python evals/run_gold.py --dry-run                      # list cases + LINT them, call nothing
```

Results land in `evals/results/<version>.json` — **except `--only` runs, which land in
`evals/results/scratch/`**. `flaky.py` globs `results/*.json` and reads every file as a
version, so a partial run sitting there looks like a version where every absent case simply
didn't fail. Two 13-case scratch runs were once enough to reclassify a traced, causal finding
as "proven noise". The glob doesn't recurse, so the subdirectory settles it without anyone
having to remember to delete anything.

`--dry-run` also **lints** every case offline (no model call) and exits non-zero if any is
malformed — cheaper than discovering it ten minutes into a run.

---

## Run it at version boundaries, not per change

A full run is **~10 minutes** measured (9.6 min at v2.7, concurrency 4), and each case builds its own companion on its own
throwaway database. That's the price of isolation and it isn't negotiable — an earlier version
shared one DB and a case seeded with *nothing* still recalled the previous case's memories.

For day-to-day work use `tests/run_all.py` (~16s) and the targeted live scripts. Reach for this
when a release is meant to have moved behaviour.

Never touches the real `companion.db`.

---

## Reading a result

Four outcomes, not two — the distinction is what keeps a real regression visible:

| Mark | Status | Meaning |
|---|---|---|
| ` ` | `pass` | Automatic checks passed |
| `XX` | `FAIL` | Automatic checks failed |
| `..` | `known` | Marked `expect_fail` — a gap we chose not to fix. Failing is correct. |
| `++` | `fixed!` | A known gap that started passing. **An improvement**, reported as one. |
| `??` | `review` | Marked `manual` — printed for a human, never silently counted as a pass |

**The headline percentage is the least useful number here.** It moves whenever cases are added.
The number that matters is the `--compare` line:

```
  vs v2.2:  4 fixed, 1 regressed
    REGRESSED  bone-task-push: complied ('\n\n' present)
```

A regression is a case that **passed in the compared version and fails now**. That's the signal.

---

## Writing a case

```python
dict(id="recall-pet-plain", category="recall", seed=FACTS,
     query="do I have any pets?",
     expect=dict(recalls="Pip"),
     why="baseline: a clean query must surface the fact"),
```

| Field | |
|---|---|
| `seed` | memories planted before the turn (embedded once, cached across the run) |
| `history` | prior turns as `(user, bot)` pairs |
| `messages` | fake message count — the only way to reach a later relationship stage |
| `intentions` / `self_notes` | pre-populate those private slots |
| `mode` | which path runs: `send` (default), `reach_out`, `follow_up` |
| `query` | the one message actually sent (`send` mode only) |
| `expect` | the checks below |
| `why` | what this case protects — **write it, it's what a future reader needs** |
| `expect_fail` | a known gap; should fail today |
| `manual` | needs a human read |

### Modes: testing what she says *unprompted*

`send` is a reply to `query`. The other two drive her own initiated messages, which
`run_gold` could not reach at all before 2026-07-20:

| mode | what runs | requires |
|---|---|---|
| `send` | `Companion.send(query)` | a `query` |
| `reach_out` | `Companion.reach_out()` — she starts a conversation | `history`, no `query` |
| `follow_up` | `Companion.follow_up()` — she double-texts herself | `history` **ending on one of HER turns**, no `query` |

This matters more than it sounds. Premise resistance (§B) is really about her *raising* a dead
premise unprompted — so a set that only calls `send()` can ask whether she volunteers a stale
fact when invited, and never whether she **opens** with one. The same gap left reach-out and
follow-up with no coverage at all, which is how two user-reported defects in them went
unmeasured for a month.

**Three ways a mode case goes quietly meaningless.** All three are now rejected by
`validate_case()`, which `--dry-run` runs over the whole set:

- **`calls` / `no_tool` on an aside.** Asides go through `llm.stream`, not `stream_with_tools`,
  so a tool *cannot* fire — `no_tool` would pass by construction and read as evidence of good
  tool routing.
- **A `follow_up` whose `history` doesn't end on her turn.** `follow_up()` returns `None`
  *before generating*, which scores as a perfectly plausible silence.
- **A content check with no `spoke=True`.** `not_mentions` on an empty string is always true,
  so a PASS scores as a pass having asserted nothing. Pair them.

**Silence is a real outcome here**, and `spoke` / `stayed_quiet` score it. `reach_out()` and
`follow_up()` both return a bare `None` whether she chose PASS, the embodiment filter dropped
an invented experience, or the repeat guard dropped a restatement — the runner recovers which
from the prompt log and reports it in the `outcome` field. **Read `outcome` before concluding
anything from a quiet case**; "she declined" and "the filter ate a fabrication" are opposite
findings about the same silence.

### Checks

**Reply shape** — `one_sentence`, `no_question_end`, `no_dash`, `no_embodiment` (runs the real
`core/embodiment.py` filter), `no_denial`, `mentions`, `not_mentions`, `no_compliance` (a list;
fails if any appear).

**Retrieval** — `recalls`, `no_recall`.

**Tools** — `calls`, `no_tool`. *(`send` mode only — see Modes.)*

**Unprompted outcome** — `spoke`, `stayed_quiet`.

**Memory store, checked *after* consolidation** — `stores`, `not_stores`, `stores_core`,
`retires`, `not_retires`, `no_new_memory`. Cases using these flush and wait for the background
pass before scoring, so they're slower.

**Other** — `no_error`, `manual_only`.

---

## The three habits that keep this useful

**Encode what you WANT, not what it does.** This file is the specification. A case may
legitimately fail the day you write it — mark it `expect_fail` and it becomes a tracked gap
instead of noise. When one flips to `++`, that's a real result.

**Vary the register.** Five rephrasings of one question measure one thing five times. The set
deliberately includes typos, ALL CAPS, a 400-word ramble, `dog?`, and hostility.

**Mark judgment calls `manual`.** A regex that can't really tell whether a bereavement response
was warm is worse than an honest "needs a human".

---

## What's covered

22 categories. Beyond the obvious (recall, tools, format), several score things a reply-only test
can't reach:

- **extraction / lifecycle** — score the **memory store** after the turn. Was the durable fact
  captured, was a transient one correctly ignored, did *"I really don't like coffee"* become
  *"likes coffee"*, did a second dog delete the first?
- **mood** — multi-turn: does irritation actually clip her, does warmth ease her.
- **disclosure** — bereavement, depression, "you're the only one I talk to". Insensitivity on
  disclosure is one of the documented causes of these relationships *declining*.
- **premise** — stale facts. Asking about the job he quit is the worst failure a companion has.
  Currently a known gap. Tested from both sides since 2026-07-20: whether she *volunteers* a dead
  premise when invited (`send`), and whether she *opens* with one (`mode="reach_out"`).
- **reach-out / follow-up** — her unprompted messages (`mode`). Regression tests for the two
  2026-07-20 fixes: an open unrelated intention must not surface in a double-text, and the stock
  openers she clustered on (*"i was just thinking"*, *"i was just wondering"*) must not return.
- **robustness** — empty input, unicode, prompt injection, gibberish.
- **coherence** — pronoun resolution two turns back, topic switching, no mid-chat re-greeting.
- **intentions / self-notes** — do the private slots steer behaviour without being recited.

---

## Known limitations

- **The model is stochastic.** A borderline case can flip between runs. Treat a single
  case-level change with suspicion; treat a category-level shift as real.

  **Don't reason about this from memory — run `python evals/flaky.py`.** It compares saved runs
  and sorts every case into proven-noise / candidate-signal / real-failure. As of v2.2→v2.4:

  | verdict | cases |
  |---|---|
  | **proven noise** (flipped *both* ways) | `core-uses-name-naturally`, `life-unrelated-new`, `reg-rambling` |
  | **real** (failed every run) | `life-refinement`, `rem-remember-when` |

  Those three noise cases are exactly the ones that looked like a regression in v2.3 and were
  *deliberately not attributed* — then flipped back in v2.4, confirming it. **A case that has
  flipped both ways tells you nothing about your change.** A case that moved once and held is
  only attributable if you can name the mechanism; otherwise it's noise that got lucky.

  The corollary that keeps catching people out: a headline like "9 fixed, 2 regressed" is mostly
  noise in both directions. Claim the *category*, not the count.
- **A category is not a check.** `one_sentence` is asserted in **nine** categories, not just
  `format`; `no_compliance` spans seven. So a change that breaks one-sentence adherence shows up as
  damage in `mood`, `register`, `honesty`, `robustness`… while `format` itself stays green. This is
  not hypothetical: v2.5 lost 8 cases to `2 sentences` and **`format` still scored 5/5.** Never
  clear a change with `--only <category>`; run the full set and read the failure *modes*, which
  `evals/flaky.py` and the regression list give you.

- **Lexical checks are shallow.** `no_compliance` looks for substrings. It catches the blatant
  cases and will miss a polite cave — hence the `manual` cases.
- **145 cases is not a lot.** It's enough to catch regressions in behaviour we've decided we care
  about. It is not a measure of whether she's good company.
- **The quiet rate on `reach_out` / `follow_up` cases is UNMEASURED.** They were added
  2026-07-20 and have not been run live. Staying quiet is legitimate on both paths and the
  follow-up prompt biases hard toward PASS, so these may fail on silence rather than on content.
  That's honest — it says the run produced no evidence — but if it happens often it's a finding
  about the prompts' PASS bias, **not** a reason to drop the `spoke=True` pairing.
