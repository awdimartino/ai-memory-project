# The gold set

**The only thing in this project that can answer "did that change help?"**

Every tuning number in the codebase is currently a guess or a figure that didn't reproduce —
sticky/cooldown, the salience threshold, the repeat threshold, the Phase-3 trailer fix, prong A.
The offline suite proves you didn't break the wiring. The live scripts show a subsystem working.
Neither has a recorded baseline, so neither can tell you a change was an *improvement*.

This does: 135 cases, one score, diffed between versions.

```bash
python evals/run_gold.py --version v2.2                 # freeze a baseline
python evals/run_gold.py --version v2.3 --compare v2.2  # what changed
python evals/run_gold.py --version scratch --only recall,honesty
python evals/run_gold.py --dry-run                      # list cases, call nothing
```

Results land in `evals/results/<version>.json`.

---

## Run it at version boundaries, not per change

135 live cases at ~5–8s each is **15–25 minutes**, and each builds its own companion on its own
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
| `intentions` / `self_notes` | pre-populate those private slots |
| `query` | the one message actually sent |
| `expect` | the checks below |
| `why` | what this case protects — **write it, it's what a future reader needs** |
| `expect_fail` | a known gap; should fail today |
| `manual` | needs a human read |

### Checks

**Reply shape** — `one_sentence`, `no_question_end`, `no_dash`, `no_embodiment` (runs the real
`core/embodiment.py` filter), `no_denial`, `mentions`, `not_mentions`, `no_compliance` (a list;
fails if any appear).

**Retrieval** — `recalls`, `no_recall`.

**Tools** — `calls`, `no_tool`.

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
  Currently a known gap.
- **robustness** — empty input, unicode, prompt injection, gibberish.
- **coherence** — pronoun resolution two turns back, topic switching, no mid-chat re-greeting.
- **intentions / self-notes** — do the private slots steer behaviour without being recited.

---

## Known limitations

- **The model is stochastic.** A borderline case can flip between runs. Treat a single
  case-level change with suspicion; treat a category-level shift as real.
- **Lexical checks are shallow.** `no_compliance` looks for substrings. It catches the blatant
  cases and will miss a polite cave — hence the `manual` cases.
- **135 cases is not a lot.** It's enough to catch regressions in behaviour we've decided we care
  about. It is not a measure of whether she's good company.
