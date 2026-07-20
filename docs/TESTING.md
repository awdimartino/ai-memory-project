# Testing

Three layers, deliberately different in cost and in what they can tell you.

| Layer | Needs LM Studio? | Cost | Answers |
|---|---|---|---|
| **Offline suite** (`tests/`) | No | ~16s | Did I break the logic? |
| **Live scripts** (`scripts/`) | Yes | 1–10 min | Does this subsystem work against a real model? |
| **Gold set** (`evals/`) | Yes | 15–25 min | Did this *version* get better or worse? |

The split exists because of the single most important v1 process lesson:

> **Verifying end-to-end against the real local stack surfaced more real bugs than code review or
> mocked testing did.** A silently-broken extraction schema, a message-ordering crash, a
> miscalibrated similarity threshold and a hallucination pattern were all invisible to reading the
> code — they only appeared by running real prompts through the real server.

So: the offline suite proves you didn't break the wiring. It cannot tell you whether she's any
good.

---

## Layer 1 — the offline suite

```bash
python tests/run_all.py              # everything, exits non-zero on any failure
python tests/run_all.py -q           # only show failures
python tests/run_all.py drives tick  # only files matching these substrings
python tests/test_drives.py          # any single file still runs standalone
```

No LM Studio, no network, no HuggingFace. Fakes for the LLM, embedder and classifier; a **real**
SQLite store on a throwaway DB.

**Discovery is a glob**, so a new `tests/test_*.py` is picked up with no edit anywhere. That
matters: the previous entry point was a hand-maintained list in HANDOFF, and it had silently
fallen two files (85 checks) behind.

### How a test file is built

```python
from _harness import case, run, temp_db, config_override
from helpers import OneHotEmbedder, ScriptedLLM, InMemoryMeta

@case
async def a_thing_that_should_be_true():
    with temp_db("mine.db") as (conn, path):
        ...
        assert something, "message shown on failure"

if __name__ == "__main__":
    raise SystemExit(run())
```

- **`_harness.py`** — the case registry and runner, `temp_db()` / `temp_dir()` (torn down even
  when a case fails), `config_override()` (restores even on failure), and a fake `Clock`.
- **`helpers.py`** — the shared fakes. Use these rather than writing your own: three separately
  drifted `FakeMeta` classes once existed, one of which was a stub whose `get()` always returned
  `None` — which silently made every read-back assertion vacuous.

**One process per file**, on purpose: a file that forgets to restore a `config` global still
can't leak into its siblings.

---

## Layer 2 — live scripts

Need LM Studio. **Stop the web server first** — two processes hitting one model has crashed it.

```bash
python scripts/rrr_diagnostic.py     # OFFLINE, read-only: repetition health of her journal
python scripts/tool_eval.py          # 30 scenarios, tool routing. NOISY — run x3
python scripts/eval_extraction.py    # is the right fact captured, is junk rejected
python scripts/eval_conversation.py  # repetition + backbone over scripted scenarios
python scripts/bakeoff_personality.py # bounce / question-ending / sameness
python scripts/emotion_eval.py       # mood channels move sensibly (no LM Studio; CPU only)
python scripts/stress_test.py        # ~30 adversarial turns, all tick jobs firing, 9 invariants
python scripts/drive_demo.py         # watch drives cross thresholds in fast-forward
python scripts/tool_smoke.py         # quick end-to-end tool check
python scripts/reminisce_smoke.py    # recover a fact scrolled out of the context window
```

`scripts/_harness.py` gives them a shared preamble — `scratch_env()` (throwaway DB, jobs off) and
`llm_client()`, which **mirrors `bootstrap.build()` exactly**. That parity matters: this eval once
built its client without the sampling penalties or retries, so it was scoring a configuration
production never runs.

Five scripts still carry their own preamble. Route them through the harness opportunistically,
when you're about to run one anyway — rewriting live code you can't execute is how silent breakage
gets in.

### Reading these honestly

- **`tool_eval.py` is noisy.** Individual runs have scored 22, 23 and 25 on unchanged code.
- **`bakeoff_personality.py` is noisier than it looks.** n=18 cannot resolve small differences;
  the documented "7% q-end" did not reproduce, with the same prompt scoring 22% in a later
  session.
- Neither can tell you a change *helped* — only that it didn't obviously break something.

---

## Layer 3 — the gold set

See [`evals/README.md`](../evals/README.md). Run it at **version boundaries**, not per change.

It's the only layer that answers "did this help?", because it's the only one with a recorded
baseline to diff against.

---

## What to run, when

| Change | Run |
|---|---|
| Refactor, no behaviour change | `tests/run_all.py` |
| Touched a prompt | `tests/run_all.py`, then the relevant live script |
| Touched memory / recall / lifecycle | `tests/run_all.py` + `eval_extraction.py` |
| Touched tools or the tool loop | `tests/run_all.py` + `tool_eval.py` **×3** |
| Touched tick jobs, drives, sleep | `tests/run_all.py` + `stress_test.py` |
| Shipping a version | the gold set, with `--compare` |

---

## Two traps

**Don't trust a test that has never failed.** Twice in this project a test passed for the wrong
reason. Verify a new test by breaking the thing it guards and watching it fail — mutation-testing
the prompt tests is how the runner's own unicode crash was found.

**Don't measure against a number you can't reproduce.** When re-measuring after a prompt change,
A/B *in the same session* against the previous prompt rather than comparing to a figure recorded
weeks ago. Model, template state and sampling all drift.
