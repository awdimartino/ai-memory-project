# Mari — a local-first AI companion

A personal AI companion for a single user, running fully local against
[LM Studio](https://lmstudio.ai/). Not an assistant — a friend. She remembers you, has moods
that persist, thinks while you're away, and occasionally messages first.

Everything runs on one machine: a 9B chat model, a small embedding model, and a ~125M emotion
classifier on CPU, inside a 16 GB VRAM budget.

```bash
python -m web.app        # web UI at http://127.0.0.1:8000   <- the main surface
python main.py           # same brain, terminal REPL
python tests/run_all.py  # offline suite, no LM Studio needed
```

Requires LM Studio running with its local server on (port 1234) and the chat + embedding models
loaded. Config comes from a git-ignored `.env` — see `.env.example`.

---

## Where to look

| If you want to… | Read |
|---|---|
| **Pick up where the last session left off** | [`HANDOFF.md` §9](HANDOFF.md) — current state, next steps, what to watch |
| Understand how the code fits together | [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) |
| Change how she behaves | [`docs/TUNING.md`](docs/TUNING.md) — symptom → knob |
| Add a tool, a job, a store, a migration | [`docs/EXTENDING.md`](docs/EXTENDING.md) |
| Know what to run and when | [`docs/TESTING.md`](docs/TESTING.md) |
| Measure whether a change helped | [`evals/README.md`](evals/README.md) — the gold set |
| Know the roadmap, and what NOT to build | [`HANDOFF.md` §8](HANDOFF.md) — incl. §G and §H |
| Understand *why* it's built this way | [`V2_PLAN.md`](V2_PLAN.md) — design decisions + full build log |
| Avoid relearning v1's lessons | [`V1_RETROSPECTIVE.md`](V1_RETROSPECTIVE.md) |

**HANDOFF.md is the living document.** README, `docs/` and this table describe how things work;
HANDOFF describes where the project *is*.

---

## What she actually does

**Memory in three tiers.** Every message is logged verbatim (episodic). At the end of a window
those are distilled into embedded facts (semantic), each passing a lifecycle decision —
duplicate, update (soft-delete the old, keep the link), or new. Identity-defining facts are
flagged `core` and injected into the prompt directly rather than waiting on a similarity search.

**Mood that persists.** A local RoBERTa GoEmotions classifier scores each of your messages into
6 channels that decay toward a baseline. Mood survives restarts and colours how she replies —
irritated means shorter and less accommodating, not just different word choice.

**Drives, not timers.** Two slow-integrating scalars — `connection` and `restlessness` — rise
while you're away and are relieved by contact. Reaching out fires on `connection`, reflecting on
`restlessness`. So *how she feels* sets the timing: a warm or sad conversation pulls a reach-out
earlier than a throwaway one. A separate `energy` reserve depletes while awake and restores while
asleep.

**An inner life you don't see.** She writes private journal entries during idle ticks, keeps a
forward agenda of things she means to bring up, and distils operating-notes about how to *be*
with you from how you actually react. All three feed back into her prompts.

**She can be quiet.** Reach-out, follow-up and even a normal reply can come back as `PASS`,
which means she says nothing.

**She sleeps.** After a long idle — or when energy runs low — she flushes pending work and
unloads the model from VRAM so the machine is yours. The heartbeat keeps ticking; the next
message wakes her.

**She can reach your phone.** Unprompted messages push via a self-hosted Bark server whenever the
chat isn't actually in front of you.

---

## Layout

```
config.py            all configuration, one place (77 knobs, .env-driven)
bootstrap.py         the single composition root — everything is wired here
main.py              terminal REPL entry point
web/app.py           FastAPI + WebSocket entry point (the main surface)

core/                domain logic. No SQL, no HTTP.
  companion.py         the facade — one turn, start to finish
  memory_manager.py    recall + consolidation + lifecycle
  emotion_manager.py   28 GoEmotions labels -> 6 decaying mood channels
  drives.py            connection / restlessness / energy
  tick.py              the heartbeat and every autonomous job
  prompts.py           every string ever sent to a model
  tools.py             the Tier-3 tool registry
  interfaces.py        store Protocols — the swap seams
  embodiment.py        filter: does this reply claim a body?
  textsim.py           lexical similarity for the repeat guards

infrastructure/      I/O adapters. Sits strictly below core/.
  db.py                connection, migrations, the SqliteStore base
  llm_client.py        LM Studio client, streaming + the tool loop
  *_store.py           one per table
  embedder.py, emotion_classifier.py, model_manager.py, notifier.py

tests/               offline suite — fakes, no LM Studio, ~16s
scripts/             live evals and probes — needs LM Studio
evals/               the gold set: does a change actually help?
archive/             v1, and scripts whose question is closed
```

The layering rule is load-bearing and dates to a v1 lesson: **`infrastructure/` sits strictly
below `core/`, and callers talk to the `Companion` facade — never through it into a store.**
That discipline is what made v1's Postgres → SQLite migration a contained change.

---

## Hardware notes

Built for an AMD Radeon 9070XT (16 GB). Consequences that shaped real decisions:

- **One model call at a time.** LM Studio can't safely serve concurrent requests to one model —
  it crashes. A single lock in `LLMClient` serializes chat against background consolidation.
- **The emotion classifier runs on CPU** (`device=-1`), leaving all VRAM for the LLMs.
- **Chat and "brain" share one model** by default, so only ~6.5 GB is resident.
- **Reasoning is off.** qwen3.5-9b is a reasoning model; it spent ~2,000 hidden tokens per
  structured call until thinking was disabled via an LM Studio template edit. See `HANDOFF.md` §4
  — that edit is required, and it is not something the app can do for you.

---

## Security

`archive/v1/infrastructure/config.py` contains a **real hardcoded HuggingFace token**. It is
git-ignored deliberately and must never be committed — it should be rotated on HuggingFace.
`.env` and `companion.db` are git-ignored too. There is no git remote; this is local-only.
