# HANDOFF — resume here

**This file is the context brief for picking the project up cold.** Keep it short. It answers
*where are we, how do I run it, what will bite me.* It is **not** the roadmap and **not** the
build log.

| you want | read |
|---|---|
| what's built / what's next | `ROADMAP.md` — pre-3.0 phases + post-3.0 backlog |
| why a decision was made, dated build log | `V2_PLAN.md` |
| architecture, extending, testing, tuning | `docs/` |
| past measurement runs | `evals/results/*.json`, `*-manual.md` |
| anything trimmed from this file | `git log -- HANDOFF.md`, `git show <commit>:HANDOFF.md` |

> **House rule that keeps being earned: measure before believing.** Plausible reasoning has been
> wrong repeatedly here, including a confident written-up hypothesis that a real A/B then refuted.
> A mechanism that fits is a hypothesis, not a finding. Build the probe first.

---

## ▶ START HERE — current as of `e098426` (2026-07-21)

**Tree is clean. Suite is 22 files / 392 checks green (~20s, fully offline).**
**The web server is running** — `python -m web.app`, port 8000. ⚠️ Check for a *listener* on
8000, not for a process (one server legitimately shows as two PIDs; see Ops).

**The project is in a deliberate manual-testing pause.** No feature work in flight.

**What changed 2026-07-21 (this session):**
- **A3 + A4 went from never-run to measured 12/12 live** (`scripts/pursuit_smoke.py`).
  `PURSUIT_UNAVAILABLE_ENABLED=true` is now set in `.env`.
- **Phone push: two defects fixed.** A 404 from Bark verified as success (httpx does not raise on
  4xx/5xx), and the unavailable event pushed an **empty body**. `/admin/test_notify` now reports
  the real result — verified live end-to-end, Bark returned 2xx.
- **Tick loop: model-calling jobs no longer run during an A4 window.** They were JIT-reloading the
  model the window had just unloaded, and could reach out while she'd said she'd stepped away.
  Mood/drive drift deliberately still run.
- **Reasoning leaked into visible replies** (user-reported). Fixed by making `done` carry the
  authoritative text. See the open item below — it is not fully fixed.

**Three things to look at first:**

1. ⚠️ **The thinking config contradicts itself — and may be causing the leak.** `.env` sets
   `NO_THINK=true` (the app appends `/no_think`), but bounded thinking was adopted 2026-07-20
   with thinking **ON** at budget 384, and this session's log proves reasoning **is** being
   generated (4,636 chars leaked in one turn). So `/no_think` is not suppressing anything, and
   the app may be sending a directive that fights the server's budget injection. **Unmeasured
   guess, stated as a guess.** Resolve which one is intended before tuning anything prompt-side.
2. ⚠️ **The gold baseline is invalidated.** v2.8 = 115/120 (95.8%) predates the bounded-thinking
   serving config. Any comparison against it is meaningless until it is re-run.
3. ⚠️ **The "haha" tic** — 47% of her replies open with "haha" (was 0%, then 1%). Unfixed. It is
   position-beats-volume, and the docs are explicit that adding another rule will not work.

---

## 1. What the project is

A **personal, local-first AI companion** named **Mari** — a friend, not an assistant — for a
single user, running fully local via **LM Studio**. Guiding principle (validated in v1): make the
**brain** (memory + emotion + conversation) trustworthy *before* the **delivery layer**
(proactivity + tools + voice).

Four pillars — **memory, emotion, proactivity, conversation quality** — all complete, plus a tool
framework (built) and voice (future). v1 is archived under `archive/v1/`; `V1_RETROSPECTIVE.md`
has the lessons.

**Hardware constraint: AMD Radeon 9070XT, 16 GB VRAM.** Small models, CPU for the tiny emotion
classifier, **one model call at a time**.

---

## 2. Current state

**Feature-complete** as of v2.1; everything since is enrichment. Full inventory in `ROADMAP.md`.

| layer | what's there |
|---|---|
| **Runtime** | single async process; FastAPI + WebSocket web UI **and** a terminal REPL sharing one `Companion` |
| **Storage** | SQLite, `PRAGMA user_version` migrations — **schema v12**; WAL |
| **Memory** | 3 tiers — episodic log, semantic recall (nomic asymmetric prefixes + contrast gate), always-injected **core** facts; lifecycle new/duplicate/update/supersede; crash-durable watermark |
| **Emotion** | RoBERTa GoEmotions on CPU → 6 mood channels, persisted, decaying |
| **Proactivity** | tick loop: reach-out, follow-up, reflection, intentions, self-notes, persona edit, pursuit mining, sleep, **unavailable + pursuit return** (A4), mood/drive drift |
| **Autonomy** | drives (A1), energy (A2), intentions + self-notes, A3 open questions, A4 stepping away |
| **Tools** | native streaming `tool_calls` loop; `get_current_time`, `reminisce` |
| **Delivery** | web UI (tabs, status panel, memory + prompt inspectors), phone push via self-hosted Bark → APNs |
| **Measurement** | 120-case gold set + runner, `flaky.py`, style scorer, manual review, ~10 probes |

---

## 3. How to run

```bash
# from repo root, with the venv python (.venv\Scripts\python.exe on Windows)
python -m web.app       # web UI at http://127.0.0.1:8000
python main.py          # same brain, terminal REPL

# OFFLINE SUITE — no LM Studio, no network. Exits non-zero on any failure.
python tests/run_all.py              # all 22 files / 392 checks (~20s)
python tests/run_all.py -q           # only show failures
python tests/run_all.py drives tick  # only files matching these substrings
# Discovery is a glob over tests/test_*.py — a new test file is picked up with no edit.
# Shared harness in tests/_harness.py, fakes in tests/helpers.py.

# LIVE (need LM Studio up). scripts/_harness.py gives them a throwaway DB + jobs off.
python scripts/pursuit_smoke.py      # A3 open-question mining + A4 unavailability, end to end
python scripts/tool_smoke.py         # tools through the real persona
python scripts/tool_eval.py          # 30-scenario tool-routing eval, per-category score
python scripts/stress_test.py        # whole-system stress + invariant checks
python scripts/rrr_diagnostic.py     # OFFLINE, read-only: repetition health of journal + self-notes

# GOLD SET — expensive, ask before running. --only saves to results/scratch/ automatically.
python evals/run_gold.py --version scratch --only premise,reach-out,follow-up
python evals/flaky.py                # ALWAYS run before attributing any gold-set movement
```

Requires **LM Studio with its local server on** (port 1234). The emotion classifier downloads
from HuggingFace on first run, then runs locally on CPU. Config is env-driven via a git-ignored
`.env` (see `.env.example`). Both entry points share `companion.db` (git-ignored).
REPL: `/exit` (flushes pending consolidation), `/reset`, `/model <name>`, `/temp <v>`.

---

## 4. Model setup

- **Chat + brain: `qwen/qwen3.5-9b`** (`BRAIN_MODEL` empty ⇒ reuse chat, so only ~6.5 GB + ~0.1 GB
  nomic resident — no second big model, no load/unload thrash).
- **Embedding: `text-embedding-nomic-embed-text-v1.5`.**
- **Emotion: `SamLowe/roberta-base-go_emotions`** (~125M) on **CPU** (`device=-1`), ~0.5 GB RAM.
- `.env`: `MODEL=qwen/qwen3.5-9b`, `BOT_NAME=Mari`, `FREQUENCY_PENALTY=0.4`,
  `PRESENCE_PENALTY=0.3`, `LLM_MAX_RETRIES=3`.

⚠️ **Thinking configuration is currently contradictory — see START HERE item 1 before changing it.**
Two mechanisms exist and the docs disagree about which is live:
- **`NO_THINK=true`** in `.env` → the app appends `/no_think` to the system message. *Currently set,
  and demonstrably not suppressing reasoning.*
- **Bounded thinking** (adopted 2026-07-20) → LM Studio Engine Protocol + `LLAMA_ARG_THINK_BUDGET`,
  budget **384**. Measured: 256 → 833c/6.5s, 384 → 1120c/8.1s, 512 → 1500c/10.0s.
- A third, older mechanism is documented in git history: an LM Studio **prompt-template edit**
  (`{% set enable_thinking = false %}`) that made consolidation ~20× faster by killing reasoning
  instance-wide. Whether it is still applied in your LM Studio install is **not visible from the
  repo** — check before trusting any latency number.

**Roleplay finetunes were tested and rejected** (`neona-12b-i1`, `rocinante-12b-v1.1`): they break
on a plain OpenAI-style chat API — generate both sides, leak the persona, or return empty.
⚠️ **Model migration is a one-way door**, not a config change: identity discontinuity is the
single best-documented way to destroy a companion relationship. And it will not fix reasoning —
that was measured.

Use `lms unload --all` between loads when comparing models (JIT keeps them resident and will
exhaust 16 GB). Bake-off harnesses in `scripts/`.

---

## 5. Key decisions & lessons

- **Capability tiers (V2_PLAN §1.1):** recall/emotion are *autonomic* pipeline stages;
  lifecycle/self-edit are *structured-output*; only external tools need true function-calling.
  **Anything consequential uses structured output, not the tool loop** — tool routing measures
  ~23/30, far too flaky to gate behaviour like going silent on the user.
- **One model call at a time** (`LLMClient` lock). Concurrent requests have crashed LM Studio.
- **Position beats volume in the prompt.** Small models follow the rules closest to the end. The
  closing reminder is load-bearing; adding to it has costs.
- **Prefix KV caching is active and worth 8x** (3.43s vs 0.42s). Anything in the cached prefix
  must be piecewise-constant, and trimming the static block saves **no** per-turn latency.
- **Deduplicating the persona's format rules cost 10 points** and was reverted. The duplication
  between "How you talk" and the closing reminder is **load-bearing**. Don't retry it.
- **Ungrounded generation on a thin window confabulates** — this has bitten memory extraction,
  self-notes, and intentions, three separate times. A3's citation guard (reject anything not cited
  to a real message id) is the structural answer; apply the same shape to anything similar.
- **Thorough testing caught what smoke tests missed:** `flush()` was never wired; consolidation
  dropped facts on error; and the lifecycle **deleted a true fact** when a second item of the same
  kind arrived ("second dog" deleted the first).
- **Calibrate, don't guess** — thresholds here are measured, and the one time a plan was written
  from a plausible story about similarity scores, measuring overturned it.

---

## 6. Known limitations (honest)

- **LM Studio instability is the biggest practical pain.** It 400s with `Engine protocol predict
  request failed: fetch failed`, and under sustained eval load has crashed outright.
  `LLMClient` retries transient failures (chat only before the first token) but cannot save a
  crashed server. **Restart LM Studio before long eval runs.**
- **Reasoning can leak into `content` with no opening `<think>`** (LM Studio bug #2147). The store
  and the final reply are protected, but **you still watch the CoT stream in before it is
  replaced**. Earliest real fix is a mid-stream reset when `</think>` arrives, which needs a signal
  threaded through the `on_token` chain.
- **Tool routing ~23/30**, weakest on `reminisce`. This is the ceiling on the tool framework, not
  a bug to tune away. Unblocked cheap lever: per-call temperature (cool the tool decision to ~0.2,
  keep the answer at 0.8) — measured a *partial* fix, 0/4 → 2/4.
- **Recall is phrasing-sensitive**; an emotional preamble can sink a query. Hybrid BM25+vector is
  the documented fix. Precision at large memory volume is unmeasured.
- **Structured decisions are probabilistic** (temp 0.2) — occasional misclassification is normal.
- **Mood drift rate is untuned** — `DECAY_RATES` were calibrated for per-message decay but run
  per-tick. Related: a **melancholy positive-feedback loop** (she acts sad → user answers a sad
  companion → classifier scores that sad) is diagnosed and **unfixed**.
- **Persona tension, unresolved:** the prompt says both "you have feelings" and "you have no
  body/life"; the model sometimes resolves it by denying feelings *or* inventing an experience.

---

## 7. Git, ops, security

- **No git remote** — local-only. `origin/main` is ~100 commits stale; do not treat it as a baseline.
- **Autonomous commits are authorized** — commit worthwhile work on `main` without asking.
- `python -m web.app` binds `WEB_HOST=127.0.0.1`. **Stop the web server before running live
  scripts** — two servers against one model has crashed LM Studio.
- **One server shows as TWO PIDs on Windows and that is normal.** `.venv/Scripts/python.exe` is a
  trampoline that spawns the real interpreter as a child (no `exec` on Windows). **Check for a
  second *listener* on port 8000, not a second process.** This caused a false alarm once.
- **Back up `companion.db` before a schema migration.** `*.db.bak-*` is git-ignored.
- ⚠️ **Security, outstanding:** `archive/v1/infrastructure/config.py` holds a **real hardcoded
  HuggingFace token**. Git-ignored deliberately, **must not be committed**, and should be
  **rotated/revoked on HuggingFace**. `.env` and `companion.db` are git-ignored too.
