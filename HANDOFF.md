# v2.0 Handoff — resume here

Short brief for picking up the AI-memory-companion (v2.0) build in a fresh session.
**The authoritative design + full build log is [`V2_PLAN.md`](V2_PLAN.md) — read it first.**
This file is the quick "where are we / how to run / what's next" summary.

---

## 1. What the project is

A **personal, local-first AI companion** named **Mari** — a friend, not an assistant —
for a single user, running fully local via **LM Studio**. Guiding principle
(validated in v1): build the **brain** (memory + emotion + conversation) so it's
trustworthy *before* the **delivery layer** (proactivity + tools + voice). v1 was
archived under `archive/v1/`; there's also a `V1_RETROSPECTIVE.md` with hard-won lessons.

Four pillars: **memory, emotion, proactivity, conversation quality**, plus later
extensions: a **modular tool framework** (web search + a Navidrome/Subsonic playlist
creator) and a **voice interface**. Build order chosen: memory lifecycle → emotion →
proactivity → tools.

Hardware constraint: **AMD Radeon 9070XT, 16 GB VRAM** — everything must fit a modest
budget (small models, CPU for the tiny emotion classifier, one model call at a time).

---

## 2. Current status — what's built and working

The **brain foundation is done and live-verified.** In place:

- **Single async runtime.** FastAPI + WebSocket **web UI** and a terminal **REPL**,
  both wired through one composition root (`bootstrap.py`) and a `Companion` facade.
- **Persistence** — SQLite with a versioned migration runner (`PRAGMA user_version`,
  schema at **v3**). Conversation survives restarts.
- **Two memory tiers:**
  - *Episodic* — full verbatim conversation log (`messages` table).
  - *Semantic* — distilled, embedded facts (`memories` table).
- **Recall (every turn, autonomic):** embed the incoming message (nomic
  asymmetric `search_query:`/`search_document:` prefixes — fixes the v1 first-person vs
  third-person mismatch), brute-force cosine KNN, inject hits into the system prompt.
- **Consolidation (end of a context window, backgrounded):** extract durable facts,
  then a **lifecycle** decision per fact — duplicate (skip) / update (soft-delete old,
  keep history via `superseded_by`) / new. Never blocks chat.
- **Persona** — emergent "friendly stranger" (Mari) in one prompt module
  (`core/prompts.py`), tuned via a model bake-off + iteration.
- **Concurrency safety** — a single model-access lock in `LLMClient` serializes ALL
  model calls (chat vs. background consolidation); LM Studio can't serve concurrent
  requests to a model (that crash was hit and fixed this session).

Not yet built: **emotion, tick/proactivity, tools, voice.** Persona self-modification
and familiarity meter are planned but not started.

---

## 3. How to run

```bash
# from repo root, with the venv python (.venv\Scripts\python.exe on Windows)
python -m web.app       # web UI at http://127.0.0.1:8000
python main.py          # same brain, terminal REPL

python tests/test_memory_lifecycle.py   # offline, no LM Studio needed
python tests/test_memory_edge.py        # offline edge-case suite (8 cases)
```

Requires **LM Studio running with its local server on** (port 1234) and the chat +
embedding models available. Config is env-driven via a git-ignored `.env` (see
`.env.example`). Both entry points share `companion.db` (git-ignored).

REPL commands: `/exit` (flushes pending consolidation), `/reset`, `/model <name>`, `/temp <v>`.

---

## 4. Model setup (current)

- **Chat model: `qwen3-8b` with reasoning disabled.** gemma-3-4b was too shallow
  (dismissive, incurious, hollow "You?" deflection). qwen3-8b is markedly smarter and
  more engaging. It's a reasoning model, so `NO_THINK=true` appends `/no_think` to the
  system message → direct answers, ~2.2s warm ttft, no reasoning latency.
  - Note: LM Studio's `enable_thinking=false` flag is a **no-op** here; `/no_think` in
    the prompt is what actually works. This build streams reasoning on a separate
    `reasoning_content` channel (not inline `<think>`), which is why a reasoning model
    with a token cap could return an empty answer.
- **Embedding: `text-embedding-nomic-embed-text-v1.5`** (CPU-friendly, small).
- **Brain (consolidation): reuses the chat model** by default (`BRAIN_MODEL` empty),
  so only ~5 GB (qwen3-8b) + ~0.5 GB (nomic) is resident — no second big model, no
  load/unload thrash. Reasoning can be re-enabled (`NO_THINK=false`) for the brain /
  future complex tasks, where latency is hidden in the background.
- `.env` currently: `MODEL=qwen3-8b`, `NO_THINK=true`, `BOT_NAME=Mari`.

To swap models one-at-a-time for testing, use `lms unload --all` between loads (LM
Studio JIT keeps them resident otherwise and will exhaust 16 GB). Bake-off + speed
harnesses live in `scripts/` (`bakeoff.py`, `bench_speed.py`, `prompt_test.py`).

---

## 5. Git state ⚠️

- **Committed:** `5669f0f` "Initial v2.0: local AI companion brain (memory + web UI)"
  — the full foundation incl. memory lifecycle, tests, and edge-case hardening.
- **Uncommitted (on disk, verified, NOT yet committed):** the qwen3-8b + `/no_think`
  model switch — `config.py` (NO_THINK), `infrastructure/llm_client.py` (no_think +
  `_prep`), `bootstrap.py`, `V2_PLAN.md` updates, and `.env` (git-ignored). **Natural
  next action: commit this.** Suggested message: "Switch chat to qwen3-8b + no-think".
- **Security note:** `archive/v1/infrastructure/config.py` holds a **real hardcoded
  HuggingFace token** and is git-ignored on purpose (specific rule in `.gitignore`).
  It should be **rotated/revoked** on HuggingFace regardless. Do not commit it.
- No git remote configured (local-only).

---

## 6. Key decisions & lessons from this session

- **Capability tiers (see V2_PLAN §1.1):** recall/emotion are *autonomic* pipeline
  stages (no tool-calling); lifecycle/self-edit are *structured-output* (reliable
  locally); only external tools need true function-calling (test before relying on it).
- **One model call at a time** (V2_PLAN §1.2) — the `LLMClient` lock. Concurrent
  requests crash LM Studio.
- **Thorough testing caught real bugs** the smoke tests missed: `flush()` was never
  wired (short sessions never consolidated); consolidation dropped facts on error (now
  re-queues); and the big one — the lifecycle wrongly **deleted a true fact** when a
  *second* item of the same kind was added ("second dog" deleted the first). Fixed in
  the decision prompt ("update only if the old fact becomes false; a second pet is new").
- **Extraction over-eagerness** fixed (was inventing persona `self` facts).
- **Threshold calibration, not guessing** — `RECALL_MIN_SIMILARITY=0.55` measured on
  nomic (real matches ~0.59–0.65, unrelated ~0.50).
- Storage is **brute-force numpy KNN** (fine at personal scale per v1); sqlite-vec
  stays a future swap behind the `MemoryStore` Protocol.

---

## 7. Known limitations / open issues (honest)

- **Hard kill loses the unconsolidated buffer** — `flush()` only runs on *graceful*
  shutdown (web shutdown event / REPL `/exit`). A crash still drops in-flight messages.
  Fix later: persist a "last consolidated message id" and resume on startup.
- **Decisions are probabilistic** (temp 0.2, not deterministic) — occasional
  misclassification possible even with the tuned prompts.
- Persona still occasionally slips a mild embodiment line ("i saw someone play it")
  and the prompt's `"you?"` example can seed an occasional bounce-back — both minor,
  prompt-tunable.
- Recall threshold calibrated on a small sample; precision at large memory volume
  unmeasured. Only qwen3-8b / gemma tested for extraction/decisions.

---

## 8. Parked ideas

- **Multi-message / "texting burst" replies** (bot sends 2–3 short bubbles in a row):
  the user asked about this, we started it, then **reverted it fully** (this handoff's
  request). Approach if revisited: model separates messages with a blank line; the web
  UI renders each as its own bubble with a short "typing" pause; store one row, split
  for display + on reload. Nothing from this remains in the code.
- Backlog (V2_PLAN §2.9): familiarity meter, status panel, presence signal, private
  thought journal, reminders, dreams, energy budget, memory salience/forgetting, etc.

---

## 9. Next steps (in plan order)

1. **Commit the uncommitted qwen3-8b/no-think change** (see §5).
2. **Emotion (pillar 2)** — keep v1's RoBERTa GoEmotions → 6 mood channels approach,
   but run it on **CPU** (the 9070XT can't use v1's CUDA path), **persist mood to DB**,
   and run a **behavioral eval** (never done in v1). See V2_PLAN §2.3.
3. Optionally first: close the **hard-kill durability** gap (§7).
4. Then proactivity (tick loop + reach-out over WebSocket) → tools framework.

Delivery-layer features stay gated behind a solid brain, per the guiding principle.
