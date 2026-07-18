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
  schema at **v4**). Conversation survives restarts.
- **Two memory tiers:**
  - *Episodic* — full verbatim conversation log (`messages` table).
  - *Semantic* — distilled, embedded facts (`memories` table).
- **Recall (every turn, autonomic):** embed the incoming message (nomic
  asymmetric `search_query:`/`search_document:` prefixes — fixes the v1 first-person vs
  third-person mismatch), brute-force cosine KNN, inject hits into the system prompt.
- **Consolidation (end of a context window, backgrounded):** extract durable facts,
  then a **lifecycle** decision per fact — duplicate (skip) / update (soft-delete old,
  keep history via `superseded_by`) / new. Never blocks chat.
- **Crash-durable consolidation (new 2026-07-18):** a persisted watermark
  (`meta.last_consolidated_msg_id`, via a reusable `MetaStore` KV table) checkpoints the
  last consolidated message; on startup the unconsolidated tail is recovered from the
  episodic log, so a hard kill no longer drops in-flight facts.
- **Persona** — emergent "friendly stranger" (Mari) in one prompt module
  (`core/prompts.py`), tuned via a model bake-off + iteration.
- **Emotion / mood (pillar 2, new 2026-07-18):** a local RoBERTa GoEmotions classifier
  on **CPU** scores each user message; 28 labels fold into **6 mood channels** (irritation,
  warmth, amusement, melancholy, unease, interest) that decay toward a baseline at per-channel
  rates. Mood is **persisted** (survives restarts) and folds into the system prompt to color
  tone. Split as `infrastructure/emotion_classifier.py` (the model) + `core/emotion_manager.py`
  (the mood logic). Graceful: if the model can't load, chat still runs.
- **Response inspector (web + REPL, new 2026-07-18):** each turn now returns a `TurnResult`
  (text, stats, recalled memories, emotion read). The web UI shows a collapsible per-turn
  panel with the **memories recalled** (+ similarity), the **emotions detected** in your
  message, and **Mari's 6-channel mood**; the REPL prints a compact one-line version.
- **Concurrency safety** — a single model-access lock in `LLMClient` serializes ALL
  model calls (chat vs. background consolidation); LM Studio can't serve concurrent
  requests to a model (that crash was hit and fixed this session).

Not yet built: **tick/proactivity, tools, voice.** Persona self-modification and
familiarity meter are planned but not started. Emotion is done but not yet wired into
a tick (mood drift on idle) — that lands with proactivity (pillar 3).

---

## 3. How to run

```bash
# from repo root, with the venv python (.venv\Scripts\python.exe on Windows)
python -m web.app       # web UI at http://127.0.0.1:8000
python main.py          # same brain, terminal REPL

python tests/test_memory_lifecycle.py   # offline, no LM Studio needed
python tests/test_memory_edge.py        # offline edge-case suite (8 cases)
python tests/test_durability.py         # offline hard-kill recovery (4 cases)
python tests/test_emotion.py            # offline mood logic, fake classifier (7 cases)
python tests/test_llm_retry.py          # offline LLM transient-retry logic (6 cases)
python scripts/emotion_eval.py          # behavioral eval: real classifier on CPU (no LM Studio)
python scripts/eval_extraction.py       # LIVE: memory-extraction quality (durable vs junk)
python scripts/eval_conversation.py     # LIVE: repetition + backbone over scripted scenarios
```

Requires **LM Studio running with its local server on** (port 1234) and the chat +
embedding models available. The emotion classifier (`SamLowe/roberta-base-go_emotions`)
downloads from HuggingFace on first run and then runs locally on CPU. Config is env-driven
via a git-ignored `.env` (see `.env.example`). Both entry points share `companion.db`
(git-ignored). Emotion can be turned off with `EMOTION_ENABLED=false`.

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
- **Emotion: `SamLowe/roberta-base-go_emotions`** (~125M), runs on **CPU** (`device=-1`),
  ~0.5 GB RAM, leaves all VRAM for the LLMs. Configurable via `EMOTION_MODEL`.
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

- ~~**Hard kill loses the unconsolidated buffer**~~ **FIXED (2026-07-18).** A persisted
  watermark (`meta.last_consolidated_msg_id`) checkpoints the last consolidated message
  id; on startup the tail (`messages_after(watermark)`) is recovered into the buffer, so
  a crash no longer drops in-flight facts. The watermark advances only on a *successful*
  consolidation and only one runs at a time, so it stays contiguous. See V2_PLAN build log.
- **Decisions are probabilistic** (temp 0.2, not deterministic) — occasional
  misclassification possible even with the tuned prompts.
- **Conversation-quality pass (2026-07-19)** fixed three complaints from real use, each with
  a live eval (`scripts/eval_conversation.py`, `scripts/eval_extraction.py`):
  - *Verbatim repetition* (the bot copied its own earlier replies, e.g. the "vending machine"
    line, and rewrote the same poem stanza 3×): added `FREQUENCY_PENALTY`/`PRESENCE_PENALTY`
    on chat + an anti-repeat persona rule. After: cross-scenario dups 0, only a rare short echo.
  - *Bad memories (too specific/temporal/self)*: rewrote `MEMORY_EXTRACTION_SYSTEM` to keep
    only timeless user-life facts, exclude current activities / app-meta / Mari's own lines /
    plans, and normalize phrasing. After: extraction eval went **8+ bad → 0 bad**.
  - *Too agreeable*: gave the persona a backbone (holds positions, doesn't cave to pressure/
    flattery, doesn't grovel), made it own its feelings (no "just a chatbot"), and wired mood
    to behavior. After: pushes back on insults, resists tasks, mood (irritation) shortens replies.
- **Remaining conversation nits (honest, low-severity):** (1) extraction still sometimes drops
  the user's **name** ("I'm Alex" read as a greeting) — strengthened the prompt with an explicit
  name example on 2026-07-19, **pending live re-verification** (LM Studio was down at commit time);
  (2) an occasional "just a chatbot" self-deprecation slip (tightened, also pending re-verify);
  (3) under enough repeated pressure it may still write a short (non-repeated) poem instead of
  fully refusing. Re-run the two eval scripts once LM Studio is healthy to confirm 1 and 2.
- Recall threshold calibrated on a small sample; precision at large memory volume
  unmeasured. Only qwen3-8b / gemma tested for extraction/decisions.
- **Recall is sensitive to query phrasing.** A clean query ("do I have any pets?") recalls
  a seeded fact at 0.638, but an emotionally-prefixed one ("I'm so excited, do I have any
  pets?") fell below the 0.55 floor and recalled nothing. Pre-existing (not emotion-related),
  but now visible in the inspector. Options later: embed only the salient clause, lower the
  floor, or re-rank.
- **LM Studio instability is the biggest practical pain.** It 400s with `Engine protocol
  predict request failed: fetch failed` — originally just on structured extraction, but under
  sustained eval load it also hit **streaming chat** and eventually **crashed** (HTTP 000, needs
  a restart). Mitigation added 2026-07-19: `LLMClient` now **retries** transient failures
  (`LLM_MAX_RETRIES=3`, chat only before the first token so it never double-streams; verified in
  `tests/test_llm_retry.py`). This makes chat/consolidation resilient to hiccups but can't save a
  fully crashed server. If it keeps happening, suspect the `response_format` grammar path or give
  qwen3-8b more headroom / a different quant. **Restart LM Studio before long eval runs.**
- **Emotion now influences behavior, not just tone.** The persona wires mood to conduct
  (irritated → shorter/less accommodating), and the eval shows irritation climbing on insults
  with visibly clipped replies. Still: "approval" from bland acknowledgements ("ok sure") nudges
  warmth up a little (mapping nuance, not clearly wrong); mood only shifts on user messages; idle
  mood-drift waits for the tick (pillar 3).

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

1. ✅ ~~Commit the qwen3-8b/no-think change~~ (landed as `73d517d`).
2. ✅ ~~Close the **hard-kill durability** gap~~ (done 2026-07-18 — watermark + tail
   recovery; §7). Reusable `MetaStore` (KV) added — emotion will use it for mood.
3. ✅ ~~Emotion (pillar 2)~~ (done 2026-07-18 — RoBERTa GoEmotions → 6 mood channels on
   CPU, persisted mood, behavioral eval passed; plus a response inspector in the web UI +
   REPL). V2_PLAN §2.3 / build log.
4. **Proactivity (pillar 3)** — the tick loop. A pluggable job scheduler that on each tick
   does: **mood drift** (decay the now-persisted mood toward baseline while idle — the
   emotion hook is ready), internal reflection, and the "should I reach out?" gate that
   pushes an unprompted, model-generated message over the **WebSocket**. Definition of done:
   a real unprompted message arrives in the UI. Pairs with **sleep/standby** (§2.8). V2_PLAN §2.4.
5. Then the tool framework (pillar 4).

Delivery-layer features stay gated behind a solid brain, per the guiding principle.
