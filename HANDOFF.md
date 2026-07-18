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
- **Tick loop (pillar 3, new 2026-07-19):** a pluggable job scheduler (`core/tick.py`) running a
  background heartbeat. Jobs: **mood drift** (decays the persisted mood toward baseline while the
  user is away), **idle consolidation** (flushes the pending buffer after a while idle), and
  **proactive reach-out** (below). A **busy guard** on `Companion` means jobs never fire mid-turn
  (idle reads 0 during a reply, even a slow generation). Started by the web app + REPL.
- **Proactive reach-out (pillar 3 payoff, new 2026-07-19):** `ReachOutJob` — after the user is idle
  past `REACHOUT_MIN_IDLE` (+ a persisted `REACHOUT_COOLDOWN` so it can't nag), `Companion.reach_out()`
  generates an unprompted message from recent context + mood and **pushes it over the WebSocket**
  (`{type:"proactive"}` → bot bubble; also logged so it replays on reconnect). Mari can reply **PASS**
  to stay quiet, and she's given **how long you've been away** (she can't see a clock) so the call is
  sensible — live she PASSed after "gonna go sleep" but checked in after an unresolved bad-day vent.
  Web-only (registered in `web/app.py` with the connection broadcaster); the REPL runs internal jobs only.

Not yet built: **tools, voice, sleep/standby.** Persona self-modification, core memory,
and the familiarity meter are planned but not started.

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
python tests/test_tick.py               # offline tick-loop scheduler + jobs + reach-out (13 cases)
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

- **Chat model: `qwen/qwen3.5-9b` with reasoning disabled** (switched 2026-07-19 after a
  personality bake-off). qwen3-8b was terse but **ended 72% of replies with a question**
  ("what's on your mind?", "you?") and used a formulaic sympathy pattern — the user's real
  complaint. qwen3.5-9b ends **0%** with a question, varies its phrasing, and scored 0/0/0 on
  the conversation eval (apology/capitulate/disclaim) vs qwen3-8b's residual slips. It's a
  reasoning model, so `NO_THINK=true` appends `/no_think` → direct answers.
  - **Roleplay finetunes were tested and rejected:** `neona-12b-i1` and `rocinante-12b-v1.1`
    both **break on our plain OpenAI-style chat API** — they generate both sides of the
    conversation ("human:/ai:"), leak the persona text, or return empty replies (chat-template
    / stop-token mismatch; they're built for SillyTavern + character cards). qwen3.5-9b stays in
    the qwen/ChatML family the app already handles cleanly.
  - Note: LM Studio's `enable_thinking=false` flag is a **no-op** here; `/no_think` in the
    prompt is what works. Reasoning streams on a separate `reasoning_content` channel.
- **Embedding: `text-embedding-nomic-embed-text-v1.5`** (CPU-friendly, small).
- **Emotion: `SamLowe/roberta-base-go_emotions`** (~125M), runs on **CPU** (`device=-1`),
  ~0.5 GB RAM, leaves all VRAM for the LLMs. Configurable via `EMOTION_MODEL`.
- **Brain (consolidation): reuses the chat model** by default (`BRAIN_MODEL` empty),
  so only ~6.5 GB (qwen3.5-9b) + ~0.1 GB (nomic) is resident — no second big model, no
  load/unload thrash. Extraction verified clean on qwen3.5-9b (eval: 15 captured / 0 bad).
- `.env` now: `MODEL=qwen/qwen3.5-9b`, `NO_THINK=true`, `BOT_NAME=Mari`. Repetition penalties
  (`FREQUENCY_PENALTY=0.4`, `PRESENCE_PENALTY=0.3`) + `LLM_MAX_RETRIES=3` also apply.

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
- **Conversation nits, re-verified 2026-07-19 (LM Studio healthy):** name-capture is now **fixed**
  — extraction reliably yields "The user's name is Alex" (full eval: 17 captured / **0 bad**), and
  the conversation eval shows near-dups 0 / cross-dups 0. Still occasional and low-severity, and
  left for the **planned self-modifying-persona + core-memory pass** (they share the tension below):
  - a rare self-deprecation slip that *denies feelings* ("i don't have feelings", "just a chat")
    when deflecting flattery (~1/31) — directly contradicts the emotion pillar, so it's the
    meaningful one to fix.
  - backbone is strong on insults/pressure but it may playfully concede a *low-stakes* opinion
    ("okay fine, you're right").
  - under repeated pressure it can still write a short (non-repeated) poem, and "prove you feel
    something" once made it invent a small emotional backstory ("i started crying").
  - **Root tension:** the persona says both "you have feelings" and "you have no body/life"; the
    model sometimes resolves it by either denying feelings *or* inventing an experience. The
    persona/core-memory work is the right place to reconcile this.
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
  with visibly clipped replies. Mood shifts on user messages, and now **drifts back toward
  baseline while idle** via the tick loop's mood-drift job. Minor: "approval" from bland
  acknowledgements ("ok sure") nudges warmth up a little (mapping nuance, not clearly wrong).
- **Mood-drift rate is untuned.** Drift decays one step per tick; at the default 60s tick that
  settles mood over several idle minutes. If it feels too fast/slow, tune `TICK_INTERVAL` or the
  per-channel `DECAY_RATES` (which were calibrated for per-message decay, not per-tick).

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
4. ✅ ~~Proactivity (pillar 3)~~ (done 2026-07-19). Internal tick loop (`core/tick.py` pluggable
   scheduler + mood-drift + idle-consolidation, busy guard) **and** the outward **reach-out**
   (`ReachOutJob` → `Companion.reach_out()` → `{type:"proactive"}` WebSocket push, PASS-escape,
   idle-duration signal, cooldown). Live-verified: Mari messages first when it's warranted.
5. **Sleep / standby (§2.8)** — the natural next tick behavior: unload the LLM(s) from VRAM when
   idle + low energy (a bot-initiated sleep job) and reload on wake ("waking up…" state). Ties to
   the energy-budget idea; the model-lifecycle piece is the `lms load/unload` we've been driving by hand.
6. **Core memory + self-modifying persona** (the user's idea, discussed) — an always-in-prompt curated
   fact block (a `core` flag on `memories`, brain-curated, capped) + Mari's editable self-description,
   both as tick reflection jobs. Also resolves the "has feelings" vs "no body" persona tension.
7. Then the tool framework (pillar 4): web search, Navidrome playlists, reminders, reminisce.

Delivery-layer features stay gated behind a solid brain, per the guiding principle.
