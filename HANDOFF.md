# v2.1 Handoff — resume here

Short brief for picking up the AI-memory-companion build in a fresh session.
**The authoritative design + full build log is [`V2_PLAN.md`](V2_PLAN.md) — read it first.**
This file is the quick "where are we / how to run / what's next" summary.

**Milestone: v2.1 is complete.** v2.0 delivered the trustworthy *brain* (memory + emotion +
conversation quality). v2.1 adds everything that makes Mari feel like a persistent companion:
**core memory, a self-modifying persona + familiarity meter, sleep/standby, a status panel +
conversation tabs, and a robust tool framework** (pillar 4). All four pillars plus the delivery
layer's first rung (tools) are built and live-verified. What remains is enrichment + reach —
see [§8 Remaining roadmap](#8-remaining-roadmap-v22).

---

## 1. What the project is

A **personal, local-first AI companion** named **Mari** — a friend, not an assistant —
for a single user, running fully local via **LM Studio**. Guiding principle
(validated in v1): build the **brain** (memory + emotion + conversation) so it's
trustworthy *before* the **delivery layer** (proactivity + tools + voice). v1 was
archived under `archive/v1/`; there's also a `V1_RETROSPECTIVE.md` with hard-won lessons.

Four pillars: **memory, emotion, proactivity, conversation quality**, plus delivery-layer
extensions: a **modular tool framework** (built) and a **voice interface** (future). Build
order followed: memory lifecycle → emotion → proactivity → conversation quality → tools.

Hardware constraint: **AMD Radeon 9070XT, 16 GB VRAM** — everything must fit a modest
budget (small models, CPU for the tiny emotion classifier, one model call at a time).

---

## 2. Current status — v2.1, built and live-verified

- **Single async runtime.** FastAPI + WebSocket **web UI** and a terminal **REPL**,
  both wired through one composition root (`bootstrap.py`) and a `Companion` facade.
- **Persistence** — SQLite with a versioned migration runner (`PRAGMA user_version`,
  schema at **v7**). Conversation survives restarts.
- **Memory tiers:**
  - *Episodic* — full verbatim conversation log (`messages` table).
  - *Semantic* — distilled, embedded facts (`memories` table), surfaced by recall.
  - *Core* — the subset of semantic facts flagged `core` (identity-defining), **always injected**
    into the prompt regardless of recall (below).
- **Recall (every turn, autonomic):** embed the incoming message (nomic
  asymmetric `search_query:`/`search_document:` prefixes — fixes the v1 first-person vs
  third-person mismatch), brute-force cosine KNN, inject hits into the system prompt.
- **Consolidation (end of a context window, backgrounded):** extract durable facts,
  then a **lifecycle** decision — duplicate (skip) / update (soft-delete old, keep history via
  `superseded_by`) / new. **Batched for speed (2026-07-18):** one extraction call, one embeddings
  call for all facts, and **one decision call covering every fact that resembles an existing memory**
  (was one call per fact); near-verbatim duplicates within a window collapse without a model call
  (`MEMORY_DUP_SIMILARITY`). Never blocks chat.
- **Core memory:** the extractor marks identity-defining facts (name, key people,
  job, where they live) as `core`; `build_system` **always injects** the core set (deduped against
  recall) so Mari never depends on a similarity search to know your name. Bounded by `CORE_MEMORY_MAX`
  — when exceeded, the brain re-ranks and demotes the least essential back to regular. Visible via the
  inspector's "Core memory" section, `GET /core`, and the REPL `/core` command. (Live: name/nurse/Seattle
  → core; hiking/coffee → regular.) This directly fixes the recall-fragility that dropped "Alex".
- **Crash-durable consolidation:** a persisted watermark
  (`meta.last_consolidated_msg_id`, via a reusable `MetaStore` KV table) checkpoints the
  last consolidated message; on startup the unconsolidated tail is recovered from the
  episodic log, so a hard kill no longer drops in-flight facts.
- **Persona** — emergent "friendly stranger" (Mari) in one prompt module
  (`core/prompts.py`), tuned via a model bake-off + iteration.
- **Emotion / mood (pillar 2):** a local RoBERTa GoEmotions classifier
  on **CPU** scores each user message; 28 labels fold into **6 mood channels** (irritation,
  warmth, amusement, melancholy, unease, interest) that decay toward a baseline at per-channel
  rates. Mood is **persisted** (survives restarts) and folds into the system prompt to color
  tone. Split as `infrastructure/emotion_classifier.py` (the model) + `core/emotion_manager.py`
  (the mood logic). Graceful: if the model can't load, chat still runs.
- **Response inspector (web + REPL):** each turn returns a `TurnResult`
  (text, stats, recalled memories, emotion read, tools called). The web UI shows a collapsible per-turn
  panel with the **memories recalled** (+ similarity), the **emotions detected** in your
  message, **Mari's 6-channel mood**, and any **tools called**; the REPL prints a compact version.
- **Concurrency safety** — a single model-access lock in `LLMClient` serializes ALL
  model calls (chat vs. background consolidation); LM Studio can't serve concurrent
  requests to a model.
- **Tick loop (pillar 3):** a pluggable job scheduler (`core/tick.py`) running a
  background heartbeat. Jobs: **mood drift** (decays the persisted mood toward baseline while the
  user is away), **drive drift** (integrates the internal drives — see below), **idle consolidation**
  (flushes the pending buffer after a while idle), **self-reflection**, **proactive reach-out**,
  **persona edit**, and **sleep**. A **busy guard** on `Companion` means jobs never fire mid-turn
  (idle reads 0 during a reply, even a slow generation).
- **Internal drives (multi-drive proactivity, arc A1):** `core/drives.py` `DriveManager` holds
  slow-integrating scalars — **connection** and **restlessness** — that rise while you're away (connection
  sped by warmth/melancholy, slowed by irritation; restlessness sped by boredom), relax while present, and
  are relieved on contact. Integrated by **elapsed wall-time** (not tick count) and **persisted** like mood;
  surfaced in the status panel. **Now load-bearing:** `ReachOutJob` fires when **connection ≥
  `DRIVE_CONNECTION_THRESHOLD`** (0.6) and `ReflectionJob` when **restlessness ≥
  `DRIVE_RESTLESSNESS_THRESHOLD`** (0.4), each discharging its drive on fire; the persisted cooldowns stay a
  hard floor, and both jobs **fall back to the old idle gate** when drives are disabled. So *how she feels*
  now sets *when* she reaches out — a warm/sad chat pulls reach-out earlier than a throwaway one. Sleep still
  runs on its idle timer (energy/body-cycles is the next slice, §8-A). Tunable via the two threshold env vars
  + the rise rates in `core/drives.py`; watch `scripts/drive_demo.py` or the panel to calibrate.
- **Sleep / standby (§2.8):** after a long idle (`SLEEP_AFTER_IDLE`, default 30 min)
  `SleepJob` calls `Companion.sleep()` — flush pending, then **unload the LLM from VRAM** via the `lms`
  CLI (`infrastructure/model_manager.py`) to free the machine. The heartbeat keeps ticking (mood still
  drifts) but the model-using jobs pause while asleep. The next message **wakes** her (`send()` reloads
  the model first; the web UI shows a "waking up…" frame for the cold-load delay). Auto-disables if `lms`
  isn't on PATH. Wake is **on-message only** — self-waking is a deliberate follow-up (see §8).
- **Self-modifying persona + familiarity meter:** `PersonaEditJob` — during idle (min-messages + long
  cooldown), `Companion.edit_persona()` rewrites a bot-owned self-description slot (`meta.persona_self`,
  injected by `build_system`) from her **thought journal** + core memories, written to herself in the
  second person. **Familiarity-gated:** a scalar from message count (`FAMILIARITY_MESSAGES`) → a label
  the edit prompt must respect, so a stranger can't write herself into a best friend. Capped
  (`PERSONA_MAX_CHARS`); can reply PASS. Visible via REPL `/persona` + `GET /persona`.
- **Self-reflection / private journal:** `ReflectionJob` — gated on the **`restlessness`
  drive** (crosses `DRIVE_RESTLESSNESS_THRESHOLD`; falls back to `REFLECT_MIN_IDLE` if drives are off, with
  `REFLECT_COOLDOWN` as a floor), `Companion.reflect()` writes a short first-person private
  thought (schema **v5** `thoughts` table + `SqliteThoughtStore`) about how she's doing, colored by and
  tagged with her current mood, avoiding recent repeats. Never shown in chat; viewable via REPL
  `/thoughts` or `GET /thoughts`. This is the substrate for reminisce and the self-modifying persona.
- **Proactive reach-out (pillar 3 payoff):** `ReachOutJob` — now gated on the **`connection`
  drive** (crosses `DRIVE_CONNECTION_THRESHOLD`; falls back to `REACHOUT_MIN_IDLE` if drives are off) with a
  persisted `REACHOUT_COOLDOWN` as a hard floor so it can't nag, `Companion.reach_out()`
  generates an unprompted message from recent context + mood and **pushes it over the WebSocket**
  (`{type:"proactive"}` → bot bubble; also logged so it replays on reconnect). Mari can reply **PASS**
  to stay quiet, and she's given **how long you've been away** (she can't see a clock) so the call is
  sensible. Web-only (needs the socket broadcaster); the REPL runs internal jobs only.
- **Follow-up messages ("double-text", 2026-07-18):** `FollowUpJob` — after Mari replies, she may fire a
  spontaneous *second* message a tick or a few later (an afterthought / addition), if she genuinely has one.
  Each user turn arms a per-turn budget (`FOLLOWUP_MAX_PER_TURN`, default 1); the job fires within a short
  `FOLLOWUP_WINDOW` (default 5 min) after her reply, gated by a per-tick `FOLLOWUP_CHANCE` (default 0.5) so
  it's off-clockwork, and `Companion.follow_up()` generates it (can reply **PASS** → closes the window; a real
  message is logged + pushed like reach-out). This replaces the old reverted "two messages split by a line
  break" approach with a **re-prompt** ("got a quick follow-up?"), which reads more natural. Distinct from
  reach-out (long idle → `connection` drive); this is the just-replied window. Web-only.
- **Web UI: status panel + conversation tabs:** a live **status panel** (right side)
  polls `GET /status` and shows Mari's whole state — awake/asleep, familiarity, the 6 mood bars, memory
  (core list + retired/superseded facts + counts), self-description, the private thought journal, and
  last tick/reach-out/reflect. Plus **conversation tabs** (left sidebar): sessions are named threads
  (schema **v7** `sessions.title`, auto-titled by the first message) you can create / switch / rename /
  delete. **Mari is one companion:** memory, mood, thoughts, persona, familiarity, and the
  consolidation/durability machinery are all the user's and **shared across every conversation**; only
  the message *thread* (history + `session_id`) is per-tab. On boot she resumes the most recent conversation.
- **Memory inspector + admin (debug, 2026-07-18):** a "memory" button in the header opens a modal that lists
  **every** memory (active + retired) and lets you **edit** a fact's text (re-embedded via
  `MemoryManager.edit_memory` so recall still matches), **delete** one (hard, unlike the lifecycle's
  soft-delete), toggle its **core** star, **clear all memories** (keeps chats/mood/persona), or **full reset**
  (wipe everything to a factory-fresh companion — `Companion.factory_reset` also resets in-memory mood/drives/
  history and starts a new conversation, no reload needed). New store ops (`all`/`update_content`/`delete`/
  `clear` on memory; `clear` on conversation/thought/meta) + HTTP routes (`GET /memories`,
  `POST /memory/{edit,delete,core,clear}`, `POST /admin/factory_reset` — confirm-gated). Both destructive
  actions confirm in the browser first.
- **Tool framework (pillar 4, 2026-07-18):** native OpenAI function-calling, verified **100%
  reliable** on qwen3.5-9b first (`scripts/tool_probe.py`) — including streamed `tool_calls` deltas.
  A hot-swappable **`ToolRegistry`** (`core/tools.py`) pairs each `Tool`'s JSON schema with an async
  handler; register one and Mari can call it next turn, nothing else changes. `LLMClient.stream_with_tools`
  runs a **streaming tool loop** that keeps token-streaming the final answer and only loops when the model
  asks for a tool. It's **robust at every seam** — malformed-arg JSON, unknown tool, and handler
  exceptions all become a result *string fed back to the model* (never an aborted turn), plus a
  `max_iters` safety net and retry-before-first-emit. Two built-in tools (`core/builtin_tools.py`):
  **`get_current_time`** (live-reliable) and **`reminisce`** (deliberate episodic recall — keyword-searches
  the full message log + private journal for "remember when…" moments, distinct from autonomic semantic
  recall). A tools-awareness block in `build_system` reconciles them with the "you just met / don't invent
  history" persona rules (reminisce recalls *real* past talks). Offline: `tests/test_tools.py` (20 cases).
  **Key finding:** reminisce is for *conversational episodes*, not *facts about the user* (those route to
  core/semantic recall) — and the model routes between them correctly on its own.

**v2.1 is feature-complete.** All four pillars + core memory + self-modifying persona + sleep + the tool
framework are built and live-verified. Everything not built is enrichment or reach (§8).

---

## 3. How to run

```bash
# from repo root, with the venv python (.venv\Scripts\python.exe on Windows)
python -m web.app       # web UI at http://127.0.0.1:8000
python main.py          # same brain, terminal REPL

python tests/test_memory_lifecycle.py   # offline, no LM Studio needed
python tests/test_memory_edge.py        # offline edge-case suite (8 cases)
python tests/test_core_memory.py        # offline core-memory flag/inject/cap (4 cases)
python tests/test_memory_admin.py       # offline memory admin: all/edit/delete/clear + factory reset (7 cases)
python tests/test_durability.py         # offline hard-kill recovery (4 cases)
python tests/test_emotion.py            # offline mood logic, fake classifier (7 cases)
python tests/test_llm_retry.py          # offline LLM transient-retry logic (6 cases)
python tests/test_tick.py               # tick scheduler + jobs (reach-out/reflection/persona/sleep/drive-drift) + familiarity (28 cases)
python tests/test_drives.py             # offline drive dynamics: rise/relax/mood-modulation/contact/discharge/persist (11 cases)
python tests/test_tools.py              # offline tool framework: registry + stream loop + reminisce (20 cases)
python scripts/tool_probe.py            # LIVE: native tool-calling reliability (probe, per-case pass rate)
python scripts/tool_smoke.py            # LIVE: tools end-to-end through the real persona (time + reminisce + no-tool)
python scripts/tool_eval.py             # LIVE: 30-scenario tool-calling eval (time/reminisce/no-tool/tricky), per-category score

python scripts/reminisce_smoke.py       # LIVE: reminisce recalls a past episode out of the context window
python scripts/drive_demo.py            # LIVE: drives observation harness — chat + away-gaps trigger reflection/reach-out
python scripts/stress_test.py           # LIVE whole-system stress + invariant checks
python scripts/emotion_eval.py          # behavioral eval: real classifier on CPU (no LM Studio)
python scripts/eval_extraction.py       # LIVE: memory-extraction quality (durable vs junk)
python scripts/eval_conversation.py     # LIVE: repetition + backbone over scripted scenarios
```

Requires **LM Studio running with its local server on** (port 1234) and the chat +
embedding models available. The emotion classifier (`SamLowe/roberta-base-go_emotions`)
downloads from HuggingFace on first run and then runs locally on CPU. Config is env-driven
via a git-ignored `.env` (see `.env.example`). Both entry points share `companion.db`
(git-ignored). Emotion can be turned off with `EMOTION_ENABLED=false`; tools with `TOOLS_ENABLED=false`.

REPL commands: `/exit` (flushes pending consolidation), `/reset`, `/model <name>`, `/temp <v>`.

---

## 4. Model setup (current)

- **⚠️ REQUIRED LM Studio template edit (huge consolidation speedup):** qwen3.5-9b is a reasoning
  model, and on the **structured (`response_format`) brain path** neither `/no_think` nor
  `enable_thinking=false` nor `chat_template_kwargs` suppresses reasoning (LM Studio bug #1990 —
  all confirmed no-ops here). Without the fix it reasons **~2,000 hidden tokens per extraction/decision
  call** → consolidation ~20× slower (~50s/call). **Fix (one-time, persists across loads):** in LM Studio
  edit qwen3.5-9b's **Prompt Template** (My Models → gear → 🧪 Advanced Config → right-click →
  *Always Show Prompt Template*, in Jinja mode) and **prepend** `{% set enable_thinking = false %}` to the
  existing template. qwen's template defaults `enable_thinking` to *on* when the var is undefined (LM Studio
  never sets it), so this initializes it false. **Verified: extraction ~50s → ~3.5s, reasoning_content 0.**
  **Tradeoff:** this disables thinking for the whole instance including chat, which modestly lowers
  tool-calling reliability (see §7). App code needs no change — it just gets fast.
- **Chat model: `qwen/qwen3.5-9b` with reasoning disabled** (switched after a
  personality bake-off). qwen3-8b was terse but **ended 72% of replies with a question**
  ("what's on your mind?", "you?") and used a formulaic sympathy pattern — the user's real
  complaint. qwen3.5-9b ends **0%** with a question, varies its phrasing, and scored 0/0/0 on
  the conversation eval (apology/capitulate/disclaim). It's a reasoning model, so `NO_THINK=true`
  appends `/no_think` → direct answers. **Tool-calling is 100% reliable on it** (§2 tool framework).
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

- **Committed (main, in order):** `5669f0f` initial v2.0 foundation → `73d517d` qwen3.5 + no-think
  → `9b4500d` crash-durable consolidation → `9f70040` emotion (pillar 2) + inspector → `15db7c5`
  conversation-quality fixes → then core memory, tick loop, reflection, reach-out, self-modifying
  persona, sleep, status panel + tabs → **`706b809` tool framework (pillar 4)**. Each pillar its own commit.
- The **user has authorized autonomous commits** — commit worthwhile work on `main` without asking
  (see the `commit-without-asking` memory). Working tree is clean as of this handoff.
- **Security note:** `archive/v1/infrastructure/config.py` holds a **real hardcoded
  HuggingFace token** and is git-ignored on purpose (specific rule in `.gitignore`).
  It should be **rotated/revoked** on HuggingFace regardless. Do not commit it.
- No git remote configured (local-only).

---

## 6. Key decisions & lessons

- **Capability tiers (see V2_PLAN §1.1):** recall/emotion are *autonomic* pipeline
  stages (no tool-calling); lifecycle/self-edit are *structured-output* (reliable
  locally); only external tools need true function-calling — which we probed at 100% before building on it.
- **Tool-loop shape (resolved):** native streaming `tool_calls` — stream the answer, loop only when the
  model requests a tool. Every failure seam (bad args, unknown tool, handler error) becomes a fed-back
  result string, never an aborted turn; a `max_iters` cap means a turn can't hang.
- **One model call at a time** (V2_PLAN §1.2) — the `LLMClient` lock. Concurrent
  requests crash LM Studio.
- **Thorough testing caught real bugs** the smoke tests missed: `flush()` was never
  wired (short sessions never consolidated); consolidation dropped facts on error (now
  re-queues); and the big one — the lifecycle wrongly **deleted a true fact** when a
  *second* item of the same kind was added ("second dog" deleted the first). Fixed in
  the decision prompt ("update only if the old fact becomes false; a second pet is new").
- **Threshold calibration, not guessing** — `RECALL_MIN_SIMILARITY=0.55` measured on
  nomic (real matches ~0.59–0.65, unrelated ~0.50).
- Storage is **brute-force numpy KNN** (fine at personal scale per v1); sqlite-vec
  stays a future swap behind the `MemoryStore` Protocol.

---

## 7. Known limitations / open issues (honest)

- **Thinking-off vs tool-calling (tradeoff from the §4 consolidation fix):** the `enable_thinking=false`
  template edit that makes consolidation ~20× faster disables reasoning on the **chat** path too, which
  hurt tool-call reliability — without reasoning she defaults to the persona's "can't sense / just met"
  rules and *under-calls* tools. Measured with `scripts/tool_eval.py` (30 scenarios): the original soft
  tools note scored **19/30** (TIME 2/8, REMINISCE 4/8). **A more imperative tools note** (`build_tools_note`,
  which now explicitly overrides those persona rules for the tool cases) recovered it to **25/30 (TIME 7/8,
  REMINISCE 5/8), with over-triggering still clean (TRICKY 6/6)** — a free fix, no VRAM cost. Remaining weak
  spot: **reminisce (~63%)**, which is inherently harder to trigger and noisier at the chat temp (0.8).
  **Two full 9B instances (thinking-on chat + thinking-off brain) is NOT an option on 16GB** — ~13GB weights
  + KV caches won't fit. Further reminisce gains would come from more prompt tuning or a lower tool-decision
  temperature (trades against personality). Backgrounded-consolidation speed was the priority and is kept.
- **Decisions are probabilistic** (temp 0.2, not deterministic) — occasional
  misclassification possible even with the tuned prompts.
- **Conversation nits (low-severity, left for later):** a rare self-deprecation slip that *denies feelings*
  ("i don't have feelings") when deflecting flattery; backbone may playfully concede a *low-stakes* opinion;
  under repeated pressure it can still write a short poem. **Root tension:** the persona says both "you have
  feelings" and "you have no body/life"; the model sometimes resolves it by denying feelings *or* inventing
  an experience. The persona/core-memory layer is the right place to keep reconciling this.
- **Recall is sensitive to query phrasing.** A clean query ("do I have any pets?") recalls a seeded fact at
  0.638, but an emotionally-prefixed one ("I'm so excited, do I have any pets?") fell below the 0.55 floor and
  recalled nothing. Now visible in the inspector. **Hybrid BM25 + vector recall (§8) is the planned fix.**
- **Recall threshold** calibrated on a small sample; precision at large memory volume unmeasured.
- **LM Studio instability is the biggest practical pain.** It 400s with `Engine protocol predict request
  failed: fetch failed`, and under sustained eval load it has hit **streaming chat** and even **crashed**
  (HTTP 000, needs a restart). Mitigation: `LLMClient` **retries** transient failures (`LLM_MAX_RETRIES=3`,
  chat only before the first token; verified in `tests/test_llm_retry.py`) — resilient to hiccups but can't
  save a fully crashed server. **Restart LM Studio before long eval runs.**
- **Mood-drift rate is untuned.** Drift decays one step per tick; at the default 60s tick that settles mood
  over several idle minutes. Tune `TICK_INTERVAL` or the per-channel `DECAY_RATES` (calibrated for
  per-message decay, not per-tick) if it feels off.
- **Multi-message / "texting burst" replies** — the original "one reply, two messages split by a blank line"
  approach was **fully reverted**. **Revisited differently (2026-07-18):** the follow-up feature (§2) sends a
  *separate, later* second message by **re-prompting** ("got a quick follow-up?") a tick or a few after the
  reply, rather than splitting one generation — reads more natural and each message is a real turn.

---

## 8. Remaining roadmap (v2.2+)

Everything below is **enrichment or reach** — the v2.1 brain is complete and none of this is on a
critical path. Grouped by theme; items the user has **already flagged as liked** are marked ★.
Full context in [`V2_PLAN.md` §2.9](V2_PLAN.md). **Delivery-layer features stay gated behind a solid
brain, per the guiding principle.**

### A. Autonomous inner life ★ (recommended next arc)
The three the user liked from the GitHub-companion research. Together they replace the single idle-timer
with something that feels alive, and they make self-wake + autonomous sleep coherent.
- **★ Multi-drive proactivity — BUILT & WIRED (2026-07-18).** Replaced the one idle gate with slow-drifting
  internal drives that cross thresholds to trigger behavior. `core/drives.py` `DriveManager` ships two drives —
  **connection** (urge to reach out; sped by warmth/melancholy, slowed by irritation) and **restlessness**
  (mental idleness → reflection; sped by boredom). They integrate by **elapsed wall-time** (deterministic,
  `TICK_INTERVAL`-independent), are **persisted** like mood, rise while away / relax while present, are relieved
  on contact (`Companion.on_user_message`), and are surfaced in `GET /status` + a "Drives" panel section. A
  `DriveDriftJob` updates them each tick. **Reach-out is gated on `connection`, reflection on `restlessness`**
  (thresholds `DRIVE_CONNECTION_THRESHOLD` 0.6 / `DRIVE_RESTLESSNESS_THRESHOLD` 0.4; discharge on fire;
  cooldowns as a hard floor; idle-gate fallback when drives are off). Tuned from the first live demo:
  `connection` has a healthy gradient (0.66 neutral vs 0.83 warm/sad at 15 min — mood modulation is real);
  restlessness rise cut 15→5/hr so it grades instead of pegging at 1.0 in 4 min. `scripts/drive_demo.py` is a
  live observation harness. **Remaining in this item:** the extra drives (mood/anxiety/busyness) are optional;
  practical tuning of the thresholds/rates is a user-testing task.
- **★ Energy / body cycles** — a fatigue/energy stat that biases toward rest, giving *autonomous* sleep an
  internal logic instead of a fixed 30-min timer.
- **★ Nightly / scheduled deep consolidation** — an "end of day" job that summarizes the day into an episodic
  day-summary (great reminisce fuel), distinct from the per-window consolidation.
- **Unlocks self-waking** (waking to reach out) — currently deferred precisely because it needs a real
  trigger (energy, or a due reminder) to be principled rather than twitchy. `wake()` is already a public seam.

### B. Memory depth
- **Memory salience / forgetting curve** — importance+recency weighting; trivial facts fade, often-recalled
  ones strengthen. Keeps the store lean, makes recall feel human.
- **Memory confidence + confirmation** — track uncertainty and occasionally double-check a shaky fact
  ("was it Kate or Katelyn?"), self-correcting the lifecycle.
- **Fact-validity windows (temporal memory)** — track *when* each fact was true; the principled version of
  the recency/conflict tie-breaker we punted on (Zep/Graphiti style).
- **"On this day" recall** — time-anchored callbacks via reminisce; pairs with proactivity.
- **Hybrid BM25 + vector recall** — keyword + semantic; directly targets the recall phrasing-sensitivity
  limit in §7.
- ~~**Memory inspector UI** — browse / edit memories + view superseded history.~~ **DONE (2026-07-18):** the
  header "memory" modal browses all memories (active + retired), edits (re-embed), deletes, toggles core, and
  clears / full-resets. Still open here: *search/filter* within the inspector (fine at current volume).

### C. Presence & timing
- **Presence signal** — the WebSocket already knows if the tab is focused / the user is typing; use it as a
  *real* "is the user here?" input to the tick/sleep logic instead of guessing from elapsed time. **Near-freebie.**
- **Time-of-day awareness** — greet differently morning vs. late night, notice patterns ("up late again").
  The `get_current_time` tool partially enables this now.
- **Do-not-disturb / time-of-day gating** for proactivity — keeps reach-out and self-wake from firing at 3am.

### D. Reach beyond the tab
- **Push notifications** (e.g. Bark) — proactive messages reach the phone instead of dying in a closed tab;
  also unlocks a genuinely *useful* self-wake.
- **Multi-channel presence** — Mari over WhatsApp / Telegram / Discord instead of only the local web UI.

### E. More tools (framework ready; paused by the user)
The pillar-4 framework is built and hot-swappable — each of these is "register a `Tool`, nothing else changes."
- **Web search**, **mood-based Navidrome/Subsonic playlist creator**, **reminder tool**, **curiosity-driven
  search** (self-initiated search + reflection during a tick), and exposing **`rewrite_self`** as a real tool
  (unifies the persona rewrite with the tool framework).

### F. Whimsical / far-future
- **Dreams** — during sleep, generate one memory-recombining "dream" she might mention on waking. One per wake.
- **Voice** (STT/TTS) — the last delivery layer; later, **acoustic emotion perception** (hearing *how* you say
  something, not just the words).
- **Embodiment** — Live2D / VRM avatar, wearable sensors.

**Recommendation:** the **A arc (multi-drive → energy → nightly consolidation)** is the highest-leverage next
move — user-picked, it makes the whole system feel alive rather than timer-driven, and it retroactively makes
self-wake and autonomous sleep coherent. Start with multi-drive proactivity (the tick already has the gate
structure to generalize). The near-freebies in C (presence signal, time-of-day awareness) can slot in alongside.
