# v2.2 Handoff — resume here

Short brief for picking up the AI-memory-companion build in a fresh session.
**The authoritative design + full build log is [`V2_PLAN.md`](V2_PLAN.md) — read it first.**
This file is the quick "where are we / how to run / what's next" summary.

**Milestone: v2.1 complete; v2.2 in progress.** v2.0 built the trustworthy *brain*; v2.1 added the
persistent-companion layer (core memory, self-modifying persona + familiarity, sleep/standby, status
panel + conversation tabs, tool framework). **v2.2 so far (2026-07-18):** **multi-drive proactivity**
(arc A1 — internal drives now gate reach-out/reflection, §8-A), **spontaneous follow-up messages**
("double-text"), an **editable memory inspector + admin** (browse/edit/delete, clear, factory reset),
and a **big consolidation speed win** (~150s → ~6.5s, via an LM Studio thinking-off template edit +
batching). All live-verified; offline suite **107 green**. **Added 2026-07-19:** phone push, a web-UI +
**iOS/Safari mobile** overhaul, arc-A2 **energy cycles**; a **memory-extraction fix** (a name buried in a 20-msg
consolidation window was silently dropped → `CONSOLIDATE_WINDOW` 10 + a stronger extraction prompt + bounded
chunks); a **personality overhaul** (hard *one-sentence* / *no-trailing-question* rules reinforced by an
end-of-prompt reminder → measured **100% one-sentence, 7% q-end**) + a **follow-up rewrite** (rarer, no invented
experiences); a new **silent-turn** capability (she can `PASS` to not reply → a faint "stayed quiet" marker); and
a **thinking-depth investigation + 3-model bake-off** that kept qwen and **parked bounded thinking** on LM Studio
(§0). §8 roadmap gained the **Generative-Agents "planning" arc** — and that arc is now **complete**: **intentions**
(a private forward agenda that makes proactivity goal-directed) plus **learned self-notes** (she distills her own
"ease off the questions" rules from how Alex reacts, injected into every user-facing prompt), closing the
memory + reflection + planning triad.

**⚠️ ACTIVE THREAD — PARKED 2026-07-19, waiting on LM Studio; full detail in §0.** Production stays **thinking-OFF
and stable on qwen3.5-9b.** This session confirmed **bounded thinking is impossible in LM Studio today** — llama.cpp
has a native reasoning budget (`thinking_budget_tokens`) but LM Studio *strips* it (probed); getting it needs
llama-server (AMD-build risk + sleep/embed rewire) or vLLM, neither worth it now → **wait for LM Studio to expose it
(bug #1838 / #1974).** A **three-model bake-off** (qwen vs gpt-oss vs Gemma 4 12B QAT) also ran: **gpt-oss rejected**
(88% question-ending — it interrogates; plus dashes), **tuned Gemma is a viable fallback** (non-interrogative, clean,
reasoning 2–4× less circular than qwen, but ~13s and its thinking-off speed is untested). **Decision: stay on qwen,
wait.** Still open + unblocked: **tool-calling prong A (per-call temperature)** — the cheap win, no migration. The
`core/prompts.py` tweak is still uncommitted. Everything else is enrichment/reach — see §8.

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

## 2. Current status (v2.2 in progress) — built and live-verified

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
  now sets *when* she reaches out — a warm/sad chat pulls reach-out earlier than a throwaway one. Tunable via
  the two threshold env vars + the rise rates in `core/drives.py`; watch `scripts/drive_demo.py` or the panel.
- **Energy / body cycle (arc A2, 2026-07-18):** the `DriveManager` also holds an **energy** reserve [0,1] that
  **depletes while awake, restores while asleep** (a body cycle, not an away-drive), shown as a status bar.
  `SleepJob` now sleeps on **low energy + a brief idle** (she's tired) as well as the long-idle VRAM trigger, so
  sleep is internally motivated, not just a 30-min timer. Self-wake stays deferred until time-of-day gating (§8-A).
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
  `FOLLOWUP_WINDOW` (**60s** since 2026-07-19; was 5 min) after her reply, gated by a per-tick `FOLLOWUP_CHANCE`
  (**0.2** since 2026-07-19; was 0.5) so it's rare and off-clockwork, and `Companion.follow_up()` generates it (can
  reply **PASS** → closes the window; a real message is logged + pushed like reach-out). **Prompt rewritten
  2026-07-19** to bias hard toward PASS and forbid inventing an experience or firing off just another question (it
  had hallucinated "I found a good spot to sit"). This replaces the old reverted "two messages split by a line
  break" approach with a **re-prompt** ("got a quick follow-up?"), which reads more natural. Distinct from
  reach-out (long idle → `connection` drive); this is the just-replied window. Web-only.
  - **↗ TODO (user-requested 2026-07-19):** extend follow-ups beyond the `FOLLOWUP_MAX_PER_TURN=1` cap to
    allow *several* chained messages with a **decaying probability** per additional message (occasionally 2–3
    short texts in a row, tapering off), instead of a hard 1-per-turn limit. Pairs with the new brevity rule
    (replies are now one sentence, so more of them can chain naturally). Tune alongside `FOLLOWUP_CHANCE`.
- **Silent turns — she can choose not to reply (2026-07-19):** the user asked for the ability *not* to respond,
  as an extension of the same `PASS` convention reach-out/follow-up use. `build_system(..., allow_silence=True)` on
  the chat path offers the option; if `Companion.send()` gets back **PASS**, it logs the user's message but **no
  reply of her own** — mood/drives still update, the message still counts for memory, and no follow-up is armed. A
  small gate in `send()` **holds the token stream** until it's clear the reply isn't "PASS", so the word never
  flashes on screen. The web UI drops the typing bubble and shows a faint **"· Mari stayed quiet"** marker.
  **Ungated for now** (she can pass on anything); add mood/low-effort gating if she goes quiet too often. Gate
  logic offline-tested; suite green. *(Uncommitted as of 2026-07-19 — user testing live.)*
- **Intentions — a private forward agenda (the "planning" pillar, 2026-07-19):** Mari keeps short notes of things
  she means to bring up or find out next time, so proactivity is *goal-directed* rather than generic. Schema **v8**
  `intentions` table + `SqliteIntentionStore` (+ Protocol): add / `active()` oldest-first (FIFO) / `fulfill` /
  `drop` / `clear`. **Minting:** `Companion.form_intentions()` asks the brain for genuinely-new intentions from the
  recent window, deduped against the open agenda and capped at `INTENTION_MAX_ACTIVE` (8, oldest dropped); driven by
  **`IntentionJob`** (idle + `INTENTION_COOLDOWN` gated, its own cadence — internal, nothing pushed). **Consumption:**
  `reach_out()` anchors on the **longest-waiting** intention ("You've had this on your mind to bring up…") and
  **fulfills it only if she actually sends**; a PASS leaves it open. **Visible** via `GET /status` + an "Intentions"
  status-panel card. **Chat + follow-up awareness:** `build_system(..., intentions=[...])` rides the open agenda
  along on every chat turn, framed *softly* ("only raise one if the conversation naturally gets there; never force
  one in"), and `build_followup_system(..., intention=...)` lets a double-text fold one in if it connects.
  **Resolution:** rather than a fuzzy match or a second model call, `form_intentions` also returns the **numbers of
  existing intentions the conversation covered** and clears them — which is how chat- and follow-up-raised items get
  retired. **Expiry:** items older than `INTENTION_MAX_AGE_DAYS` (7) are dropped before each pass, so a stale agenda
  neither lingers nor crowds the cap. **Prompt gotcha (fixed):** the first draft said "most reflections add none",
  which set the default to empty and made thinking-off qwen return `[]` even on a window full of follow-ups (the same
  shallow-pass failure as the memory-extraction bug); rebalanced with concrete examples → 3/3 on a rich window, `[]`
  on a barren meta-chat. `tests/test_intentions.py` (**33 offline cases**). **Live-verified:** formed "ask how
  Deadlock is going" from the real log; resolution correctly returned `[1]` when a conversation covered the dye
  intention; and with an open agenda injected, chat showed **0/7 shoehorning into unrelated messages** with
  personality intact (7/8 one-sentence, 1/8 q-end).
- **Learned self-notes — reflection that changes behavior (2026-07-19):** the closed loop *experience → principle
  → future behavior*, and the second half of the planning arc. Mari distills short operating-notes about **how to
  BE with this user** ("He gets annoyed when you ask questions — react with an opinion instead"), which ride in
  **every user-facing prompt**. Sibling to the persona edit: that one rewrites **who she is** from her private
  thoughts, this one rewrites **how she acts** from how the user actually responded to her. **Storage:** a single
  `self_notes` MetaStore slot (no table) — `Companion.update_self_notes()` **rewrites the whole list wholesale**,
  so revising, adding and dropping notes all happen in one call with no dedupe pass or fuzzy match. Capped at
  `SELFNOTES_MAX_CHARS` (400). Driven by **`SelfNotesJob`** (idle + `SELFNOTES_COOLDOWN` 30min, internal).
  **Injected** via `build_system(..., self_notes=...)`, threaded through chat + reach-out + follow-up.
  **Visible** in `GET /status` + a "Lessons learned" panel card. **Two prompt bugs found and fixed live:**
  (1) notes came back in *first person with quotes* ("Push back when I ask a question") — ambiguous about who "I"
  is, fixed with an explicit person rule → **0/8 slips**; (2) tightening that rule pushed the PASS instruction off
  the end and barren-window PASS collapsed **3/3 → 1/3** (she manufactured filler lessons), fixed by moving a
  *discriminating* rule to the very end ("a conversation only teaches you something when they reacted to HOW you
  talked to them; ordinary small talk teaches you nothing") — the recency lesson again. **Voice guard:** these
  notes are addressed *to* her, so a correct one never names her; when `BOT_NAME` appears the model has written
  the note to the **user** instead ("You get annoyed when Mari asks questions"), which would inject the lesson
  **backwards** — `update_self_notes` detects that and drops the pass rather than teaching her the inverse.
  `tests/test_self_notes.py` (**52 offline cases**). **Live-verified:** perfect discrimination —
  **9/9 barren PASS, 9/9 real lessons caught, 0/9 voice slips** across 6 scenarios; correctly *revised* a note the
  conversation disproved (2/2); and injecting notes **did not** regress personality (**8/8 one-sentence, 0/8
  q-end**, vs. a 7/8 and 1/8 baseline).
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
python tests/test_notifier.py           # offline phone-push (Bark) via httpx MockTransport (6 cases)
python tests/test_durability.py         # offline hard-kill recovery (4 cases)
python tests/test_emotion.py            # offline mood logic, fake classifier (7 cases)
python tests/test_llm_retry.py          # offline LLM transient-retry logic (6 cases)
python tests/test_tick.py               # tick scheduler + all jobs (reach-out/reflection/persona/sleep/drive-drift/follow-up) + energy-sleep + familiarity (42 cases)
python tests/test_drives.py             # offline drive + energy dynamics: rise/relax/mood-modulation/contact/discharge/deplete/restore/persist (15 cases)
python tests/test_tools.py              # offline tool framework: registry + stream loop + reminisce (20 cases)
python scripts/tool_probe.py            # LIVE: native tool-calling reliability (probe, per-case pass rate)
python scripts/tool_smoke.py            # LIVE: tools end-to-end through the real persona (time + reminisce + no-tool)
python scripts/tool_eval.py             # LIVE: 30-scenario tool-calling eval (time/reminisce/no-tool/tricky), per-category score
python scripts/bench_specdec.py         # LIVE: tok/s A/B for spec decoding — DONE: 45 base, +27% predictable / -50% creative → not used (§0)
python scripts/probe_reasoning_control.py  # LIVE (needs thinking ON): reasoning-cap knobs — DONE: all no-ops for qwen3.5 (§0)

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
  persona, sleep, status panel + tabs → **`706b809` tool framework (pillar 4)** → the v2.2 arc (multi-drive,
  energy, follow-up, inspector, phone push) → **`55bcd77`** model-bake-off tooling + §0 → **`afe3552`** iOS/Safari
  UI fixes → **`cc410e8`** memory-extraction + personality fixes. Each feature its own commit.
- The **user has authorized autonomous commits** — commit worthwhile work on `main` without asking
  (see the `commit-without-asking` memory). **⚠️ Uncommitted as of 2026-07-19:** the **silent-turn feature** + the
  **time-tool over-use fix** + these HANDOFF/roadmap updates (`core/prompts.py`, `core/companion.py`, `web/app.py`,
  `web/static/index.html`, `HANDOFF.md`) — the user is testing silent-turns live before committing.
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

**Resolved 2026-07-19 (was here):** thinking-off **personality** issues (long replies, reflexive "what's on your
mind?" questions) are **fixed** via hard one-sentence / no-question rules + an end-of-prompt reminder (100%
one-sentence, 7% q-end); the follow-up **hallucination** ("found a spot to sit") is fixed by the follow-up rewrite;
the **memory-extraction miss** (a stated name dropped in a big consolidation window) is fixed (window 10 + stronger
prompt). The tool-calling / bounded-thinking tradeoff below is now **parked in §0** (waiting on LM Studio). The rest
still stand:

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

### 0. thinking depth + tool-calling reliability — PARKED 2026-07-19 (waiting on LM Studio)
Production is **thinking-OFF and stable on qwen3.5-9b.** After a full investigation this session the decision is to
**stay on qwen thinking-off and WAIT for LM Studio to expose a reasoning budget** rather than migrate serving or swap
the model. Resume when LM Studio ships budget control (watch bug-tracker **#1838 / #1974**); or, independently, do
tool-calling **prong A** below (no migration, still worthwhile).

**Bounded thinking — confirmed impossible in LM Studio today (probed 2026-07-19).**
- llama.cpp *added a real reasoning budget* in March 2026: `--reasoning-budget` (launch) + per-request
  **`thinking_budget_tokens`** — a sampler that injects the end-of-thought token once the budget is hit (the TIL
  `ThinkingTokenBudgetProcessor` technique, native), and it works on Qwen3-family GGUF. **But LM Studio does not
  surface it.** `scripts/probe_reasoning_control.py` (now also tests `thinking_budget_tokens` 64/256) with thinking
  ON: baseline **5212** reasoning chars; every knob — `reasoning_effort`, `reasoning_budget`, `max_thinking_tokens`,
  `thinking_budget`, **`thinking_budget_tokens`** — left reasoning ~2700–3700 chars and did **not** scale with the
  budget (64 and 256 both ~3500). All stripped. So bounded thinking needs **llama-server** (real cost: AMD Vulkan/ROCm
  build risk on the 9070XT + rewiring the `lms` sleep/unload in `model_manager.py` + a 2nd server for nomic embeddings)
  or **vLLM** — both abandon LM Studio's conveniences. Not worth it now → **wait for #1838/#1974**, which would give
  bounded thinking on the *current* stack with zero migration and no personality risk.
- Still-settled dead ends (don't relitigate): **spec decoding** net loss (see below); **reasoning_effort/budget API
  knobs** no-ops for qwen (bug #1990). Thinking stays template-global binary: OFF (~3–9s) / ON (unbounded 20–45s).

**Model alternatives explored 2026-07-19 — qwen stays; tuned-Gemma is the fallback.**
- **qwen3.5-9b is current-gen, not old.** It's the **March 2026 Qwen 3.5 small series** (0.8/2/4/9B) and beats
  gpt-oss-120b on several benches; **gpt-oss (Aug 2025) is the OLDER model.** No "newness" reason to switch.
- **Three-model bake-off** via new `scripts/model_tryout.py` (per-model 11-question run: answer + reasoning-chars +
  wall-time, reasoning split from the answer) + `scripts/bakeoff_personality.py` (bounce / q-end / sameness). Outputs
  in `bakeoff/gpt-oss-20b_results.md`, `bakeoff/gemma-4-12b-qat_results.md` (+ `_v1_liveprompt` backup):
  - **gpt-oss-20b — REJECTED on personality.** Fast (3.6s low / 6.5s medium), the `reasoning_effort` dial *genuinely
    works* (bat_ball WRONG→OK low→medium), hallucination-disciplined. BUT **88% q-end** (interrogates — the exact
    thing that killed qwen3-8b), em-dashes everywhere, occasional harmony-format `500`s. Not Mari.
  - **Gemma 4 12B QAT — viable challenger.** `google/gemma-4-12b-qat` (Q4_0, 6.66 GiB). **0% bounce / low q-end**
    (non-interrogative like qwen), **0 dashes**, reasoning **2–4× less circular than qwen** (~1300–2000 chars,
    *completes* answers, ~13s vs qwen 20–45s — validates "qwen thinks in circles"). Two initial flaws (over-refused
    casual Qs like jet-lag/story; "That's the worst" sympathy reflex ×5) **were fixed** with a Gemma-specific persona
    variant **`bakeoff/gemma_persona.txt`** (loaded via the new **`PERSONA_FILE`** env override in both scripts — live
    prompt untouched): narrowed task-refusal to real deliverables + added a "vary stock reactions" rule → jet-lag/story/
    bat-ball now engage, cover-letter still refused, sympathy varied; q-end stayed 11%. The "no body" honesty was
    **kept** (user's call — it's true). **Only open gap: SPEED ~13s** (thinking-on; efficient but not fast). **NOT yet
    tested: Gemma thinking-OFF** (needs the same template edit as qwen) — the deciding data point if switching is ever
    revisited (goal: ~5s + still coherent). Thinking on/off is template-controlled (chat_template_kwargs/reasoning_effort
    are no-ops, same as qwen).

**Tool-calling reliability — plan still valid, unstarted.** `tool_eval.py` noisy (22/23/25, run ×3). **TIME
inconsistency = temperature** (0.8 too hot for routing; 0.2 → TIME 7/8) → **prong A: per-call temperature** in
`infrastructure/llm_client.py` (cool the tool-decision pass ~0.2–0.3, keep the answer at 0.8). No serving change,
independent of the thinking question — the cheapest real win, do it anytime. **REMINISCE (~3–4/8) = reasoning** (the
model *affirms* instead of *retrieving*) — the case bounded thinking would help, so it's blocked on the same wait.

**`core/prompts.py` still has the UNCOMMITTED `build_tools_note` change** (left on purpose; strengthened reminisce
rule + few-shot calibration; scored 22–23 vs 25 baseline — decide revert/rework; **not** committed this session).

**Production config right now:** qwen3.5-9b, thinking OFF (template `{% set enable_thinking = false %}`),
`NO_THINK=true`, spec decoding OFF, `TEMPERATURE=0.8`. ⚠️ **The template was flipped to `= true` for the 2026-07-19
probe; flip it back to `= false` in LM Studio to restore thinking-off** (verify: `probe_reasoning_control.py` baseline
reads ~0).

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
- **★ Energy / body cycles — BUILT (2026-07-18).** `DriveManager` gained an `energy` reserve [0,1] that
  **depletes while awake and restores while asleep** (rates in `core/drives.py`, per elapsed wall-time so
  they're `TICK_INTERVAL`-independent; persisted + shown as a bar in the status panel). `SleepJob` now fires on
  **low energy + a brief idle gap** (she's tired — `ENERGY_SLEEP_THRESHOLD` / `ENERGY_SLEEP_MIN_IDLE`) *in
  addition* to the long-idle VRAM-freeing trigger, so sleep has an internal logic rather than only a 30-min
  timer. The `min_idle` gap means she never nods off mid-conversation (the busy guard zeroes idle during a turn).
  Offline-tested (deplete/restore/clamp/persist + the sleep triggers); default rates model a rough day/night and
  are tunable. **Still deferred: self-wake** (see below).
- **★ Nightly / scheduled deep consolidation** — an "end of day" job that summarizes the day into an episodic
  day-summary (great reminisce fuel), distinct from the per-window consolidation.
- **Self-waking (deferred, next after A3)** — waking to reach out when rested (energy high) + missing the user
  (connection high). Energy (the gate that stops her waking exhausted) now exists; what's still missing is
  **time-of-day / do-not-disturb gating** so she doesn't wake at 3am. Build that first, then a `WakeJob`
  (web-registered like reach-out) that checks energy + connection + a cooldown. `wake()` is already a public seam.

**★ Completing the Generative Agents cognitive loop (idea pass 2026-07-19).** Arc A gave Mari drives + energy; the
triad from Park et al.'s "Smallville" agents is **memory (§B) + reflection (her journal) + planning — the one she's
missing.** Adding planning makes the inner life *goal-directed*:
- **★ Intentions / a private agenda — BUILT + COMPLETE 2026-07-19** (schema v8 + `IntentionStore` + `IntentionJob`;
  full design in §2). Proactivity is goal-directed: reach-out anchors on the longest-waiting intention, chat and
  follow-up carry the agenda *softly*, stale items expire, and `form_intentions` clears whatever the conversation
  actually covered. **Possible later:** relevance-ranked pick instead of FIFO, and surfacing fulfilled intentions as
  a "we covered this" history in the UI.
- **★ Learned operating-notes (reflection that changes behavior) — BUILT + COMPLETE 2026-07-19** (`self_notes`
  meta slot + `SelfNotesJob` + a voice guard; full design in §2). The self-improving closed loop is closed: she now
  distills her own "ease off the questions" lessons from how the user reacts, and they steer every user-facing
  prompt — the lessons we'd been hand-tuning, learned unsupervised. **Possible later:** let a note *decay* if the
  behavior it describes stops recurring, and show which note changed a given reply in the prompt-inspector tab (§8-C).
- **Relational continuity — NEW (small).** Let persisted mood/drives shape her *engagement level* (reply length,
  silent turns, warmth) across sessions, not just word choice within one: a little cool after a rough exchange,
  warming back over the next chats. Uses the silent-turn seam built 2026-07-19.

### B. Memory depth
- **Memory salience / forgetting curve** — importance+recency weighting; trivial facts fade, often-recalled
  ones strengthen. Keeps the store lean, makes recall feel human. *This is exactly the Generative Agents
  memory-stream ranking — recency × **importance** × relevance; score importance at consolidation, weight recall by it.*
- **Memory confidence + confirmation** — track uncertainty and occasionally double-check a shaky fact
  ("was it Kate or Katelyn?"), self-correcting the lifecycle.
- **Fact-validity windows (temporal memory)** — track *when* each fact was true; the principled version of
  the recency/conflict tie-breaker we punted on (Zep/Graphiti style).
- **"On this day" recall** — time-anchored callbacks via reminisce; pairs with proactivity. Extend to
  **spontaneous recall**: she surfaces a salient memory *unprompted* ("this reminds me of when you said…"), not
  only time-anchored callbacks (needs the importance score above).
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
- **Surface her inner life** — expose a light "current preoccupation" in the status panel from her latest
  thought/intention ("mulling over your jacket project", "curious about shooters"), so she reads as continuously
  present, not idle-until-pinged. (Open-LLM-VTuber surfaces the AI's unspoken thoughts.)
- **Prompt inspector (dev/transparency, user-requested 2026-07-19)** — a per-turn view (like the existing response
  inspector's `<details>`) that shows the **exact assembled prompt** sent to the model — system persona + tools
  note + core/recalled memories + mood + the history window + the user turn — formatted readable. Great for
  debugging persona/memory/tool injection. Server would need to return the built `system` (+ message list) on the
  `done` frame (or a `GET /prompt/{turn}`); the UI adds a second collapsible tab next to "inspect".

### D. Reach beyond the tab
- ~~**Push notifications** (e.g. Bark)~~ **DONE + LIVE-VERIFIED (2026-07-19):** a **reach-out** now pushes to the
  user's iPhone via a **self-hosted Bark server** (on their Raspberry Pi `alex-pi:8090`, also a tailnet node) →
  APNs. Wired as a config-gated `PhonePush` (`infrastructure/notifier.py`) on the reach-out path:
  `_notify_reachout` in `web/app.py` broadcasts to open tabs AND POSTs `{title, body, url, group, icon}` to
  `NOTIFY_URL` (no-op when unset; failures logged + swallowed so they can't break a reach-out; a trailing slash
  on the url is stripped — Bark routes `/<key>/` differently). Tapping the notification opens the web UI over
  **Tailscale** (`NOTIFY_UI_URL` deep-link). `NOTIFY_ICON` adds a custom notification image — served from
  `web/static/` (now mounted at `/static`) so the phone fetches it over Tailscale; iOS caches it by URL.
  Follow-ups stay in-tab (not pushed). `POST /admin/test_notify` fires a one-off push on demand. iOS forces APNs
  (the one unavoidable online hop; Bark can E2E-encrypt so Apple sees only ciphertext). Offline-tested via httpx
  MockTransport (6 cases). Confirmed end-to-end on the user's phone, icon included. Groundwork for a genuinely
  *useful* self-wake later. **Gotcha for next time:** the Bark device key must be issued *by* the self-hosted
  server (register the iOS app against `alex-pi:8090`); a key from the public `api.day.app` returns
  "failed to get device token from database".
- **Multi-channel presence** — Mari over WhatsApp / Telegram / Discord instead of only the local web UI.

### E. More tools (framework ready; paused by the user)
The pillar-4 framework is built and hot-swappable — each of these is "register a `Tool`, nothing else changes."
- **Web search**, **mood-based Navidrome/Subsonic playlist creator**, **reminder tool**, **curiosity-driven
  search** (self-initiated search + reflection during a tick), and exposing **`rewrite_self`** as a real tool
  (unifies the persona rewrite with the tool framework).
- **Autonomy framing (2026-07-19):** the **curiosity search** is where **autotelic** behavior lands — *she* picks
  what to chase from her own interests/intentions (§A), looks it up via web search between chats, and brings it
  back ("got curious about Deadlock — Valve's hero shooter, right?"). The **playlist** becomes a real autonomy
  signal when a tick job decides to build one *unprompted* ("made you something from the stuff you've been into"),
  not only on request — Mari's version of AIRI's autonomous *acts*.

### F. Whimsical / far-future
- **Dreams** — during sleep, generate one memory-recombining "dream" she might mention on waking. One per wake.
- **Voice** (STT/TTS) — the last delivery layer; later, **acoustic emotion perception** (hearing *how* you say
  something, not just the words).
- **Embodiment** — Live2D / VRM avatar, wearable sensors.

**Recommended order from here (updated 2026-07-19):**
1. **§0 is PARKED** — bounded thinking waits on LM Studio (#1838/#1974); personality (1-sentence/no-question),
   memory-extraction fix, and silent-turns all landed this session. No action there until LM Studio ships it.
2. ~~**Complete the Generative Agents loop in arc A**~~ — **DONE 2026-07-19**: ★ intentions/planning *and* learned
   self-notes both shipped, so memory + reflection + planning are all in place. What still feeds this arc:
   **A3 nightly consolidation** (day-summaries = intention + reminisce fuel) and **relational continuity**.
3. **§B memory upgrade** — importance-weighted recall (the GA ranking) + spontaneous recall. The deepest item.
4. **World-reaching tools (§E)** — web search unlocks the **autotelic curiosity loop**; the Navidrome playlist
   becomes an autonomous act. Both need the tool built first.
5. **Slot in the C near-freebies** (presence signal, time-of-day awareness, surfaced inner life) alongside — cheap.
Practical tuning in real use: drive thresholds/rates, `FOLLOWUP_CHANCE`, and **silent-turn frequency** (add the
mood/low-effort gating if she goes quiet too often).
