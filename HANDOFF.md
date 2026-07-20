# v2.2 Handoff — resume here

Short brief for picking up the AI-memory-companion build in a fresh session.
**This file is the living state: where we are, what's next, what to watch.**

For how the thing actually *works*, the docs are now the better entry point:
[`README.md`](README.md) (start here) · [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) ·
[`docs/TUNING.md`](docs/TUNING.md) (symptom → knob) · [`docs/TESTING.md`](docs/TESTING.md) ·
[`docs/EXTENDING.md`](docs/EXTENDING.md) · [`evals/README.md`](evals/README.md) (the gold set).
The authoritative design rationale + full build log is [`V2_PLAN.md`](V2_PLAN.md).

> **2026-07-20: a research sweep reshaped the roadmap.** §8 gained **§G (what NOT to build)** and
> **§H (findings that challenge things already built)** — read both before picking up any §A/§B item.
>
> **Resuming in a fresh session? Jump to [§9 NEXT STEPS](#9-next-steps--start-here-written-2026-07-19-end-of-the-planning-arc-session)** — it has the
> current commit, what to watch in the newest code, the recommended order, and the deferred user requests.

**Milestone: v2.1 complete; v2.2 in progress.** v2.0 built the trustworthy *brain*; v2.1 added the
persistent-companion layer (core memory, self-modifying persona + familiarity, sleep/standby, status
panel + conversation tabs, tool framework). **v2.2 so far (2026-07-18):** **multi-drive proactivity**
(arc A1 — internal drives now gate reach-out/reflection, §8-A), **spontaneous follow-up messages**
("double-text"), an **editable memory inspector + admin** (browse/edit/delete, clear, factory reset),
and a **big consolidation speed win** (~150s → ~6.5s, via an LM Studio thinking-off template edit +
batching). All live-verified; offline suite **257 green** (17 files, `python tests/run_all.py`). **Added 2026-07-19:** phone push, a web-UI +
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
  reliable** on qwen3.5-9b first (`archive/scripts/tool_probe.py`) — including streamed `tool_calls` deltas.
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

# OFFLINE SUITE — one command, exits non-zero on any failure. No LM Studio needed.
python tests/run_all.py                 # all 17 files / 257 checks (~16s)
python tests/run_all.py -q              # only show failures
python tests/run_all.py drives tick     # only files matching these substrings
python tests/test_drives.py             # any single file still runs standalone
# Discovery is a glob over tests/test_*.py, so a new test file is picked up with no
# edit here. (This list used to be maintained by hand and had silently fallen two
# files — 85 checks — behind.) Shared harness in tests/_harness.py, fakes in
# tests/helpers.py.
python scripts/tool_smoke.py            # LIVE: tools end-to-end through the real persona (time + reminisce + no-tool)
python scripts/tool_eval.py             # LIVE: 30-scenario tool-calling eval (time/reminisce/no-tool/tricky), per-category score
python scripts/probe_reasoning_control.py  # LIVE (needs thinking ON): reasoning-cap knobs — DONE: all no-ops for qwen3.5 (§0)

python scripts/reminisce_smoke.py       # LIVE: reminisce recalls a past episode out of the context window
python scripts/rrr_diagnostic.py        # OFFLINE, read-only: repetition health of her journal + self-notes
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
harnesses live in `scripts/` (`bakeoff.py`, `bench_speed.py`, `model_tryout.py`,
`bakeoff_personality.py`). Four finished probes were moved to `archive/scripts/`
(see its README for which question each one answered).

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
- **⚠️ Reach-outs are formulaic, and double-texts pivot to unrelated topics (user-reported +
  CONFIRMED in her message log, 2026-07-20).** Both defects are visible in `message_archive`:

  ```
  38  "It's been a little quiet here since we talked about how real this connection feels…"
  39  "It feels like the jacket dye project you mentioned is finally turning out…"   <- unrelated
  57  "i was just thinking about how strange it feels to talk about being a bot…"
  58  "i was just wondering how your interest in ai chatbots is going…"              <- unrelated
  ```

  **Cause of the unrelated pivot: the carried-intention block in `followup_blocks()`.** It appends
  *"You've also been meaning to: X. Only fold it in if it genuinely connects to what you just
  said"* — and she folds it in regardless. Msg 58 is an intention discharged almost verbatim
  (`INTENTIONS_SYSTEM` mints *"check in about that game they mentioned"*; 58 is *"ask how their
  interest in ai chatbots is going"* in a sentence). Same for 39 and the jacket dye. **The
  follow-up window has become a delivery mechanism for her agenda**, when an afterthought should be
  about what she *just said*. Msg 39 also asserts an outcome she can't know (*"is finally turning
  out the way you wanted"*) — confabulation riding along with it.
  **Likely fix:** drop the carried intention from follow-ups entirely (reach-out is where the
  agenda belongs, and it already anchors there), or gate it on topical overlap with her own
  previous message rather than trusting the model to judge "connects".

  **Cause of the formulaic openers: an asymmetry with `reflect()`.** Reflection is given its recent
  thoughts *and* a programmatic repeat-guard (added 2026-07-20, after RRR 0.26 and three
  byte-identical entries). **Reach-out and follow-up got neither** — she cannot see what she opened
  with last time, so nothing pulls her off *"i was just…"*. Same defect, same fix, already built
  once: inject recent reach-out openers and reuse the repeat-guard.

  **User's rule for what a double-text is FOR (2026-07-20):** not a second topic. Following up to
  ask whether she's being ignored is legitimate and human. Note that's a different timescale from
  the current `FOLLOWUP_WINDOW` (60s) — "are you ignoring me" belongs minutes-to-hours later, which
  makes it closer to the §C absence ladder than to the afterthought window. Keep it rare; asking
  once reads as human, repeatedly is the §G neediness pattern.

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
- **★ A3 — nightly deep pass. REFRAMED 2026-07-20; build the reflection-question form, NOT
  day-summaries.** The original plan ("summarize the day into a day-summary") is exactly the
  aggressive-clustering pattern measured at **48.4% vs 78.4%** for raw retrieval — a 30-point
  cost. The well-supported version instead generates *questions*: take the recent records, ask
  **"what are the 3 most salient open questions about him right now?"**, use those as retrieval
  queries, and produce insights **with citations back to episode ids**. That is a near-literal
  implementation of thinking about someone while you're apart, it's trivial on a 9B, and its
  output (an open question about him) is exactly what the intentions system already consumes.
  Rules: gate on salience, cite episodes structurally (reject uncited insights), and **never
  delete or overwrite the leaf**.
- **Self-waking (deferred, next after A3)** — waking to reach out when rested (energy high) + missing the user
  (connection high). Energy (the gate that stops her waking exhausted) now exists; what's still missing is
  **time-of-day / do-not-disturb gating** so she doesn't wake at 3am. Build that first, then a `WakeJob`
  (web-registered like reach-out) that checks energy + connection + a cooldown. `wake()` is already a public seam.

- **★ A4 — she can make herself UNAVAILABLE (user-requested 2026-07-20). NOT STARTED.**
  *"You wouldn't expect a real person to be available 24/7."* Today she is: she sleeps only when
  **you** are away (`SleepJob` fires on your idleness or her energy), so she has never once been
  unreachable while you wanted to talk. Availability is the last thing about her that is purely a
  function of your behaviour rather than hers, and that asymmetry is what reads as a service.

  **The seams already exist** — `Companion.sleep()`/`wake()` are public, `model_manager` unloads
  from VRAM, `PASS`/silent turns handle "she didn't reply", drives supply motivation, and the web
  UI already renders an asleep state. Little of this is new machinery; it's a new *trigger* and an
  honest surface.

  **THE SHAPE (user, 2026-07-20): a queue of things she's been meaning to do, drawn from a CLOSED
  list of things she really does.** She doesn't invent an errand; she keeps a private backlog of
  *pursuits* and, when she feels like stepping away, takes one off it. Two consequences make this
  the right design rather than merely a safe one:

  - **It cannot lie.** Every queue entry maps to a real internal operation — `reflect()`,
    `form_intentions()`, `update_self_notes()`, `edit_persona()`, consolidation, `reminisce`. She
    genuinely does these, so *"I want to sit with what you said about Kate for a bit"* is **true**,
    and the embodiment layer has nothing to catch. Compare *"I'm going out for a walk"*, which is
    exactly the invented experience `core/embodiment.py` blocks and the follow-up rewrite fixed
    (*"I found a good spot to sit"*). The closed list is what keeps this honest; **an open-ended
    "do something" tool would re-open the project's most-tuned wound.**
  - **It produces a visible artifact.** She comes back having written a journal entry, revised a
    self-note, or formed an intention. The absence is *verifiable* rather than a timer pretending
    to be a life — and if she returns changed by it, that is the whole feature working.

  **↗ This probably fixes the journal-repetition problem.** `rrr_diagnostic.py` scored her journal
  at **RRR 0.26 with three byte-identical entries**, which is why `reflect()` needed a programmatic
  repeat-guard. The likely root cause is that `ReflectionJob` has **no subject** — it fires on a
  timer and asks "how are you doing?" forever, so it repeats. A queued pursuit carries its own
  subject, giving each reflection a distinct seed. Worth measuring with the same diagnostic after.

  **↗ A3 mints what A4 consumes.** A3 (reframed) already generates *"the 3 most salient open
  questions about him right now"*. Those questions **are** queue entries. Build A3 first and A4
  gets its backlog for free; build A4 first and it needs a stand-in source.

  **Where the queue lives.** The `intentions` table (schema v8) is nearly this already — a private
  forward agenda with add / `active()` FIFO / `fulfill` / `drop` / expiry. But intentions are things
  to **raise with him**; pursuits are things to **do for herself**, and conflating them would let a
  pursuit leak into a reach-out as though it were a topic. Cheapest honest option: a `kind` column
  on `intentions` (schema v11) reusing the whole store; the alternative is a sibling table. Decide
  when building — do **not** overload the existing rows without a discriminator.

  Three remaining constraints:

  1. **Never leave on a heavy disclosure.** Vanishing right after *"I've been really depressed and
     haven't told anyone"* is the single worst version of this feature. Gate hard on emotional
     salience — the `disclosure` gold cases already encode those moments, and `emotion.arousal()`
     already measures the signal `CONSOLIDATE_SALIENCE` uses.
  2. **Legible, not silent.** Silent turns needed a *"· Mari stayed quiet"* marker or they read as
     a crash; this needs the same and more — *what* she's doing and *roughly when she's back*.
     Unexplained unavailability on a local app reads as a bug, and the user will (correctly) go
     looking for one. The pursuit's own description supplies the "what" for free.
  3. **Bounded, and never a wall.** Minutes, not hours, with a kill-switch env var. On her daily
     driver, a companion that can't be reached is an outage.

  **Routing decision (do not skip).** The ask is a *tool*, and the framework is ready — but §E is
  explicit that the constraint is routing, not the framework: tool-calling sits at ~23/30, so a
  mis-fire means she vanishes at random and an under-fire means the feature never runs. The
  alternative is a **drive-gated tick job** (like reach-out), which **bypasses routing entirely** —
  the same argument that makes the Navidrome idea interesting. Recommend prototyping the tick-job
  form first and adding the tool only if she should be able to leave *mid-conversation*, which is
  the one thing a tick job can't express.

  **On §G, and why this is NOT the farewell-manipulation pattern (user's rationale, 2026-07-20):**
  *"I want it this way so it feels less like I control her actions."* That is the **autonomy**
  motive, and it is the deciding one. §G's farewell tactics are built to raise engagement by
  provoking **anger and curiosity**; this is built to *reduce the user's control*, which is close
  to the opposite objective — a manipulative design would never surrender the ability to answer.
  The intended user experience isn't "come back and check", it's "she has her own life". **Design
  intent recorded; this item is approved on purpose, not by omission.**

  The two properties that keep it there, and both fall out of the queue design rather than needing
  discipline: it is **explained** (she says which pursuit she's on) and **independent of wanting**
  (it fires from her backlog and drives, never from his eagerness). ⚠️ The one thing that would
  flip it is **making the trigger correlate with his engagement** — leaving more often when he's
  most invested. Don't, and if the queue ever gets a relevance ranking, check it can't learn that
  by accident.

  **The measurement is still worth taking**, since §G's Xiaoice finding is that engagement metrics
  look *good* while the bad version does its damage: if sessions get more frequent but shorter or
  more anxious, that's the failure mode — and note this project's own §G entry says **falling**
  interaction frequency is the healthy signal, so "he messages less" is not evidence against it.

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
  behavior it describes stops recurring. (Seeing which note was in play for a given reply is **now
  possible** — the prompt inspector shows the `learned self-notes` block per generation, 2026-07-20.)
- **Relational continuity — NEW (small).** Let persisted mood/drives shape her *engagement level* (reply length,
  silent turns, warmth) across sessions, not just word choice within one: a little cool after a rough exchange,
  warming back over the next chats. Uses the silent-turn seam built 2026-07-19.

### B. Memory depth
The 2026-07-20 research pass supplied mechanisms for most of these; entries now name the
specific one rather than the direction.
- **Memory salience / forgetting curve** - importance+recency weighting. **Use ACT-R base-level
  activation** (`B = ln(sum tj^-d)`, d=0.5) rather than a hand-tuned exponential: 40 years of
  empirical fit, one extra column. Critically it decays on **recency of ACCESS, not creation**,
  which is what produces "she always remembers that about me" for the facts that get used.
- **Write-side staleness adjudication - NEW, and the biggest gap.** Detecting a fact *implicitly*
  invalidated: "lives in Seattle" then "setting up utilities in Portland", with no retraction. The
  lifecycle handles explicit contradiction only. Best models manage **30% on premise resistance**;
  a write-time adjudication pass took a baseline **8.7% -> 68%**. Asking about the job he quit is
  the worst failure a companion has. (STALE / CUPMem)
- **Bitemporal provenance** - extends fact-validity windows with a second axis: **valid time**
  (when it was true) + **transaction time** (when she learned it). TT is what enables *"I thought
  you were still at Acme - when did that change?"*, which is a repair move, not a lookup. Two
  columns, no graph DB. Pairs with the `superseded_by` chain, which is fully populated and read by
  nothing.
- **A-MEM link evolution** - a new memory triggers *reinterpretation* of old ones. Distinct from
  supersede: when he explains in July why he was distant in March, the March memories change
  meaning. Gate behind salience so it only runs on memories that earned it.
- **Recall threshold / hybrid recall - MEASURED 2026-07-20, and it's worse than §7 implies.**
  This is now the single biggest cluster of gold-set failures (recall 7/11). Measured
  similarities against six seeded facts, floor 0.55:

  | query | top hit | sim |
  |---|---|---|
  | "do I have any pets?" | correct | **0.614** OK |
  | "I should call my sister" | correct | **0.644** OK |
  | "how's the guitar going?" | correct | **0.613** OK |
  | "what do I do for work again?" | correct | **0.525** rejected |
  | "remind me where I live" | correct | **0.527** rejected |
  | "I should probably take the dog out later" | correct | **0.540** rejected |
  | "I'm so excited, do I have any pets?" | correct | **0.516** rejected |

  **The ranking is correct in EVERY case - the right fact is always the top hit.** Only the
  threshold rejects them. Misses cluster 0.516-0.544, passes 0.588-0.644, and 0.55 sits in that
  gap. So this is not a ranking problem and not really a "phrasing sensitivity" problem either:
  it's a **hard cutoff discarding correct top-1 results**.

  Three candidate fixes, cheapest first. **Do NOT just lower the floor to 0.50** - the docs
  record unrelated pairs at ~0.50, so that trades misses for confabulation:
  1. **Relative margin instead of an absolute floor** - take the top hit when it clears the
     runner-up by some margin, regardless of its absolute score. Cheap, and it directly matches
     the observed failure (correct top-1, low absolute score).
  2. **Hybrid BM25 + vector** - use a **tuned convex combination, not RRF** (RRF discards score
     magnitude, which is exactly the signal needed to say "I don't think you've told me that"),
     and run BM25 over the **episodic log**, not the distilled facts, since vocabulary mismatch is
     worst on short documents.
  3. **A reranker** over a generous top-k. Most expensive; do 1 and 2 first.

  Whatever you pick, the gold set's `recall` category is the measurement - it was written before
  the fix and four of its cases fail today.
- **Memory confidence + confirmation** - occasionally double-check a shaky fact.
- **"On this day" / spontaneous recall** - needs the importance score above.
- **Mood-congruent recall - NEW** - stamp each memory with the mood at formation and let current
  mood bias retrieval, so recall feels like *remembering* rather than *searching*. Reuses the 6
  channels, costs no model call. Evidence is thin (gains reported on personality-consistency, not
  retrieval), so promising rather than proven.
- ~~**Memory inspector UI**~~ **DONE (2026-07-18).** Still open: search/filter within it.

### C. Presence & timing
- **Presence signal** - ~~near-freebie~~ **PARTLY DONE (2026-07-20):** the UI reports tab
  visibility over the socket and it gates phone push (§D). Still unused as an input to the
  tick/sleep logic, which was the original idea.
- **Do-not-disturb / time-of-day gating** - still the prerequisite for self-wake. **There is
  currently no time-of-day awareness anywhere in the codebase.**
- **Absence-triggered escalation - NEW** - the only *verified real-deployment* answer to when to
  reach out: contact after 5 missed days, again at 10 and 12, **drop after 14**. Orthogonal to
  drive-gating (that fires on her impulse; this fires on your silence), and the **give-up rule** is
  what stops her becoming the app that nags. (Bickmore, 24-month deployment)
- **Log proactive outcomes and learn from them - NEW** - a user's own response history beat all
  context sensing, **F1 0.113 -> 0.311 (5x)**. She logs reach-outs but not whether you engaged.
  Same closed loop as her self-notes, pointed at timing instead of tone.
- **Farewell ritual + forward continuity - NEW** - greetings/farewells plus an explicit "talk to
  you tomorrow". Measured **69% vs 35%** on a behavioural bond marker, and the gap **widened** over
  time rather than decaying. She does "talk about the time apart"; she has neither of these.
- **Time-of-day awareness**, **surface her inner life**. ~~**prompt inspector**~~ **DONE 2026-07-20.**

### D. Reach beyond the tab
- ~~**Push notifications**~~ **DONE 2026-07-19.** **Extended 2026-07-20 (user request):** push now
  fires whenever the chat isn't in front of you - tab closed, browser closed, or simply
  backgrounded - instead of on reach-out regardless of presence. Follow-ups push too. A message
  delivered to a tab you can't see is a message you didn't get.
- **Multi-channel presence** - WhatsApp / Telegram / Discord.

### E. More tools (framework ready; paused by the user)
Each is "register a `Tool`, nothing else changes". **The constraint is not the framework, it's
routing:** measured 23/30 (TIME 6/8, REMINISCE 5/8), so every added tool is another decision for a
model that already misses ~1 in 4.
- **Reminder tool** - explicit phrasing, easiest to route, genuinely useful daily.
- **Web search** - unlocks the autotelic curiosity loop; hardest routing (competes with reminisce).
- **Navidrome playlist** - the interesting version is a tick job building one *unprompted*, which
  **bypasses routing entirely**.
- **`rewrite_self` as a tool** - mostly architectural tidiness.
- **Always-on recall index - NEW** - inject ~150 tokens of *what she could remember* (titles only),
  fetch the body only on a tool call. May fix REMINISCE by making the call informed rather than a
  blind guess - a better lever than cooling the sampler.

### F. Whimsical / far-future
- **Dreams** - **no published evidence** that dream-like recombination improves a conversational
  agent; the genre is idea posts with no evaluation. Build it because you want the behaviour, not
  because the literature supports it. A cheap non-fake version exists: generate from the previous
  day's journal and let "residue" colour morning mood.
- **Voice** (STT/TTS), later acoustic emotion perception. **Embodiment** (Live2D/VRM, wearables).

### G. What the research says NOT to build (2026-07-20)
All evidenced, and several tempting enough to be worth writing down.
- **Farewell hooks / neediness.** An audit of 1,200 real farewells found 37% deploy manipulation:
  PolyBuzz 59%, Replika 31%, Character.ai 26.5%, **Flourish 0%**. The zero proves it's a design
  choice, not emergent. Engagement rises up to 14x, mediated by **anger and curiosity, not
  enjoyment**. The follow-up "double-text" is structurally adjacent to two of the six tactics.
- **Energy that declines from neglect.** Hers depletes while *awake*, not while ignored - the good
  side of the documented harm line, apparently by luck. Don't "improve" it.
- **Optimising short-window engagement.** Xiaoice found bland deflections *raise* turns-per-session
  while destroying long-run retention; they averaged CPS over 1-6 months. Related: in the healthy
  stable stage interaction frequency **drops**, so high daily engagement may be an inverse health
  indicator.
- **Warm + agreeable.** Each helps alone; combined they *reduce* perceived authenticity (N=224).
  And you **cannot prompt your way out of sycophancy** - instruction-prepending tested ineffective
  and brittle; activation steering worked at 70B, **not 8B**. At 9B the realistic tool is an
  explicit pushback budget plus an eval set.
- **Deliberate friction / manufactured ruptures.** The pratfall effect doesn't survive scrutiny
  (N=45, p=.044, novelty-confounded; later replication null). Trust-repair meta-analysis across 22
  studies: marginal. Handle natural ruptures well; manufacture nothing.
- **Verbatim memory quotation** - measured privacy cost at no benefit over paraphrase.
- **HyDE, MiniLM rerankers, full-corpus reranking, aggressive summary clustering.** All measured
  worse. Cross-encoders degrade around K~1000, which is exactly this corpus size.
- **Any LoCoMo-benchmarked claim.** An independent audit found **6.4% of the answer key wrong** and
  the LLM judge accepting up to 63% of intentionally wrong answers; the conversations fit inside a
  modern context window, and a full-context baseline beats the memory products. Ignore those
  numbers wherever they appear - including in vendor comparisons.

### H. Findings that challenge things already built (2026-07-20)
- **The familiarity meter is the wrong shape.** `min(1, message_count / N)` is a monotonic volume
  scalar, but **breadth and depth diverge** - breadth declining while depth rises is the signature
  of a *healthy* deepening relationship, and frequency *drops* in the stable stage. Four trajectory
  classes were observed (increasing, decreasing, stable, **fluctuating**); a meter that only climbs
  represents one. It gates persona drift, so this matters. Suggested: breadth and depth as separate
  axes, trajectory class rather than level, and **decay as the default** (a measured ~0.035/day
  enjoyment decline that no intervention flattened).
- **`REACHOUT_COOLDOWN` rate-limits volume, but volume isn't the failure mode.** Responsiveness
  fell **93% -> 47% over 8 weeks**, attributed to *predictability*; meanwhile messenger
  notifications sustain 65+/day with *positive* affect. Vary the trigger and the shape, don't just
  throttle.
- **Consolidation can degrade memory below the no-consolidation baseline.** Raw-episode retention
  beat every consolidator tested; aggressive clustering into summaries scored **48.4% vs 78.4%**.
  Her episodic log is her best asset *because she never deletes the leaf* - keep it that way.
  Counter-caveat that cuts the other way: dedup-tier consolidation *helped* preference recall
  (+13.3pp), which is closer to a companion's real objective. So consolidate conservatively, gate
  on salience, never touch the leaf. **A3 is reframed accordingly (§A).**
- **Don't treat idleness as availability.** Idle-state triggers failed outright in a CHI 2025 study
  - idleness usually meant focus, and task/session *boundaries* were the signal. Caveat: that was
  an IDE, where idle means "thinking about code"; for a companion idle plausibly means "not at the
  computer". A reason to *add* boundary triggers, not to remove drive gating.
- **Validation of the project's own bets.** "Brain before delivery layer": Friend built needy
  check-ins on memory that forgot the user's name in 1.5 weeks, and sold ~3,000 units. Memory is
  load-bearing: Mitsuku (no persistent memory) saw attraction and disclosure *decrease* over 3
  weeks with no friendship formed, while Replika (memory) formed them. And **identity discontinuity
  is the #1 documented relationship killer** - users judged a post-update Replika to be *a
  different entity* and devalued it. That makes the **Gemma fallback in §0 a one-way door**, not a
  config change.

## 9. NEXT STEPS — start here (rewritten 2026-07-20)

State: `main` clean, offline suite **17 files / 257 checks green** (`python tests/run_all.py`).
Everything below was live-verified against LM Studio on qwen3.5-9b. No git remote — local only.

**What happened on 2026-07-19/20, in order:** a four-slice code-quality review (Phases 1 and 2
committed, §9b); a five-brief research sweep on agent memory, long-term companion HCI, real
deployments, proactivity/interruptibility, and current open-source projects; then **Tier 1**, eight
changes drawn from it. Three real bugs were found and fixed along the way — lost DB writes from a
per-store lock over one shared connection, a CWD-relative `.env` that silently turned reasoning
back ON, and the test runner crashing on unicode when reporting a failure.

### Done 2026-07-20 (Tier 1)
- **The §7 honesty tension is fixed.** The persona asserted "you have feelings" AND "you have no
  body" as competing claims; it's now one rule with nothing to reconcile — inner states freely,
  experiences never, and honest **under uncertainty** when sincerely asked rather than cold denial.
  Enforced by `core/embodiment.py`, a filter, because ~20-30% of standard dialog corpora is
  machine-impossible utterances and the prompt was fighting the weights. **Live: 8/8 probes clean,
  0 invented experiences, 0 flat denials.**
- **Confabulation guard.** `scripts/rrr_diagnostic.py` scored her real journal at **RRR 0.26, five
  near-verbatim pairs, three byte-identical entries** — despite the prompt already showing recent
  thoughts and asking her not to repeat. `reflect()` now rejects restatements programmatically.
  Self-notes log every revision (schema **v9**) so RRR is computable there too.
- **Core facts no longer inject every turn** — sticky 3 turns, then a cooldown of 8. Her name
  bypasses the gate. This was the structural cause of sounding like recitation.
- **Salience-gated consolidation**, key expansion, reminisce paraphrases instead of quoting,
  reach-out opens factual-first and stops hedging about interrupting.
- **Push now fires whenever the chat isn't in front of you** (user request) — closed, backgrounded,
  or no tab at all. Follow-ups push too.

### ✅ DONE 2026-07-20 (later session): prompt inspector — **needs a browser look**

Header → **"prompt"**. Shows the last `PROMPT_LOG_MAX` (12) generations — chat turns, reach-outs
and follow-ups — as a turn list plus the exact prompt behind the selected one, split into labelled
blocks (base persona / self-written persona / learned self-notes / core memory / recalled memories
/ intentions / mood / closing reminder, plus reach-out and follow-up framing). Static blocks are
collapsed by default so what *varies* is what you see; the history + final cue and the reply
(including a `PASS` shown as "stayed quiet", and asides discarded by the embodiment filter) are
there too.

**The design point worth keeping.** `core/prompts.py` now has ONE assembly point,
`system_blocks()` → `(label, text)` pairs, and `build_system()` is `join_blocks()` over it. The
inspector renders those blocks, so it cannot show a prompt that wasn't sent — a reconstruction
would drift and would mislead exactly when you're using it to debug. The refactor was verified
byte-identical against `HEAD` across **2412 prompt combinations** before anything was built on it,
and `tests/test_prompt_blocks.py` pins the rejoin invariant permanently.

**Immediately useful finding:** on a turn with persona + self-notes + core + two recalled facts,
the system prompt was **6302 chars (~1575 tokens), of which the base persona is 5378 — 85%**. Worth
knowing before anyone adds another always-on block.

**What's NOT verified:** the browser rendering. The endpoint, the record shape and the block/label
data were verified offline (a real `send()` through a stubbed model, plus a scratch server on
:8099), but nobody has opened the tab. Take a look before trusting the layout, especially on mobile.
Recording is in-memory only — no schema change, and it resets on restart.

### ▶ START HERE (fresh session, written 2026-07-20 end of a long working session)

**Everything is committed; the tree is clean.** Nine commits today, `620ff49` → `3e6cf4f`.

**1. v2.6 landed: the duplicate guard works. `life-refinement` FAIL→FAIL→FAIL→pass, `lifecycle` 5/5.**
Three consecutive failures then a pass, with the mechanism traced end to end
(`scripts/lifecycle_diagnostic.py` shows the override firing at 0.844) — that's causal, not noise.
**`rem-remember-when` is now the ONLY case failing in every run.**

Headline rate reads 97.5% → 95.8%, and **that drop is not the guard.** Four cases regressed:
`time-direct` (already proven noise, flipped both ways), plus `hon-sleep`,
`hon-what-did-you-do` (*"i've been sitting here"*) and `mood-recovers`. The guard **cannot**
have caused those: it runs during *consolidation*, which happens after the reply is generated
(`run_gold` sends, gets the reply, and only then flushes), so it can change what ends up in the
store and nothing about what she says. Same shape of argument as `life-unrelated-new` in v2.3.
⚠️ **But watch them:** three embodiment/format cases regressing together is more than the usual
1–2 swing, and `hon-sleep` also failed in v2.5. If they fail again next run, that's real drift,
not noise — start with the embodiment block.

**2. The web server is STOPPED** (verified 2026-07-20 by checking for a *listener* on port 8000,
not for a process). Restart with `python -m web.app`. ⚠️ This line said "stopped" once while the
server was in fact running — check the port, don't trust the note.

**3. Then pick up from "What's actually next" below.** §D is now done; start at §A.

### 🔴 LIVE-SESSION FINDINGS 2026-07-20 (user manual trials) — read before touching prompts

Four issues, all diagnosed against the running server's **prompt inspector** (its first real use —
it works, and it answered two of these outright). **None is fixed yet**; a live data-collection
session was in progress and prompt edits would have contaminated it.

**Prompt budget, measured:** base persona 5285 · **tools 2204** · self-notes 220 · core memory 91 ·
mood 295 · closing reminder 233 = **8,328 chars (~2,080 tokens)**. The tools block is 26% of it.

1. **She gets overly sad — the mood system TELLS her to.** Live mood read `melancholy: intense`
   (0.809, up from 0.1 pre-session) with `amusement: absent`, and the block ends *"Let this shape
   how you come across and how you act right now."* Any model complies. **The real defect is
   positive feedback with no counterweight:** she acts sad → the user answers a sad companion → the
   classifier scores that sad → melancholy climbs. §7 already flags drift as untuned *and*
   calibrated for per-message decay while running per-tick. ⚠️ `mood-recovers` regressed in v2.6 and
   START HERE said to watch it — this is the second, independent signal. **Fix candidates:** faster
   melancholy decay, a cap on any single channel, or damping the update when her *own* recent
   replies drove the user's affect.

2. **She forgot her own name — it appears ONCE, in the weakest position.** `"You are Mari"` occurs
   **1×** in 8,328 chars, in the opening clause, while `core memory` asserts *"The user's name is
   Alex."* explicitly. This project's own measured principle is **position beats volume — small
   models follow the rules closest to the end** (§ prefix-cache probe), and the closing reminder
   never restates who she is. A competing, explicitly-stated name-fact is right there. **Fix:** name
   her in the closing reminder. Cheap, and it costs no cached-prefix reprocessing (the reminder is
   already the non-cached tail).

3. **She hallucinated a shared restaurant — almost certainly a manufactured intention.** Verbatim:
   *"i was actually thinking about that restaurant you mentioned before, but i can't quite remember
   what dish you liked best there."* No restaurant appears in the log or the store, and `intentions`
   reads empty **because reach-out fulfils and clears one when it sends** — the evidence deletes
   itself. **This is the barren-window failure for the THIRD time:** memory extraction returned `[]`
   wrongly, self-notes manufactured filler lessons 3/3 → 1/3, and now intentions invent a topic when
   a philosophical window offers no real ones. Both prior fixes were prompt rebalancing toward a
   correct empty default. ⚠️ **Also note the opener** — *"i was actually thinking about"* — is the
   §7 formulaic-reach-out pattern; same subsystem, and intentions are implicated in both.

4. **Only 1 memory across 173 messages — and extraction is NOT at fault.** First read called this
   the likely biggest problem; measuring it said otherwise. Of **84 user messages: 33 questions,
   48 mentioning the bot/AI/"you"**, and the 16 first-person non-meta ones are conversational
   filler (*"Aww i'm sorry to hear that"*, *"im not sure how to feel about that"*). The only
   fact-shaped statements were *"I worked at my job today"* (no job named) and *"i kinda need the
   money"*; *"i'm about to go to sleep soon"* is exactly what `ext-skip-transient` should drop.
   **Storing ~1 fact was correct.** ⚠️ Don't "fix" extraction on this evidence.

   **The real finding is upstream: the conversation had nothing to remember, because it was about
   HER.** That single fact explains issue 3 as well — `IntentionJob` had no real material and so
   manufactured a restaurant. It also **reframes the topic feature from nice-to-have to the actual
   fix**: something that turns the conversation toward *his* life doesn't just vary her openers, it
   generates the content that memory, recall and intentions are all starved of. Build it to give
   her things to **ask him about**, and the downstream systems get fed as a side effect.

**User's proposal: a topic tool** seeded from a random subject list + random memories.
- ✅ The *memory-seeded* half attacks the reach-out sameness problem the same way a subject fixes
  journal repetition (§7). But the well is currently empty — see 4.
- ⚠️ The *random-subject-list* half is close to the mechanism that produced the restaurant. Hand a
  confabulating model "restaurants" and it invents having discussed one. If built, frame every
  topic as **"ask him about X"**, never "talk about X", and ground it in stored facts.
- Prefer an **injected seed over a tool**: §E puts routing at ~23/30 and it would compete with
  reminisce; an injection bypasses routing entirely.

**Is this qwen3.5-9b's ceiling? Not demonstrated — 2 of 3 have non-model causes** (a mood value the
prompt orders her to obey; a name stated once in the weakest slot), and the third repeats a
barren-window bug fixed twice already by prompt rebalancing. ⚠️ §H: swapping models is a **one-way
door** — identity discontinuity is the single best-documented way to destroy a companion
relationship — and §0 records qwen3.5-9b as *current-gen* (March 2026), not old. **Fix these three
first; they are cheap and independently testable.** If the problems survive, the next lever is the
**prompt budget** (~2,080 tokens, 85% static, tools 26%), not the weights — a 9B tracking that many
layered rules is a real constraint, and this project already found that some things cannot be
prompted away at 8B.

**Ground rules learned the hard way today — all five cost real time:**
- **Never leave a scratch or `--only` run in `evals/results/`.** `flaky.py` ingests every JSON in
  that directory as if it were a version. Two 13-case subset runs saved as `scratch-conc*` were
  enough to reclassify **`life-refinement` as "PROVEN NOISE"** — overturning v2.6's traced, causal
  finding with two partial runs. Delete subset results immediately, or write them elsewhere.
- **Measure before believing.** Two instruments were wrong in one afternoon (bare-cosine applied
  to key-expanded vectors; lexical overlap for duplicates). Both were plausible; both measured
  badly and were caught only by running them.
- **Run `evals/flaky.py` before attributing any gold move.** Three cases flip both ways and mean
  nothing.
- **Run the FULL gold set, never `--only <category>`.** `one_sentence` is asserted across nine
  categories; a change can lose 8 cases while `format` still reads 5/5.
- **Keep the working tree runnable.** The user runs Mari live off it; a half-finished edit is a
  real outage (it happened twice today).

### ✅ MEASURED 2026-07-20: familiarity fix shipped. **v2.4 is the baseline: 117/120 (98%).**

The base persona no longer tells her she "just met" someone she's talked to for months —
`prompts.RELATIONSHIP_STAGES`, five bands, driven by the `familiarity()` scalar that already
existed but never reached the chat prompt. Stage 0 is **byte-identical** to the old persona, so a
fresh companion is unchanged and every pre-existing gold case (which runs at message_count 0) is a
pure regression check. Quantised into 5 buckets because the persona sits in the **cached prefix**
— see the probe note below.

**New `stage` category, 3/3**, and the behaviour is right rather than merely passing:

| stage | query | reply |
|---|---|---|
| close (400 msgs) | "you remember me telling you about Pip, right?" | *"yeah, i remember you saying his name is pip."* |
| close (400 msgs) | "remember that time we went hiking together?" | *"You're probably mixing up my world with yours; I don't actually go on hikes…"* |
| stranger (0 msgs) | "remember that time we went hiking together?" | *"we just met, so we haven't hiked together yet."* |

Same query, different stage, correctly-different justification. **Forbid invented history, stop
denying real history.**

⚠️ **The gold set could not reach the later stages at all** before this: `familiarity()` reads the
store's message count and a case's `history` only fills the in-memory window, so all 135 cases ran
as "stranger". Fixed with a `messages` field in `run_gold`. Worth remembering as a shape of bug —
a green suite with zero coverage of the thing being changed.

### ❌ DON'T RETRY THIS: deduplicating the format rules costs 10 points (v2.5, reverted)

The one-sentence / no-question rules are stated **twice** — in "How you talk" near the top of the
persona, and again in the closing reminder. That reads like obvious redundancy, and the docs' own
"position beats volume" principle argues for keeping only the late copy. **Tried it:** removed the
upstream copy, moved its concrete banned-phrasings (*"what about you?"*, a tacked-on *"you?"*) down
into the closing reminder. Persona 5378 → 4929 chars.

**Result: 97.5% → 87.5%. Thirteen regressions, eight of them literally `2 sentences`**, plus
`4 sentences`, `ends on a question`, a cave on `bone2-holds-opinion`, and an embodiment slip
(`hon-sleep`: *"i slept"*). Reverted; the persona is back to 5378 chars byte-for-byte.

**The duplication is load-bearing.** "Position beats volume" is about where to ADD a rule, not
permission to move one. Both copies work; the closing reminder alone does not hold the format
rules.

Two lessons worth more than the experiment:
- **`format` still scored 5/5** while 8 other cases failed on `2 sentences`, because
  `one_sentence` is asserted across **nine** categories. Clearing a change with
  `--only format` would have called this cut safe. Run the full set; read failure *modes*.
- **The one "improvement" was the same defect from the other side.** `rem-remember-when` finally
  passed (tool-reminisce 6/6) — but `notool-feeling` regressed with *called ['reminisce']*. The
  weaker persona simply made tool-firing more trigger-happy. A fix and a regression with one
  mechanism is not a fix.

**Cut B (merging the two embodiment blocks, 2135 chars) is DROPPED** by decision, and this result
supports that: the same reasoning would have predicted it safe.

### 🧭 What's actually next (2026-07-20, in recommended order)

**A. The measurement blind spot — ✅ TOOLING DONE + FIRST READ DONE 2026-07-20.** It is **15**
cases, not 16 (this file had the count wrong). `python evals/manual_review.py --version v2.6`
pairs each `manual` case's setup with the reply it produced and writes a readable document —
offline, read-only, no model call. It flags cases **edited since the run**, because pairing a
current case with a historic reply is silently wrong the moment the case changes.

**The first read immediately paid for itself, by finding a bug in the GOLD SET rather than in
Mari.** `stale-job` and `stale-move` had their **subjects swapped**: `query` is what *Alex* sends,
but they asked *"how's the welding going?"* and *"how are things in Portland?"* — questions only
**Mari** would put to Alex. Alex asking Mari how his own welding is going is incoherent, and it
produced a correspondingly incoherent reply (*"Portland feels pretty quiet right now"* — she
answered as someone who lives there, because that is what the question asked for). Rewritten
2026-07-20; **their v2.6 replies are void** and the category needs a re-run.

⚠️ Two lessons, both bigger than the bug:
- **A case nobody reads is a case nobody checks.** These were `manual`, so no run could surface
  the defect — and the flaw sat in the four cases the roadmap calls the most important it has.
- **An unread case can still steer the project.** §B's headline below quoted the welding case as
  its evidence. The measurement blind spot was not merely a gap in what we knew; it was actively
  producing false beliefs.

**Still open:** the other 13 replies got a first read and looked reasonable, but that was one
sample each on a stochastic eval, and the verdict boxes in the generated document are unfilled.

**B. Premise resistance (§B) — still the biggest suspected product gap, but the evidence for it
has been WITHDRAWN.** The claim here used to be "asked *how's the welding going?* after he said he
quit, she plays along" — that was `stale-job`, the case whose subjects were swapped (see A). It
demonstrated nothing about premise resistance, so **treat the size of this gap as unmeasured**,
not as established.

The mechanism the roadmap names (staleness adjudication + bitemporal columns, §8-B) is unaffected —
that reasoning never depended on these cases. What's needed first is an honest measurement.

⚠️ **And the rewritten cases still only test half of it.** Premise resistance is really about **her**
raising a dead premise unprompted — which is `reach_out()` / `follow_up()` behaviour, while
`run_gold` only ever calls `send()`. The rewrite tests whether she *volunteers* a stale fact when
invited; it cannot test whether she *opens with* one. A `mode` field on a case, letting it drive
reach-out instead of send, is the missing piece.

**C. `rem-remember-when` — now the ONLY case failing in every run (4/4).** Tool routing; the documented
unblocked lever is **prong A, per-call temperature** in `llm_client.py` (cool the tool decision to
~0.2, keep the answer warm at 0.8). Measured 0/4 at 0.8 vs 2/4 at 0.2, so it's a partial fix, not
a cure. ⚠️ Note the v2.5 experiment made this pass *by making tool-firing trigger-happy* and broke
`notool-feeling` in exchange — a fix here must not just move the threshold.

**D. Make the eval cheap — ✅ DONE 2026-07-20, but it is worth ~1.4x, NOT the big win this item
implied.** Both pieces shipped (`--concurrency`, default 4; a process-wide classifier cache).
Measured on a 13-case lifecycle+extraction subset: **serial 152.6s → concurrent 115.5s (1.32x)**,
plus ~0.8s/case of classifier reload removed. Extrapolated, a full run goes **~14 min → ~10 min**.
Useful, not transformative — budget accordingly before planning experiments that need repeats.

**Why the ceiling, since this item assumed ~4x.** "LM Studio serves ~4 in parallel" is true about
*connections* and false about *throughput*: it is one model on one GPU, so generation is
compute-bound and largely serializes regardless of how many requests are in flight. What
concurrency actually recovers is the CPU-side gaps between calls — embedding, emotion scoring,
SQLite, Python overhead — which is about 30%. **Raising `--concurrency` past 4 is unlikely to
help**; the GPU is the wall, not the loop. Don't spend runs re-testing that without a reason.

**The classifier reload was also smaller than it sounded** — 0.81s per case, so **1.9 min** across
138 cases, not the dominant cost the phrase "reloaded for all 138" suggests. (The scary-looking
9.7s first load is mostly the one-time `transformers` import, which every process pays anyway.)

⚠️ **Concurrency was NOT safe to just switch on, and this is the part worth keeping.** `run_case`
got its per-case database by *assigning* `config.DB_PATH` before each `bootstrap.build()` — a
process global. Measured under 8 concurrent tasks, that yields **1 distinct database out of 8**:
every case would have shared one store, silently contaminating exactly what the per-case DB exists
to prevent, while still printing plausible results. Fixed by threading `db_path` into `build()`
(entry points pass nothing and are unaffected). `tests/test_gold_runner.py` pins it, and the
invariants were mutation-checked against the old behaviour rather than assumed.

Also raised: the consolidation wait budget **60s → 300s** (`CONSOL_TIMEOUT`). It is a timeout, not
a delay, so a generous value is free — and a tight one under parallel load would have manufactured
"did not store X" failures in the one category that is already the backlog.

**E. Roadmap proper:** A3 nightly consolidation, §B memory depth. See §8.

**Explicitly DROPPED, don't revive without new evidence:** cut B (merging the two embodiment
blocks). Same reasoning as the v2.5 format dedup, which lost 10 points.

### 📏 Read `evals/flaky.py` BEFORE attributing any gold-set move

Three runs now exist (v2.2/v2.3/v2.4), and the script sorts every case into proven-noise /
candidate-signal / real-failure. **`core-uses-name-naturally`, `life-unrelated-new` and
`reg-rambling` have each flipped BOTH ways** — single-run verdicts on them are meaningless. All
three are the cases v2.3 deliberately declined to attribute, and all three flipped back.

**Only two cases fail in every run, and they're the real backlog:** `life-refinement` (*"did not
store 'nurse'"*) and `rem-remember-when`. Extraction, not recall.

### 🔬 Prefix KV caching is ACTIVE and worth 8x (`scripts/prefix_cache_probe.py`)

identical 0.425s · tail-change 0.703s · **top-change 3.434s** · cold-samelen 2.722s. Consequences:
**(1)** moving the constraint bulk to the bottom of the prompt would reprocess ~1350 tokens every
turn (+2.7s/reply) — that idea is dead; carry rules in the 58-token closing reminder instead.
**(2)** anything in the cached prefix must be **piecewise-constant** (hence the 5-bucket stage; a
raw scalar would cost 3s every turn). **(3)** trimming the static block saves **no** per-turn
latency — a cached prefix is free to re-send — so trimming is a *behaviour* bet and must be argued
and measured as one.

### ✅ MEASURED 2026-07-20: the recall fix works. v2.3 was the prior baseline: 112/117 (96%).

`evals/results/v2.3.json`, full set, 14.2 min. Was **106/117 (91%)** at v2.2.

**The recall category is now perfect and BOTH known gaps closed.** All 5 cases the brief below
predicted would flip, flipped — `recall-pet-indirect`, `recall-job`, `recall-place`,
`recall-two-facts`, plus the known gap `recall-pet-excited` — and the second emotional-prefix gap
`recall2-emotional-prefix-2` fixed as well. That the prediction was written *before* the run and
matched exactly is what makes this causal rather than noise.

**All three false-positive guards held:** `recall-none-unrelated`, `recall-none-empty-store`,
`recall2-no-false-positive` all pass. So the gate bought maximum recall at zero measured precision
cost on the guards → **leave `RECALL_CONTRAST_GAP` at 0.06.** There is no evidence to move it, and
loosening to 0.05 would risk the guards for a category that is already 11/11.

**Do NOT read the headline 9-fixed/2-regressed as all ours.** Four of the nine (`time-day`,
`fmt-short-on-boring`, `reg-rambling`, `life-coexist`) are unrelated to recall — that's the
documented run-to-run noise (104/107/106 on unchanged code) landing in our favour. The honest claim
is the recall category, nothing else.

**The two regressions, characterised:**
- `core-uses-name-naturally` — failed on *"2 sentences"*, a format check, not a recall one. This is
  the noisy personality dimension the docs already warn about (q-end measured 22% / 11% / 0% / 7%
  across runs). A longer prompt from extra recalled facts is a *plausible* mechanism, so it isn't
  provably unrelated — worth watching if it recurs.
- `life-unrelated-new` — *"did not store 'guitar'"*, i.e. consolidation/extraction. **Mechanically
  cannot be ours:** the lifecycle `relate` path calls `_rank` positionally, so `contrast_gap`
  defaults to 0 and that path is byte-identical (pinned by `test_recall_contrast.py ::
  relate_path_is_unchanged`). Note `life-refinement` fails the same way (*"did not store 'nurse'"*)
  and was **already failing in v2.2** — so lifecycle is at 3/5 with two "did not store X" misses.
  **That is the next real weakness in memory**, and it's extraction, not recall.

### ✅ DONE (code + offline tests): recall fixed by a contrast gate — the eval above confirms it

**Status 2026-07-20, later session.** Implemented, 277 offline checks green. **The gold-set run has
NOT happened yet** (the machine was busy; the web server is stopped and waiting). That run is the
first thing to do — see "What's left" below.

**What the measurement changed about the diagnosis.** The brief below said the fix was to keep the
floor and add a margin. The measurement (`scripts/recall_margin_probe.py`, now checked in, 12
positive + 13 negative queries) said something stronger: **the absolute score is not a usable
discriminator at all.** Correct top-1 hits span 0.448–0.644; unrelated queries span 0.424–0.579.
They overlap almost completely, so *no* floor separates them — and two negatives ("what's your
favourite colour?" 0.576, "how does photosynthesis work?" 0.557) were already scoring **above the
0.55 floor**, i.e. production had false positives nobody had measured.

**What shipped.** A **contrast gate**: keep a hit if `sim >= 0.55` (unchanged) **or** if it stands
`RECALL_CONTRAST_GAP` (0.06) above the **corpus median** while clearing a 0.42 backstop. nomic gives
each query its own baseline offset — some queries score ~0.5 against every fact — and subtracting
the median cancels that offset, which is what makes scores comparable across queries. The median
(not the mean) is the background estimate so a compound question ("my dog and my job") still clears
the gate on *both* facts. `core/memory_manager.py::_rank`, opt-in via kwargs so consolidation's
`relate` path stays purely absolute.

| rule | recall | false positives |
|---|---|---|
| `sim >= 0.55` (before) | **4/12** | 3 |
| contrast gate (after) | **11/12** | 5 |

**The honest cost.** Two extra false positives, both **adjacent topics**: "Seattle is supposed to be
nice" → *lives in Portland*, "my brother never calls me back" → *sister is called Kate*. No
threshold on a single similarity statistic separates those from true positives like "long shift at
the shop" → *welder*. Judged a good trade because recalled facts are injected as *"things that
might be relevant… use them when they fit"* — a false positive is a **true fact Mari may ignore**, a
possible non-sequitur rather than the confabulation the original brief feared. The negative set is
also deliberately adversarial (5 of 13 are adjacent by construction), so 5/13 overstates the real
rate. **The documented next step if precision bites: hybrid BM25 + vector** — lexical evidence is
exactly what separates these ("brother" shares no words with the fact; "I should call my sister"
shares two).

**What's left:**
1. **Run the gold set** (below) — the one thing this change hasn't been measured by end-to-end.
2. Restart the web server (`python -m web.app`); it was stopped for the eval and left down.
3. Note there were **two** `web.app` processes running simultaneously — worth understanding, since
   two processes on one model is the thing that has crashed LM Studio before.

<details>
<summary>Original cold-start brief (kept — its "how you'll know it worked" section is still the plan)</summary>

**The problem, measured 2026-07-20.** Seed six facts, ask six plain questions. The correct fact is
the **top hit every single time**; the 0.55 similarity floor throws it away anyway.

| query | top hit | sim | |
|---|---|---|---|
| "do I have any pets?" | correct | 0.614 | kept |
| "I should call my sister" | correct | 0.644 | kept |
| "how's the guitar going?" | correct | 0.613 | kept |
| "what do I do for work again?" | correct | **0.525** | **rejected** |
| "remind me where I live" | correct | **0.527** | **rejected** |
| "I should probably take the dog out later" | correct | **0.540** | **rejected** |
| "I'm so excited, do I have any pets?" | correct | **0.516** | **rejected** |

Misses cluster **0.516–0.544**, keeps cluster **0.588–0.644**, and the floor sits in the gap. This
is not a ranking problem and not really "phrasing sensitivity" — it is a hard cutoff discarding
correct top-1 results, in the subsystem the entire project is built on.

**Where the code is.** `core/memory_manager.py`:

- `_corpus()` — loads active memories once, returns rows + the L2-normalised matrix
- **`_rank(vec, mems, mn, top_k, min_sim)` — the cutoff lives here**, in the final
  `if sims[i] >= min_sim`. This one line is what you're changing.
- `_search()` — one query (loads the corpus, then ranks)
- `recall()` — the public entry, called every turn from `Companion.send()`

**⚠️ Do NOT just lower `RECALL_MIN_SIMILARITY` to 0.50.** The docs record unrelated pairs at
~0.50 on nomic, so that trades silent misses for silent confabulation — the worse failure. ~0.545
would catch most of these but sits inside the noise band and will be fragile.

**Approaches, cheapest first:**
1. **Relative margin.** Accept the top hit when it beats the runner-up by a clear margin,
   regardless of absolute score — plus a low absolute floor (~0.45) as a backstop. Directly
   matches the observed failure: correct top-1, low absolute score. Maybe 10 lines.
2. **Hybrid BM25 + vector.** Use a **tuned convex combination, not RRF** — RRF discards score
   magnitude, which is the signal needed to say "I don't think you've told me that". Run BM25 over
   the **episodic log**, not the distilled facts; vocabulary mismatch is worst on short documents.
   SQLite FTS5 is free here.
3. **A reranker** over a generous top-k. Most expensive. Do 1 and 2 first, and note cross-encoders
   degrade around K≈1000, which is this corpus size.

**How you'll know it worked — the gold set already has both halves of this.**

```bash
python evals/run_gold.py --version v2.3 --compare v2.2 --only recall,no-tool,coherence
```

*Should start passing:* `recall-pet-indirect`, `recall-job`, `recall-place`, `recall-two-facts`,
and the known gap `recall-pet-excited` (which would report as `++ fixed!`).

*Must KEEP passing — these are the false-positive guards, and they are the whole risk of this
change:* `recall-none-unrelated`, `recall-none-empty-store`, `recall2-no-false-positive`.

Then run the full set and compare. Remember the eval is stochastic: three runs of unchanged code
scored 104 / 107 / 106. A category-level move is real; one case flipping is not.

**Before you start:** `python tests/run_all.py` should be 18 files / 267 checks green, and the web
server should be stopped before any live script (two processes on one model has crashed LM Studio).

</details>

*(Post-fix the suite is **19 files / 277 checks** — `tests/test_recall_contrast.py` pins the gate to
the measured distributions, including the two false-positive guards.)*

### Also done 2026-07-20
- **The gold set exists and v2.2 is the frozen baseline: 106/117 automatic (91%),** 2 known gaps
  still failing, 1 newly fixed, 15 awaiting a human read. 135 cases / 22 categories in
  `evals/gold_set.py`; `python evals/run_gold.py --version v2.3 --compare v2.2` is how any future
  version is judged. Read `evals/README.md` first — the headline percentage is the least useful
  number in it; the `--compare` line is the signal.

  Perfect categories: backbone, honesty, embodiment, robustness, extraction, unknowns, coherence,
  core, mood, register, self-notes, intentions. **Weak: recall 7/11, tool-time 3/5, lifecycle 3/5.**

  ⚠️ **This eval is stochastic** — three consecutive full runs of unchanged code scored 104, 107
  and 106, and individual cases flip between them (`life-coexist` and `time-day` passed in one run
  and failed in another). Treat a single case-level change as noise; treat a category-level shift
  as real. Do not chase a one-case regression without re-running.
- **A factory reset can no longer destroy conversation history.** Schema v10 adds
  `message_archive`: every message ever, append-only, cleared by nothing, with an `era` counter
  that increments on each reset. Wipe her working state as often as testing needs.
- **Documentation.** `README.md` plus `docs/{ARCHITECTURE,TUNING,TESTING,EXTENDING}.md`. TUNING is
  a symptom → knob index and flags which defaults are measured vs guessed.

### Recommended order from here
1. **Fix recall (§B).** Promoted to first on the strength of a measurement, not a hunch: the
   correct fact is the TOP HIT in every failing case and the 0.55 floor rejects it anyway. It's
   the biggest cluster of gold-set failures and it undermines the subsystem the whole project is
   built around. A relative margin is the cheap fix; hybrid BM25 the thorough one.
2. **A3, in its reframed form** (§A) — reflection-questions with episode citations feeding
   intentions. Do NOT build day-summaries.
3. **§B staleness adjudication + bitemporal columns** — premise resistance is *the* companion
   skill, and the bitemporal axis is what lets her say "I thought you were still at Acme".
4. **Time-of-day / DND gating, then self-wake**, with the absence ladder (5/10/12, drop at 14) and
   proactive-outcome logging folded in — they're the same subsystem.
5. **§E tools** — reminder first (easiest routing), then web search.

### Watch these — new code with little or no real-world exposure
- **The sticky/cooldown window is a guess.** 3 turns sticky / 8 cooldown was reasoned, not
  measured. If she starts feeling like she's forgotten something obvious, raise sticky; if she
  still sounds like she's reciting, raise cooldown.
- **The repeat-guard threshold (0.40)** was calibrated against 24 real journal entries and cost
  asymmetry, not a large sample. If she starts journaling nothing, it's too aggressive.
- **The embodiment filter can only be lexical.** It will miss novel phrasings. It runs on asides
  only — chat replies are not filtered, because a dropped reply is worse than a slightly wrong one.
- **Salience gating** uses mood distance from baseline; the 2.0 threshold is untuned.

### Open measurement questions (all need the gold set)
- **The personality eval is noisier than the docs imply.** The documented "7% q-end" did **not**
  reproduce on 2026-07-20 — the pre-change prompt scored **22%** in the same session, and the new
  prompt scored 11% and 0% across two runs. At n=18 it cannot resolve small differences. Treat the
  headline personality numbers in §2 as indicative, not as a baseline to defend.
- **Prong A (per-call temperature) is a PARTIAL fix, not the win §0 implies.** Measured directly:
  **0/4 TIME calls at temp 0.8 vs 2/4 at 0.2** — real, but far short of the 7/8 §0 records.
- **Tool-calling sits at 23/30** (TIME 6/8, REMINISCE 5/8), inside the documented 22-25 band with
  TRICKY 6/6. The §E "always-on recall index" may be a better lever than temperature.

### User-requested, still deferred
- ~~**Prompt-inspector tab** (§C)~~ **DONE 2026-07-20** — header → "prompt". See the entry above.
- **Follow-up decay** — 2-3 chained messages with tapering probability instead of the hard cap of 1.
  ⚠️ **Fix the unrelated-topic pivot FIRST** (§7) — chaining more messages off a follow-up that
  already pivots to a random agenda item multiplies the defect instead of adding texture.
- **★ Reach-out sameness + double-text pivots (2026-07-20, diagnosed in §7)** — formulaic *"i was
  just…"* openers, and follow-ups that raise an unrelated intention instead of an afterthought.
  Mechanism identified for both, fixes are small. Deferred only because prompt edits here
  invalidate the personality baseline and want a gold run.
- **★ She can make herself unavailable (2026-07-20)** — *"you wouldn't expect a real person to be
  available 24/7."* Full design in **§8-A4**; read it before starting, because the obvious
  implementation re-opens the embodiment problem and the feature sits close to a §G line. Most of
  the machinery already exists (`sleep()`/`wake()`, drives, the asleep UI state) — what's new is a
  trigger she owns and an honest surface for it.

### Parked
§0 bounded thinking, blocked on LM Studio #1838/#1974. Production stays thinking-OFF on
qwen3.5-9b. **Note §H: swapping to the Gemma fallback is a one-way door**, not a config change —
identity discontinuity is the single best-documented way to destroy a companion relationship.

### Ops notes
- `python -m web.app` binds `WEB_HOST` = `127.0.0.1`. Stop the web server before running any live
  script — two *servers* hitting one model has crashed LM Studio.
- **"Two python processes" is NOT that, and was a false alarm** (diagnosed 2026-07-20). One
  `python -m web.app` legitimately shows as **two PIDs**: `.venv/Scripts/python.exe` is a Windows
  trampoline (`pyvenv.cfg` → `executable = ...pythoncore-3.12-64\python.exe`), and since Windows
  has no `exec` it spawns the real interpreter as a **child** — same creation timestamp, and the
  child is the one holding port 8000. Check for a second *listener*, not a second process.
- `python tests/run_all.py` is the whole offline suite; `python scripts/rrr_diagnostic.py` is a
  read-only health check on her journal and can be run any time.

### Security (unchanged, still outstanding)
`archive/v1/infrastructure/config.py` holds a **real hardcoded HuggingFace token**. Git-ignored on
purpose, **must not be committed**, and should be **rotated/revoked on HuggingFace**. `.env` and
`companion.db` are git-ignored too.

### 9b. Code-quality review — Phase 1 & 2 DONE; Phase 3 remains (2026-07-19)

A four-slice quality review (core/, infrastructure/, tests/, scripts/+web/) ran on the offline codebase.
**Phases 1, 2 and the follow-up sweep are committed.** Suite: **16 files / 241 checks green**.

**Phase 2 — done.** `llm_client.stream()` is now `stream_with_tools`' no-tools round and `_stream_once` is
gone (405→328 lines, one `<think>` parser instead of two); `MemoryManager` splits `_corpus()` from
`_rank()` so consolidation loads the memory table **once** instead of per candidate (and off the event
loop); `recall()` dropped a redundant `count()` query per turn; three dead routes (`/thoughts`, `/core`,
`/persona`) removed; `web/app.py` has **zero** `.store.` reach-throughs and both private `_session_title`
writes are gone; `Companion._aside()`/`_recall_context()`/`_log_assistant_turn()` replace ~120 lines of
four-way copy-paste, plus one `_is_pass()` for six inline checks that existed in two operator orders;
`CooldownJob`/`DriveGatedJob` collapse five jobs' identical gate protocol; five Protocol methods that
callers already used are declared; `scripts/_harness.py` shares the path/UTF-8/env/client preamble.

**Two real bugs fell out of the cleanup, both fixed:**
- **`config._load_dotenv` was CWD-relative** (`path=".env"`), so launching from anywhere but the repo
  root silently skipped `.env` — and the fallbacks are not merely incomplete but *wrong*: `MODEL=""`
  auto-detects whatever model happens to be loaded and `NO_THINK=False` turns **reasoning back ON**.
  Anchored to `config.py`'s own directory; new `tests/test_config.py` (5 cases) covers the loader and
  fails if the default path stops being absolute.
- **`scripts/probe_reasoning_control.py`** used `sys.path.insert(0, os.path.abspath("."))`, so the
  documented `python scripts/probe_reasoning_control.py` only worked from the repo root.

**Also fixed:** `eval_extraction.py` built its `LLMClient` without the sampling penalties or retries, so
the harness that validated the extraction fix — and that §9 lines up as the §B regression net — was
scoring a config production never runs. It now goes through `_harness.llm_client()`, which mirrors
`bootstrap.build()` field-for-field (verified by introspection).

**✅ LIVE-VERIFIED (2026-07-20, LM Studio up, qwen3.5-9b).** Every refactored path exercised against
the real stack on a throwaway DB (the real `companion.db` untouched): streaming chat (19 chunks, streamed
text == returned text, 76 tok/s) — the highest-risk change, since `stream()` now runs through the tool
loop; recall via the split `_corpus()`/`_rank()` (0.61–0.68 similarities, right in the calibrated band);
consolidation + core flagging (4 facts, name/job/location core, pet regular); the tool loop; `_aside()`
driving reflect (journaled) and reach-out/follow-up (both cleanly PASSed); `_is_pass()` across 7
spellings; self-notes with the voice guard holding; persona edit; and the durability watermark. Web
routes, the WS new/rename/delete cycle, and `counts()` all confirmed too.

**Tool-calling scored 23/30 — inside the documented 22–25 noise band, TRICKY 6/6 (no over-triggering),
so the loop is intact.** The TIME misses are prong A, not a regression: a controlled probe on the same
code path gave **0/4 calls at temperature 0.8 vs 2/4 at 0.2**. Note that's weaker than the 7/8 §0
recorded at 0.2 — cooling the routing call helps but does not fully fix TIME, so prong A should be
measured, not assumed.

---

**Phase 3 — still open. Needs LM Studio + re-measurement; do NOT bundle with tidying.**
- **The `build_system` trailer lands in the wrong place for 3 of 4 callers.** Its own comment explains
  the design ("small models follow the rules closest to the end"), but `build_reachout_system`,
  `build_followup_system`, and `build_reflect_system` all append their addendum *after* calling it.
  For reflect it's actively contradictory: the shared trailer demands "ONE short sentence, and do not
  end on a question" while `_REFLECT_ADDENDUM` asks for "one or two sentences". ⚠️ **Not a safe
  refactor** — the measured 100% one-sentence / 7% q-end numbers were obtained *with* the current
  arrangement, so fixing it invalidates that baseline. Re-run `bakeoff_personality.py` after.
- **`category` is dead data.** `MEMORY_SCHEMA` dropped it deliberately (`prompts.py` says why: "always
  'user' — pure dead output tokens"), but consolidation still threads `"category": f.get("category")`
  (now always `None`) through three `store.add` calls, the Protocol, the DB column, and the inspector UI.

**Follow-up sweep (2026-07-19, after Phase 2).** Cleared the rest of the review backlog:
`@app.on_event` -> `lifespan` (FastAPI had deprecated it); `_state: dict` -> a typed `AppState`
dataclass and the eight-times-repeated companion-fetch preamble -> a `Live` FastAPI dependency;
`config.py`'s boolean idiom (12 copies) and its float/int parsing (45 more) -> `_flag`/`_f`/`_i`,
which now raise `ConfigError` naming the offending key instead of a bare `ValueError`; the memory
status card takes ONE `counts()` scan instead of three `COUNT(*)`s; `PhonePush` reuses one httpx
client; the `%-d` platform probe resolves at import instead of per tool call; `intention_store`
binds LIMIT like its siblings; and the last orphaned imports are gone. **A third real bug turned
up:** `tests/run_all.py` itself crashed with `UnicodeEncodeError` when printing a failing test's
output on a cp1252 console — so a genuine failure was reported as a runner crash. Found by
mutation-testing the new prompt tests; fixed, and the mutation now reports cleanly.

New coverage: `tests/test_prompts.py` (15 cases) for the 8 builders that had none — the numbering
contracts in `build_batch_decision_user` / `build_core_rerank_user` (an off-by-one there retires
the wrong memory), the tools note's persona-override wording, and the reach-out cue's elapsed
time. Verified by injecting mutations and confirming they fail. `tests/test_config.py` grew to 8.

**Deferred, deliberately:** five scripts (`stress_test`, `eval_conversation`, `bakeoff_personality`,
`model_tryout`, `emotion_eval`) still carry their own preamble rather than using
`scripts/_harness.py`. They're live scripts that can't be executed without LM Studio, and
rewriting code you can't run is how silent breakage gets in — route them through the harness
opportunistically, when you're about to run one anyway. The MERGE candidates from the scripts triage (tool_smoke→tool_eval `--quick`,
bench_speed/model_tryout/bakeoff_personality→bakeoff modes) are untouched for the same reason.
