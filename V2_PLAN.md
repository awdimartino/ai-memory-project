# ai-memory-project — V2.0 Plan

**Status:** planning (no code yet). This document is the deliberate answer to the open questions in the V1 retrospective (`V1_RETROSPECTIVE.md`), grounded in the decisions made during v2 planning.

---

## 0. Locked decisions (from planning)

| Decision | Choice |
|---|---|
| LLM serving | **Fully local** (LM Studio / OpenAI-compatible), continuing v1's deliberate local-first stance |
| Interface | **Python-native web UI: FastAPI + WebSocket**, minimal frontend; voice-in/voice-out slots in later behind the same pipeline |
| First goal | A **solid, trustworthy base**: (1) realistic personality, (2) fast memory recall + saving, (3) a working tick loop |
| Pillar build order | **Memory lifecycle → Emotion persistence+eval → Proactivity → Tool framework** |
| What a tick does | **All four**: internal reflection, mood drift, proactive messages, self-initiated tool use |
| Persona identity | **Emergent, self-modifying prompt.** Simple starting persona ("be friendly, but a stranger"); the bot rewrites a dedicated section of its own prompt over time during idle ticks as it gets to know the user. Low priority vs. the basics. Scaffolding already exists in v1 (`prompt_builder` injects a `Your self described personality` slot). |
| Emotion | **Keep the v1 approach** (RoBERTa GoEmotions → 6 channels + decay). It's lightweight and low-overhead — important on limited hardware. Add persistence; run the classifier on **CPU** (AMD 9070XT can't use the v1 CUDA `device=0` path, and CPU keeps all 16GB VRAM free for LLMs). |
| Model choice | **Decide empirically via an early bake-off** across LM Studio models. Primary axis: **personality vs. speed.** |
| Tech continuity | **Open to rethinking anything** — each subsystem below flags its trade-off explicitly |
| Overriding priority | **Get the skeleton running end-to-end as fast as possible.** Minimal Phase 0, real chat against the real stack ASAP, then layer. |
| V2 finish line | **Bare-bones vertical slice + a rigorous model bake-off first.** Get chat + autonomic memory + emotion + basic tick + logging genuinely good, thoroughly test personality & performance across multiple models, *then* decide how far to scale. Front-load only the cheap seams; defer heavy/uncertain work. |
| Runtime | **Single async process** (FastAPI/asyncio); blocking work (RoBERTa/torch) offloaded to a threadpool; a **priority queue** arbitrates the one-model-at-a-time GPU. Brain reachable only via the `Companion` facade so a future brain-daemon split stays a contained change. |
| Capability model | **Three tiers by mechanism** (see §1.1): autonomic pipeline stages (recall/emotion), deliberate-internal via structured output (lifecycle/self-edit), deliberate-external via true tool-calling (music/web/reminders). Only the last is uncertain on local models — test it before betting on it. |

**Hardware constraint:** AMD Radeon 9070XT (16GB VRAM), not a high-end machine. Everything must fit a modest VRAM/compute budget — favors small models, CPU for the tiny emotion classifier, and on-demand model load/unload (see Sleep mode, §2.8).

**Guiding principle carried from v1 (validated under real use):** the companion is a *brain* (memory + emotion + conversation) with a *delivery layer* (proactivity + tools + voice) on top. The brain must be trustworthy before the delivery layer earns its keep.

---

## 1. Architecture — the "do it right from day one" lessons

These are v1 lessons that cost real rework the first time. They are non-negotiable starting conditions for v2, from the first commit.

- **Strict layering + one composition root.** `infrastructure/` (persistence, LLM client, embedder, emotion model) sits strictly below `core/` (domain managers, facade, chat loop, tick loop), wired in one place. This is what made v1's Postgres→SQLite swap a contained change — keep it.
- **Store vs. Manager separation, with a Store `Protocol` defined day one.** A Store does only persistence (queries in, plain records out); a Manager coordinates a Store with collaborators (embedder, LLM, emotion). The Protocol documents the swap seam *and* enables an in-memory fake for deterministic tests. V1 added this late — v2 starts with it.
- **A deliberate facade / entry-point API from day one.** No reach-through coupling (`loop.companion.manager.sub.method()`). Callers talk to the `Companion` facade only.
- **All prompt text lives in exactly one place.** One prompt module owns every string sent to any model — persona, extraction, reflection, tool-use. No inline prompt strings drifting into managers.
- **Scoped logging from commit one.** Scope DEBUG to our own package(s); pin httpx / huggingface_hub / openai client loggers to WARNING. Never `basicConfig(level=DEBUG)` globally. No `print()` debugging that has to be cleaned up later.
- **Secrets via `.env` / real secrets flow from the start.** No hardcoded tokens in config files (a v1 open issue).

### 1.1 Capability tiers: autonomic vs. deliberate (key scaling decision)

Not everything the companion does is a "tool." Capabilities split into **three tiers by mechanism**, mirroring how a brain works — and this split *de-risks the whole project*, because the core brain lives in the two reliable tiers.

| Tier | Mechanism | Examples | Local-model reliability |
|---|---|---|---|
| **1 — Autonomic (involuntary)** | Pipeline stages that run around every cognition step; the model never "chooses" them | **Recall** (embed context → vector search → inject into system prompt), **emotional reaction**, **memory formation** (backgrounded extraction) | N/A — deterministic code, no model choice |
| **2 — Deliberate-internal** | Model chooses, but expresses it by **filling a JSON schema** (schema-constrained decoding) | **Memory lifecycle** (new/update/supersede), **self-edit** of the persona section | **Reliable** locally (v1 proved schema-constrained output works and even suppresses `<think>`) |
| **3 — Deliberate-external** | Model emits a real **tool/function call** | **Navidrome/music**, **web search**, **reminders** | **Uncertain** — must be tested per model before we lean on it |

Consequences:
- **Recall is not a tool.** It happens automatically every turn, like memory surfacing unbidden. Memory works regardless of whether the chosen model can function-call.
- **Only Tier 3 gates on tool-calling reliability.** Flaky function-calling limits delivery-layer extras, never whether the companion remembers you or stays in character. This is why the bake-off scores tool-calling as a **separate** axis (§2.1 / Phase 0).
- This resolves the earlier "is everything a tool?" question: **no** — autonomic functions are the involuntary substrate, structured-output is the reliable middle, and true tools are reserved for external, deliberate actions.

### 1.2 Runtime & concurrency (decided: single process, brain extractable)

- **Physical constraint:** one GPU ⇒ **one model call at a time**, no matter the architecture. Everything else is about what wraps that serialization point.
- **Decision: single async process** (FastAPI + asyncio). LLM calls are async HTTP to LM Studio (never block the loop); the one blocking dependency (RoBERTa on CPU / torch) runs in a **threadpool executor**. The UI, presence pings, and WebSocket stay responsive during long generations.
- **Model-access arbiter = a priority queue, not a boolean lock.** Contention order: **user turn > proactive reply > background reflection.** This is the scalable replacement for v1's single skip-if-busy lock.
- **Sleep fits naturally:** the process stays alive and lightweight while models are unloaded, so the tick/heartbeat keeps running to decide when to wake. The brain sleeps; the heartbeat doesn't.
- **Kept-open future path:** the web layer talks to the brain **only through the `Companion` facade**, never reaching into managers. That single discipline makes a later "brain daemon + UI client" split a contained change — we pay nothing for it now but don't foreclose it.

### 1.3 Scaling seams to establish early (cheap now, expensive to retrofit)

Front-load these even in the bare-bones slice; skip the heavy stuff until testing informs it.

- **Internal event bus (pub/sub).** Subsystems subscribe to events (`UserMessageReceived`, `MemoryExtracted`, `MoodChanged`, `TickFired`, `WentToSleep`) instead of cross-calling. The status panel, logging, and future features become free subscribers. Nearly impossible to retrofit once many subsystems cross-call.
- **DB migrations from commit one.** Store-everything + evolving features ⇒ many schema changes. A simple versioned migration runner now.
- **Model/role registry (declarative).** `role → model → VRAM cost → residency policy`, not hardcoded names. Drives the §2.8 lifecycle manager and the bake-off config.
- **Tick as a pluggable job scheduler**, not a hardcoded sequence (v1 hardcoded decay→think→unprompted). New autonomy behaviors register as jobs.
- **Defer, don't build yet:** the full Tier-3 tool framework, any multi-process split, most of the §2.9 backlog.
- **Persist any pending/queued state to the DB, not just memory.** V1 lost in-progress classification batches on an ungraceful kill. Anything queued (unclassified messages, pending ticks) is durable.

---

## 2. Subsystem plans (with trade-offs flagged — everything is on the table)

### 2.1 Conversation & personality (core, first)

- **Model-per-role split (recommended, carried from v1):** a **non-reasoning instruct model** for casual chat (latency), a **reasoning / schema-constrained model** for structured brain calls (extraction, lifecycle decisions, reflection). V1 confirmed schema-constrained decoding suppresses `<think>` leakage, so reasoning models are fine there.
- **Message shape rule (hard-won in v1):** keep `[system, ...history, user]` with the last turn always `user`. Fold retrieved memories, mood, and internal thoughts into the **system** message — never inject extra assistant turns (local chat templates reject it).
- **Anti-hallucination persona (v1 lesson):** instruct *what to do instead* (deflect warmly, redirect to the user), not just *what not to do*; and explicitly tell it not to over-explain its AI nature. Test with **varied natural phrasing**, not one named example.
- **Emergent, self-modifying persona (decided).** The system prompt has two parts: a small **fixed core** ("be friendly with the user, but be a stranger" + the hard anti-hallucination/format rules) and a **bot-owned self-description section** the companion rewrites over time. It starts nearly empty and fills in as the bot gets to know the user. The v1 `prompt_builder` already reserves this slot (`Your self described personality: PLACEHOLDER`).
  - **When it changes:** only during idle ticks when the user is (seemingly) absent — never mid-conversation.
  - **Gated by familiarity** (see §2.9 backlog): a stranger shouldn't rewrite itself into a close friend on day one. The familiarity level bounds how far/fast the self-description may drift.
  - **Self-editing is a Tier-2 deliberate-internal action** (§1.1): the bot rewrites the slot via a **schema-constrained** call during reflection, not free-form tool-calling — so it works reliably on local models regardless of their function-calling ability.
  - **Priority: low** relative to the basics. The slot exists from day one, but the autonomous rewriting can land after the core is solid.

### 2.2 Memory: retrieval (core, first) + lifecycle (pillar 1)

Fast recall/saving is a core goal; lifecycle is the first pillar. These share the same store, so design together.

- **Retrieval perspective mismatch (top v1 open issue).** First-person query ("do I have pets?") embeds badly against third-person memory ("Alex has a dog"). **Recommendation:** use an embedding model with native **asymmetric query/document prefixes** (e.g. `query:` / `passage:` style — nomic-embed, e5, bge, Qwen3-embedding all support this) *and* normalize memory phrasing at write time. This is a stack change from v1's embedder — flagged because you're open to rethinking.
- **Thresholds are model-specific — measure, don't guess.** Any similarity cutoff must be calibrated empirically on real query/memory pairs for the chosen embedder (v1's 0.70 guess silently dropped real matches). Budget a live measurement pass.
- **Brute-force KNN is fine.** No ANN index at personal scale (hundreds–low-thousands). Don't over-engineer.
- **Lifecycle (pillar 1) — the biggest unfinished brain work.** On each extraction: retrieve related existing memories, and have the brain model classify the new candidate as **new / duplicate / update / contradiction(supersede)**. Superseding **soft-deletes** (mark inactive, keep history) rather than hard-deletes. This fixes both "no lifecycle" and "dedup is threshold-only, not contradiction-aware" in one mechanism.
- **Housekeeping is backgrounded + batched (v1 lesson).** Extraction/lifecycle runs off the response path (every N turns or async), so it never adds felt latency between turns. Pending batch state is persisted (see §1).
- **Storage:** SQLite + sqlite-vec is validated and stays the default. Only revisit if scope adds multi-device sync (single-file embedded DB wouldn't fit that).

### 2.3 Emotion (pillar 2 — keep, persist, evaluate)

- **Decision: keep the v1 approach.** Local RoBERTa GoEmotions classifier → 28 labels mapped into **6 mood channels** (irritation, warmth, amusement, melancholy, unease, interest), each with its own decay rate toward a baseline (`CHANNEL_MAP` / `DECAY_RATES` / `BASELINE_STATE` in v1's `emotion_manager`). Chosen because it's **lightweight and adds almost no overhead** — critical on limited hardware. This closes the v1 "RoBERTa vs LLM-mood" open question in favor of RoBERTa.
- **Run it on CPU.** V1 used `device=0` (CUDA); the 9070XT can't. The model is ~125M params — CPU inference per message is fine, and it keeps the full 16GB VRAM for the LLMs. (Optional later: DirectML/ROCm, but not needed.)
- **Two changes v2 commits to:** (a) **persist mood to the DB** so it survives restarts (v1 reset to baseline every launch); (b) run a **behavioral eval pass** — v1 never checked whether the channels actually move sensibly on real conversation. Re-confirm/tune `PULL_STRENGTH`, decay rates, and the 28→6 mapping against real transcripts.
- **Mood drift** (a tick behavior) lives here: decay toward baseline during idle ticks; the current mood also colors proactivity tone and reach-out probability (see §2.4).

### 2.4 Tick system / proactivity (core loop first, pillar 3 for outward behavior)

The tick loop is core (you named it a core aspect), but its *outward* proactive behavior is pillar 3 — build the loop early, enable reach-out later.

- **A tick does four things:** internal reflection (consolidate/update memory), mood drift, deciding whether to **proactively message** you, and optional **self-initiated tool use**.
- **Proactivity UX (v1 open question, now answered by the web UI choice):** the tick pushes proactive messages over the **WebSocket** to the web UI. This is the reason the REPL was the wrong surface and the web UI is right.
- **Rate/appropriateness guardrails:** proactive reach-out needs throttling and a "should I even?" gate so it's a companion, not a nag. Design this into the tick decision, not as an afterthought.
- **Durable tick state:** pending/scheduled ticks persist to the DB (no silent loss on kill).
- **V1 warning:** the v1 tick built prompts but never called the model or sent anything — proactivity was pure stub. V2's definition of done for this pillar is a real model call producing a real pushed message.

### 2.5 Tool framework — Tier-3 deliberate-external only (pillar 4, last)

Scope note: this framework is **only for Tier-3 external tools** (§1.1). Recall/emotion are Tier-1 pipeline stages; lifecycle/self-edit are Tier-2 structured-output — none of those go through here.

- **Modular, hot-swappable** tool registry: each tool is a self-contained plugin exposing a schema + handler; add/remove without touching core.
- **Gated on the bake-off's tool-calling score.** How ambitious this layer can be depends on how reliably the chosen model function-calls (tested in Phase 0). If local tool-calling is weak, fall back to a constrained menu / structured-output routing rather than open-ended function-calling.
- **Architecturally invasive (v1 flag):** it touches the LLM turn-handling loop and interacts with streaming. Decide the turn-loop shape early even though it's built last.
- **First tenants:** web search; **Navidrome playlist creator** (Subsonic API, Raspberry Pi) that builds playlists from your real library using taste stored in memory; **reminders**.
- Ties back to the tick: self-initiated tool use means the tick loop can invoke a tool and then tell you about it.

### 2.6 Web UI (delivery surface, built alongside the core)

- **FastAPI + WebSocket.** WebSocket is bidirectional so it carries both user messages and **proactive pushes** from the tick loop. Minimal server-rendered frontend (vanilla or HTMX) — no JS build step.
- **Design for voice later:** structure the pipeline as `input → (text | STT) → brain → (text | TTS) → output` so speech slots in behind the same core without a rewrite.

### 2.7 Conversation logging + reminisce (decided)

- **Store everything.** Full, verbatim conversation records are persisted in a **separate section of the DB** from the distilled memories. This gives two tiers:
  - **Episodic** — raw conversation logs (and the bot's own idle thoughts/journal), timestamped.
  - **Semantic** — the distilled, canonical facts that memory recall searches (§2.2).
- **Recall vs. reminisce are different reads.** Fast in-conversation recall hits the semantic tier (small, embedded, calibrated). The **reminisce tool** reads the episodic tier — reflecting on *past conversations* rather than isolated facts. This is the natural home for "on this day" callbacks and for the bot bringing up something you talked about weeks ago.
- **Cheap to build first, valuable later.** Logging everything costs almost nothing now and is impossible to backfill later — so start capturing full logs from Phase 1 even though the reminisce tool itself is a Phase 5 tenant.

### 2.8 Sleep / standby mode + model lifecycle (decided)

The single most hardware-relevant subsystem, and it doubles as the VRAM manager.

- **What it does:** unload the LLM(s) from VRAM/RAM so the machine is free for other work ("standby"), and reload on demand when you return or the bot wants to act.
- **Two triggers:** (a) **user-initiated** ("go to sleep" / closing the UI), and (b) **bot-initiated** — during a tick the companion may *decide* to sleep if nothing interesting is happening (ties to the "energy budget" idea, §2.9).
- **Mechanism:** LM Studio exposes model load/unload via its REST API and the `lms` CLI (JIT model loading). A small **model-lifecycle manager** in `infrastructure/` owns "which models are resident right now." The same manager lets us **juggle roles within 16GB** — e.g. don't keep the chat and brain models resident simultaneously if they don't fit; load the brain model only for the (backgrounded) extraction pass, then release it.
- **Wake behavior:** a proactive push, or the next user message, triggers reload. Expect the v1 **one-time cold-load delay** (tens of seconds) on wake — surface it in the UI as a "waking up…" state rather than looking hung.
- **Optional flourish:** generate one "dream" on the way into or out of sleep (see §2.9).

### 2.9 Idea backlog (brainstorm — not yet committed)

Tiered by fit-to-effort. None are on the critical path; pull them in as the core solidifies.

**High fit-to-effort (cheap, each unlocks something already wanted):**
- **Episodic/semantic split** — already promoted to §2.7.
- **Familiarity meter** — a slow, persistent scalar (stranger → acquaintance → friend) that **gates how far the self-modifying persona (§2.1) may drift.** The mechanism behind "be a stranger, warm up over time."
- **Status panel (web UI)** — live view of resident models, mood channels, memory count, last tick, sleep/energy state. Directly attacks v1's "invisible bugs" pain.
- **Presence signal** — the WebSocket already knows if the tab is focused / the user is typing, giving a real "is the user here?" input to the tick/sleep logic instead of guessing from elapsed time.
- **Private thought journal** — persist tick reflections as the bot's own diary; reminisce can read the bot's past thoughts, not just conversations.
- **Reminder tool** — bot sets and proactively surfaces reminders; genuinely useful day to day.
- **Self-editing as a tool** — the persona rewrite is a `rewrite_self` tool call (unifies §2.1 with §2.5).

**Medium:**
- **Memory salience / forgetting curve** — importance+recency weight; trivial memories fade, often-recalled ones strengthen. Keeps the store lean, makes recall feel human.
- **Memory confidence + confirmation** — bot tracks uncertainty and occasionally double-checks a shaky fact ("was it Kate or Katelyn?"); self-corrects the lifecycle.
- **Curiosity-driven search** — during a tick the bot follows something it got curious about (self-initiated web search) and brings it up later. Combines self-initiated tool use + web search + reflection.
- **Mood-based Navidrome playlists** — the playlist tool reads current emotion + taste memory ("you seemed down, made you something").
- **Memory inspector (web UI)** — browse/search/edit memories and view superseded history; supports debugging the lifecycle work.
- **"On this day" recall** — reminisce surfaces time-anchored callbacks; pairs with proactivity.

**Whimsical / low priority:**
- **Dreams** — during sleep, generate one memory-recombining "dream" the bot may mention on waking. One gen per wake.
- **Energy budget** — ticks deplete a small energy stat; low energy biases the bot toward choosing sleep, giving autonomous sleep an internal logic instead of a random roll.
- **Time-of-day awareness** — greets differently morning vs. late night; notices patterns ("up late again").

### 2.10 Testing (v1 had none — decide upfront)

- **Deterministic layer:** the in-memory fake Store (from the day-one Protocol) lets us unit-test manager logic — lifecycle decisions, retrieval ranking, mood math — with fixed inputs, no model.
- **Live smoke/eval layer:** a small committed set of scripted prompts run **manually against the real local stack, per subsystem** — promoted out of v1's throwaway scratch scripts into a real directory. This is where the non-deterministic model behavior gets checked.
- **The overriding v1 process lesson:** live-test each subsystem against the real model stack **early and per-subsystem**, not once at the end. Every load-bearing v1 bug (broken extraction schema, template crash, miscalibrated threshold, hallucination) was invisible to code review and mocked tests.

---

## 3. Phased build plan

**Overriding priority: get the skeleton talking end-to-end as fast as possible.** Phase 0 stays minimal; the goal is real chat against the real stack early, then layer. Each phase ends with a **live-test checkpoint against the real local stack** before moving on.

**Phase 0 — Skeleton + model bake-off.** Two parallel tracks:
- *Frame:* layering, composition root, Store `Protocol` + SQLite store + **migration runner**, facade, scoped logging, `.env`, single prompt module, **event bus**, **model/role registry**, single async runtime with the **priority-queue arbiter** (§1.2), FastAPI+WebSocket chat loop wired to LM Studio. Minimal — just a trustworthy frame that can hold a conversation.
- *Bake-off (rigorous — this is a headline deliverable):* score the LM Studio models that fit 16GB on a repeatable rubric, pick a **chat** model, a **brain** (structured-output) model — possibly the same one — and an **embedding** model. Record winners, quant, VRAM, and measured latencies as the baseline config.

  | Axis | What it measures |
  |---|---|
  | Personality | Holds character, no disclaimer-bot, no fabricated experiences — tested with **varied phrasing** (v1 lesson), not one example |
  | Speed | Tokens/sec + time-to-first-token **with the full persona system prompt loaded** |
  | Instruction-following | Respects format rules (no em-dashes, short replies, `[system, …history, user]` shape) |
  | Structured output | Tier-2 reliability — schema-constrained extraction / lifecycle |
  | Tool-calling | Tier-3 reliability — **separate** probe; gates how far the tool framework can go |
  | VRAM | Footprint at the quant that fits 16GB (informs residency policy) |

  **Chat/personality candidates currently loaded in LM Studio** (discovered 2026-07-17): `qwen3.5-9b`, `qwen3-8b`, `qwen/qwen3-14b`, `neona-12b-i1`, `qwen3-4b-rpg-roleplay-v2`, `gemma-3-4b-it`, `llama-3.2-3b-instruct` (+ 1b), `qwen2.5-0.5b-instruct`. **Embedding candidates:** `text-embedding-nomic-embed-text-v1.5`, `text-embedding-qwen3-embedding-0.6b`, `text-embedding-all-minilm-l6-v2`. (qwen3.x are reasoning models — the client already strips `<think>`.)

**Phase 0 progress (2026-07-17):** Built the **first chat slice** — REPL-first, per decision. Files: `config.py` (env-driven), `core/prompts.py` (the "friendly, but a stranger" seed persona as Mari), `core/companion.py` (facade: history + `[system, …history, user]`), `infrastructure/llm_client.py` (OpenAI-SDK stream + `<think>` strip + TTFT/tok-s timing, auto-detects loaded model), `main.py` (composition root + REPL with `/model` `/temp` `/reset` for live model comparison). Verified end-to-end against live LM Studio (llama-3.2-3b: ~134 tok/s). No DB / memory / emotion / tick yet — deliberately. Also removed the stale `config.py` entry from `.gitignore` (it no longer holds secrets; those live in the git-ignored `.env`).

**Bake-off + prompt iteration (2026-07-17):** Built `scripts/bakeoff.py` (quality+speed, 11 probes, `lms unload --all` between models so only one is resident) and `scripts/prompt_test.py` (fast focused-probe iteration, no unloading). Full transcripts in `bakeoff/results.md`. Findings: reasoning models are a poor fit for casual chat (qwen3.5-9b spent its whole token budget "thinking" and never answered; qwen3-8b/14b usable but carry think-latency). `qwen3-4b-rpg-roleplay-v2` leaks control tokens and fabricates a body — disqualified. The two biggest failures (snapping into assistant mode on a task request; fabricating physical activities/favorites) were **prompt problems, not model problems**. Iterated the seed prompt v1→v4 (give positive alternatives + example phrasings; explicit "no body/backstory/favorites"; never guess appearance; refuse tasks without offering tips). **Both issues now fixed** on `gemma-3-4b-it` (fast chat pick) and `qwen3-8b` (brain pick); `llama-3.2-3b` is a weaker follower (still leaks task-help + mild AI-disclaimers). **Chat-role lean: `gemma-3-4b-it`** (best personality-per-token + fast, ~130 tok/s), though it dodges hard reasoning (bat/ball) — that's what the reasoning brain model is for. **Open nit:** models occasionally still emit an em-dash despite the rule; cheapest fix is to strip em/en-dashes from output in code rather than fight it in the prompt.

**Conversation logging + web UI slice (2026-07-17):** Added the **episodic tier** and a **web interface**, and moved the core to the async runtime (§1.2). New/changed: `infrastructure/db.py` (SQLite connect + `PRAGMA user_version` migration runner — schema v1 = `sessions` + `messages`), `infrastructure/conversation_store.py` (`SqliteConversationStore`, UTC timestamps, write lock), `core/interfaces.py` (`ConversationStore` Protocol — the swap seam, defined up front this time), `bootstrap.py` (single composition root + scoped logging, shared by both entry points), async `infrastructure/llm_client.py` (`AsyncOpenAI`) and `core/companion.py` (logs every turn, seeds history from the store on startup so conversation **survives restarts**), `web/app.py` (FastAPI + WebSocket, one shared Companion, `asyncio.Lock` serializing generations = seed of the priority-queue arbiter) + `web/static/index.html` (minimal streaming chat, replays history on reconnect). Deps added: fastapi, uvicorn[standard]. Verified end-to-end: REPL (`python main.py`) and web (`python -m web.app` → http://127.0.0.1:8000) both stream, persist, and carry context across restarts; WebSocket round-trip confirmed (stream + done stats + history replay). DB path is `companion.db` (git-ignored). Not yet built: semantic memory tier, retrieval, emotion, tick — this slice is the substrate they'll build on.

**Semantic memory tier — Tier-1 recall + consolidation (2026-07-17):** Added durable, embedded facts on top of the episodic log. Per the user's memory constraints: **recall runs every turn** (cheap), **consolidation runs only at the end of a context window** and **backgrounded** (never blocks chat). New/changed: schema **v2** = `memories(content, category, embedding BLOB, active, ...)`; `infrastructure/embedder.py` (LM Studio `/embeddings`, nomic **asymmetric prefixes** `search_query:`/`search_document:` — fixes the v1 perspective-mismatch bug); `infrastructure/memory_store.py` + `MemoryStore` Protocol; `core/memory_manager.py` (recall = embed query + brute-force numpy cosine KNN; consolidate = Tier-2 schema-constrained extraction); `core/prompts.py` (extraction system prompt + JSON schema wrapped under `memories`, + `build_system()` folding recalled facts into the system message); `llm_client.structured()`; `core/companion.py` (recall before generate, `asyncio.create_task` consolidation at window boundary, `flush()` for shutdown). **VRAM-safe default:** `BRAIN_MODEL` empty ⇒ reuse the chat model (gemma) for consolidation, so no second large model loads (~3GB resident: gemma + nomic). Set `BRAIN_MODEL=qwen3-8b` for sharper extraction if headroom allows. **Verified live:** gemma extracted 5 clean canonical facts from a transcript and correctly dropped a transient ("tired today"); first-person query "do I have any pets?" recalls "The user owns a dog named Rufus" (0.650 top); full turn: Mari naturally answered "Rufus… you're a nurse" with no retrieval leakage. **Threshold calibrated** on nomic: real ~0.59-0.65, unrelated ~0.50 ⇒ `RECALL_MIN_SIMILARITY=0.55` (small sample; refine later). Storage is brute-force numpy (v1 blessed this at personal scale); sqlite-vec remains a future swap behind the Protocol.

**Bugfix (2026-07-17): short replies were invisible.** The `<think>`-strip streamer buffered the reply start until it reached 7 chars before deciding it wasn't a reasoning block, so any reply under ~7 chars ("Yep.") never streamed a token — an empty bubble in the web UI ("not responding"), even though the model answered. Since the persona is tuned for short replies, this hit often. Fixed in `llm_client.stream`: buffer only while the text is still a possible prefix of `<think>`, flush the moment it clearly isn't, and always flush any remainder at stream end. Verified short + long replies stream correctly via REPL and web WebSocket.

**Bugfix (2026-07-17): concurrent model calls stalled/crashed LM Studio.** Live testing hit a hang mid-conversation. Root cause: background **consolidation** and **live chat** both call the same model, and the web layer only serialized chat-vs-chat — nothing stopped a consolidation from hitting LM Studio at the same time as the next chat turn. Confirmed empirically: two concurrent requests to one LM Studio model raise `APIConnectionError` and, in this session, took the LM Studio server down entirely. Fix: a single `asyncio.Lock` inside `LLMClient` that **both** `stream()` and `structured()` acquire, so no two model calls ever overlap (the §1.2 "one model at a time" arbiter, seeded). Consolidation now simply waits its turn between chat turns. (Full verification pending LM Studio restart — the crash left its server down.)

**Memory lifecycle — pillar 1 (2026-07-17):** Consolidation no longer only-adds. For each extracted fact, `MemoryManager` finds related active memories (cosine KNN above `MEMORY_RELATE_SIMILARITY=0.6`) and, if any, asks the brain for a Tier-2 decision — **duplicate** (skip), **update** (soft-delete the old via `deactivate(id, superseded_by=new_id)`, keep history), or **new** (insert). No related memory ⇒ insert directly (no LLM call, efficient). Changes: schema **v3** adds `memories.superseded_by`; `MemoryStore.deactivate()` + Protocol; `llm_client.structured_json()` (generic single-object schema call, shares the model lock); `MEMORY_DECISION_SYSTEM`/`_SCHEMA` + `build_decision_user()`; `_search()` refactor shared by recall and relate. **Verified offline** (no LM Studio needed — the payoff of the store Protocol): `tests/test_memory_lifecycle.py` uses the real SQLite store with a fake embedder (topic→one-hot) + fake LLM (scripted), and asserts new/update-supersede/duplicate/unrelated-new all behave correctly (NY→Boston supersede links + history kept, duplicate not inserted, dog coexists). First committed test in the suite. v3 migration confirmed non-destructive on the real `companion.db` (3 memories + 80 messages preserved). **Live-verified (2026-07-18):** LM Studio back up. (1) **Concurrency fix confirmed** — chat + consolidation fired simultaneously both completed serialized (~8.5s), no crash, server stayed up (the exact collision that crashed it before). (2) **Lifecycle quality confirmed with real gemma** — "moved to Boston" correctly superseded "New York" (retired + linked), a restatement was correctly skipped as duplicate, an unrelated fact added as new. (3) Found + fixed **extraction over-eagerness**: gemma occasionally invented a persona `self` fact ("Mari is an AI companion") not present in the transcript; tightened `MEMORY_EXTRACTION_SYSTEM` ("only what's actually stated; don't add persona/general knowledge; self-facts only if the user says something new about Mari") — re-tested clean (user-only facts, small talk ignored, no persona leak). Offline `test_memory_lifecycle.py` still passes.

**Edge-case hardening pass (2026-07-18):** A deliberate thoroughness pass after the initial smoke tests. **Bugs found + fixed:** (1) `Companion.flush()` existed but was never called ⇒ sessions shorter than a window (20 msgs) never consolidated — now wired into web `shutdown` and REPL `/exit`. (2) A failed consolidation silently **dropped** the message chunk (facts lost) — now re-queues the chunk for a later retry. (3) **Data-loss bug in lifecycle decisions (the important one):** gemma treated an ADDITIONAL item of the same kind ("a second dog named Lucy") as a *replacement*, deleting the first dog. Fixed by rewriting `MEMORY_DECISION_SYSTEM` with the explicit rule "update only if the existing memory becomes FALSE; a second pet/friend/hobby is 'new' (both true)" + contrast examples. **Coverage now:** `tests/test_memory_edge.py` (8 offline logic cases: superseded-exclusion, recall threshold/order, within-batch dedup, coexist-on-new, bad/garbage decision targets, empty/blank facts, empty extraction) + `tests/test_memory_lifecycle.py` (4 cases) + live behavioral matrix (coexist dog+cat ✓, two dogs ✓, preference reversal ✓, healthcare→nurse refinement ✓, flush-short-session ✓) + concurrency regression ✓. **Known remaining limitations (honest):** `_unconsolidated` still lost on a hard/ungraceful kill (flush only runs on graceful shutdown); recall threshold calibrated on a small sample; decisions are model-dependent and probabilistic (temp 0.2, not deterministic) so occasional misclassifications are possible; only gemma tested for decisions; recall precision at large memory volume unmeasured.

**Chat model upgrade → qwen3-8b + no-think (2026-07-18):** Live testing showed gemma-3-4b was too shallow for conversation (dismissive, incurious, hollow deflection). Compared gemma vs qwen3-8b vs qwen3-14b on the actual failure cases from the log; both qwen models were dramatically more engaging (real world knowledge about the user's music/games, genuine curiosity, actually suggests topics). **Reasoning disabled via `/no_think`** — empirically, LM Studio's `enable_thinking=false` flag is a no-op (this build streams reasoning on a separate `reasoning_content` channel), but `/no_think` in the system message zeroes reasoning → ~2.2s warm ttft, no latency penalty. Picked **qwen3-8b** (near-14b quality, ~5GB, leaves room for the embedder). Implemented: `LLMClient(no_think=...)` appends `/no_think` to the system message for all calls; `config.NO_THINK`; `.env` now `MODEL=qwen3-8b`, `NO_THINK=true`. Verified: engaged chat with no reasoning leak + consolidation still yields valid JSON facts. Reasoning stays available (flip `NO_THINK=false`) for the brain / complex tasks later. **Minor caveat:** qwen occasionally still slips a mild embodiment line ("i saw someone play it once") and the prompt's "you?" example still seeds an occasional bounce — both low-severity, prompt-tunable.

**Next: emotion (pillar 2)** — persist the 6-channel mood + eval; or continue hardening memory. Delivery layer (proactivity/tools) still gated behind a solid brain.

**Phase 1 — Core brain (the "solid base").** Chat with the emergent persona prompt (fixed core + empty self-description slot; anti-hallucination + message-shape rules). Fast memory save + recall with a calibrated embedder (measure thresholds live). Backgrounded/batched extraction. **Full conversation logging from day one** (episodic tier, §2.7). Basic tick loop running internally (reflection + mood drift, no outward messages yet). **Model-lifecycle manager** (§2.8) minimally in place so roles fit VRAM.

**Phase 2 — Memory lifecycle (pillar 1).** Contradiction-aware new/duplicate/update/supersede on extraction; soft-delete history. Deterministic tests on the fake store + live eval.

**Phase 3 — Emotion persistence + eval (pillar 2).** Persist the 6-channel mood to DB; run the behavioral eval; tune `PULL_STRENGTH`/decay/mapping against real transcripts. Wire mood drift into the tick.

**Phase 4 — Proactivity + sleep (pillar 3).** Tick decides + pushes real proactive messages over WebSocket, with throttling/appropriateness gate keyed off mood/presence. Full **sleep/standby** (§2.8): user- and bot-initiated model unload, "waking up…" state on reload. Definition of done: a real, unprompted, model-generated message arrives in the UI, and the bot can put itself to sleep.

**Phase 5 — Tool framework (pillar 4).** Registry + turn-loop integration; verify local tool-calling live; ship web search, **Navidrome playlist creator**, and **reminisce** (reads the episodic log); expose `rewrite_self` so the emergent persona (§2.1) can evolve; enable self-initiated tool use in the tick.

**Later — Voice, and pull from the §2.9 backlog** (familiarity meter, status panel, presence signal are the cheap early wins). STT/TTS slot in behind the existing pipeline.

---

## 4. Open questions still to decide

**Resolved during planning:** persona (emergent/self-modifying, §2.1), emotion mechanism (keep RoBERTa on CPU, §2.3), interface, LLM serving, build order, **runtime/concurrency** (single process + priority queue, §1.2), **capability model** (three tiers, §1.1), **v2 finish line** (bare-bones slice + rigorous bake-off, then reassess).

**Answered by testing, not by decision (Phase 0 bake-off):**
- **Chat/brain/embedding model choices** — including whether chat and brain can be the *same* model to save VRAM (v1 used llama-3.1-8b for both).
- **Embedding model** — commit to an asymmetric-prefix model (nomic/e5/bge/Qwen3-embedding; v1 used `text-embedding-qwen3-embedding-0.6b`) to fix perspective mismatch; confirm in the bake-off.
- **How far Tier-3 tools can go** — set by the measured tool-calling score.
- **VRAM residency policy** — which models may be resident together; what loads/unloads per role.

**Still open (decide after the basics work):**
- **Familiarity → persona coupling:** what the familiarity meter gates and how fast it rises.
- **Proactivity cadence:** default tick interval and reach-out frequency so it feels alive but not naggy.
- **Tool-call turn loop shape:** streaming + tool-call interaction pattern, before Phase 5.
- **Backlog triage:** which §2.9 items are in-scope vs. parked.
