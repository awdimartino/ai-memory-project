# ai-memory-project — Roadmap

**North star:** A companion *you* actually talk to every day — one that reliably remembers you, has a consistent emotional presence, occasionally reaches out on its own, can take real actions through pluggable tools, and eventually listens and speaks. Local, private, single-user.

**Guiding principle:** The companion is a *brain* (memory + emotion + conversation) with a *delivery layer* (tools + proactivity + voice) on top. The brain must be trustworthy before the delivery layer is worth building — a companion that can't remember isn't worth giving a voice.

**Project profile:**
- **Purpose:** Personal companion — private, local-first, day-to-day use. Not a product or demo.
- **Priority style:** Balanced/phased — fix the critical bug first, then layer features and quality.
- **Interface goal:** Eventually a voice interface, kept local/single-user.
- **Core focus (all matter):** Memory, Emotion, Proactivity, Conversation quality.
- **Storage:** Embedded, in-process — no separate server to launch. Migrating from PostgreSQL/pgvector to **SQLite + sqlite-vec** (one local file, covers relational data and vector search).

---

## Phase 1 — Migrate storage & make the foundation trustworthy
*Goal: move to an embedded database with nothing to launch, and prove the memory loop works end-to-end. The recall fix folds into the rewrite rather than preceding it.*

**Storage migration (do this first — it rewrites the code the bug lives in):**
- [x] **Move to SQLite + sqlite-vec** — embedded, in-process, one local file, no separate server. `database.py` now uses `sqlite3` and loads the `sqlite-vec` extension.
- [x] **Port the schema** — memories now use a `vec0` virtual table (`vec_memories`, cosine) for embeddings joined by rowid to a `memories` metadata table; `conversations`/`messages` are plain SQLite tables with ISO-8601 UTC timestamps.
- [x] **Decide on existing data** — went with a clean start (new `companion.db`); no Postgres→SQLite import written. Old data was unrecoverable memory anyway.

**Fold in the recall fix + hygiene during the rewrite:**
- [x] **Fix the recall bug as part of the rewrite.** New `memory_exists` returns a real boolean via a cosine KNN; verified end-to-end (store → recall) and with a dedup regression check.
- [x] **Match the embedding dimension** — single config knob `EMBED_DIM`; confirmed 1024 against the live LM Studio server.
- [x] **Fix the top-level-array schema bug** (found on the live run) — a top-level `array` JSON schema made LM Studio's grammar collapse to `[]`, so classification never extracted anything. `BRAIN_RESPONSE_FORMAT` now wraps the list under a required `memories` key; `llm_client` parses accordingly. Live run then extracted + stored a real memory.
- [x] **Calibrate the recall threshold** — 0.70 was too strict for qwen3-embedding-0.6b (related queries score ~0.63–0.81). Now config-driven `MEMORY_RECALL_THRESHOLD` (0.5). Verified: 3 of 4 natural rephrasings recall correctly.
- [x] **Kill the debug noise.** Replaced the `print("TESTING")`/`"FETCHING"`/raw-dump traces with a togglable `logging` setup (`DEBUG_MODE`). Live stream output kept.
- [x] **Honest dependencies & config.** Dropped `psycopg2`/`pgvector`, added `sqlite-vec`; `torch`/`transformers`/`tqdm` were already installed and are now declared. HF token/endpoint read from env with fallback; DB password gone (SQLite has no auth).

**Structural cleanup (do it now, before Phases 4–6 pile on coupling):**
- [x] **Give `Companion` a real facade.** Added `ensure_conversation()`, `classify()`, `decay()`, `minutes_since_*()`; `ChatLoop`/`TickSystem` no longer reach through to managers.
- [x] **Define a `MemoryStore` interface** — `core/interfaces.py` `MemoryStoreProtocol`; `MemoryManager` type-hints against it.
- [x] **Consolidate prompt construction.** Tick prompts moved into `PromptBuilder.build_thought_prompt` / `build_unprompted_prompt`.
- [x] **Clear import noise** — removed `from unittest import result`, unused `pyexpat`/`openai`/`PromptBuilder`/`MemoryRecord` imports.

**Runtime fixes (found running `main.py` end-to-end for the first time):**
- [x] **Logging flood** — `basicConfig(DEBUG)` had turned on DEBUG for every library (httpcore/httpx/huggingface), burying the `You:` prompt. Now scoped: only `core`/`infrastructure` follow `DEBUG_MODE`; third-party libs pinned to WARNING.
- [x] **Critical prompt-ordering crash** — memories/thoughts were injected as an `assistant` turn *before* the user query, producing `[system, assistant, user]`, which the Qwen chat template rejects ("No user query found in messages"). Would crash every turn that recalled a memory. Fixed: memories/thoughts now fold into the system message → `[system, …history, user]`.
- [x] **`<think>` reasoning blocks** — qwen3.5-9b is a reasoning model and emits `<think>…</think>` before answers (no prompt flag reliably disables it). `LLMClient.stream` now hides the block from both live output and the stored reply.
- [x] **Robustness** — empty input is skipped; `Ctrl+C`/EOF exits gracefully; polluted dev `companion.db` cleared. Verified a clean in-character reply via `main.py`.

*Why first: every later phase reads from or writes to memory, and it should sit on the final storage substrate before Phase 2 invests in memory quality. The structural cleanup is far cheaper now than after tools/proactivity add more reach-through coupling. Nothing above is worth building on a memory loop that drops everything.*

> **Model note:** `BOT_MODEL` switched from `qwen/qwen3.5-9b` to `llama-3.2-3b-instruct`. qwen spent ~40s/turn generating a hidden `<think>` block, so the (streamed) answer only appeared after a long silence — looked like no streaming. llama has no reasoning block: warm first token ~0.7–2.5s and it streams live. `BRAIN_MODEL` stays qwen3.5 (JSON grammar suppresses its reasoning; runs after the reply). Both models stay resident together — no reload thrash. Residual: one-time ~30s cold load per LM Studio session, and ~8s synchronous classification between turns → addressed by the Phase 2 async-classification item.

---

## Phase 2 — Memory that earns trust
*Goal: it remembers the right things, recalls them at the right moments, and doesn't drown in duplicates or stale facts.*

- [ ] **Real dedup** — semantic ("I love hiking" vs "hiking is my favorite") rather than exact-match.
- [ ] **Retrieval quality** — relevance thresholds so weak matches don't pollute the prompt; tune top-k; consider recency/importance weighting so recent and salient memories surface first.
  - *Live finding (Phase 1):* first-person queries against third-person memories score poorly ("do I have any pets?" vs "Alex has a dog named Rufus" = 0.35, below unrelated controls). Fixes to try: normalize perspective when storing, and use Qwen3-embedding's instruction-prefixed query embeddings (asymmetric query/document embedding).
- [ ] **Memory lifecycle** — let facts be *updated* or *superseded* (job changed, moved), and handle contradictions instead of stacking both versions. Optional gentle "forgetting."
- [x] **Behavioral testing against live models** — verified memory extraction quality, personality/tone, and hallucination resistance with real Companion turns (not just plumbing). Memory: transient statements ("birthday coming up") correctly skipped, durable ones ("birthday is July 25th") correctly saved with proper rephrasing — 6/6 cases correct. Personality: casual, in-character, no robotic markers, matches user energy. **Finding + fix:** open-ended questions ("what did you do today," "favorite food") triggered soft hallucination — fabricated mundane experiences (pizza preferences, watching shows) despite the system prompt forbidding it; the literal named example (skydiving) was already handled correctly, but softer natural phrasing wasn't. Added an explicit "don't have a body/senses, don't invent an answer" rule to `BOT_PROMPT`; verified it eliminates fabrication with zero personality regression and — better than an earlier prototype — without leaning on clinical "I'm just a program" disclaimers.
- [x] **Smarter classification cadence** — resolved the `chat_loop` TODO. Classification now runs in batches (`config.CLASSIFY_BATCH_SIZE`, default 6 messages = 3 exchanges) instead of every turn: `ConversationManager` accumulates an in-memory pending batch, `Companion.maybe_classify()` fires once the threshold is hit, `Companion.flush_pending_classification()` catches any partial batch on graceful exit (Ctrl+C/EOF) so nothing is silently dropped. Fixed a latent bug found along the way: `MessageRecord.id` was never populated (`store_message` didn't return the row id) — now it does, so `MemoryRecord.origin_id` has real provenance. Verified offline (no call below threshold, one call at threshold, flush catches partial batches) and live (3 turns → exactly 1 brain-model call instead of 3).
  - *Known limitation:* the pending batch is in-memory only — an ungraceful process kill (not Ctrl+C/EOF) loses an in-progress partial batch. Acceptable for a personal local app; revisit if it becomes annoying.

*Why here: memory is the top-named focus and the project's whole reason to exist. Get it genuinely good before layering personality and proactivity on it.*

---

## Phase 3 — Conversation quality
*Goal: turns feel coherent, in-character, and grounded in what it knows.*

- [ ] **Prompt assembly balance** — tune how memories, emotion state, and history share the context budget (currently a fixed last-10 window plus whatever memories return).
- [ ] **Persona consistency** — make Mari feel like the same entity across sessions; audit the system prompt for drift and contradiction.
- [ ] **Grounding** — ensure recalled memories are actually *used* naturally in replies, not ignored or awkwardly recited.

*Why here: this is the payoff surface for Phases 1–2. Better memory only helps if the conversation layer uses it well.*

---

## Phase 4 — Modular tool framework
*Goal: a plug-in system where tools are self-contained and hot-swappable — drop one in and the companion can use it; pull it out and it's gone, with zero changes to the core.*

**The framework (the actual deliverable):**
- [ ] **Self-contained tool modules** — each tool lives in its own module and declares everything it needs: name, description, parameter schema (OpenAI function-calling format, which LM Studio/Qwen speak natively), and its handler.
- [ ] **Add/remove at will** — auto-discovery so registering is just placing a module in the tools folder; enable/disable via config or a runtime toggle, no core edits. The framework code never changes when the tool roster does.
- [ ] **The agentic turn loop** — the LLM client handles: model requests a tool → framework dispatches to the handler → result feeds back → model continues (chaining allowed). Reconciling this with the existing *streaming* replies is the fiddly part — budget for it.
- [ ] **Guardrails** — per-tool timeouts, graceful failure (a broken or offline tool degrades the turn, never crashes it), and a confirmation seam for side-effecting actions.

**First tenants (proof the framework works):**
- [ ] **Web search** — read-only, network-only, no auth. The clean first proof.
- [ ] **Navidrome playlist creator** — talks to the self-hosted Navidrome server on a Raspberry Pi over its **Subsonic API** to read the available library and build playlists. "Based on your taste" couples to **Memory (Phase 2)** — music preferences live there and inform selection. Has side effects (creates playlists on the server), so it exercises the confirmation guardrail.

*Why here: tool use extends what the model does in conversation, so it wants stable conversation (Phase 3) underneath. It's otherwise independent — if a tool becomes the thing you most want, this phase can jump the queue.*

---

## Phase 5 — Emotional presence with continuity
*Goal: mood feels persistent and shapes tone believably.*

- [ ] **Persist emotional state across sessions** (and likely across ticks) so the companion doesn't reset to baseline every launch.
- [ ] **Tune the decay/pull dynamics** so moods shift at a believable pace — noticeable but not erratic.
- [ ] **Emotion → behavior coupling** — let mood color word choice and openness, and optionally influence what memories feel salient.

*Why here: emotion is a modifier on conversation. It's most rewarding once conversation quality (Phase 3) is solid, and it feeds naturally into proactivity next.*

---

## Phase 6 — Proactivity (bring the ticks to life)
*Goal: the companion feels "alive" during silence — it reflects, and sometimes reaches out.*

- [ ] **Implement `think_tick`** — actually call the LLM to reflect during idle time (currently a stub that builds a prompt and does nothing), possibly forming new memories or intentions.
- [ ] **Implement `unprompted_message_tick`** — initiate a message after a meaningful idle gap, drawing on memory + current mood (and possibly tools, e.g. "I found a playlist for how you're feeling").
- [ ] **Delivery plumbing** — a clean way for an async message to surface into the interface without clobbering the input prompt (the tick/chat lock is the seam to build on).
- [ ] **Guardrails** — frequency limits and mood/context gating so it's endearing, not annoying.

*Why here: proactivity consumes everything below it — memory to have something to say, emotion to have a reason, conversation quality so it lands well, and optionally tools to act.*

---

## Phase 7 — Voice interface
*Goal: talk to it and hear it back.*

- [ ] **Speech-to-text** for input (the repo shows traces of an earlier voice iteration to learn from — or deliberately not repeat).
- [ ] **Text-to-speech** for replies, ideally streaming so it speaks as it generates.
- [ ] **Interaction handling** — barge-in / interruption, and deciding how spoken proactive messages (Phase 6) announce themselves.

*Why last: voice is a presentation layer over a finished companion. It multiplies the value of everything before it and adds nothing on its own.*

---

## Woven through every phase
- [ ] **Tests** — start with the memory loop (store, dedup, recall) once Phase 1 stabilizes it; grow coverage as each subsystem settles.
- [ ] **Resilience** — graceful handling when the local LLM server, DB, or embedder is down (a personal daily-driver needs to fail softly).

---

## Sequence at a glance

| Phase | Focus |
|---|---|
| 1 | Storage migration (SQLite + sqlite-vec) — fix recall, hygiene |
| 2 | Memory quality |
| 3 | Conversation quality |
| 4 | Modular tool framework (web search + Navidrome playlists) |
| 5 | Emotional presence / continuity |
| 6 | Proactivity (ticks) |
| 7 | Voice interface |
