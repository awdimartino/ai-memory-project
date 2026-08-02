# ai-memory-project — V1 Retrospective & Handoff

**Purpose of this document:** a self-contained brief for planning v2.0 in a fresh session. It covers what the project is trying to be, what v1 actually built, and — most importantly — what was learned the hard way so v2 doesn't have to relearn it. This is a knowledge handoff, not a prescription; v2 may reasonably choose a different stack, scope, or priority order.

---

## 1. The vision

A **personal AI companion** — not an assistant, a friend — for a single user, running fully **local-first** on their own machine. Not a product, not a demo: something the user actually talks to day to day.

Four pillars, all considered core (not optional add-ons):
- **Memory** — durable, long-term recall of facts/preferences/goals about the user.
- **Emotion** — a persistent, evolving mood that colors tone and personality.
- **Proactivity** — the companion can reflect and occasionally reach out unprompted, not just respond.
- **Conversation quality** — replies feel coherent, in-character, and genuinely use what it knows.

Plus two extension goals layered on top once the core is solid:
- A **modular tool framework** — hot-swappable tools (add/remove without touching core code). First planned tenants: web search, and a playlist creator that talks to the user's self-hosted **Navidrome** server (Subsonic API) on a Raspberry Pi, building playlists from the user's actual music library based on taste stored in memory.
- A **voice interface** (speech in, speech out), eventually — kept local/single-user.

**The one architectural principle that held up under real use:** the companion is a *brain* (memory + emotion + conversation) with a *delivery layer* (tools + proactivity + voice) on top. The brain has to be trustworthy before the delivery layer is worth building — a companion that can't remember isn't worth giving a voice. This ordering proved correct in practice: every bug found by live-testing was in the brain layer, and fixing those was consistently higher-leverage than any delivery-layer feature would have been.

---

## 2. What v1 actually was (stack snapshot)

- **Language/runtime:** Python, single-process CLI (`main.py` — a terminal REPL).
- **LLM serving:** OpenAI-SDK client pointed at a **local LM Studio** server, not any cloud API. Separate models for chat and for memory-extraction ("brain") calls; a separate local embedding model.
- **Emotion:** a local HuggingFace RoBERTa GoEmotions classifier running on GPU, reduced from 28 emotion labels into 6 mood channels with decay-toward-baseline dynamics.
- **Storage:** migrated mid-project from PostgreSQL + pgvector to **SQLite + sqlite-vec** — a single embedded file, no server process to launch. This was a deliberate, explicit priority ("I don't want to have to launch a separate program on my desktop") and it worked well.
- **Architecture:** layered — `infrastructure/` (persistence, LLM client, embedder) sits strictly below `core/` (domain managers for memory/conversation/emotion, a `Companion` facade, the chat loop, and a background "tick" thread), composed via dependency injection from one root (`main.py`).

---

## 3. Durable technical lessons

These are the findings most likely to still matter regardless of what stack v2 chooses, because they're about *local LLM serving and small-model behavior* generally, not about this specific codebase.

### Local LLM serving (LM Studio-specific, but likely generalizes to other local OpenAI-compatible servers)

- **A top-level-array JSON schema can silently break structured extraction.** Constraining a response to a bare `array` schema let the grammar collapse to the trivially-valid empty array `[]` on every call — the model never generated real output, and this was invisible without inspecting raw responses. Fix: wrap the array under a required object key (e.g. `{"memories": [...]}`) instead of using a bare top-level array schema.
- **Chat templates can be strict about message ordering.** Many local templates require the *last* message to be a `user` turn and will outright reject unusual orderings (e.g. `[system, assistant, user]`), throwing a template-rendering error instead of a graceful failure. Practical rule: don't inject extra "assistant" context turns before the final user message — fold auxiliary context (retrieved memories, internal thoughts, etc.) into the **system** message instead, and keep the shape `[system, ...history, user]`.
- **Reasoning models emit large hidden `<think>...</think>` blocks by default, and no prompt-level flag reliably disabled it.** Neither an in-prompt `/no_think` directive nor an `enable_thinking: False` `extra_body` flag cleanly suppressed it in testing — the latter just leaked raw unlabeled reasoning text instead of clean tags. A reasoning model can spend 30-40+ seconds generating hidden reasoning before a short visible reply appears; over a streaming connection this reads as "not streaming" even though it technically is (the visible answer streams fine once it starts — it just starts very late). For latency-sensitive casual chat, prefer a **non-reasoning instruct model**. Reserve reasoning models for calls where quality matters more than latency (e.g. structured extraction) — and note that JSON-schema-constrained decoding appears to suppress the "thinking" tendency as a side effect, probably because the grammar leaves no slot for free text.
- **Multiple models can stay resident simultaneously without reload thrashing.** It's safe to route different roles (fast casual-chat model, smarter extraction model, embedding model) to different concurrently-loaded models without paying a swap-latency penalty on every call.
- **Expect a one-time cold-load delay** (tens of seconds) the first time any given model is used in a server session. This is irreducible infrastructure latency, not a bug — don't waste time trying to "fix" it in application code.
- **Always empirically verify the actual local server + model combination.** Assumptions carried over from hosted-API experience (array-only schemas being fine, lenient message ordering, prompt flags working as documented) did not hold locally. Budget explicit time to run real prompts through the real server per subsystem, early — see the process lesson in §4.

### Embeddings / retrieval

- **Similarity thresholds are not portable across embedding models — measure, don't guess.** For the embedding model used in v1, genuinely related query/document pairs scored cosine similarity ~0.63–0.81, and unrelated pairs scored ~0.42. A "reasonable-sounding" 0.70 cutoff silently excluded real matches; the bug was invisible without directly measuring real similarity values on real data.
- **Perspective mismatch is a real, easy-to-miss retrieval failure mode.** A first-person query ("do I have any pets?") can embed *worse* against a third-person stored memory ("the user has a dog named Rufus") than genuinely unrelated text does. If keeping semantic memory retrieval in v2, plan for this explicitly — options include instruction-prefixed/asymmetric query-vs-document embeddings (several modern embedding models support this natively) or normalizing memory phrasing/perspective at write time.
- **Brute-force KNN (no ANN index) was plenty fast at personal/single-user memory volumes.** Don't over-engineer indexing for a scale of hundreds-to-low-thousands of memories.

### Prompt / personality engineering (small local instruct models, ~3B–9B parameters)

- **Small models default toward fabricating plausible personal experiences on open-ended questions**, even when explicitly told not to claim physical/human experiences. A prompt that only forbids a *named* example (e.g. "you haven't been skydiving") reliably blocks that literal example but lets the same underlying tendency leak through on differently-phrased questions ("what did you do today?", "favorite food?"). **Test with varied natural phrasing, not just the one example you're worried about** — a single named-example test can pass while the general behavior still fails.
- **There's a genuine tension between "don't fabricate" and "don't sound like a disclaimer-bot."** An instruction that's too strong ("you have no body, don't answer") pushes small models toward repeatedly explaining they're "just a program" / "exist as code" — which reads as robotic and can violate a separate persona rule ("don't reference your AI nature unless asked"). What actually worked: instructing not just *what not to do*, but *what to do instead* (play it off, redirect the question back to the user, stay warm) **and** explicitly telling it not to over-explain its nature unless asked directly. An instruction that only says "don't invent an answer" without offering a positive alternative behavior was insufficient and the model filled the gap with either fabrication or robotic disclaimers.

### Logging / dev ergonomics

- **`logging.basicConfig(level=DEBUG)` enables DEBUG for every imported library**, not just your own code. With an HTTP-heavy stack (an OpenAI-style client, httpx, huggingface_hub, etc.), this floods the terminal enough that a perfectly-working interactive app can look completely broken/unresponsive. Scope DEBUG to your own top-level package(s) explicitly and pin known-noisy third-party loggers to WARNING from the start.
- **A synchronous secondary LLM call after the main reply (e.g. classification/extraction) adds real, felt latency between turns**, even when the main reply itself streams fine. Batching that kind of "housekeeping" call (run it once every N turns instead of every turn) or backgrounding it meaningfully improves perceived responsiveness, and is usually easy to justify since the information needed (recent conversation) doesn't go stale between turns.

---

## 4. Architectural lessons

### Worked well — worth keeping regardless of stack choice

- **Strict layering + one composition root.** `infrastructure/` (persistence, LLM, embedding) sat strictly below `core/` (domain/orchestration), wired together in a single place. This made a full storage-engine migration (Postgres → SQLite) a contained, low-risk change — only the store classes and the connection wrapper needed rewriting; nothing in the domain layer changed.
- **Store vs. Manager separation.** A Store does *only* persistence (SQL/queries in, plain records out); a Manager coordinates a Store with other collaborators (embedder, LLM client). This split is exactly what made the storage swap safe — the swap seam was already explicit.
- **An interface/Protocol for the store**, defining the exact contract a persistence layer must satisfy. This documents the swap seam and would enable a fast in-memory fake for tests. (In v1 this was added *after* the fact, mid-project — better to define it alongside the store from day one.)

### Didn't work well initially — worth doing differently from the start

- **A facade was added late to stop reach-through coupling.** Early on, callers reached two levels deep into collaborators (e.g. `chat_loop.companion.conversation_manager.some_method()`). This kind of coupling is easy to introduce accidentally and gets progressively more expensive to unwind as more callers accumulate. Design the facade/entry-point API surface deliberately from day one instead of exposing manager objects directly to callers.
- **Debug `print()`s accumulate fast during exploratory development** and are easy to forget about. Wire up scoped logging from the very first commit rather than "printf-debug now, clean up later" — the cleanup pass is real work that's easy to skip.
- **Prompt-construction logic drifted into multiple places** (a dedicated prompt-builder class, plus ad-hoc inline strings elsewhere) before being consolidated. Decide on day one that *all* text sent to a model lives in exactly one place, so persona/voice stays consistent and auditable as the project grows.

### The most important process lesson

**Verifying end-to-end against the real local model stack surfaced more real, load-bearing bugs than code review or offline/mocked testing did.** A silently-broken extraction schema, a message-ordering crash enforced by the chat template, a miscalibrated similarity threshold, and a hallucination pattern were all invisible from reading the code or from tests using fake/mocked models — they only appeared by running real prompts through the real local server. Local LLM serving has genuine, non-obvious constraints (grammar/schema quirks, template strictness, reasoning-model latency behavior, model-specific similarity distributions) that simply don't show up any other way.

**Recommendation for v2: budget explicit time to live-test each subsystem against the real model stack early, per subsystem — not just once at the end.** It consistently paid for itself in this project.

---

## 5. Known open issues / unfinished business in v1

So v2 doesn't have to rediscover these:

- **Retrieval perspective mismatch** (first-person query vs. third-person stored memory) — identified, not resolved.
- **No memory lifecycle** — memories can be created but never updated, superseded, or forgotten; contradictory facts would simply stack up over time.
- **Dedup is similarity-threshold-only** — not contradiction-aware, not robust to significant rewording below the threshold.
- **Proactivity was never implemented**, only stubbed — the background "tick" system's reflection and unprompted-message hooks built prompts but never actually called the model or sent anything.
- **No tool-use framework was built** — it was scoped (modular, hot-swappable, web search + Navidrome playlist creator as first tenants) but implementation never started.
- **Emotional state is in-memory only** — resets to baseline on every process restart; never persisted.
- **No voice interface** was started.
- **No automated test suite** — verification was done via one-off scratch scripts run against the live stack during development, never promoted into a real committed test directory.
- **Batch-classification pending state is in-memory only** — an ungraceful process kill (not a clean exit) silently drops an in-progress partial batch of unclassified messages.
- **A HuggingFace API token was hardcoded** in a config file (not committed to git, but never rotated, and not handled through a real secrets/`.env` flow either).

---

## 6. Open questions worth deciding explicitly for v2

These aren't answered here on purpose — v1 either answered them implicitly (by accident of what was convenient at the time) or never resolved them at all, and they deserve a deliberate decision this time:

- **Model-per-role split.** V1 landed on a non-reasoning small instruct model for casual chat (for latency) and a reasoning model for structured extraction (quality matters more there, and latency was hidden by schema-constrained decoding + backgrounding). Is that still the right split? Does v2 want different models, a hosted API instead of fully local, or one model doing both?
- **How much of the emotion system is worth keeping as-is?** It was functional but was never behaviorally evaluated the way memory/personality/hallucination were — worth an explicit eval pass if carried forward.
- **What should proactivity actually look like from a UX standpoint?** A terminal REPL may not be the right surface for "the companion reaches out on its own" — does that need a persistent background service, a notification system, a different interface entirely?
- **Tool framework scope and design.** Still only planned, never built. It touches the LLM client's turn-handling loop and interacts with streaming — worth deciding the shape of this early since it's architecturally invasive.
- **Testing strategy.** V1 relied on hand-written scratch scripts run against the live server rather than a committed, repeatable test suite. Worth deciding upfront how to test deterministically against something that's fundamentally non-deterministic (mocking? fixtures? a small reserved local model for tests?).
- **Storage engine fit going forward.** SQLite + sqlite-vec worked well for the single-device, single-user case and is a reasonable default to keep — but re-confirm it still fits if v2's scope changes significantly (e.g. multi-device sync would need rethinking an embedded single-file database).

---

## 7. Where more detail lives

If the v1 codebase or its `ROADMAP.md` is available to the new session, it has the full phase-by-phase plan, exact before/after transcripts for the hallucination-prompt fix, and the specific config values (thresholds, batch sizes, model names) arrived at during development — useful as concrete reference points even if v2 rebuilds from scratch.
