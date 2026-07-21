# ROADMAP — ai-memory-project (Mari)

**What this file is:** the at-a-glance view of what's built and what's next.
**Pre-3.0** is everything shipped, folded back into the phases it belongs to.
**Post-3.0** is the forward backlog — new ideas, plus what those phases left behind.

**Where detail lives — this file does not duplicate it:**
- `HANDOFF.md` — session pickup brief ("▶ START HERE"), current state, how to run, §8 design notes.
- `V2_PLAN.md` — the design decisions and the dated build log. Provenance for every "DONE" below.
- `docs/` — ARCHITECTURE, EXTENDING, TESTING, TUNING.

> **Status vocabulary.** This project distinguishes *built* from *measured*, and the
> distinction has been load-bearing more than once. **DONE** = built and live-verified.
> **DONE (unmeasured)** = built, tests pass, never scored against reality. **PARTIAL** =
> shipped in part, remainder moved to post-3.0.

---

## Pre-3.0 — shipped

### Phase 0 — Skeleton + model bake-off — **DONE (partial by design)**
Composition root, Store `Protocol` + SQLite + migration runner, facade, scoped logging,
`.env`, single prompt module, async runtime, FastAPI+WebSocket wired to LM Studio.
Model bake-off scored on a repeatable rubric → **qwen3.5-9b** (chat + brain),
**nomic-embed-text-v1.5** (embeddings), RoBERTa GoEmotions on CPU (emotion).

⚠️ **Three Phase-0 deliverables were never built** and are carried to post-3.0: the
**event bus**, the **model/role registry**, and the **priority-queue arbiter** (§1.2).
The arbiter exists only as a plain `asyncio.Lock` whose own comment calls it *"the seed of"*.

### Phase 1 — Core brain — **DONE**
Emergent persona prompt (fixed core + self-description slot, anti-hallucination and
message-shape rules). Three memory tiers: episodic log, semantic recall (asymmetric
`search_query:`/`search_document:` prefixes), and always-injected **core** facts.
Backgrounded batched extraction with a crash-durable watermark. Tick loop running
internally. Model-lifecycle manager via the `lms` CLI.

### Phase 2 — Memory lifecycle (pillar 1) — **DONE**
Contradiction-aware new/duplicate/update/supersede, soft-delete with `superseded_by`
history. Deterministic offline tests on the fake store, plus live eval.

### Phase 3 — Emotion persistence + eval (pillar 2) — **DONE**
6-channel mood persisted to the `MetaStore` and reloaded on boot (v1 reset every launch).
Per-channel decay, behavioral eval over a scripted set, mood folded into the system prompt
without ever being named literally.

### Phase 4 — Proactivity + sleep (pillar 3) — **DONE**
Tick loop pushes real unprompted messages over the WebSocket, gated on drives and mood.
Sleep/standby unloads the model from VRAM and reloads on wake. Phone push on reach-out
(self-hosted Bark → APNs), presence-gated so it only fires when the chat isn't in front of you.

### Phase 5 — Tool framework (pillar 4) — **PARTIAL**
Registry + native streaming `tool_calls` turn loop, verified live. Shipped tenants:
`get_current_time`, `reminisce`.

⚠️ **The constraint is routing, not the framework** — measured ~23/30. Remaining tenants
(web search, Navidrome, `rewrite_self`) moved to post-3.0 for that reason, not for effort.

### Phase 6 — Autonomy, measurement, and polish — **DONE** *(not in the original plan; shipped anyway)*
Everything after the four pillars closed:

| area | what shipped |
|---|---|
| **Drives (arc A1)** | connection / restlessness drives, mood-modulated, gating reach-out and reflection |
| **Energy (arc A2)** | energy reserve depleting awake, restoring asleep; biases sleep |
| **Planning triad** | **intentions** (private forward agenda) + **learned self-notes** — completes memory + reflection + planning |
| **A3 — open-question mining** | grounded questions about the user, each rejected unless cited to a real message id |
| **A4 — self-directed unavailability** | she can step away for a bounded window to do one of a closed set of real things; VRAM freed for the window |
| **Self-modifying persona** | persona edit + familiarity + 5-band relationship stage |
| **Silent turns** | she can decline to reply (`PASS`) |
| **Recall quality** | contrast gate — gate on margin above the corpus median, not absolute similarity (recall 4/12 → 11/12) |
| **Measurement** | 120-case gold set + runner, `flaky.py`, style scorer, manual review, ~10 diagnostic probes |
| **Test suite** | 22 files / 392 checks, fully offline (no LM Studio, no network) |
| **Web UI** | streaming chat, conversation tabs, status panel, memory inspector + admin, prompt inspector, mobile/iOS pass |
| **Bounded thinking** | adopted 2026-07-20 — thinking ON, `LLAMA_ARG_THINK_BUDGET`, budget 384 |

**Current measured quality:** gold set **v2.8 = 115/120 (95.8%)**.
⚠️ That baseline is **invalidated** — it predates the bounded-thinking serving config and must
be re-run before any comparison against it means anything.

---

## Post-3.0 — backlog

Nothing below is owed. V2_PLAN calls the project **feature-complete** at v2.1:
*"Everything not built is enrichment or reach."* This is a menu.

### ★ NEW — Topic tool / topic seed *(user request, 2026-07-21)*
**Goal:** give her something to talk about. She struggles to originate topics — a training
artifact, not a memory failure. Returns a mix of **random topics** (the weather, a game, a
random thing) and **things the user has mentioned before**.

**Why it may matter more than it sounds.** The live-session diagnosis found the real problem
upstream: *"the conversation had nothing to remember, because it was about HER."* A feature
that turns the conversation toward *his* life doesn't just vary her openers — it generates the
material that memory, recall, and intentions are all starved of. It was explicitly reframed
*"from nice-to-have to the actual fix."*

⚠️ **Three constraints on record. Read them before building — two cut against the obvious design:**
1. **Frame every topic as "ask him about X", never "talk about X."** The random-subject half is
   close to the mechanism that produced the fabricated restaurant: hand a confabulating model
   "restaurants" and it invents having discussed one.
2. **Ground the memory-seeded half in stored facts** — and note the well is currently shallow,
   which this feature is partly meant to fix (chicken-and-egg: expect it to improve as it runs).
3. **An injected seed is preferred over a tool.** Tool routing is ~23/30 and a topic tool would
   compete with `reminisce`; an injection bypasses routing entirely. *If built as a tool anyway,
   that is a deliberate choice against this note, not an oversight — measure the routing cost.*

### ★ NEW — Let her know she can sleep, and choose it *(user request, 2026-07-21)*
**Practical case:** the user says goodnight. She should acknowledge that resting is an
option for her too, and be able to actually go into standby if she decides to.

**The blocker is not the trigger — it's that the persona says she doesn't sleep.**
`core/prompts.py` "No body, no life outside these conversations" lists *"you don't go
places, do things, eat, **sleep**, own stuff, or have a day."* That line predates standby
and was never reconciled with it. **She really does sleep** — `SleepJob` unloads the model
from VRAM, and the UI has an asleep state and a "waking up… reloading the model" message.
So today she will deny something that is true.

Standby belongs with functional emotions ("real at the level she has them"), **not** with
walks and meals — it is the one embodied-sounding thing that actually happens to her. Same
principle A4 rests on: an absence is honest when the thing she does is real.

**Measured 2026-07-21: the embodiment filter already draws the right line and needs no
change.** It blocks the false human framing (`"i slept well"`, `"i woke up"`, `"i took a
nap"`) and allows the true one (`"i'll go quiet too"`, `"i'm going to sleep too"`,
`"i might rest while you do"`) — because lived past experience is a fabrication while
entering standby is a real future action.

**Shape:**
1. Carve sleep out of the "no body" list. Keep human framing forbidden (no nights, beds,
   dreams, mornings); allow going quiet / powering down, which is true.
2. On a departure cue, offer her the choice via **structured output, not a tool** — same
   reasoning A4 used: routing is ~23/30, far too flaky to gate whether she goes silent.
   Keep the mechanical long-idle trigger as the VRAM backstop so she cannot refuse her way
   into never sleeping.
3. Note A2 energy already exists, so "tired" is arguably already true at the level she has
   it — a STATE, which the three-tier rule already permits.

⚠️ **Not a free edit: changing the persona invalidates the personality baseline.** Re-run
`scripts/bakeoff_personality.py` after, and note the project's own rule that prompt rules
nearest the end carry the most weight.

### §8-B — Memory depth *(largest coherent cluster)*
| item | status | note |
|---|---|---|
| Write-side staleness adjudication | NOT STARTED | *"the biggest gap"* — catch facts *implicitly* invalidated with no retraction. Cited lift **8.7% → 68%** |
| Bitemporal provenance | NOT STARTED | valid time + transaction time. *"Two columns, no graph DB"*. Enables *"when did that change?"* |
| Salience / forgetting curve | NOT STARTED | ACT-R base-level activation, decaying on recency of **access**. *"One extra column."* Gates the two below |
| A-MEM link evolution | NOT STARTED | new memories reinterpret old ones; gate behind salience |
| Hybrid recall (BM25 + vector) | PARTIAL | contrast gate shipped; BM25 over the episodic log is the documented fallback. **Tuned convex combination, not RRF** |
| Mood-congruent recall | NOT STARTED | *"promising rather than proven"* |
| Memory confidence + confirmation | NOT STARTED | occasionally double-check a shaky fact |
| Inspector search/filter | NOT STARTED | inspector itself done |

⚠️ **The justification for this whole cluster is currently unmeasured.** §B's headline evidence
was withdrawn — the `stale-job` gold case had its subjects swapped. The instrument exists
(`mode` field on gold cases) but **has never been run**. Measure before building.

### §8-C — Presence & timing
- **Time-of-day / do-not-disturb gating** — NOT STARTED. *"There is currently no time-of-day
  awareness anywhere in the codebase."* **Hard prerequisite for self-wake.**
- **Absence-triggered escalation** — NOT STARTED. From a 24-month real deployment: contact after
  5 missed days, again at 10 and 12, **drop at 14**. The give-up rule is what prevents nagging.
- **Log proactive outcomes and learn** — NOT STARTED. Response history beat all context sensing,
  **F1 0.113 → 0.311**. She logs reach-outs but not whether you engaged.
- **Farewell ritual + forward continuity** — NOT STARTED. Measured **69% vs 35%** on a bond
  marker, and the gap widened over time.
- **Presence as a tick/sleep input** — PARTIAL. Tab visibility gates phone push; unused elsewhere.

### §8-A — Autonomy remainders
- **Self-waking** — BLOCKED on time-of-day gating above. `wake()` seam already exists.
- **Relational continuity** — NOT STARTED (small). Let persisted mood/drives shape engagement
  level across sessions.
- Intentions: relevance-ranked pick instead of FIFO. Self-notes: let a note decay.

### §8-E — More tools *(the rest of Phase 5)*
Reminder tool (easiest routing, most useful daily) · web search (unlocks curiosity; hardest
routing) · Navidrome playlist — *the interesting version is a tick job building one unprompted,
which bypasses routing* · `rewrite_self` as a tool (tidiness; behavior already exists) ·
always-on recall index (~150 tokens of titles only).

### Architecture debt *(carried from Phase 0)*
- **§1.2 priority-queue arbiter** — designed, seeded as a plain lock, never built. Contention
  order was to be user turn > proactive reply > background reflection. No bug attributed to it yet.
- **§1.3 event bus** — NOT STARTED. Doc warns it is *"nearly impossible to retrofit once many
  subsystems cross-call"* — that risk has now materialized.
- **§1.3 model/role registry** — NOT STARTED. Today `MODEL`/`BRAIN_MODEL` are plain env strings.

### §8-H — Rework of things already built
- **The familiarity meter is the wrong shape.** `min(1, count/N)` only climbs, but breadth and
  depth diverge and four trajectory classes exist. **It gates persona drift, so this matters.**
- **`REACHOUT_COOLDOWN` throttles the wrong variable** — responsiveness fell 93% → 47% from
  *predictability*, not volume. *"Vary the trigger and the shape, don't just throttle."*
- **Don't treat idleness as availability** — add boundary triggers, don't remove drive gating.

### Quality / tuning — open
- ⚠️ **The "haha" tic** — 47% of replies open with it (was 0%, then 1%). Position-beats-volume;
  *"adding another rule will not work."*
- ⚠️ **Melancholy positive-feedback loop** — she acts sad → user answers a sad companion →
  classifier scores that sad. Fix candidates listed in §8-G2; none chosen.
- **Mood drift rate untuned** — `DECAY_RATES` calibrated per-message, running per-tick.
- **Reasoning leak, residual** — the visible reply is now reconciled on `done`, but the CoT is
  still *watched* streaming in first. Earliest fix is a mid-stream reset when `</think>` arrives.
- **Re-baseline the gold set** at thinking-on; sweep the thinking budget below 384 (96/128/192).
- **Per-call temperature (prong A)** — cool the tool decision to ~0.2, keep the answer at 0.8.

### Code quality — Phase 3 *(Phases 1 & 2 shipped)*
Deliberately not bundled with tidying: *"needs LM Studio + re-measurement."*
- **`build_system`'s trailer lands in the wrong place for 3 of 4 callers** — `build_reachout_system`,
  `build_followup_system` and `build_reflect_system` all append *after* it. For reflect it is
  actively contradictory (the trailer demands one sentence; the addendum asks for one or two).
  ⚠️ **Not a safe refactor** — it invalidates the personality baseline. Re-run
  `scripts/bakeoff_personality.py` after.
- **`category` is dead data** — always `None`, still threaded through three `store.add` calls, the
  Protocol, the DB column, and the inspector UI.
- Five live scripts still carry their own preamble instead of `scripts/_harness.py`. Deferred on
  purpose: *"rewriting code you can't run is how silent breakage gets in."*

### §8-D / §8-F — Reach and far future
Multi-channel presence (WhatsApp/Telegram/Discord) · dreams *(evidence-negative — build it
because you want the behaviour, not because the literature supports it)* · voice STT/TTS,
then acoustic emotion · embodiment (Live2D/VRM, wearables).

---

## Do not revive

HANDOFF §8-G records things **deliberately closed**, several after costing real time. Read it
before reopening any of: farewell hooks / manufactured neediness, energy that declines from
neglect, warm+agreeable combined (*"you cannot prompt your way out of sycophancy"* — steering
worked at 70B, not 8B), deliberate friction, verbatim memory quotation, HyDE / MiniLM
rerankers / aggressive summary clustering, any LoCoMo-benchmarked claim (6.4% of its answer key
is wrong), **deduplicating the persona's format rules** (cost 10 points, reverted — the
duplication is load-bearing), and **model migration to fix reasoning** (measured not to work;
identity discontinuity is the best-documented way to destroy a companion relationship).

## Housekeeping

- `archive/v1/infrastructure/config.py` contains a **real hardcoded HuggingFace token**.
  Git-ignored deliberately, but it should be **rotated/revoked** on HuggingFace.
