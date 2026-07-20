# Architecture

How the pieces fit, and why. For *what to change* see [TUNING.md](TUNING.md); for *how to add
something* see [EXTENDING.md](EXTENDING.md).

---

## The one rule

```
        web/app.py            main.py
             \                  /
              \                /
            Companion (core/companion.py)     <- the facade. Everything goes through it.
                     |
        core/  managers: memory, emotion, drives, tick, tools
                     |
   infrastructure/  stores, LLM client, embedder, classifier, notifier
                     |
                  SQLite + LM Studio
```

**`infrastructure/` sits strictly below `core/`, and nothing reaches through the facade into a
store.** This is a v1 lesson that cost real rework: reach-through coupling
(`loop.companion.manager.sub.method()`) is easy to introduce and expensive to unwind. When it was
respected, swapping Postgres for SQLite touched only the store classes.

`bootstrap.py` is the **only** place anything is wired. Both entry points call `build()`.

Two related disciplines, same origin:
- **All prompt text lives in `core/prompts.py`.** No inline strings anywhere else.
- **Store contracts are Protocols** in `core/interfaces.py`. They're what makes the offline test
  suite possible — and a method a caller depends on but that isn't declared there is a fake
  waiting to pass vacuously.

---

## One turn, start to finish

`Companion.send()` — the path every user message takes:

1. **Wake if asleep.** Reload the model into VRAM first; the UI shows "waking up…".
2. **Recall (autonomic).** Embed the message with nomic's asymmetric `search_query:` prefix,
   brute-force cosine KNN over active memories. No tool call, no model choice — it just happens,
   like memory surfacing unbidden.
3. **Core facts.** Identity facts injected directly, subject to a sticky/cooldown window so they
   aren't in *literally* every prompt. The name bypasses that gate.
4. **Emotion (autonomic).** RoBERTa scores the message; 28 labels fold into 6 mood channels.
5. **Build the prompt.** `build_system()` assembles: persona → tools note → her self-description →
   learned self-notes → core facts → recalled facts → open intentions → mood → a terse
   end-of-prompt reminder.
6. **Generate.** Streams through the tool loop when tools are registered. She may reply `PASS`,
   which means silence — a gate holds the token stream so the word never flashes on screen.
7. **Log.** Both turns to the episodic store; the assistant turn is skipped on a silent turn.
8. **Relieve drives**, arm a possible follow-up, and maybe kick off consolidation.

Message shape is always `[system, ...history, user]`. Never inject an extra assistant turn —
local chat templates reject it. That's a v1 lesson with a crash behind it.

### Why the prompt is ordered that way

Small models follow the rules **nearest the generation point** far better than the same rules
buried 1,500 tokens up. That's why there's a terse reminder appended at the very end, and why the
persona was restructured rather than made louder. Three separate prompt bugs were fixed by
*moving* a rule to the end, not strengthening it.

---

## Memory

Three tiers, deliberately separate:

| Tier | Table | What it is | Read by |
|---|---|---|---|
| **Episodic** | `messages` | The **working** log — what she operates on | `reminisce`, consolidation |
| **Semantic** | `memories` | Distilled embedded facts | autonomic recall, every turn |
| **Core** | `memories.core` | Identity facts | injected directly, not searched |
| **Archive** | `message_archive` | Every message ever, **append-only** | nothing yet, by design |

**The archive is the one thing no admin operation clears.** `messages` is working state and a
factory reset wipes it — which is necessary, because testing needs a clean slate often. The
archive is the record of what was actually said, and losing that to a test reset would be
unrecoverable. It's deliberately FK-free and denormalized (sessions get deleted; an archive that
breaks when its parent row goes is not an archive).

An `era` column increments on each factory reset, so a future look-back can tell that a
discontinuity happened rather than reading across it as one continuous relationship — which
matters, since identity discontinuity is the best-documented way to damage this kind of
relationship (HANDOFF §H). The counter lives in the archive itself, not `meta`, because `meta` is
one of the things a reset clears.

Nothing reads it yet, on purpose: searching pre-reset conversations right after a reset would
defeat the reset. `Companion.search_archive()` and `archive_stats()` exist for when that changes.

**The episodic log is the most valuable thing here, and nothing ever deletes from it.** Research
in 2025–26 repeatedly found that LLM-driven consolidation degrades memory *below* a raw-episode
baseline — aggressive clustering into summaries measured 48.4% vs 78.4%. Distillation is additive;
the leaf survives. See HANDOFF §H.

**Consolidation** runs backgrounded at the end of a window (default 10 messages, or earlier when
the conversation is emotionally charged — see salience gating). It's batched: one extraction call,
one embeddings call, and one lifecycle-decision call covering every fact that resembles an
existing memory. Near-verbatim duplicates inside a window collapse with no model call.

**Lifecycle** decides per fact: *duplicate* (skip), *update* (soft-delete the old, link
`superseded_by`), or *new*. The critical rule, learned from a data-loss bug: update only when the
old fact becomes FALSE. A second dog is `new`, not a replacement.

**Durability.** A persisted watermark (`meta.last_consolidated_msg_id`) marks the last consolidated
message. The unconsolidated set is just "messages newer than the watermark", recovered from the
episodic log at startup — so a hard kill drops nothing. The watermark only advances on success,
and consolidation is serialized, so it can never jump a failed chunk.

**Retrieval is brute-force numpy KNN.** At personal scale (hundreds to low thousands) that's fine
and exact — no ANN recall loss. Don't optimise it.

---

## Emotion, drives, energy

**Mood** — 6 channels (irritation, warmth, amusement, melancholy, unease, interest), each with its
own decay rate toward a baseline. Persisted, so it survives restarts. Folded into the prompt to
colour tone; **never named literally**.

**Drives** — `connection` and `restlessness`, slow scalars in [0,1] that rise while you're away and
relax while you're present. They integrate by **elapsed wall-time**, not tick count, so behaviour
is independent of `TICK_INTERVAL` and deterministic under a fake clock. Mood modulates the rate:
warmth and melancholy speed `connection`, irritation slows it.

**These gate behaviour.** Reach-out fires when `connection` crosses its threshold, reflection when
`restlessness` does. Each discharges its drive on firing. The persisted cooldown remains a hard
floor. When drives are disabled everything falls back to the old idle gates.

**Energy** is different: it ignores idleness entirely and tracks only awake/asleep, depleting
while awake and restoring while asleep. It is deliberately **not** coupled to neglect — an agent
that visibly declines when ignored is the documented dependency hook (HANDOFF §G).

---

## The tick loop

`core/tick.py` — a pluggable job scheduler, not a hardcoded sequence. A job that raises is logged
and skipped; it never kills the heartbeat.

Three job shapes:

- **`Job`** — runs on its interval and self-gates. Mood drift, drive drift, idle consolidation,
  follow-up, sleep.
- **`CooldownJob`** — the shared protocol the autonomy jobs all wanted: asleep/busy guard → does
  she want to? → persisted wall-clock cooldown → **stamp the attempt** → act. Stamping *before*
  acting is load-bearing: it enforces the cooldown even when she declines, and stops a slow
  generation letting the next tick start a second one.
- **`DriveGatedJob`** — a `CooldownJob` whose trigger is a drive crossing a threshold instead of an
  idle timer.

A **busy guard** on `Companion` means `idle_seconds()` reads 0 during a turn, so nothing fires
mid-reply even during a slow generation.

Jobs needing a surface to push to (reach-out, follow-up) are registered by `web/app.py`, not
shared bootstrap — the REPL runs internal jobs only.

---

## Her inner life

Four private subsystems, all feeding back into her prompts:

| Subsystem | Storage | What it is |
|---|---|---|
| **Thought journal** | `thoughts` table | Short first-person reflections written during idle ticks, tagged with mood. Never shown in chat. |
| **Intentions** | `intentions` table | A forward agenda — things she means to raise. Reach-out anchors on the longest-waiting one; chat carries the agenda *softly*. |
| **Self-notes** | `meta.self_notes` | Operating-rules about how to BE with you, distilled from how you react. Injected into every user-facing prompt. |
| **Persona** | `meta.persona_self` | A self-description she rewrites during idle, gated by a familiarity meter so a stranger can't write herself into a best friend. |

Two guards exist because both failed in practice:

- **Repeat guard.** A reflection restating a recent one is rejected *programmatically*. The prompt
  already showed her recent thoughts and asked her not to repeat, and it didn't hold — measured
  RRR 0.26 with three byte-identical entries. Repetition in self-generated content predicts it's
  **wrong**, not important. Run `python scripts/rrr_diagnostic.py` any time.
- **Voice guard.** A self-note is addressed *to* her, so a correct one never names her. When
  `BOT_NAME` appears, the model wrote the note about the user instead — which would inject the
  lesson backwards — and the pass is dropped.

---

## Honesty: the three-tier rule

The persona used to assert both "you have feelings" and "you have no body", and the model resolved
that contradiction by either denying feelings or inventing experiences. It's now one rule:

| Tier | Rule |
|---|---|
| **Inner states** | Free. Moods, reactions, what stuck with her. |
| **Experiences** | Never. No walks, meals, weather, sleep, sensory events. |
| **Sincere questions** | Honest *under uncertainty* — not cold denial. |

The middle tier is enforced by **`core/embodiment.py`, a filter, not just the prompt**: roughly a
quarter of a base model's dialogue prior is machine-impossible utterances, so a prompt rule is
fighting the weights. It runs on **asides only** (reach-out, follow-up, reflection), where staying
quiet is already valid. Chat replies aren't filtered — a dropped reply is worse than a slightly
wrong one.

---

## Tools (Tier 3)

Capabilities split three ways by mechanism, and only the last needs function-calling:

1. **Autonomic** — recall, emotion. Pipeline stages; the model never chooses them.
2. **Deliberate-internal** — lifecycle decisions, persona edit, self-notes. Expressed by filling a
   JSON schema. Reliable locally.
3. **Deliberate-external** — real tool calls. The only tier gated on function-calling reliability.

This split de-risks the project: **memory works regardless of whether the model can call tools.**

`LLMClient.stream_with_tools` streams the answer and only loops when the model asks for a tool.
Every failure seam — malformed args, unknown tool, handler exception — becomes a result *string*
fed back to the model, never an aborted turn, with a `max_iters` cap so a turn can't hang.

Current tools: `get_current_time` and `reminisce` (deliberate episodic search, distinct from
autonomic recall). Routing is the constraint, not the framework — measured 23/30.

---

## Persistence

SQLite, one file, WAL, with a versioned migration runner keyed on `PRAGMA user_version`.
**Append to `MIGRATIONS`; never edit an existing entry.** Schema is at **v9**.

| v | Added |
|---|---|
| 1 | `sessions`, `messages` |
| 2 | `memories` |
| 3 | `superseded_by` |
| 4 | `meta` + the consolidation watermark |
| 5 | `thoughts` |
| 6 | `memories.core` |
| 7 | `sessions.title` (conversation tabs) |
| 8 | `intentions` |
| 9 | injection bookkeeping + `self_notes_log` |

**All five stores share one connection and therefore one write lock** (`SqliteStore` in `db.py`).
This is not cosmetic: per-store locks over a shared connection lost rows under concurrent writes —
measured 594/600 with `SystemError`s. See `tests/test_store_concurrency.py`.

Writes go through `asyncio.to_thread`; the one large read (`store.active()`) does too.

---

## Conversations

Sessions are named threads you can create, switch, rename and delete. **Mari is one companion
across all of them** — memory, mood, thoughts, persona, familiarity and the consolidation
machinery are global. Only the message thread is per-tab.
