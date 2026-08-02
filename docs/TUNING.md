# Tuning

97 knobs live in `config.py`, all overridable from `.env`. This is the reverse index:
**symptom → what to change.**

> ⚠️ **Most of these numbers are reasoned, not measured.** Where a value was actually
> measured it says so. Before and after any tuning that's meant to change behaviour, run the
> gold set (`evals/README.md`) — otherwise you're trading one guess for another.

---

## She sounds like she's reciting facts about me

Core memories used to be injected on **every single turn**, which is the structural cause of
this — no amount of persona wording fixes a fact that is always present.

| Knob | Default | Effect |
|---|---|---|
| `CORE_STICKY_TURNS` | 3 | Once injected, stays this many turns so it can't flicker out mid-topic |
| `CORE_COOLDOWN_TURNS` | 8 | After that, held back until this many turns have passed |
| `CORE_ALWAYS_PATTERN` | `name is` | Facts matching this bypass the gate entirely |

**Still reciting?** Raise `CORE_COOLDOWN_TURNS`.
**Seems to have forgotten something obvious?** Raise `CORE_STICKY_TURNS`.

Both defaults are guesses. The name bypass exists because rotating someone's name out of context
is a downgrade, not variety.

---

## She doesn't remember something I told her

Work down this list — the causes are quite different:

1. **Was it ever extracted?** Check the memory inspector (header → "memory"), or
   `scripts/eval_extraction.py`. Extraction deliberately ignores transient states ("tired
   today"), anything about the companion or the app, and one-off plans.
2. **Was it a *big, noisy* window?** A lone durable fact drowns in banter — this is the
   documented "name the user was dropped" bug. `CONSOLIDATE_WINDOW` (10) is small for this reason;
   don't raise it much.
3. **Is recall failing to surface it?** This was the most likely cause until 2026-07-20, when a
   **contrast gate** fixed it (`RECALL_CONTRAST_GAP`). If you're tuning recall, read this first —
   the intuitive knob is the wrong one.

   The measurement (`python scripts/recall_margin_probe.py`, 12 positive + 13 negative queries
   over the gold set's six facts) killed the obvious fix: **the absolute score barely
   discriminates at all.**

   | | range |
   |---|---|
   | correct top-1 hits | 0.448 – 0.644 |
   | unrelated queries | 0.424 – 0.579 |

   Those overlap almost completely, so **no floor separates them**. At 0.55 the correct fact was
   the top hit every time and thrown away anyway (4/12 recall) — while "what's your favourite
   colour?" scored 0.576 and got in.

   ⚠️ **Don't lower `RECALL_MIN_SIMILARITY` to 0.50.** It scores worse on both counts than the
   gate, because it's the wrong instrument, not the wrong number.

   What works is **contrast**: how far the top hit stands above the corpus *median*. nomic gives
   each query its own baseline offset (some score ~0.5 against everything), and subtracting the
   median cancels it. Same measurement: **11/12 recall**. Knobs:

   - `RECALL_CONTRAST_GAP` (0.06) — how far above the median a hit must stand. **This is the
     precision/recall dial.** Lower admits adjacent topics ("getting a cat" needs 0.054); higher
     starts dropping real hits ("long shift at the shop" → welder sits at 0.061).
   - `RECALL_CONTRAST_FLOOR` (0.42) — backstop; standing out means little when everything scores badly.
   - `RECALL_CONTRAST_MIN_CORPUS` (3) — below this the median *is* the top hit, so only the floor applies.

   **Still imperfect, and known:** adjacent topics are what leak through — "my brother never calls
   me back" pulls in "sister is called Kate". No threshold on a single similarity statistic
   separates those from true positives; the fix is lexical evidence (**hybrid BM25 + vector**, the
   documented next step), since "brother" shares no words with the fact while "I should call my
   sister" shares two. Note the cost is bounded: recalled facts are injected as *"things that
   might be relevant… use them when they fit"*, so a false positive is a true fact the companion can
   ignore — a possible non-sequitur, not a fabrication.
4. **Should it be core?** Toggle the star in the inspector, or raise `CORE_MEMORY_MAX` (12).

---

## Memory is filling with junk

- `CONSOLIDATE_SALIENCE` (2.0) — an emotionally charged window consolidates *early*; a flat one
  waits for the full window. **Raise** to require a stronger signal.
- `SALIENCE_MIN_MESSAGES` (4) — never fire below this, however charged.
- The window remains a hard ceiling, so salience can only make her save **sooner, never less**.
- If the junk is *wrong* rather than trivial, that's the extraction prompt in `core/prompts.py`,
  not a knob.

---

## She replies too long / asks too many questions

These are **prompt** properties, not knobs. The rules live in `SYSTEM_PROMPT` and in the terse
reminder appended at the very end of `build_system()`.

**The single most useful thing to know here:** when a prompt rule isn't holding, the fix is
almost always **moving it nearer the end**, not making it louder. That failure recurred three
times during the build.

⚠️ The documented "100% one-sentence / 7% q-end" figures **did not reproduce** on 2026-07-20 —
the same prompt measured 22% q-end in a fresh session. `scripts/bakeoff_personality.py` at n=18
cannot resolve small differences. Treat those numbers as indicative and re-measure with the gold
set.

Sampling knobs that also matter: `TEMPERATURE` (0.8), `FREQUENCY_PENALTY` (0.4),
`PRESENCE_PENALTY` (0.3). The penalties are anti-repetition; research suggests 0.05–0.1 is more
typical for personality, but 0.4 was chosen deliberately and hasn't been re-tested.

---

## She reaches out too much / not enough

Reach-out fires on the **`connection` drive**, not a timer — so mood affects the timing.

| Knob | Default | Effect |
|---|---|---|
| `DRIVE_CONNECTION_THRESHOLD` | 0.6 | Higher = reaches out less often |
| `REACHOUT_COOLDOWN` | — | A hard floor, persisted across restarts |
| `REACHOUT_MIN_IDLE` | — | Fallback gate when drives are disabled |
| `DRIVES_ENABLED` | true | Off → everything reverts to idle timers |

Rise rates are module constants in `core/drives.py`, not env vars. `connection` is ~2.4/hr;
`restlessness` was cut 15→5/hr after it pegged at 1.0 within 4 minutes.

⚠️ **Read HANDOFF §H before tuning this.** Cooldown limits *volume*, but volume is not the
failure mode — **predictability** is. A digital-health deployment saw responsiveness fall 93% →
47% over 8 weeks from habituation, while messenger notifications sustain 65+/day with *positive*
affect. Varying the trigger and the shape beats throttling.

---

## She's quiet too often

Silent turns are **ungated on purpose** — she can `PASS` on anything. There's no knob. If it's
too frequent, add mood or low-effort gating in `Companion.send()`.

Follow-ups (the "double-text") do have knobs: `FOLLOWUP_CHANCE` (0.2), `FOLLOWUP_WINDOW` (60s),
`FOLLOWUP_MAX_PER_TURN` (1), `FOLLOWUP_MIN_DELAY`.

---

## She journals the same thought repeatedly

Diagnose first: `python scripts/rrr_diagnostic.py` — offline, read-only, safe any time.

A restated reflection is now rejected programmatically. The threshold is `REPEAT_THRESHOLD` in
`core/textsim.py` (0.40), **calibrated against 24 real journal entries** plus cost asymmetry:
rejecting a fresh thought costs one skipped entry; storing a repeat pollutes the journal that
feeds the persona edit.

**If she stops journaling entirely, it's too aggressive** — raise it.

Remember what the rate *means*: repetition in self-generated content predicts the content is
**wrong**, not that it's important.

---

## She invents things she did

`core/embodiment.py` filters asides. It's lexical and conservative, and it will miss novel
phrasings — add a pattern to `_EMBODIED` when you see one, and check `_ALLOWED` still exempts
idioms like "I see what you mean" and "that's been sitting with me".

It runs on **asides only**. Chat replies are unfiltered, deliberately: a dropped reply is worse
than a slightly wrong one, whereas a dropped reach-out costs nothing.

The persona side is the three-tier rule in `SYSTEM_PROMPT` — states freely, experiences never,
honest-under-uncertainty when sincerely asked.

---

## She won't use a tool / uses one when she shouldn't

Measured 23/30 (`scripts/tool_eval.py`, TIME 6/8, REMINISCE 5/8). **The eval is noisy — run it
×3.**

- **Under-calling TIME** is temperature: measured **0/4 at 0.8 vs 2/4 at 0.2**. Prong A
  (per-call temperature) is the intended fix and is *unbuilt*. Note it's a **partial** fix —
  §0 records 7/8 at 0.2, which did not reproduce.
- **Under-calling REMINISCE** is a knowledge problem: she doesn't know there's anything to look
  up. HANDOFF §E's always-on index is likely a better lever than temperature.
- **Over-triggering** is currently clean (TRICKY 6/6). Guard it when changing the tools note.
- `TOOLS_ENABLED=false` disables the whole tier at zero cost.

---

## She's slow

- **Check reasoning is off first.** qwen3.5-9b spends ~2,000 hidden tokens per structured call
  otherwise, making consolidation ~20× slower. Requires an LM Studio template edit
  (`{% set enable_thinking = false %}`) — see HANDOFF §4. Verify with
  `scripts/probe_reasoning_control.py`; the baseline should read ~0.
- Consolidation is backgrounded and never blocks chat. If it *feels* slow, it's the chat path.
- `LLM_MAX_RETRIES` (3) — LM Studio intermittently 400s; chat only retries before the first
  visible token.
- Don't bother with speculative decoding: measured +27% on predictable text, **−50% on
  creative**, and chat is the creative path.

---

## Notifications

- `NOTIFY_URL` — Bark device endpoint. Unset = the whole feature is a no-op.
- `NOTIFY_UI_URL` — tap-to-open deep link (Tailscale).
- `NOTIFY_ICON` — served from `web/static/`; iOS caches it by URL.
- `POST /admin/test_notify` fires one on demand.

Push fires whenever the chat **isn't in front of you** — closed, backgrounded, or no tab at all.
The tab reports visibility over the WebSocket. Gotcha: the Bark device key must be issued *by*
your self-hosted server, not `api.day.app`.

---

## Sleep

- `SLEEP_AFTER_IDLE` (30 min) — the VRAM-freeing trigger.
- `ENERGY_SLEEP_THRESHOLD` (0.15) + `ENERGY_SLEEP_MIN_IDLE` (120s) — the "she's tired" trigger.
- Rates (`ENERGY_DEPLETE_PER_HOUR` 0.07 / `ENERGY_RESTORE_PER_HOUR` 0.15) are module constants
  in `core/drives.py`, modelling roughly a 14-hour day.
- Auto-disables if the `lms` CLI isn't on PATH.

Because the defaults are slow, energy-sleep rarely beats the 30-minute idle trigger in short
sessions. Its real payoff is as the gate for a future self-wake.

---

## She mines the wrong open questions / never mines any (§A3)

`PursuitMiningJob` reads the recent window and asks for at most 3 salient open questions
about you, each cited to a real message id. An uncited question, or one citing an id outside
the fetched window, is rejected **before storage** — not a prompt request, a hard check in
`Companion.mine_open_questions()`.

| Knob | Default | Effect |
|---|---|---|
| `PURSUIT_MINING_ENABLED` | true | Off disables the job entirely |
| `PURSUIT_MINING_MIN_IDLE` | 1800s | How long you must be away first |
| `PURSUIT_MINING_COOLDOWN` | 86400s | The "nightly deep pass" cadence |
| `PURSUIT_WINDOW_MESSAGES` | 40 | How far back to mine |
| `PURSUIT_MIN_MESSAGES` | 8 | Never mine a window this thin (a barren-window guard, same shape as the extraction/self-notes/intentions bugs this project already fixed twice) |
| `PURSUIT_MAX_ACTIVE` | 5 | Caps the open backlog (oldest dropped) |
| `PURSUIT_SALIENCE` | = `CONSOLIDATE_SALIENCE` | Reuses the same "did something actually happen" floor |

**Nothing being mined is usually correct** — most windows are small talk with no real
open thread. Only worry if a window that clearly needed follow-up produced nothing;
check `PURSUIT_MIN_MESSAGES` isn't being tripped first.

---

## She makes herself unavailable too often / not at all / at the wrong times (§A4)

**Off by default** (`PURSUIT_UNAVAILABLE_ENABLED=false`) — this is a new, live-untested
feature; turn it on deliberately.

She steps away only to do one of a **closed set of real things** (journal, revise her
self-notes/persona, or sit with an A3-mined open question) — never an invented errand.
Gated on **`restlessness`**, not `connection`: restlessness is idle/boredom-driven, not
warmth-driven, so the trigger can't correlate with how invested you are in the conversation
(the property that would turn this into the manipulative pattern HANDOFF's research sweep
warns about — see §G there before loosening this).

| Knob | Default | Effect |
|---|---|---|
| `PURSUIT_UNAVAILABLE_ENABLED` | **false** | The kill switch |
| `PURSUIT_UNAVAILABLE_MIN_IDLE` | 900s | Fallback idle gate (drives are the real gate) |
| `PURSUIT_UNAVAILABLE_COOLDOWN` | 21600s (6h) | Hard floor over the drive |
| `PURSUIT_UNAVAILABLE_THRESHOLD` | 0.8 | Higher than `DRIVE_RESTLESSNESS_THRESHOLD` (0.4, journaling's own gate) — stepping away entirely is a bigger deal than a private journal entry |
| `PURSUIT_UNAVAILABLE_CALM_CEILING` | = `CONSOLIDATE_SALIENCE` | Refuses to fire if something heavy just happened (same signal A3 uses, as a ceiling instead of a floor) |
| `PURSUIT_JOURNAL_MIN/MAX`, `PURSUIT_SELFNOTES_MIN/MAX`, `PURSUIT_PERSONA_MIN/MAX`, `PURSUIT_SIT_MIN/MAX` | seconds | Per-pursuit "educated random" duration range — all minutes, never hours |

**On interrupt: it's fine to make you wait.** A message sent while she's away gets a canned
acknowledgment (no model call, works even with the model unloaded) and the real reply arrives
once the window ends — see `PursuitReturnJob`. This was a deliberate product choice, not a
limitation: raise it with whoever owns this app before "fixing" it to auto-interrupt.

Not yet measured live: whether the thresholds above actually produce a good rhythm. Start
conservative (raise the cooldown / threshold further) and loosen only after watching it fire
a few times for real.

---

## I need to wipe her for testing, but keep the conversations

You can. **A factory reset does not destroy conversation history.**

| Operation | Wipes | Keeps |
|---|---|---|
| `POST /memory/clear` | semantic memories | conversations, mood, persona, thoughts |
| `POST /admin/factory_reset` | memories, working log, sessions, thoughts, intentions, all meta | **`message_archive` — every message ever** |

Every message is written to both the working log and an append-only archive in the same
transaction. Clearing touches only the working log. Each reset opens a new `era`, so the record
shows where the discontinuity was.

```bash
python -c "import sys;sys.path.insert(0,'.');from infrastructure.db import connect;from infrastructure.conversation_store import SqliteConversationStore as S;print(S(connect('companion.db')).archive_eras())"
```

`GET /status` reports `archive.total` and a row per era. Nothing reads the archive back into her
context yet — that's deliberate, since recalling pre-reset conversations immediately after a reset
would defeat the point of resetting.

⚠️ **Backup note:** the DB runs in WAL mode. Copying `companion.db` alone can miss recent commits
still in `companion.db-wal` — copy all three files, or run `PRAGMA wal_checkpoint(TRUNCATE)` first.

## Things that are NOT knobs

Worth knowing so you don't go looking:

- **Drive rise rates, energy rates, mood decay rates** — module constants.
- **Personality, backbone, the honesty rule** — `core/prompts.py`.
- **Whether she can be silent** — hardcoded as always allowed.
- **The layering rules** — architectural, see ARCHITECTURE.md.
