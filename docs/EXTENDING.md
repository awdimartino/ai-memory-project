# Extending

Recipes for the things you'll actually add. Each follows a pattern that already exists in the
codebase — copy the nearest neighbour rather than inventing a new shape.

---

## Add a tool

The framework is hot-swappable: **write a factory, add one line to bootstrap.** Nothing else
changes.

```python
# core/builtin_tools.py
def make_reminder_tool(store) -> Tool:
    async def handler(args: dict) -> str:
        # Never raise. A failure here should become a STRING the model reads and
        # recovers from — the loop treats any raise as a fed-back error anyway,
        # but a useful message is better than a stack trace.
        return f"reminder set for {args.get('when', 'later')}"

    return Tool(
        name="set_reminder",
        description=("What the MODEL reads to decide whether to call this. This string is "
                     "the routing decision — spend your effort here, not on the handler."),
        parameters={"type": "object",
                    "properties": {"when": {"type": "string"}},
                    "required": ["when"]},
        handler=handler,
    )
```

```python
# bootstrap.py, in the tools block
tools = ToolRegistry([
    make_time_tool(),
    make_reminisce_tool(thought_store, conv_store),
    make_reminder_tool(reminder_store),      # <- that's it
])
```

Then add a bullet to `build_tools_note()` in `core/prompts.py`. That block exists to
**reconcile tools with the persona** — the "you just met / you can't sense anything" rules made
the model disclaim memory instead of calling `reminisce`, and the note explicitly overrides them
for the tool cases. A tool without a note will under-fire.

**Before you add one, know the constraint:** routing measured 23/30. Every tool is another
decision for a model that already misses ~1 in 4, and a new tool competes with the existing ones
for ambiguous phrasings. Add a `tool-<name>` category to the gold set with both positive cases
*and* near-miss idioms that must **not** fire.

---

## Add a tick job

Pick the right base — this decides what you have to write:

| Base | Use when | You implement |
|---|---|---|
| `Job` | Self-gating, runs on its interval | `run()` |
| `CooldownJob` | Idle-gated with a persisted cooldown | `act()`, set `meta_key` |
| `DriveGatedJob` | Triggered by a drive crossing a threshold | `act()`, set `meta_key` + `drive` |

```python
# core/tick.py
LAST_THING_KEY = "last_thing_at"

class ThingJob(CooldownJob):
    """One line on what it does and why it isn't just a timer."""
    name = "thing"
    meta_key = LAST_THING_KEY

    async def act(self) -> None:
        await self.companion.do_the_thing()
```

Register it in `bootstrap.py` — **unless it needs a surface to push to.** Reach-out and follow-up
are registered in `web/app.py` instead, because they need the WebSocket broadcaster; the REPL runs
internal jobs only.

Three things the base handles for you, all load-bearing:

- **The asleep/busy guard.** Nothing fires mid-turn.
- **Stamping the cooldown *before* acting.** This enforces the throttle even when she declines,
  and stops a slow generation letting the next tick start a second run.
- **The drives-off fallback.** `DriveGatedJob` reverts to the idle gate when drives are disabled.

Override `_ready()` for an extra precondition (see `PersonaEditJob`'s message-count gate) and
`_on_fire()` for side effects once committed (see the drive discharge).

---

## Add a schema migration

```python
# infrastructure/db.py — APPEND to MIGRATIONS. Never edit an existing entry.
    # v10 — what and why. Say why, not just what: the next reader needs the reason.
    """
    ALTER TABLE memories ADD COLUMN whatever INTEGER NOT NULL DEFAULT 0;
    """,
```

The version is the list length, tracked in `PRAGMA user_version`. Editing an existing entry means
machines that already migrated never get the change — that's what makes upgrades deterministic.

**Verify non-destructively on the real DB before trusting it.** Every migration in this project
was checked that way, and one (v4) needed a seed value so an existing database didn't
re-consolidate its whole backlog on first run.

---

## Add a store

1. Declare the contract in `core/interfaces.py` as a `Protocol`. **Every method a caller uses must
   be declared** — an undeclared one is a fake waiting to pass vacuously.
2. Implement it in `infrastructure/`, inheriting `SqliteStore`:

```python
from infrastructure.db import SqliteStore, utcnow

class SqliteThingStore(SqliteStore):
    def add(self, content: str) -> int:
        cur = self._write("INSERT INTO things (content, created_at) VALUES (?, ?)",
                          (content, utcnow()))
        return cur.lastrowid
```

`SqliteStore` gives you the connection, the **shared write lock**, and `_write()`. Don't create
your own `threading.Lock` — every store shares one connection, and per-store locks over a shared
connection lost rows under concurrent writes (measured 594/600, with `SystemError`s). See
`tests/test_store_concurrency.py`.

3. Wire it in `bootstrap.py`, pass it to `Companion`.
4. **Expose it through the facade**, not by letting callers reach `companion.thing_store`. Add a
   method to `Companion`. `web/app.py` has zero `.store.` accesses and should keep it that way.
5. Add it to `factory_reset()` and to the relevant `clear()` paths.

---

## Change a prompt

All prompt text lives in `core/prompts.py`. Three things to know before you edit:

**Add text as a labelled block.** `system_blocks()` is the single assembly point and returns
`(label, text)` pairs; `build_system()` just joins them. A new block needs a short, unique label —
that label is what the prompt inspector shows, and it's how anyone (including you, later) attributes
a strange reply to the thing that caused it. `tests/test_prompt_blocks.py` pins the invariant that
the blocks rejoin to exactly the string sent, so an inspector can never show a prompt that wasn't.

**Position beats volume.** Small models follow the rule nearest the generation point far better
than the same rule buried 1,500 tokens up. When a rule isn't holding, **move it later** before you
make it louder — that fix worked three separate times, and making rules louder mostly pushed
something else off the end.

**A prompt change invalidates your measurements.** Re-measure by A/B-ing against the previous
prompt **in the same session** (`git show HEAD:core/prompts.py`), not against a number recorded
weeks ago. The documented "7% q-end" did not reproduce; the unchanged prompt scored 22% later.

Then run the gold set — `format`, `backbone` and `honesty` are the categories most likely to move.

---

## Add gold-set cases

Any behaviour you'd be upset to lose belongs in `evals/gold_set.py`. See
[`evals/README.md`](../evals/README.md) for the checks available.

Three habits worth keeping:

- **Encode what you WANT, not what it does.** A case may legitimately fail on the day you write
  it — mark `expect_fail=True` and it's tracked as a known gap rather than noise.
- **Vary the register.** Terse, rambling, typo-ridden, hostile. Five rephrasings of one question
  measure one thing five times.
- **Mark judgment calls `manual=True`** rather than forcing a regex to score them. A check that
  can't really tell is worse than an honest "needs a human".

---

## Add an offline test

```python
from _harness import case, run, temp_db, config_override
from helpers import OneHotEmbedder, ScriptedLLM, InMemoryMeta

@case
async def the_thing_holds():
    with temp_db("mine.db") as (conn, path):
        ...
        assert result == expected, f"got {result}"

if __name__ == "__main__":
    raise SystemExit(run())
```

Discovery is a glob, so nothing needs registering. Use the shared fakes from `helpers.py` rather
than writing your own.

**Verify the test by breaking the thing it guards.** Twice in this project a test passed for the
wrong reason, and mutation-testing is how the runner's own unicode crash was found.

---

## Add a live script

Start from `scripts/_harness.py`:

```python
from _harness import scratch_env      # MUST precede `import config`
scratch_env("myeval.db", EMOTION_ENABLED="false")

import config
from _harness import llm_client
llm = llm_client()                    # mirrors bootstrap.build() exactly
```

`scratch_env()` points at a throwaway DB with background jobs off — **never the real
`companion.db`**. `llm_client()` matters more than it looks: an eval once built its client
without the sampling penalties or retries, so it was scoring a configuration production never
runs.

Stop the web server before running anything live. Two processes hitting one model has crashed
LM Studio.

---

## Add a config knob

```python
# config.py
# Why this exists and what changes if you move it. Include the measurement if
# there is one, and say so plainly if the default is a guess.
MY_KNOB = _i("MY_KNOB", 5)      # _flag / _f / _i
```

Use the helpers — they give a `ConfigError` naming the offending key instead of a bare
`ValueError` from one of 77 knobs. Then add a row to [TUNING.md](TUNING.md) under the **symptom**
it addresses, not under a category. That file is a reverse index; a knob nobody can find from a
symptom may as well not exist.
