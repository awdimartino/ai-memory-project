"""Offline coverage for labelled prompt blocks + the inspector's prompt log.

The prompt inspector's whole claim is "this is what was actually sent". That claim
rests on ONE invariant, which is what most of this file pins:

    join_blocks(system_blocks(...)) == build_system(...)

If those ever diverge, the inspector shows a plausible reconstruction instead of
the real prompt — and it lies precisely when you're using it to debug a strange
reply, which is worse than having no inspector at all.

Run:  python tests/test_prompt_blocks.py
"""
import itertools

from _harness import case, run  # also puts the repo root on sys.path

from core.prompts import (
    build_followup_system,
    build_reachout_system,
    build_system,
    followup_blocks,
    join_blocks,
    reachout_blocks,
    system_blocks,
)

# Every argument that can add, remove or reorder a block.
MATRIX = dict(
    memories=[[], ["fact one", "fact two"]],
    mood=[None, "You're feeling warm toward them."],
    core=[None, [], ["The user's name is Alex"]],
    persona=[None, "", "I've gotten more relaxed with them."],
    tools=[None, [], ["get_current_time", "reminisce"]],
    allow_silence=[False, True],
    intentions=[None, [], ["ask how the dye turned out"]],
    self_notes=[None, "", "He gets annoyed when you ask questions."],
)


def _combos():
    keys = list(MATRIX)
    for values in itertools.product(*(MATRIX[k] for k in keys)):
        yield dict(zip(keys, values))


@case
async def blocks_rejoin_to_the_exact_prompt():
    """THE invariant: the inspector's blocks must reassemble the real prompt."""
    n = 0
    for kw in _combos():
        mem, mood = kw.pop("memories"), kw.pop("mood")
        assert join_blocks(system_blocks(mem, mood, **kw)) == build_system(mem, mood, **kw), (
            f"blocks != build_system for {kw}")
        n += 1
    assert n > 500, f"matrix collapsed to {n} combinations — it should be exhaustive"


@case
async def aside_blocks_rejoin_too():
    """Reach-out and follow-up compose extra blocks on top; same invariant.

    Only reach-out takes an `intention` now: the follow-up path deliberately carries no
    agenda (2026-07-20), so the two are exercised with different keyword sets rather
    than a shared one.
    """
    for mem, mood, core, persona, intent, notes in itertools.product(
            MATRIX["memories"], MATRIX["mood"], MATRIX["core"], MATRIX["persona"],
            [None, "ask how it went"], MATRIX["self_notes"]):
        base = dict(core=core, persona=persona, self_notes=notes)
        ro = dict(base, intention=intent)
        assert join_blocks(reachout_blocks(mem, mood, **ro)) == build_reachout_system(mem, mood, **ro)
        assert join_blocks(followup_blocks(mem, mood, **base)) == build_followup_system(mem, mood, **base)


@case
async def labels_are_unique_and_nonempty():
    """Duplicate labels would make attribution ambiguous, which is the point of them."""
    for kw in _combos():
        mem, mood = kw.pop("memories"), kw.pop("mood")
        labels = [lbl for lbl, _ in system_blocks(mem, mood, **kw)]
        assert all(labels), "a block shipped without a label"
        assert len(labels) == len(set(labels)), f"duplicate block labels: {labels}"


@case
async def only_present_context_produces_a_block():
    """An absent input must not leave an empty labelled block behind."""
    bare = dict(system_blocks([], None))
    assert set(bare) == {"base persona", "closing reminder"}, f"unexpected bare blocks: {list(bare)}"

    full = dict(system_blocks(["a fact"], "mood text", core=["core fact"],
                              persona="persona text", tools=["reminisce"],
                              intentions=["an intention"], self_notes="a note"))
    for label in ("tools", "self-written persona", "learned self-notes", "core memory",
                  "recalled memories", "intentions", "mood"):
        assert label in full, f"{label} missing when its input was supplied"


@case
async def block_content_carries_its_input():
    blocks = dict(system_blocks(["oat milk"], None, core=["name is Alex"],
                                intentions=["ask about the dye"], self_notes="ease off"))
    assert "oat milk" in blocks["recalled memories"]
    assert "name is Alex" in blocks["core memory"]
    assert "ask about the dye" in blocks["intentions"]
    assert "ease off" in blocks["learned self-notes"]


@case
async def closing_reminder_is_last():
    """It's last on purpose — small models follow the rules nearest the end best."""
    for kw in _combos():
        mem, mood = kw.pop("memories"), kw.pop("mood")
        assert system_blocks(mem, mood, **kw)[-1][0] == "closing reminder"


# --- relationship stage (the familiarity fix) ------------------------------------

@case
async def stage_zero_is_the_old_persona_unchanged():
    """A fresh companion must see EXACTLY what it saw before this feature existed.

    That's the safety net for the whole change: only long-lived relationships get
    new text, so the gold set (which runs at message_count 0) is a pure regression
    check rather than a measurement of the new stages.
    """
    from core.prompts import SYSTEM_PROMPT, build_persona
    assert build_persona(0.0) == SYSTEM_PROMPT
    assert dict(system_blocks([], None))["base persona"] == SYSTEM_PROMPT


@case
async def every_stage_keeps_the_anti_confabulation_rule():
    """What changes is the JUSTIFICATION, never the rule itself."""
    from core.prompts import RELATIONSHIP_STAGES

    for bound, label, opening, rule in RELATIONSHIP_STAGES:
        flat = " ".join(rule.split())          # the rule wraps across lines
        assert "invent shared history" in flat, f"{label}: lost the anti-confabulation rule"
        assert "claim" in flat, f"{label}: lost the false-claim handling"


@case
async def only_the_stranger_stage_denies_a_shared_past():
    """The bug: telling her she 'just met' someone she's talked to for months."""
    from core.prompts import RELATIONSHIP_STAGES

    denies = [i for i, (_b, _l, opening, rule) in enumerate(RELATIONSHIP_STAGES)
              if "just met" in " ".join((opening + rule).split())]
    assert denies == [0], f"stages denying a shared past: {denies} (should be only stage 0)"


@case
async def stage_bands_match_the_familiarity_label():
    """The panel's label and the prompt's framing must come from one table."""
    from core.companion import familiarity_label
    from core.prompts import relationship_stage

    for f in (0.0, 0.149, 0.15, 0.399, 0.40, 0.699, 0.70, 0.899, 0.90, 1.0):
        assert familiarity_label(f) == relationship_stage(f)[0], f"labels diverge at {f}"


@case
async def stage_is_quantised_so_the_kv_prefix_is_stable():
    """The persona sits in the CACHED PREFIX, so it must be piecewise-constant.

    Measured (scripts/prefix_cache_probe.py): changing the first character of a
    ~1500-token prompt costs 3.43s TTFT vs 0.42s for a cache hit. A stage that
    changed with every message would pay that on every single turn; five buckets pay
    it four times in the life of a relationship. This test fails if anyone
    interpolates the raw scalar or a message count into the persona.
    """
    from core.prompts import build_persona

    seen = {build_persona(i / 200) for i in range(201)}   # 201 distinct familiarities
    assert len(seen) == 5, f"persona takes {len(seen)} distinct forms, expected 5 buckets"


@case
async def familiarity_reaches_every_generation_path():
    from core.prompts import (build_persona, followup_blocks, reachout_blocks,
                              reflect_blocks, system_blocks)

    close = build_persona(1.0)
    assert close != build_persona(0.0), "stage 1.0 should differ from stage 0"
    for name, blocks in (
            ("chat", system_blocks([], None, familiarity=1.0)),
            ("reach-out", reachout_blocks([], None, familiarity=1.0)),
            ("follow-up", followup_blocks([], None, familiarity=1.0)),
            ("reflection", reflect_blocks([], None, [], familiarity=1.0))):
        assert dict(blocks)["base persona"] == close, f"{name} ignored familiarity"


# --- the inspector's prompt log --------------------------------------------------

class _FakeMeta:
    def get(self, *_a, **_k):
        return None


def _companion():
    """A Companion built with everything optional switched off — the log is pure state."""
    from core.companion import Companion

    return Companion(llm=None, store=None, memory=None, meta=_FakeMeta(), session_id=1)


@case
async def prompt_log_is_newest_first():
    c = _companion()
    for i in range(3):
        c._record_prompt("chat", [("base persona", f"p{i}")], [{"role": "user", "content": str(i)}])
    kinds = [p["messages"][0]["content"] for p in c.prompt_log()]
    assert kinds == ["2", "1", "0"], f"log not newest-first: {kinds}"


@case
async def prompt_log_is_bounded():
    """It's a debugging window, not a record — it must not grow forever."""
    import config

    c = _companion()
    for i in range(config.PROMPT_LOG_MAX + 8):
        c._record_prompt("chat", [], [{"role": "user", "content": str(i)}])
    log = c.prompt_log()
    assert len(log) == config.PROMPT_LOG_MAX, f"log grew to {len(log)}"
    assert log[0]["messages"][0]["content"] == str(config.PROMPT_LOG_MAX + 7), "dropped the newest"


@case
async def record_is_returned_so_the_reply_can_be_attached():
    c = _companion()
    rec = c._record_prompt("reach-out", [("mood", "warm")], [])
    assert rec["reply"] is None, "reply should start unset (None == stayed quiet)"
    rec["reply"] = "hey"
    assert c.prompt_log()[0]["reply"] == "hey", "record isn't the one in the log"


@case
async def every_generation_path_records_a_prompt():
    """All four user-facing generations must reach `_aside`/`_record_prompt` correctly.

    This is the regression test for a real shipped bug: `reflect()` still called
    `_aside(system, cue)` after the signature became `(blocks, cue, kind)`, so every
    reflection raised TypeError. It survived because `test_tick.py` exercises
    `ReflectJob` against a FAKE companion — the job was covered, the method wasn't.
    Reach-out and follow-up were covered end-to-end elsewhere, which is exactly why
    those two were caught and this one wasn't.

    The tick loop swallows job exceptions ("one bad job must not stop the
    heartbeat"), so the only symptom was that she quietly stopped journaling — the
    invisible failure mode this test exists to prevent.
    """
    import asyncio

    from helpers import FakeConv, FakeMemory, InMemoryMeta

    from core.companion import Companion

    REPLY = "wednesdays have a strange shape to them"

    class FakeLLM:
        model = "fake"

        async def stream(self, messages, on_token):
            # Deliberately unlike the seeded recent thought below: reflect() drops a
            # reply that restates one (the programmatic repeat guard).
            await on_token(REPLY)
            return REPLY, {"tokens": 1}

    class FakeThoughts:
        def __init__(self):
            self.added = []

        def recent(self, _n):
            return [{"content": "an earlier thought"}]

        def add(self, text, dom=None):
            self.added.append(text)

    async def noop(_t):
        pass

    def build():
        return Companion(FakeLLM(), FakeConv(), FakeMemory(), InMemoryMeta(), 1,
                         history=[{"role": "user", "content": "hey"}],
                         thoughts=FakeThoughts())

    c = build()
    await c.send("hey", noop)
    assert c.prompt_log()[0]["kind"] == "chat", "send() did not record a chat prompt"

    c = build()
    await c.reach_out()
    assert c.prompt_log()[0]["kind"] == "reach-out", "reach_out() did not record"

    c = Companion(FakeLLM(), FakeConv(), FakeMemory(), InMemoryMeta(), 1,
                  history=[{"role": "assistant", "content": "hi"}], thoughts=FakeThoughts())
    await c.follow_up()
    assert c.prompt_log()[0]["kind"] == "follow-up", "follow_up() did not record"

    c = build()
    out = await c.reflect()          # the path that was broken
    assert out == REPLY, f"reflect() failed: {out!r}"
    rec = c.prompt_log()[0]
    assert rec["kind"] == "reflection", f"reflect() recorded {rec['kind']!r}"
    labels = [b["label"] for b in rec["blocks"]]
    assert "reflection framing" in labels, f"reflection blocks missing framing: {labels}"
    assert "recent thoughts" in labels, f"recent thoughts not carried: {labels}"
    # And the blocks must still rejoin to what was actually sent.
    assert join_blocks([(b["label"], b["text"]) for b in rec["blocks"]]) == \
        rec["messages"][0]["content"], "reflection blocks don't match the sent system message"


@case
async def messages_are_snapshotted_not_aliased():
    """The log must show what was SENT, not whatever the list became afterwards."""
    c = _companion()
    messages = [{"role": "user", "content": "original"}]
    c._record_prompt("chat", [], messages)
    messages[0]["content"] = "mutated later"
    messages.append({"role": "user", "content": "appended later"})
    logged = c.prompt_log()[0]["messages"]
    assert len(logged) == 1 and logged[0]["content"] == "original", f"log aliased the caller's list: {logged}"


if __name__ == "__main__":
    raise SystemExit(run())
