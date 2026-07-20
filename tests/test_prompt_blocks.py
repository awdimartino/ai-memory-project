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
    """Reach-out and follow-up compose extra blocks on top; same invariant."""
    for mem, mood, core, persona, intent, notes in itertools.product(
            MATRIX["memories"], MATRIX["mood"], MATRIX["core"], MATRIX["persona"],
            [None, "ask how it went"], MATRIX["self_notes"]):
        kw = dict(core=core, persona=persona, intention=intent, self_notes=notes)
        assert join_blocks(reachout_blocks(mem, mood, **kw)) == build_reachout_system(mem, mood, **kw)
        assert join_blocks(followup_blocks(mem, mood, **kw)) == build_followup_system(mem, mood, **kw)


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
