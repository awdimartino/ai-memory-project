"""Offline coverage for learned self-notes (experience -> principle -> future behavior):
update_self_notes (write / PASS / cap / empty-history), injection into every user-facing prompt
(chat + reach-out + follow-up), and the idle+cooldown gating of SelfNotesJob.

Uses a real Companion wired to tiny fakes, plus a fake clock for the job gates.
Run:  python tests/test_self_notes.py
"""
import asyncio

import _harness  # noqa: F401  — imported for its sys.path side effect
from helpers import FakeConv, FakeMemory, InMemoryMeta

import config  # noqa: E402
from core.companion import SELF_NOTES_KEY, Companion  # noqa: E402
from core.prompts import (  # noqa: E402
    build_followup_system, build_reachout_system, build_self_notes_system, build_self_notes_user,
    build_system)
from core.tick import LAST_SELFNOTES_KEY, SelfNotesJob  # noqa: E402

_checks = 0


def check(cond, msg):
    global _checks
    assert cond, msg
    _checks += 1


class FakeLLM:
    def __init__(self, reply="You ask him too many questions — ease off."):
        self.model = "fake"
        self._reply = reply
        self.seen = []          # every system prompt this LLM was handed
        self.asked = []         # ...and the user turn that came with it

    async def stream(self, messages, on_token):
        self.seen.append(messages[0]["content"])
        self.asked.append(messages[-1]["content"])
        await on_token(self._reply)
        return self._reply, {"ttft": 0, "tok_per_s": 0, "tokens": 1, "estimated": True}

    async def structured_json(self, messages, schema, model=None):
        return {"intentions": [], "resolved": []}


HIST = [{"role": "user", "content": "stop asking me so many questions"},
        {"role": "assistant", "content": "fair enough"}]


def _companion(llm, meta, history=HIST):
    return Companion(llm, FakeConv(), FakeMemory(), meta, 1, history=history)


async def main():
    # --- 1) update_self_notes writes the slot ---
    meta = InMemoryMeta()
    llm = FakeLLM()
    out = await _companion(llm, meta).update_self_notes()
    check(out == "You ask him too many questions — ease off.", f"returns the new notes: {out}")
    check(meta.get(SELF_NOTES_KEY) == out, "persisted to the meta slot")
    check("stop asking me so many questions" in llm.asked[0], "the conversation is what it learns from")

    # the current notes are fed back in so a rewrite can revise rather than duplicate
    meta = InMemoryMeta({SELF_NOTES_KEY: "He likes short replies."})
    llm = FakeLLM()
    await _companion(llm, meta).update_self_notes()
    check("He likes short replies." in llm.asked[0], "existing notes are shown for revision")

    # --- 2) PASS leaves the slot untouched ---
    meta = InMemoryMeta({SELF_NOTES_KEY: "He likes short replies."})
    check(await _companion(FakeLLM("PASS"), meta).update_self_notes() is None, "PASS -> None")
    check(meta.get(SELF_NOTES_KEY) == "He likes short replies.", "PASS does not clobber existing notes")
    for variant in ("PASS.", " pass ", "Pass!"):
        meta = InMemoryMeta({SELF_NOTES_KEY: "keep me"})
        check(await _companion(FakeLLM(variant), meta).update_self_notes() is None, f"{variant!r} is a PASS")
        check(meta.get(SELF_NOTES_KEY) == "keep me", f"{variant!r} preserved the slot")

    meta = InMemoryMeta({SELF_NOTES_KEY: "keep me"})
    check(await _companion(FakeLLM("   "), meta).update_self_notes() is None, "empty reply -> None")
    check(meta.get(SELF_NOTES_KEY) == "keep me", "empty reply does not clobber")

    # --- 2b) a note written in the wrong voice is dropped, not injected backwards ---
    # The model occasionally addresses the note to the USER ("You get annoyed when <bot> asks
    # questions"), which would teach her the inverse of the lesson. Her name is the tell, so the
    # fixtures are built from config.BOT_NAME rather than a hardcoded name — the guard keys off it.
    bot = config.BOT_NAME
    for backwards in (f"You get annoyed when {bot} asks questions — push back on them gently.",
                      f"{bot.lower()} should stop asking so many questions.",
                      f"They like it when {bot} has an opinion."):
        meta = InMemoryMeta({SELF_NOTES_KEY: "keep me"})
        check(await _companion(FakeLLM(backwards), meta).update_self_notes() is None,
              f"inverted note rejected: {backwards[:40]!r}")
        check(meta.get(SELF_NOTES_KEY) == "keep me", "a rejected note leaves the good notes in place")

    meta = InMemoryMeta()
    ok_note = "They get annoyed when you ask questions — react with an opinion instead."
    check(await _companion(FakeLLM(ok_note), meta).update_self_notes() == ok_note,
          "a correctly-voiced note still passes the guard")
    # A word that merely EMBEDS the bot name must not trip the guard — it is word-bounded.
    embeds = f"{bot.lower()}ade"   # e.g. "companionade": contains the name but isn't the name
    meta = InMemoryMeta()
    check(await _companion(FakeLLM(f"They like {embeds} on everything."), meta).update_self_notes(),
          "a word containing the name as a substring is not a false positive")

    # --- 3) hard cap keeps the injected block small ---
    meta = InMemoryMeta()
    long = "x" * (config.SELFNOTES_MAX_CHARS + 500)
    out = await _companion(FakeLLM(long), meta).update_self_notes()
    check(len(out) <= config.SELFNOTES_MAX_CHARS, f"capped to {config.SELFNOTES_MAX_CHARS}, got {len(out)}")
    check(len(meta.get(SELF_NOTES_KEY)) <= config.SELFNOTES_MAX_CHARS, "stored value is capped too")

    # --- 4) nothing to learn from ---
    meta = InMemoryMeta()
    llm = FakeLLM()
    check(await _companion(llm, meta, history=[]).update_self_notes() is None, "empty history -> no-op")
    check(llm.seen == [], "empty history costs no model call")

    # --- 5) injection into every user-facing prompt ---
    NOTE = "Ease off the questions — he finds them tiring."
    s = build_system([], None, self_notes=NOTE)
    check(NOTE in s, "chat prompt carries the notes")
    check("how to be with them" in s.lower(), "framed as learned behavior, not fact or identity")
    check(NOTE not in build_system([], None), "no notes arg injects nothing")
    check(NOTE not in build_system([], None, self_notes=""), "empty notes injects nothing")
    check(NOTE in build_reachout_system([], None, self_notes=NOTE), "reach-out carries the notes")
    check(NOTE in build_followup_system([], None, self_notes=NOTE), "follow-up carries the notes")

    # notes and persona are distinct slots that coexist
    both = build_system([], None, persona="You've grown fond of him.", self_notes=NOTE)
    check("You've grown fond of him." in both and NOTE in both, "persona + notes both present")

    # ... and reach both live generation paths through the Companion, not just the builders
    meta = InMemoryMeta({SELF_NOTES_KEY: NOTE})
    llm = FakeLLM("sure")
    c = _companion(llm, meta, history=[{"role": "user", "content": "hey"}])
    await c.send("hey", lambda t: asyncio.sleep(0))
    check(NOTE in llm.seen[-1], "send() injects the notes")

    meta = InMemoryMeta({SELF_NOTES_KEY: NOTE})
    llm = FakeLLM("been thinking about you")
    await _companion(llm, meta, history=[{"role": "user", "content": "later"}]).reach_out()
    check(NOTE in llm.seen[-1], "reach_out() injects the notes")

    meta = InMemoryMeta({SELF_NOTES_KEY: NOTE})
    llm = FakeLLM("oh also")
    await _companion(llm, meta, history=[{"role": "assistant", "content": "hi"}]).follow_up()
    check(NOTE in llm.seen[-1], "follow_up() injects the notes")

    # --- 6) prompt builders steer away from the two known failure modes ---
    sysmsg = build_self_notes_system(400).lower()
    check("400" in build_self_notes_system(400), "char budget is stated to the model")
    check("belong in memory" in sysmsg, "explicitly excludes user facts (those are memories)")
    check("self-description" in sysmsg, "explicitly excludes identity (that's the persona slot)")
    check("your own behavior" in sysmsg, "the notes are about her behavior")
    user = build_self_notes_user("cur notes", HIST)
    check("cur notes" in user and "stop asking me so many questions" in user, "user turn has both inputs")
    check("(none yet" in build_self_notes_user("", HIST), "first run is framed as first-time, not empty-default")

    # --- 7) SelfNotesJob gating: asleep / busy / not-idle-enough / cooldown ---
    class C:
        def __init__(self, meta, idle=999, asleep=False, busy=False):
            self.meta, self._idle, self._asleep, self._busy = meta, idle, asleep, busy
            self.calls = 0

        def is_asleep(self):
            return self._asleep

        def is_busy(self):
            return self._busy

        def is_unavailable(self):
            return False   # CooldownJob skips every model call during an A4 window

        def idle_seconds(self):
            return self._idle

        async def update_self_notes(self):
            self.calls += 1

    now = [100000.0]  # well past the cooldown, as a real epoch clock always is on a fresh install

    def clock():
        return now[0]

    def job(c):
        return SelfNotesJob(c, 1.0, min_idle=300, cooldown=1800, clock=clock)

    c = C(InMemoryMeta(), idle=999)
    await job(c).run()
    check(c.calls == 1, "runs when idle and off cooldown")
    await job(c).run()
    check(c.calls == 1, "second run inside the cooldown is skipped")
    now[0] += 1801
    await job(c).run()
    check(c.calls == 2, "runs again once the cooldown elapses")

    for label, kw in (("asleep", {"asleep": True}), ("busy", {"busy": True}), ("not idle", {"idle": 10})):
        c = C(InMemoryMeta(), **kw)
        await job(c).run()
        check(c.calls == 0, f"skipped while {label}")
        check(c.meta.get_json(LAST_SELFNOTES_KEY, 0) == 0, f"{label}: cooldown stamp not burned on a skip")

    print(f"OK - {_checks} checks passed")


asyncio.run(main())
