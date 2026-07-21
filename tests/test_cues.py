"""Offline coverage for core/cues.py — the farewell cue that gates the sleep decision.

The asymmetry that matters: a FALSE POSITIVE puts her to sleep in the middle of a live
conversation (the user then waits out a model reload to say one more thing), while a
false negative just means she times out into standby later, which is the behaviour that
already existed. So the negative cases below carry more weight than the positive ones,
and the past tense ("last night", "how was your night") is the failure mode to beat.

Run:  python tests/test_cues.py
"""
from _harness import case, run  # also puts the repo root on sys.path

from core.cues import farewell_cue


SIGNOFFS = [
    "goodnight",
    "good night mari",
    "gnight",
    "g'night",
    "night night",
    "alright, i'm going to bed",
    "im gonna go to sleep",
    "heading to bed, long day",
    "i'm off to bed",
    "gonna hit the hay",
    "calling it a night",
    "time for bed i think",
    "talk to you tomorrow",
    "see ya tomorrow",
    "catch you in the morning",
    "sleep well",
    "going to get some sleep",
    # Added after the decision was measured to discriminate, which made a broader
    # cue safe. All of these missed the first, tighter version.
    "hit the hay",
    "i'm sleeping so you should too",
    "bedtime",
    "gotta sleep",
    "i should sleep",
    "i need to go to bed",
    "you should get some rest",
    "you should sleep too",
    "sweet dreams",
    "rest up",
    "signing off for the night",
    "turning in",
    "ready for bed",
    "night",
    "ok night",
]

NOT_SIGNOFFS = [
    # past tense — the dominant false positive, and the expensive one
    "last night was rough",
    "how was your night?",
    "i had the strangest dream the other night",
    "i didn't sleep well",
    "i was up all night with the migration",
    "i work the night shift this week",
    "that show gave me night terrors as a kid",
    # future, but about him, not about leaving now
    "tomorrow i have a client call i'm dreading",
    # ordinary conversation with no cue at all
    "what do you like to do?",
    "i've been walking the long way home past the river",
    "",
    # Added alongside the broadened patterns — these are what a loose cue risks.
    "i can't sleep lately",
    "my sleep schedule is wrecked",
    "i've been having trouble sleeping",
    "the kids went to bed an hour ago",
    "i'm not going to bed yet",
    "i don't want to sleep",
    "if i go to bed now i'll just lie there",
    "how was your day",
    "she's sleeping over at a friend's",
]


@case
async def signoffs_are_detected():
    missed = [t for t in SIGNOFFS if farewell_cue(t) is None]
    assert not missed, f"missed farewell cue in: {missed}"


@case
async def non_signoffs_are_not_detected():
    wrong = [(t, farewell_cue(t)) for t in NOT_SIGNOFFS if farewell_cue(t) is not None]
    assert not wrong, f"false farewell cue (would sleep mid-conversation): {wrong}"


@case
async def past_tense_night_never_fires():
    """Singled out because it is the one that puts her to sleep mid-sentence."""
    for t in ["last night", "the other night", "yesternight", "all night",
              "how was the night", "at night i can't focus"]:
        assert farewell_cue(t) is None, f"{t!r} read as a goodbye"


@case
async def the_matched_phrase_is_returned():
    """The cue is logged, so it has to identify itself usefully."""
    cue = farewell_cue("ok i'm going to bed, night")
    assert cue and cue.lower() in ("i'm going to bed", "im going to bed", "night"), cue


# --- the decision itself (Companion.maybe_sleep_on_farewell) ----------------------

class FakeManager:
    def __init__(self):
        self.unloaded = 0

    async def unload_all(self):
        self.unloaded += 1

    async def load(self, models):
        pass


class ChoiceLLM:
    """Returns a fixed sleep choice, and records that it was even asked."""
    def __init__(self, choice="SLEEP", boom=False):
        self.choice, self.boom, self.calls = choice, boom, 0

    async def structured_json(self, messages, schema, model=None):
        self.calls += 1
        if self.boom:
            raise RuntimeError("brain offline")
        return {"choice": self.choice}


def _companion(llm, manager=None):
    from core.companion import Companion
    c = Companion(llm=llm, store=None, memory=None, meta=None, session_id=1,
                  model_manager=manager if manager is not None else FakeManager())
    c.flush = _noop_flush.__get__(c)   # sleep() flushes; there's no store here
    return c


async def _noop_flush(self):
    return None


@case
async def a_goodnight_can_put_her_to_sleep():
    mgr = FakeManager()
    c = _companion(ChoiceLLM("SLEEP"), mgr)
    slept = await c.maybe_sleep_on_farewell("goodnight mari", "night, i'll go quiet too.")
    assert slept is True
    assert c.is_asleep() and mgr.unloaded == 1, "she should have actually powered down"


@case
async def she_can_decline_and_stay_up():
    mgr = FakeManager()
    c = _companion(ChoiceLLM("STAY"), mgr)
    slept = await c.maybe_sleep_on_farewell("goodnight", "night.")
    assert slept is False
    assert not c.is_asleep() and mgr.unloaded == 0
    # The mechanical SLEEP_AFTER_IDLE trigger is untouched, so declining only ever
    # delays standby — it can't keep the model resident forever.


@case
async def no_cue_means_no_model_call():
    """The gate's whole purpose: ordinary turns must not pay a structured call."""
    llm = ChoiceLLM("SLEEP")
    c = _companion(llm)
    assert await c.maybe_sleep_on_farewell("what do you like to do?", "not much") is False
    assert llm.calls == 0, "a decision was made on a turn with no farewell in it"
    assert not c.is_asleep()


@case
async def a_failed_decision_leaves_her_awake():
    c = _companion(ChoiceLLM(boom=True))
    assert await c.maybe_sleep_on_farewell("goodnight", "night") is False
    assert not c.is_asleep(), "a brain error must not strand her asleep"


@case
async def she_does_not_sleep_twice_or_while_away():
    llm = ChoiceLLM("SLEEP")
    c = _companion(llm)
    c._asleep = True
    assert await c.maybe_sleep_on_farewell("goodnight", "") is False
    assert llm.calls == 0, "already asleep -> don't even ask"


if __name__ == "__main__":
    raise SystemExit(run())
