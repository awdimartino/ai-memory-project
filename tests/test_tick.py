"""Offline coverage for the tick loop (proactivity scaffolding).

Drives the scheduler with a fake clock (deterministic, no real waiting) and the
two internal jobs with fakes: mood drift only fires when the user is idle and a
failing job never stops the heartbeat. The one lifecycle test uses a tiny real
interval to confirm start/stop.

Run:  python tests/test_tick.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.emotion_manager import EmotionManager
from core.tick import (
    IdleConsolidationJob,
    Job,
    MoodDriftJob,
    PersonaEditJob,
    ReachOutJob,
    ReflectionJob,
    SleepJob,
    TickLoop,
)


class Clock:
    def __init__(self, t=100.0):
        self.t = t

    def __call__(self):
        return self.t


class CountingJob(Job):
    def __init__(self, name, interval=0.0):
        self.name = name
        self.interval = interval
        self.runs = 0

    async def run(self):
        self.runs += 1


class FailingJob(Job):
    name = "boom"
    interval = 0.0

    async def run(self):
        raise RuntimeError("boom")


class FakeCompanion:
    def __init__(self, idle=0.0, pending=0, asleep=False):
        self._idle = idle
        self._pending = pending
        self.flushed = 0
        self._asleep = asleep

    def idle_seconds(self):
        return self._idle

    def is_asleep(self):
        return self._asleep

    def pending_count(self):
        return self._pending

    async def flush(self):
        self.flushed += 1
        self._pending = 0


class InMemoryMeta:
    def __init__(self):
        self._d = {}

    def get_json(self, k, default=None):
        return self._d.get(k, default)

    def set_json(self, k, v):
        self._d[k] = v


class FakeClassifier:
    def classify(self, text):
        return []


CASES = []


def case(fn):
    CASES.append(fn)
    return fn


@case
async def scheduler_respects_interval():
    clk = Clock()
    j = CountingJob("j", interval=10.0)
    loop = TickLoop([j], interval=10.0, clock=clk)
    await loop.run_due()          # first run (no prior) -> fires
    assert j.runs == 1
    await loop.run_due()          # no time elapsed -> skip
    assert j.runs == 1
    clk.t += 10.0
    await loop.run_due()          # interval elapsed -> fires
    assert j.runs == 2
    clk.t += 5.0
    await loop.run_due()          # 5 < 10 -> skip
    assert j.runs == 2, j.runs


@case
async def zero_interval_runs_every_tick():
    clk = Clock()
    j = CountingJob("j", interval=0.0)
    loop = TickLoop([j], interval=1.0, clock=clk)
    for _ in range(3):
        await loop.run_due()
    assert j.runs == 3, j.runs


@case
async def failing_job_does_not_stop_others():
    good = CountingJob("good", 0.0)
    loop = TickLoop([FailingJob(), good], interval=1.0, clock=Clock())
    await loop.run_due()          # must not raise
    assert good.runs == 1


@case
async def mood_drift_only_when_idle():
    meta = InMemoryMeta()
    emo = EmotionManager(FakeClassifier(), meta, pull_strength=0.4, noise_floor=0.05)
    emo.state["irritation"] = 0.9
    comp = FakeCompanion(idle=10.0)
    job = MoodDriftJob(comp, emo, interval=0.0, idle_after=90.0)

    await job.run()               # idle 10 < 90 -> no drift
    assert emo.state["irritation"] == 0.9
    assert meta.get_json("mood_state") is None

    comp._idle = 100.0
    await job.run()               # idle 100 >= 90 -> drift toward baseline
    assert emo.state["irritation"] < 0.9, emo.state["irritation"]
    assert meta.get_json("mood_state") is not None


@case
async def mood_drift_noop_without_emotion():
    job = MoodDriftJob(FakeCompanion(idle=1000.0), None, interval=0.0, idle_after=90.0)
    await job.run()               # emotion disabled -> no crash, nothing to do


@case
async def idle_consolidation_flushes_when_idle_and_pending():
    comp = FakeCompanion(idle=1000.0, pending=4)
    job = IdleConsolidationJob(comp, interval=0.0, idle_after=180.0)
    await job.run()
    assert comp.flushed == 1 and comp.pending_count() == 0


@case
async def idle_consolidation_noop_when_active_or_empty():
    active = FakeCompanion(idle=5.0, pending=4)
    await IdleConsolidationJob(active, 0.0, 180.0).run()
    assert active.flushed == 0, "must not consolidate while user is active"
    empty = FakeCompanion(idle=1000.0, pending=0)
    await IdleConsolidationJob(empty, 0.0, 180.0).run()
    assert empty.flushed == 0, "nothing pending -> no flush"


@case
async def busy_guard_makes_idle_zero():
    # A real Companion reports 0 idle while a turn is in flight, so the tick never
    # fires mid-reply (even during a slow generation). No I/O in the constructor.
    from core.companion import Companion
    c = Companion(llm=None, store=None, memory=None, meta=None, session_id=1)
    c._last_activity -= 500.0            # pretend the last turn ended long ago
    assert c.idle_seconds() > 400.0
    c._busy = True
    assert c.idle_seconds() == 0.0, "busy turn must read as not idle"


class ReachCompanion:
    def __init__(self, idle, reach_result, asleep=False):
        self._idle = idle
        self.meta = InMemoryMeta()
        self._reach_result = reach_result
        self.reach_calls = 0
        self._asleep = asleep

    def idle_seconds(self):
        return self._idle

    def is_asleep(self):
        return self._asleep

    async def reach_out(self):
        self.reach_calls += 1
        return self._reach_result


class WallClock:
    # Realistic wall-clock magnitude: with last-attempt defaulting to 0, a companion
    # that has never reached out is always past the cooldown (as with real time.time()).
    def __init__(self, t=1_000_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _reach_job(comp, clock, pushes):
    async def notify(m):
        pushes.append(m)
    return ReachOutJob(comp, notify, interval=0.0, min_idle=900.0, cooldown=7200.0, clock=clock)


@case
async def reach_out_gated_by_idle():
    comp = ReachCompanion(idle=10.0, reach_result="hey")   # user still around
    pushes = []
    await _reach_job(comp, WallClock(), pushes).run()
    assert comp.reach_calls == 0 and pushes == [], "should not reach out while active"


@case
async def reach_out_gated_by_cooldown():
    comp = ReachCompanion(idle=1000.0, reach_result="hey")
    comp.meta.set_json("last_reachout_at", 1_000_000.0 - 100.0)  # attempted 100s ago (< 7200)
    pushes = []
    await _reach_job(comp, WallClock(), pushes).run()
    assert comp.reach_calls == 0 and pushes == [], "cooldown should block a second attempt"


@case
async def reach_out_fires_and_pushes():
    comp = ReachCompanion(idle=1000.0, reach_result="hey, was thinking about that game")
    pushes = []
    await _reach_job(comp, WallClock(), pushes).run()
    assert comp.reach_calls == 1
    assert pushes == [{"type": "proactive", "content": "hey, was thinking about that game"}], pushes
    assert comp.meta.get_json("last_reachout_at") == 1_000_000.0


@case
async def reach_out_decline_still_sets_cooldown():
    comp = ReachCompanion(idle=1000.0, reach_result=None)  # Mari stays quiet
    pushes = []
    await _reach_job(comp, WallClock(), pushes).run()
    assert comp.reach_calls == 1, "should have attempted"
    assert pushes == [], "declined -> no push"
    assert comp.meta.get_json("last_reachout_at") == 1_000_000.0, "cooldown set even on decline"


class ReflectCompanion:
    def __init__(self, idle, asleep=False):
        self._idle = idle
        self.meta = InMemoryMeta()
        self.reflect_calls = 0
        self._asleep = asleep

    def idle_seconds(self):
        return self._idle

    def is_asleep(self):
        return self._asleep

    async def reflect(self):
        self.reflect_calls += 1
        return "i've been wondering what they do all day"


@case
async def reflection_gated_by_idle():
    comp = ReflectCompanion(idle=30.0)   # only 30s away, min_idle 120
    job = ReflectionJob(comp, interval=0.0, min_idle=120.0, cooldown=600.0, clock=WallClock())
    await job.run()
    assert comp.reflect_calls == 0, "should not reflect while barely idle"


@case
async def reflection_gated_by_cooldown():
    comp = ReflectCompanion(idle=1000.0)
    comp.meta.set_json("last_reflect_at", 1_000_000.0 - 60.0)  # reflected 60s ago (< 600)
    job = ReflectionJob(comp, interval=0.0, min_idle=120.0, cooldown=600.0, clock=WallClock())
    await job.run()
    assert comp.reflect_calls == 0, "cooldown should block back-to-back reflection"


@case
async def reflection_fires_and_sets_cooldown():
    comp = ReflectCompanion(idle=1000.0)
    job = ReflectionJob(comp, interval=0.0, min_idle=120.0, cooldown=600.0, clock=WallClock())
    await job.run()
    assert comp.reflect_calls == 1
    assert comp.meta.get_json("last_reflect_at") == 1_000_000.0


class _Store:
    def __init__(self, n):
        self.n = n

    def message_count(self):
        return self.n


class PersonaCompanion:
    def __init__(self, idle, messages, asleep=False):
        self._idle = idle
        self.meta = InMemoryMeta()
        self.store = _Store(messages)
        self.edit_calls = 0
        self._asleep = asleep

    def idle_seconds(self):
        return self._idle

    def is_asleep(self):
        return self._asleep

    async def edit_persona(self):
        self.edit_calls += 1
        return "you've grown fond of them"


def _persona_job(comp, clock):
    return PersonaEditJob(comp, interval=0.0, min_idle=300.0, cooldown=3600.0,
                          min_messages=20, clock=clock)


@case
async def persona_edit_gated_by_idle():
    comp = PersonaCompanion(idle=60.0, messages=100)   # not idle enough
    await _persona_job(comp, WallClock()).run()
    assert comp.edit_calls == 0


@case
async def persona_edit_gated_by_min_messages():
    comp = PersonaCompanion(idle=1000.0, messages=5)   # too early in the relationship
    await _persona_job(comp, WallClock()).run()
    assert comp.edit_calls == 0


@case
async def persona_edit_gated_by_cooldown():
    comp = PersonaCompanion(idle=1000.0, messages=100)
    comp.meta.set_json("last_persona_edit_at", 1_000_000.0 - 60.0)  # edited 60s ago (< 3600)
    await _persona_job(comp, WallClock()).run()
    assert comp.edit_calls == 0


@case
async def persona_edit_fires_and_sets_cooldown():
    comp = PersonaCompanion(idle=1000.0, messages=100)
    await _persona_job(comp, WallClock()).run()
    assert comp.edit_calls == 1
    assert comp.meta.get_json("last_persona_edit_at") == 1_000_000.0


@case
async def familiarity_scales_with_messages():
    import config
    from core.companion import Companion, familiarity_label
    config.FAMILIARITY_MESSAGES = 400
    c = Companion(llm=None, store=_Store(0), memory=None, meta=None, session_id=1)
    assert c.familiarity() == 0.0
    c.store.n = 200
    assert abs(c.familiarity() - 0.5) < 1e-9
    c.store.n = 100000
    assert c.familiarity() == 1.0  # capped
    assert "stranger" in familiarity_label(0.0)
    assert "close" in familiarity_label(0.95)


class SleepCompanion:
    def __init__(self, idle, asleep=False):
        self._idle = idle
        self._asleep = asleep
        self.sleep_calls = 0

    def idle_seconds(self):
        return self._idle

    def is_asleep(self):
        return self._asleep

    async def sleep(self):
        self.sleep_calls += 1
        self._asleep = True


@case
async def sleep_gated_by_idle():
    comp = SleepCompanion(idle=100.0)   # not idle long enough (sleep_after 1800)
    await SleepJob(comp, interval=0.0, sleep_after=1800.0).run()
    assert comp.sleep_calls == 0


@case
async def sleep_skipped_when_already_asleep():
    comp = SleepCompanion(idle=99999.0, asleep=True)
    await SleepJob(comp, interval=0.0, sleep_after=1800.0).run()
    assert comp.sleep_calls == 0


@case
async def sleep_fires_after_long_idle():
    comp = SleepCompanion(idle=99999.0)
    await SleepJob(comp, interval=0.0, sleep_after=1800.0).run()
    assert comp.sleep_calls == 1 and comp.is_asleep()


@case
async def model_jobs_skip_while_asleep():
    # reflection and reach-out must not fire a model call while she's asleep
    refl = ReflectCompanion(idle=99999.0, asleep=True)
    await ReflectionJob(refl, 0.0, 1.0, 1.0, clock=WallClock()).run()
    assert refl.reflect_calls == 0, "reflection should skip while asleep"

    reach = ReachCompanion(idle=99999.0, reach_result="hi", asleep=True)
    await _reach_job(reach, WallClock(), []).run()
    assert reach.reach_calls == 0, "reach-out should skip while asleep"


@case
async def start_stop_lifecycle():
    j = CountingJob("j", interval=0.0)
    loop = TickLoop([j], interval=0.01)      # real clock, tiny interval
    loop.start()
    await asyncio.sleep(0.05)
    await loop.stop()
    assert j.runs >= 1, j.runs
    await loop.stop()                        # idempotent


async def main() -> int:
    failed = 0
    for fn in CASES:
        try:
            await fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
