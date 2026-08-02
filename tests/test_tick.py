"""Offline coverage for the tick loop and every job that hangs off it.

Drives the scheduler with a fake clock (deterministic, no real waiting) and each
job with fakes. Covers the loop itself (interval respect, a failing job never
stopping the heartbeat, start/stop via one tiny real interval) plus all eight
jobs: mood drift, drive drift, idle consolidation, reach-out, reflection,
persona edit, follow-up, and sleep — including their idle/cooldown/drive gates,
the asleep and busy guards, and familiarity scaling.

Run:  python tests/test_tick.py
"""
import asyncio

from _harness import case, config_override, run  # also puts the repo root on sys.path
from helpers import FakeClassifier, InMemoryMeta

from core.emotion_manager import EmotionManager
from core.tick import (
    DriveDriftJob,
    FollowUpJob,
    IdleConsolidationJob,
    Job,
    MoodDriftJob,
    PersonaEditJob,
    PursuitMiningJob,
    PursuitReturnJob,
    ReachOutJob,
    ReflectionJob,
    SleepJob,
    TickLoop,
    UnavailableJob,
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


class NotAway:
    """Default `is_unavailable()` for the job fakes: she has not stepped away.

    Every CooldownJob (plus follow-up and idle consolidation) now checks this, so a
    fake missing it raises AttributeError instead of quietly failing a check. Fakes
    that exercise the A4 window override it; the rest inherit False.
    """
    def is_unavailable(self):
        return False


class FakeCompanion(NotAway):
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


class FakeDrives:
    """Records update() calls so the job's plumbing (idle + mood + asleep pass-through) is testable."""

    def __init__(self):
        self.updates = []

    async def update(self, idle_seconds, mood=None, asleep=False):
        self.updates.append((idle_seconds, mood, asleep))
        return {"connection": 0.1, "restlessness": 0.2, "energy": 0.9}


class DriveCompanion(NotAway):
    def __init__(self, idle, emotion=None, asleep=False):
        self._idle = idle
        self.emotion = emotion
        self._asleep = asleep

    def idle_seconds(self):
        return self._idle

    def is_asleep(self):
        return self._asleep


@case
async def drive_drift_updates_with_idle_and_no_mood():
    drives = FakeDrives()
    comp = DriveCompanion(idle=500.0)               # emotion disabled -> mood None, awake
    await DriveDriftJob(comp, drives, interval=0.0).run()
    assert drives.updates == [(500.0, None, False)], drives.updates


@case
async def drive_drift_passes_mood_and_asleep():
    drives = FakeDrives()

    class Emo:
        state = {"warmth": 0.5}

    comp = DriveCompanion(idle=500.0, emotion=Emo(), asleep=True)
    await DriveDriftJob(comp, drives, interval=0.0).run()
    assert drives.updates == [(500.0, {"warmth": 0.5}, True)], drives.updates


@case
async def drive_drift_noop_without_drives():
    # Drives disabled -> the job is a harmless no-op (mirrors mood-drift without emotion).
    await DriveDriftJob(DriveCompanion(idle=500.0), None, interval=0.0).run()


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


class ReachCompanion(NotAway):
    def __init__(self, idle, reach_result, asleep=False, busy=False):
        self._idle = idle
        self.meta = InMemoryMeta()
        self._reach_result = reach_result
        self.reach_calls = 0
        self._asleep = asleep
        self._busy = busy

    def idle_seconds(self):
        return self._idle

    def is_asleep(self):
        return self._asleep

    def is_busy(self):
        return self._busy

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
    comp = ReachCompanion(idle=1000.0, reach_result=None)  # the companion stays quiet
    pushes = []
    await _reach_job(comp, WallClock(), pushes).run()
    assert comp.reach_calls == 1, "should have attempted"
    assert pushes == [], "declined -> no push"
    assert comp.meta.get_json("last_reachout_at") == 1_000_000.0, "cooldown set even on decline"


class FakeDriveState:
    """Minimal drive stand-in for gating tests: get() reads a level, discharge() records + zeroes."""

    def __init__(self, **vals):
        self.vals = dict(vals)
        self.discharged = []

    def get(self, name):
        return self.vals.get(name, 0.0)

    async def discharge(self, name):
        self.discharged.append(name)
        self.vals[name] = 0.0


def _reach_drive_job(comp, drv, pushes, threshold=0.6):
    async def notify(m):
        pushes.append(m)
    return ReachOutJob(comp, notify, interval=0.0, min_idle=900.0, cooldown=7200.0,
                       drives=drv, threshold=threshold, clock=WallClock())


@case
async def reach_out_gated_by_low_connection_drive():
    # idle is high, but the connection drive is below threshold -> no reach-out (drive gates now)
    comp = ReachCompanion(idle=1000.0, reach_result="hey")
    drv = FakeDriveState(connection=0.3)
    pushes = []
    await _reach_drive_job(comp, drv, pushes).run()
    assert comp.reach_calls == 0 and pushes == [], "low connection should not reach out"
    assert drv.discharged == []


@case
async def reach_out_fires_on_high_connection_and_discharges():
    # idle is LOW (would fail the old gate) but connection is high -> the drive is the gate
    comp = ReachCompanion(idle=5.0, reach_result="hey, was thinking about you")
    drv = FakeDriveState(connection=0.8)
    pushes = []
    await _reach_drive_job(comp, drv, pushes).run()
    assert comp.reach_calls == 1
    assert pushes == [{"type": "proactive", "content": "hey, was thinking about you"}]
    assert drv.discharged == ["connection"], "firing must discharge the drive"
    assert comp.meta.get_json("last_reachout_at") == 1_000_000.0


@case
async def reach_out_drive_still_respects_cooldown():
    comp = ReachCompanion(idle=1000.0, reach_result="hey")
    comp.meta.set_json("last_reachout_at", 1_000_000.0 - 100.0)  # 100s ago (< 7200)
    drv = FakeDriveState(connection=0.9)
    pushes = []
    await _reach_drive_job(comp, drv, pushes).run()
    assert comp.reach_calls == 0 and drv.discharged == [], "cooldown is a hard floor over the drive"


@case
async def reach_out_skips_when_busy():
    # drive is high but a turn is in flight -> must not fire mid-reply (drives relieve at turn end)
    comp = ReachCompanion(idle=1000.0, reach_result="hey", busy=True)
    drv = FakeDriveState(connection=0.9)
    pushes = []
    await _reach_drive_job(comp, drv, pushes).run()
    assert comp.reach_calls == 0 and pushes == [], "must not reach out mid-turn"


class FollowCompanion(NotAway):
    """Fake for the follow-up job: exposes the small surface the job reads + follow_up()."""

    def __init__(self, left, since, asleep=False, busy=False, result="oh and one more thing"):
        self._left = left
        self._since = since
        self._asleep = asleep
        self._busy = busy
        self._result = result
        self.follow_calls = 0
        self.cancelled = 0

    def is_asleep(self):
        return self._asleep

    def is_busy(self):
        return self._busy

    def followups_pending(self):
        return self._left

    def seconds_since_reply(self):
        return self._since

    def cancel_followups(self):
        self.cancelled += 1
        self._left = 0

    async def follow_up(self):
        self.follow_calls += 1
        if self._result:
            self._left -= 1
        else:
            self._left = 0
        return self._result


def _follow_job(comp, pushes, chance=0.5, rng=lambda: 0.0):
    async def notify(m):
        pushes.append(m)
    return FollowUpJob(comp, notify, interval=0.0, chance=chance,
                       min_delay=30.0, window=300.0, rng=rng)


@case
async def follow_up_fires_within_window_and_pushes():
    comp = FollowCompanion(left=1, since=60.0)          # in-window, has budget
    pushes = []
    await _follow_job(comp, pushes, rng=lambda: 0.0).run()   # 0.0 < 0.5 chance -> fire
    assert comp.follow_calls == 1
    assert pushes == [{"type": "proactive", "content": "oh and one more thing"}], pushes


@case
async def follow_up_gated_when_none_left():
    comp = FollowCompanion(left=0, since=60.0)
    await _follow_job(comp, []).run()
    assert comp.follow_calls == 0


@case
async def follow_up_gated_before_min_delay():
    comp = FollowCompanion(left=1, since=10.0)           # 10 < 30 min_delay
    await _follow_job(comp, []).run()
    assert comp.follow_calls == 0


@case
async def follow_up_window_expiry_cancels():
    comp = FollowCompanion(left=1, since=600.0)          # 600 > 300 window
    await _follow_job(comp, []).run()
    assert comp.follow_calls == 0 and comp.cancelled == 1


@case
async def follow_up_chance_gate_can_skip():
    comp = FollowCompanion(left=1, since=60.0)
    await _follow_job(comp, [], rng=lambda: 0.9).run()   # 0.9 >= 0.5 -> skip this tick
    assert comp.follow_calls == 0 and comp.cancelled == 0  # still eligible next tick


@case
async def follow_up_skips_when_asleep_or_busy():
    asleep = FollowCompanion(left=1, since=60.0, asleep=True)
    await _follow_job(asleep, []).run()
    assert asleep.follow_calls == 0
    busy = FollowCompanion(left=1, since=60.0, busy=True)
    await _follow_job(busy, []).run()
    assert busy.follow_calls == 0


class ReflectCompanion(NotAway):
    def __init__(self, idle, asleep=False, busy=False):
        self._idle = idle
        self.meta = InMemoryMeta()
        self.reflect_calls = 0
        self._asleep = asleep
        self._busy = busy

    def idle_seconds(self):
        return self._idle

    def is_asleep(self):
        return self._asleep

    def is_busy(self):
        return self._busy

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


@case
async def reflection_gated_by_low_restlessness_drive():
    comp = ReflectCompanion(idle=1000.0)               # idle high, but drive low
    drv = FakeDriveState(restlessness=0.2)
    job = ReflectionJob(comp, interval=0.0, min_idle=120.0, cooldown=600.0,
                        drives=drv, threshold=0.4, clock=WallClock())
    await job.run()
    assert comp.reflect_calls == 0 and drv.discharged == []


@case
async def reflection_fires_on_high_restlessness_and_discharges():
    comp = ReflectCompanion(idle=5.0)                  # idle low; the drive is the gate
    drv = FakeDriveState(restlessness=0.6)
    job = ReflectionJob(comp, interval=0.0, min_idle=120.0, cooldown=600.0,
                        drives=drv, threshold=0.4, clock=WallClock())
    await job.run()
    assert comp.reflect_calls == 1
    assert drv.discharged == ["restlessness"]
    assert comp.meta.get_json("last_reflect_at") == 1_000_000.0


@case
async def reflection_skips_when_busy():
    comp = ReflectCompanion(idle=1000.0, busy=True)
    drv = FakeDriveState(restlessness=0.9)
    job = ReflectionJob(comp, interval=0.0, min_idle=120.0, cooldown=600.0,
                        drives=drv, threshold=0.4, clock=WallClock())
    await job.run()
    assert comp.reflect_calls == 0, "must not reflect mid-turn"


class _Store:
    def __init__(self, n):
        self.n = n

    def message_count(self):
        return self.n


class PersonaCompanion(NotAway):
    def __init__(self, idle, messages, asleep=False, busy=False):
        self._idle = idle
        self.meta = InMemoryMeta()
        self.store = _Store(messages)
        self.edit_calls = 0
        self._asleep = asleep
        self._busy = busy

    def idle_seconds(self):
        return self._idle

    def is_asleep(self):
        return self._asleep

    def is_busy(self):
        return self._busy

    def message_count(self):
        # The job reads this off the facade rather than reaching into `.store`.
        return self.store.message_count()

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
async def persona_edit_skips_when_busy():
    # Inherited from CooldownJob. Redundant with the idle gate on a real Companion
    # (idle reads 0 mid-turn) but explicit, so it holds if that guard ever changes.
    comp = PersonaCompanion(idle=1000.0, messages=100, busy=True)
    await _persona_job(comp, WallClock()).run()
    assert comp.edit_calls == 0, "must not rewrite her persona mid-turn"


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
    from core.companion import Companion, familiarity_label
    with config_override(FAMILIARITY_MESSAGES=400):
        c = Companion(llm=None, store=_Store(0), memory=None, meta=None, session_id=1)
        assert c.familiarity() == 0.0
        c.store.n = 200
        assert abs(c.familiarity() - 0.5) < 1e-9
        c.store.n = 100000
        assert c.familiarity() == 1.0  # capped
    assert "stranger" in familiarity_label(0.0)
    assert "close" in familiarity_label(0.95)


class SleepCompanion:
    def __init__(self, idle, asleep=False, unavailable=False):
        self._idle = idle
        self._asleep = asleep
        self._unavailable = unavailable
        self.sleep_calls = 0

    def idle_seconds(self):
        return self._idle

    def is_asleep(self):
        return self._asleep

    def is_unavailable(self):
        return self._unavailable

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
async def sleep_skipped_when_unavailable():
    # §A4: sleep and "stepped away" are mutually exclusive states.
    comp = SleepCompanion(idle=99999.0, unavailable=True)
    await SleepJob(comp, interval=0.0, sleep_after=1800.0).run()
    assert comp.sleep_calls == 0, "must not also fall asleep while already unavailable"


@case
async def sleep_fires_after_long_idle():
    comp = SleepCompanion(idle=99999.0)
    await SleepJob(comp, interval=0.0, sleep_after=1800.0).run()
    assert comp.sleep_calls == 1 and comp.is_asleep()


class FakeEnergy:
    def __init__(self, e):
        self._e = e

    def energy(self):
        return self._e


@case
async def sleep_fires_when_tired_and_briefly_idle():
    # energy below threshold + past the small idle gap, but WELL short of sleep_after
    comp = SleepCompanion(idle=200.0)
    job = SleepJob(comp, 0.0, 1800.0, drives=FakeEnergy(0.1),
                   energy_threshold=0.15, energy_min_idle=120.0)
    await job.run()
    assert comp.sleep_calls == 1 and comp.is_asleep(), "tired + idle gap should sleep early"


@case
async def sleep_not_when_tired_but_mid_activity():
    comp = SleepCompanion(idle=30.0)  # < energy_min_idle -> she's basically still here
    job = SleepJob(comp, 0.0, 1800.0, drives=FakeEnergy(0.05),
                   energy_threshold=0.15, energy_min_idle=120.0)
    await job.run()
    assert comp.sleep_calls == 0, "must not nod off mid-conversation even if exhausted"


@case
async def sleep_not_when_energized_and_briefly_idle():
    comp = SleepCompanion(idle=200.0)  # < sleep_after and not tired
    job = SleepJob(comp, 0.0, 1800.0, drives=FakeEnergy(0.9),
                   energy_threshold=0.15, energy_min_idle=120.0)
    await job.run()
    assert comp.sleep_calls == 0


@case
async def model_jobs_skip_while_asleep():
    # reflection and reach-out must not fire a model call while she's asleep
    refl = ReflectCompanion(idle=99999.0, asleep=True)
    await ReflectionJob(refl, 0.0, 1.0, 1.0, clock=WallClock()).run()
    assert refl.reflect_calls == 0, "reflection should skip while asleep"

    reach = ReachCompanion(idle=99999.0, reach_result="hi", asleep=True)
    await _reach_job(reach, WallClock(), []).run()
    assert reach.reach_calls == 0, "reach-out should skip while asleep"


class FakeArousal:
    def __init__(self, arousal=0.0):
        self._arousal = arousal

    def arousal(self):
        return self._arousal


class MiningCompanion(NotAway):
    def __init__(self, idle, asleep=False, busy=False, emotion=None):
        self._idle = idle
        self.meta = InMemoryMeta()
        self.mine_calls = 0
        self._asleep = asleep
        self._busy = busy
        self.emotion = emotion

    def idle_seconds(self):
        return self._idle

    def is_asleep(self):
        return self._asleep

    def is_busy(self):
        return self._busy

    async def mine_open_questions(self):
        self.mine_calls += 1
        return ["a question"]


def _mining_job(comp, salience=0.0):
    return PursuitMiningJob(comp, interval=0.0, min_idle=1800.0, cooldown=86400.0,
                            salience=salience, clock=WallClock())


@case
async def pursuit_mining_gated_by_idle():
    comp = MiningCompanion(idle=100.0)   # min_idle 1800
    await _mining_job(comp).run()
    assert comp.mine_calls == 0


@case
async def pursuit_mining_no_salience_floor_fires_freely():
    comp = MiningCompanion(idle=99999.0, emotion=FakeArousal(0.0))  # salience=0.0 -> no gate
    await _mining_job(comp, salience=0.0).run()
    assert comp.mine_calls == 1


@case
async def pursuit_mining_gated_by_salience_floor():
    comp = MiningCompanion(idle=99999.0, emotion=FakeArousal(0.5))   # below the floor
    await _mining_job(comp, salience=2.0).run()
    assert comp.mine_calls == 0, "a flat window should not be mined"


@case
async def pursuit_mining_fires_above_salience_floor():
    comp = MiningCompanion(idle=99999.0, emotion=FakeArousal(3.0))   # above the floor
    await _mining_job(comp, salience=2.0).run()
    assert comp.mine_calls == 1
    assert comp.meta.get_json("last_pursuit_mine_at") == 1_000_000.0


@case
async def pursuit_mining_skips_when_asleep():
    comp = MiningCompanion(idle=99999.0, asleep=True)
    await _mining_job(comp).run()
    assert comp.mine_calls == 0


# --- §A4: UnavailableJob / PursuitReturnJob ---------------------------------------

class UnavailableCompanion:
    def __init__(self, idle, unavailable=False, pending=False, asleep=False, busy=False,
                 go_result=True, reason="journaling", eta=120.0):
        self._idle = idle
        self.meta = InMemoryMeta()
        self._unavailable = unavailable
        self._pending = pending
        self._asleep = asleep
        self._busy = busy
        self._go_result = go_result
        self._reason = reason
        self._eta = eta
        self.go_calls = 0
        self.emotion = None

    def idle_seconds(self):
        return self._idle

    def is_asleep(self):
        return self._asleep

    def is_busy(self):
        return self._busy

    def is_unavailable(self):
        return self._unavailable

    def has_pending_message(self):
        return self._pending

    def unavailable_reason(self):
        return self._reason

    def unavailable_eta_seconds(self):
        return self._eta

    async def go_unavailable(self):
        self.go_calls += 1
        if self._go_result:
            self._unavailable = True
        return self._go_result


def _unavailable_job(comp, drv=None, threshold=0.8, calm_ceiling=0.0, pushes=None):
    async def notify(m):
        (pushes if pushes is not None else []).append(m)
    return UnavailableJob(comp, notify, interval=0.0, min_idle=900.0, cooldown=21600.0,
                          drives=drv, threshold=threshold, calm_ceiling=calm_ceiling, clock=WallClock())


@case
async def unavailable_gated_by_low_restlessness_drive():
    comp = UnavailableCompanion(idle=1000.0)
    drv = FakeDriveState(restlessness=0.3)   # below the (higher-than-reflection) threshold
    pushes = []
    await _unavailable_job(comp, drv=drv, threshold=0.8, pushes=pushes).run()
    assert comp.go_calls == 0 and pushes == []


@case
async def unavailable_fires_on_high_restlessness_and_notifies():
    comp = UnavailableCompanion(idle=5.0)   # idle is irrelevant; the drive is the gate
    drv = FakeDriveState(restlessness=0.9)
    pushes = []
    await _unavailable_job(comp, drv=drv, threshold=0.8, pushes=pushes).run()
    assert comp.go_calls == 1
    assert drv.discharged == ["restlessness"]
    assert pushes == [{"type": "unavailable", "reason": "journaling", "eta_secs": 120,
                       "push": "stepped away — journaling (back in 2m)"}]


@case
async def unavailable_notice_carries_phone_wording():
    """The UI renders this event from reason/eta and needs no `content` — but the phone
    push body came from `content`, so stepping away fired a BLANK notification."""
    comp = UnavailableCompanion(idle=5.0, reason="sitting with a question", eta=3600.0)
    pushes = []
    await _unavailable_job(comp, drv=FakeDriveState(restlessness=0.9), pushes=pushes).run()
    body = pushes[0]["push"]
    assert body.strip(), "an unavailable notice must carry a non-empty phone body"
    assert "sitting with a question" in body and "1h" in body, body


@case
async def unavailable_notice_survives_a_missing_reason():
    comp = UnavailableCompanion(idle=5.0, reason=None, eta=45.0)
    pushes = []
    await _unavailable_job(comp, drv=FakeDriveState(restlessness=0.9), pushes=pushes).run()
    assert pushes[0]["reason"] == "taking a moment", "None reason must not reach the UI"
    assert pushes[0]["push"] == "stepped away — taking a moment (back in 45s)"


@case
async def unavailable_declining_sends_no_notice():
    comp = UnavailableCompanion(idle=5.0, go_result=False)
    drv = FakeDriveState(restlessness=0.9)
    pushes = []
    await _unavailable_job(comp, drv=drv, threshold=0.8, pushes=pushes).run()
    assert comp.go_calls == 1 and pushes == [], "a decline (PASS) must not announce anything"


@case
async def unavailable_refuses_while_already_unavailable():
    comp = UnavailableCompanion(idle=5.0, unavailable=True)
    drv = FakeDriveState(restlessness=0.9)
    await _unavailable_job(comp, drv=drv, threshold=0.8).run()
    assert comp.go_calls == 0, "must not stack a second unavailable window on the first"


@case
async def unavailable_refuses_with_a_pending_message():
    comp = UnavailableCompanion(idle=5.0, pending=True)
    drv = FakeDriveState(restlessness=0.9)
    await _unavailable_job(comp, drv=drv, threshold=0.8).run()
    assert comp.go_calls == 0


@case
async def unavailable_refuses_right_after_a_heavy_disclosure():
    comp = UnavailableCompanion(idle=1000.0)
    comp.emotion = FakeArousal(3.0)   # something heavy just happened
    drv = FakeDriveState(restlessness=0.9)
    await _unavailable_job(comp, drv=drv, threshold=0.8, calm_ceiling=2.0).run()
    assert comp.go_calls == 0, "never step away right after a heavy disclosure"


@case
async def unavailable_fires_once_calm_again():
    comp = UnavailableCompanion(idle=1000.0)
    comp.emotion = FakeArousal(0.1)   # calm
    drv = FakeDriveState(restlessness=0.9)
    await _unavailable_job(comp, drv=drv, threshold=0.8, calm_ceiling=2.0).run()
    assert comp.go_calls == 1


@case
async def unavailable_skips_when_busy():
    comp = UnavailableCompanion(idle=1000.0, busy=True)
    drv = FakeDriveState(restlessness=0.9)
    await _unavailable_job(comp, drv=drv, threshold=0.8).run()
    assert comp.go_calls == 0, "must not step away mid-turn"


class ReturnCompanion:
    def __init__(self, unavailable, busy=False, pending=False, end_result=None):
        self._unavailable = unavailable
        self._busy = busy
        self._pending = pending
        self._end_result = end_result
        self.end_calls = 0

    def is_unavailable(self):
        return self._unavailable

    def is_busy(self):
        return self._busy

    def has_pending_message(self):
        return self._pending

    async def end_unavailable(self):
        self.end_calls += 1
        return self._end_result


class FakeTurnResult:
    def __init__(self, text, silent=False):
        self.text = text
        self.silent = silent


@case
async def pursuit_return_waits_out_the_window():
    comp = ReturnCompanion(unavailable=True, pending=True)
    pushes = []
    async def notify(m):
        pushes.append(m)
    await PursuitReturnJob(comp, notify, interval=0.0).run()
    assert comp.end_calls == 0 and pushes == [], "must not return before the window elapses"


@case
async def pursuit_return_noop_without_a_pending_message():
    comp = ReturnCompanion(unavailable=False, pending=False)
    pushes = []
    async def notify(m):
        pushes.append(m)
    await PursuitReturnJob(comp, notify, interval=0.0).run()
    assert comp.end_calls == 0 and pushes == []


@case
async def pursuit_return_delivers_the_real_reply():
    comp = ReturnCompanion(unavailable=False, pending=True,
                          end_result=FakeTurnResult("here's my real answer"))
    pushes = []
    async def notify(m):
        pushes.append(m)
    await PursuitReturnJob(comp, notify, interval=0.0).run()
    assert comp.end_calls == 1
    assert pushes == [{"type": "delayed_reply", "content": "here's my real answer"}]


@case
async def pursuit_return_says_nothing_on_a_silent_turn():
    comp = ReturnCompanion(unavailable=False, pending=True,
                          end_result=FakeTurnResult("", silent=True))
    pushes = []
    async def notify(m):
        pushes.append(m)
    await PursuitReturnJob(comp, notify, interval=0.0).run()
    assert comp.end_calls == 1 and pushes == [], "a silent return must not push an empty reply"


@case
async def pursuit_return_skips_while_busy():
    comp = ReturnCompanion(unavailable=False, busy=True, pending=True)
    pushes = []
    async def notify(m):
        pushes.append(m)
    await PursuitReturnJob(comp, notify, interval=0.0).run()
    assert comp.end_calls == 0


@case
async def start_stop_lifecycle():
    j = CountingJob("j", interval=0.0)
    loop = TickLoop([j], interval=0.01)      # real clock, tiny interval
    loop.start()
    await asyncio.sleep(0.05)
    await loop.stop()
    assert j.runs >= 1, j.runs
    await loop.stop()                        # idempotent


# --- the A4 window silences model calls, but NOT her inner life -------------------
# go_unavailable() unloads the model so the GPU is free for the window. Nothing was
# stopping the other jobs from firing during it, so a background job JIT-reloaded the
# model seconds later (idle consolidation, near-certain: its 180s threshold is inside
# most windows and being unavailable requires the user to be idle), and reach-out /
# follow-up could send an unprompted message right after she'd said she'd stepped away.

@case
async def unavailable_silences_every_model_calling_job():
    away = {"is_unavailable": lambda: True}

    reach = ReachCompanion(idle=99999.0, reach_result="hi")
    reach.is_unavailable = away["is_unavailable"]
    pushes = []
    await _reach_job(reach, WallClock(), pushes).run()
    assert reach.reach_calls == 0, "must not reach out while she's stepped away"
    assert pushes == []

    refl = ReflectCompanion(idle=99999.0)
    refl.is_unavailable = away["is_unavailable"]
    await ReflectionJob(refl, interval=0.0, min_idle=120.0,
                        cooldown=600.0, clock=WallClock()).run()
    assert refl.reflect_calls == 0, "must not reflect during the window"

    foll = FollowCompanion(left=1, since=60.0)
    foll.is_unavailable = away["is_unavailable"]
    fpushes = []
    await _follow_job(foll, fpushes).run()
    assert foll.follow_calls == 0, "a double-text while away contradicts the window"

    cons = FakeCompanion(idle=1000.0, pending=4)
    cons.is_unavailable = away["is_unavailable"]
    await IdleConsolidationJob(cons, 0.0, 180.0).run()
    assert cons.flushed == 0, "consolidation would reload the model it just unloaded"


@case
async def unavailable_does_not_freeze_mood_or_drives():
    """The counterweight to the test above: her inner state must keep moving while
    she's away, exactly as it does while asleep. Both drift jobs are deliberately
    plain Jobs (no model call), so the CooldownJob guard must not reach them."""
    meta = InMemoryMeta()
    emo = EmotionManager(FakeClassifier(), meta, pull_strength=0.4, noise_floor=0.05)
    emo.state["melancholy"] = 0.9

    comp = FakeCompanion(idle=1000.0)
    comp.is_unavailable = lambda: True
    await MoodDriftJob(comp, emo, interval=0.0, idle_after=90.0).run()
    assert emo.state["melancholy"] < 0.9, "mood must still decay while she's away"

    drives = FakeDrives()
    dcomp = DriveCompanion(idle=1000.0)
    dcomp.is_unavailable = lambda: True
    await DriveDriftJob(dcomp, drives, interval=0.0).run()
    assert len(drives.updates) == 1, "drives must keep integrating while she's away"


@case
async def pursuit_return_still_runs_while_unavailable():
    """PursuitReturnJob is the one job that MUST survive the guard — it's what ends
    the window. It does its own is_unavailable() check for the opposite reason."""
    pushes = []

    async def notify(m):
        pushes.append(m)

    comp = ReturnCompanion(unavailable=True, pending=True)
    await PursuitReturnJob(comp, notify, interval=0.0).run()
    assert comp.end_calls == 0, "still inside the window -> don't return yet"
    comp._unavailable = False
    await PursuitReturnJob(comp, notify, interval=0.0).run()
    assert comp.end_calls == 1, "window over -> deliver the pending reply"


if __name__ == "__main__":
    raise SystemExit(run())
