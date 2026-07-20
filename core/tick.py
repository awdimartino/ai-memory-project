"""The tick loop (pillar 3, proactivity) — an internal heartbeat.

A small **pluggable job scheduler**, not a hardcoded sequence (V2_PLAN §1.3): the
loop wakes on a cadence and runs whichever jobs are due. New autonomy behaviors
register as jobs without touching the loop.

This first slice ships two *internal* jobs — mood drift and idle consolidation —
that only act once the user has been away for a while, so nothing fires
mid-conversation. The outward reach-out (an unprompted message pushed over the
WebSocket) is a later slice that plugs in as another job.

Model calls made by jobs go through the same `LLMClient` lock as chat, so the
"one model at a time" rule holds. A job that raises is logged and skipped; it
never kills the loop.
"""
import asyncio
import logging
import random
import time

logger = logging.getLogger(__name__)


class Job:
    """A unit of autonomous work. Subclasses set `name` and implement `run`.

    `interval` is the minimum seconds between runs (0 = every tick). Any further
    gating (e.g. "only when the user is idle") lives inside `run`.
    """
    name: str = "job"
    interval: float = 0.0

    async def run(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError


class TickLoop:
    def __init__(self, jobs, interval: float, clock=time.monotonic):
        self.jobs = list(jobs)
        self.interval = interval
        self.clock = clock
        self._last: dict[str, float] = {}
        self._task: asyncio.Task | None = None
        self._stop = asyncio.Event()

    async def run_due(self) -> None:
        """Run every job whose interval has elapsed. One tick. Testable directly."""
        now = self.clock()
        for job in self.jobs:
            last = self._last.get(job.name)
            if last is not None and now - last < job.interval:
                continue
            self._last[job.name] = now
            try:
                await job.run()
            except Exception:  # noqa: BLE001 - one bad job must not stop the heartbeat
                logger.exception("tick job %r failed", job.name)

    async def _loop(self) -> None:
        logger.info("tick loop started (interval %.0fs, jobs: %s)",
                    self.interval, ", ".join(j.name for j in self.jobs))
        while not self._stop.is_set():
            await self.run_due()
            try:  # interruptible sleep so stop() is responsive
                await asyncio.wait_for(self._stop.wait(), timeout=self.interval)
            except asyncio.TimeoutError:
                pass

    def register(self, job: "Job") -> None:
        """Add a job. Entry points use this to plug in surface-specific jobs (e.g. the
        web app adds the reach-out job with its WebSocket notifier) before start()."""
        self.jobs.append(job)

    def start(self) -> None:
        if self._task is None:
            self._stop.clear()
            self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        self._stop.set()
        if self._task is not None:
            await self._task
            self._task = None
        logger.info("tick loop stopped")


class DriveDriftJob(Job):
    """Integrate Mari's internal drives + energy each tick (multi-drive proactivity).

    Away-drives rise while the user is away (connection scaled by mood, restlessness by boredom)
    and relax while present; energy depletes while awake and restores while asleep. Reach-out and
    reflection gate on these drives (this job just keeps them current). Runs even while asleep:
    it's cheap state with no model call, and energy recharging + connection building during sleep
    is the basis for a future principled self-wake.
    """
    name = "drive_drift"

    def __init__(self, companion, drives, interval: float):
        self.companion = companion
        self.drives = drives
        self.interval = interval

    async def run(self) -> None:
        if self.drives is None:
            return
        mood = self.companion.emotion.state if self.companion.emotion is not None else None
        # Pass asleep so energy restores during sleep and depletes while awake.
        state = await self.drives.update(
            self.companion.idle_seconds(), mood, self.companion.is_asleep())
        logger.info("drive drift: connection %.2f restlessness %.2f energy %.2f",
                    state.get("connection", 0), state.get("restlessness", 0), state.get("energy", 1))


class MoodDriftJob(Job):
    """While the user is away, decay mood one step toward baseline each run, so it
    settles over time instead of freezing where the last message left it."""
    name = "mood_drift"

    def __init__(self, companion, emotion, interval: float, idle_after: float):
        self.companion = companion
        self.emotion = emotion
        self.interval = interval
        self.idle_after = idle_after

    async def run(self) -> None:
        if self.emotion is None:
            return
        if self.companion.idle_seconds() < self.idle_after:
            return  # user is active; per-message reaction is handling mood
        mood = await self.emotion.drift()
        dom = max(mood, key=mood.get)
        logger.info("mood drift (idle): %s %.2f", dom, mood[dom])


LAST_REACHOUT_KEY = "last_reachout_at"


class ReachOutJob(Job):
    """Once the urge to connect is strong enough, maybe send an unprompted message.

    Gated on the **`connection` drive** (multi-drive proactivity): the drive integrates how
    long the user's been away *and* how she feels — a warm or sad conversation makes her miss
    them faster than a throwaway one — so this fires sooner then, later otherwise. The drive
    resets to baseline on contact, so it inherently can't cross mid-conversation. When drives
    are disabled it falls back to the old idle gate (`idle >= min_idle`). The persisted
    wall-clock cooldown is a hard floor either way, so she can't nag even if the drive spikes.

    The attempt marks the cooldown *and* discharges the drive **before** generating, so a
    decline or a slow generation can't start a second reach-out. `companion.reach_out()` may
    still decline (PASS). `notify` pushes the message to the surface (the WebSocket broadcaster).
    """
    name = "reach_out"

    def __init__(self, companion, notify, interval: float, min_idle: float,
                 cooldown: float, drives=None, threshold: float = 0.6, clock=time.time):
        self.companion = companion
        self.notify = notify
        self.interval = interval
        self.min_idle = min_idle
        self.cooldown = cooldown
        self.drives = drives
        self.threshold = threshold
        self.clock = clock

    def _wants_to(self) -> bool:
        """Drive threshold when drives are on; else the old idle gate (graceful fallback)."""
        if self.drives is not None:
            return self.drives.get("connection") >= self.threshold
        return self.companion.idle_seconds() >= self.min_idle

    async def run(self) -> None:
        # is_busy(): drive-gated jobs no longer inherit the idle busy-guard (drives are only
        # relieved at the END of send()), so guard explicitly against firing mid-turn.
        if self.companion.is_asleep() or self.companion.is_busy() or not self._wants_to():
            return
        now = self.clock()
        last = self.companion.meta.get_json(LAST_REACHOUT_KEY, 0) or 0
        if now - last < self.cooldown:
            return
        # Mark the attempt (cooldown + drive discharge) before generating: enforces the
        # cooldown even on a decline, and stops a slow generation from starting a second one.
        await asyncio.to_thread(self.companion.meta.set_json, LAST_REACHOUT_KEY, now)
        if self.drives is not None:
            await self.drives.discharge("connection")
        message = await self.companion.reach_out()
        if message:
            await self.notify({"type": "proactive", "content": message})


class FollowUpJob(Job):
    """After Mari replies, maybe fire a spontaneous follow-up a tick or a few later — like
    double-texting to add or elaborate on a thought.

    Distinct from reach-out (which is about long idle): this only fires in a short WINDOW right
    after her own message, while she's still the last speaker. Per-turn budget + a per-tick
    CHANCE keep it occasional and off-clockwork; `companion.follow_up()` generates and may PASS.
    Web-only (needs the socket broadcaster). `rng`/`clock` are injectable for tests.
    """
    name = "follow_up"

    def __init__(self, companion, notify, interval: float, chance: float,
                 min_delay: float, window: float, rng=random.random, clock=time.monotonic):
        self.companion = companion
        self.notify = notify
        self.interval = interval
        self.chance = chance
        self.min_delay = min_delay
        self.window = window
        self.rng = rng
        self.clock = clock

    async def run(self) -> None:
        c = self.companion
        if c.is_asleep() or c.is_busy() or c.followups_pending() <= 0:
            return
        elapsed = c.seconds_since_reply()
        if elapsed > self.window:
            c.cancel_followups()  # the moment has passed; don't follow up on a stale reply
            return
        if elapsed < self.min_delay:
            return
        if self.rng() >= self.chance:
            return  # not this tick — another roll next tick while still in the window
        message = await c.follow_up()
        if message:
            await self.notify({"type": "proactive", "content": message})


LAST_REFLECT_KEY = "last_reflect_at"


class ReflectionJob(Job):
    """While the user is away, have Mari write a short private thought to her journal.

    Gated on the **`restlessness` drive** (mental idleness) — same shape as reach-out but
    internal: nothing is pushed to the user, the thought is just stored via
    `companion.reflect()`. Falls back to the idle gate when drives are off; the persisted
    cooldown is a hard floor. A reflection only *partly* discharges restlessness (she can
    still be restless), so she may journal a few times over a long absence.
    """
    name = "reflection"

    def __init__(self, companion, interval: float, min_idle: float,
                 cooldown: float, drives=None, threshold: float = 0.4, clock=time.time):
        self.companion = companion
        self.interval = interval
        self.min_idle = min_idle
        self.cooldown = cooldown
        self.drives = drives
        self.threshold = threshold
        self.clock = clock

    def _wants_to(self) -> bool:
        """Drive threshold when drives are on; else the old idle gate (graceful fallback)."""
        if self.drives is not None:
            return self.drives.get("restlessness") >= self.threshold
        return self.companion.idle_seconds() >= self.min_idle

    async def run(self) -> None:
        # is_busy() guard: drive-gated, so it no longer inherits the idle busy-guard (drives
        # are only relieved at the end of send()) — don't reflect mid-turn.
        if self.companion.is_asleep() or self.companion.is_busy() or not self._wants_to():
            return
        now = self.clock()
        last = self.companion.meta.get_json(LAST_REFLECT_KEY, 0) or 0
        if now - last < self.cooldown:
            return
        await asyncio.to_thread(self.companion.meta.set_json, LAST_REFLECT_KEY, now)
        if self.drives is not None:
            await self.drives.discharge("restlessness")
        await self.companion.reflect()


LAST_INTENTION_KEY = "last_intention_at"


class IntentionJob(Job):
    """While the user is away, note forward intentions from the recent conversation — things Mari
    means to bring up or find out next time. Idle + cooldown gated (its own cadence); internal (no
    surface push). Reach-out consumes these to make its messages purposeful (the "planning" pillar)."""
    name = "intention"

    def __init__(self, companion, interval: float, min_idle: float, cooldown: float, clock=time.time):
        self.companion = companion
        self.interval = interval
        self.min_idle = min_idle
        self.cooldown = cooldown
        self.clock = clock

    async def run(self) -> None:
        if self.companion.is_asleep() or self.companion.is_busy():
            return
        if self.companion.idle_seconds() < self.min_idle:
            return
        now = self.clock()
        last = self.companion.meta.get_json(LAST_INTENTION_KEY, 0) or 0
        if now - last < self.cooldown:
            return
        await asyncio.to_thread(self.companion.meta.set_json, LAST_INTENTION_KEY, now)
        await self.companion.form_intentions()


LAST_PERSONA_EDIT_KEY = "last_persona_edit_at"


class PersonaEditJob(Job):
    """During idle, let Mari rewrite her own self-description (the self-modifying persona).

    Slow and rare: needs a minimum amount of history first (`min_messages`), then a long
    cooldown between edits. `companion.edit_persona()` reads her thoughts + core memories and
    is gated by familiarity so a stranger doesn't rewrite herself into a best friend.
    """
    name = "persona_edit"

    def __init__(self, companion, interval: float, min_idle: float, cooldown: float,
                 min_messages: int, clock=time.time):
        self.companion = companion
        self.interval = interval
        self.min_idle = min_idle
        self.cooldown = cooldown
        self.min_messages = min_messages
        self.clock = clock

    async def run(self) -> None:
        if self.companion.is_asleep() or self.companion.idle_seconds() < self.min_idle:
            return
        if self.companion.store.message_count() < self.min_messages:
            return  # too early in the relationship to have a developed self
        now = self.clock()
        last = self.companion.meta.get_json(LAST_PERSONA_EDIT_KEY, 0) or 0
        if now - last < self.cooldown:
            return
        await asyncio.to_thread(self.companion.meta.set_json, LAST_PERSONA_EDIT_KEY, now)
        await self.companion.edit_persona()


class SleepJob(Job):
    """Put Mari into standby: unload the LLM from VRAM to free the machine. Two triggers:
    a **long idle** (the practical VRAM-freeing one) OR **low energy** while briefly idle
    (the body-cycle one — she's tired, arc A2). The energy path still needs a small idle gap
    (`energy_min_idle`) so she never nods off mid-conversation; the busy guard already zeroes
    idle during a turn. The heartbeat keeps ticking (mood/energy still move); a message wakes her."""
    name = "sleep"

    def __init__(self, companion, interval: float, sleep_after: float,
                 drives=None, energy_threshold: float = 0.15, energy_min_idle: float = 120.0):
        self.companion = companion
        self.interval = interval
        self.sleep_after = sleep_after
        self.drives = drives
        self.energy_threshold = energy_threshold
        self.energy_min_idle = energy_min_idle

    async def run(self) -> None:
        if self.companion.is_asleep():
            return
        idle = self.companion.idle_seconds()
        tired = (self.drives is not None
                 and self.drives.energy() <= self.energy_threshold
                 and idle >= self.energy_min_idle)
        if idle < self.sleep_after and not tired:
            return
        await self.companion.sleep()


class IdleConsolidationJob(Job):
    """Once the user has been away a while, consolidate any pending messages so
    facts get saved without waiting for a full context window."""
    name = "idle_consolidation"

    def __init__(self, companion, interval: float, idle_after: float):
        self.companion = companion
        self.interval = interval
        self.idle_after = idle_after

    async def run(self) -> None:
        if self.companion.is_asleep():
            return  # a model call would defeat standby
        if self.companion.idle_seconds() < self.idle_after:
            return
        if self.companion.pending_count() == 0:
            return
        n = self.companion.pending_count()
        await self.companion.flush()
        logger.info("idle consolidation: flushed %d pending message(s)", n)
