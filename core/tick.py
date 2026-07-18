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
    """Integrate Mari's internal drives each tick (multi-drive proactivity, arc A1).

    Drives rise while the user is away (connection scaled by mood, restlessness by
    boredom) and relax while present. This "observe first" slice only *updates and
    surfaces* the drives — the behaviors still fire on their own idle gates — so a bad
    drive can't yet cause or suppress an action. Runs even while asleep: drives are
    cheap state with no model call (like mood drift), and letting connection keep
    building while she sleeps is exactly the future basis for a principled self-wake.
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
        state = await self.drives.update(self.companion.idle_seconds(), mood)
        top = max(state, key=state.get)
        logger.info("drive drift: %s %.2f", top, state[top])


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
        if self.companion.is_asleep() or not self._wants_to():
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
        if self.companion.is_asleep() or not self._wants_to():
            return
        now = self.clock()
        last = self.companion.meta.get_json(LAST_REFLECT_KEY, 0) or 0
        if now - last < self.cooldown:
            return
        await asyncio.to_thread(self.companion.meta.set_json, LAST_REFLECT_KEY, now)
        if self.drives is not None:
            await self.drives.discharge("restlessness")
        await self.companion.reflect()


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
    """After a long idle, put Mari into standby: unload the LLM from VRAM to free the
    machine. The heartbeat keeps ticking (mood still drifts); the next message wakes her."""
    name = "sleep"

    def __init__(self, companion, interval: float, sleep_after: float):
        self.companion = companion
        self.interval = interval
        self.sleep_after = sleep_after

    async def run(self) -> None:
        if self.companion.is_asleep():
            return
        if self.companion.idle_seconds() < self.sleep_after:
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
