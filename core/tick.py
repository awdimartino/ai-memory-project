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
    """Once the user has been away a while, maybe send an unprompted message.

    A cheap gate first (idle >= min_idle, and a persisted cooldown so attempts can't
    hammer — restart-safe via wall clock), then one model call via `companion.reach_out()`,
    which may decline. The attempt timestamp is written *before* generating, so the
    cooldown holds even on a decline and two reach-outs can't overlap. `notify` pushes
    the message to the surface (the web app's WebSocket broadcaster).
    """
    name = "reach_out"

    def __init__(self, companion, notify, interval: float, min_idle: float,
                 cooldown: float, clock=time.time):
        self.companion = companion
        self.notify = notify
        self.interval = interval
        self.min_idle = min_idle
        self.cooldown = cooldown
        self.clock = clock

    async def run(self) -> None:
        if self.companion.idle_seconds() < self.min_idle:
            return
        now = self.clock()
        last = self.companion.meta.get_json(LAST_REACHOUT_KEY, 0) or 0
        if now - last < self.cooldown:
            return
        # Mark the attempt before generating: enforces the cooldown even if she declines,
        # and stops a slow generation from letting the next tick start a second reach-out.
        await asyncio.to_thread(self.companion.meta.set_json, LAST_REACHOUT_KEY, now)
        message = await self.companion.reach_out()
        if message:
            await self.notify({"type": "proactive", "content": message})


class IdleConsolidationJob(Job):
    """Once the user has been away a while, consolidate any pending messages so
    facts get saved without waiting for a full context window."""
    name = "idle_consolidation"

    def __init__(self, companion, interval: float, idle_after: float):
        self.companion = companion
        self.interval = interval
        self.idle_after = idle_after

    async def run(self) -> None:
        if self.companion.idle_seconds() < self.idle_after:
            return
        if self.companion.pending_count() == 0:
            return
        n = self.companion.pending_count()
        await self.companion.flush()
        logger.info("idle consolidation: flushed %d pending message(s)", n)
