"""The tick loop (pillar 3, proactivity) — an internal heartbeat.

A small **pluggable job scheduler**, not a hardcoded sequence (V2_PLAN §1.3): the
loop wakes on a cadence and runs whichever jobs are due. New autonomy behaviors
register as jobs without touching the loop.

Jobs come in three shapes. Plain `Job`s run on their interval and self-gate
(mood drift, drive drift, idle consolidation, follow-up, sleep). `CooldownJob`
adds the shared protocol the autonomy jobs all wanted — asleep/busy guard, an
idle gate, a persisted wall-clock cooldown, stamp-before-acting — and
`DriveGatedJob` swaps that idle gate for an internal drive crossing a threshold,
so *how she feels* sets the timing rather than a fixed timer.

Internal jobs act only once the user has been away, so nothing fires
mid-conversation. Outward ones (reach-out, follow-up) need a surface to push to,
so the web entry point registers those rather than shared bootstrap.

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


class CooldownJob(Job):
    """Base for the idle-gated, cooldown-throttled autonomy jobs.

    Five jobs repeated this protocol verbatim, so it lives here once:

        asleep / mid-turn guard -> does she want to? -> persisted wall-clock
        cooldown -> stamp the attempt -> act

    **Stamping before acting is load-bearing**, not an ordering detail: it enforces
    the cooldown even when she declines (PASS), and stops a slow generation from
    letting the next tick start a second one.

    The cooldown is persisted (in the MetaStore, under `meta_key`) and read in wall
    time, so a restart doesn't reset the throttle.

    Subclasses set `name` + `meta_key` and implement `act()`; they may override
    `_wants_to()` (what triggers it), `_ready()` (extra preconditions), and
    `_on_fire()` (side effects once it's committed to acting).
    """
    meta_key: str = ""

    def __init__(self, companion, interval: float, min_idle: float,
                 cooldown: float, clock=time.time):
        self.companion = companion
        self.interval = interval
        self.min_idle = min_idle
        self.cooldown = cooldown
        self.clock = clock

    def _wants_to(self) -> bool:
        """Left alone long enough? Drive-gated subclasses replace this with a threshold."""
        return self.companion.idle_seconds() >= self.min_idle

    def _ready(self) -> bool:
        """Extra precondition beyond wanting to (e.g. enough history). Default: always."""
        return True

    async def _on_fire(self) -> None:
        """Committed to acting, cooldown already stamped — discharge drives etc."""

    async def act(self) -> None:  # pragma: no cover - overridden
        raise NotImplementedError

    async def run(self) -> None:
        c = self.companion
        # is_busy(): drive-gated jobs don't inherit the idle busy-guard (drives are
        # only relieved at the END of send()), so guard explicitly against firing
        # mid-turn. Harmless for the idle-gated ones, whose idle already reads 0.
        if c.is_asleep() or c.is_busy() or not self._wants_to() or not self._ready():
            return
        now = self.clock()
        if now - (c.meta.get_json(self.meta_key, 0) or 0) < self.cooldown:
            return
        await asyncio.to_thread(c.meta.set_json, self.meta_key, now)
        await self._on_fire()
        await self.act()


class DriveGatedJob(CooldownJob):
    """A CooldownJob triggered by an internal drive crossing a threshold.

    The drive replaces the idle gate — so *how she feels* sets the timing, not a
    fixed timer — and falls back to that idle gate when drives are disabled. Firing
    discharges the drive by its own per-drive amount (connection fully, restlessness
    only partly, so she may journal a few times over a long absence). The persisted
    cooldown stays a hard floor over the drive, so a spike still can't make her nag.
    """
    drive: str = ""

    def __init__(self, companion, interval: float, min_idle: float, cooldown: float,
                 drives=None, threshold: float = 0.6, clock=time.time):
        super().__init__(companion, interval, min_idle, cooldown, clock)
        self.drives = drives
        self.threshold = threshold

    def _wants_to(self) -> bool:
        if self.drives is not None:
            return self.drives.get(self.drive) >= self.threshold
        return super()._wants_to()

    async def _on_fire(self) -> None:
        if self.drives is not None:
            await self.drives.discharge(self.drive)


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


class ReachOutJob(DriveGatedJob):
    """Once the urge to connect is strong enough, maybe send an unprompted message.

    Gated on the **`connection` drive**: it integrates how long the user's been away
    *and* how she feels — a warm or sad conversation makes her miss them faster than a
    throwaway one — so this fires sooner then, later otherwise. The drive resets on
    contact, so it inherently can't cross mid-conversation.

    `companion.reach_out()` may still decline (PASS). `notify` pushes a real message to
    the surface (the WebSocket broadcaster).
    """
    name = "reach_out"
    meta_key = LAST_REACHOUT_KEY
    drive = "connection"

    def __init__(self, companion, notify, interval: float, min_idle: float,
                 cooldown: float, drives=None, threshold: float = 0.6, clock=time.time):
        super().__init__(companion, interval, min_idle, cooldown, drives, threshold, clock)
        self.notify = notify

    async def act(self) -> None:
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


class ReflectionJob(DriveGatedJob):
    """While the user is away, have Mari write a short private thought to her journal.

    Same shape as reach-out but internal: nothing is pushed to the user, the thought is
    just stored via `companion.reflect()`. Gated on **`restlessness`** (mental idleness),
    which only partly discharges — so she may journal a few times over a long absence.
    """
    name = "reflection"
    meta_key = LAST_REFLECT_KEY
    drive = "restlessness"

    def __init__(self, companion, interval: float, min_idle: float,
                 cooldown: float, drives=None, threshold: float = 0.4, clock=time.time):
        super().__init__(companion, interval, min_idle, cooldown, drives, threshold, clock)

    async def act(self) -> None:
        await self.companion.reflect()


LAST_INTENTION_KEY = "last_intention_at"


class IntentionJob(CooldownJob):
    """While the user is away, note forward intentions from the recent conversation — things Mari
    means to bring up or find out next time. Idle + cooldown gated (its own cadence); internal (no
    surface push). Reach-out consumes these to make its messages purposeful (the "planning" pillar)."""
    name = "intention"
    meta_key = LAST_INTENTION_KEY

    async def act(self) -> None:
        await self.companion.form_intentions()


LAST_SELFNOTES_KEY = "last_self_notes_at"


class SelfNotesJob(CooldownJob):
    """While the user is away, distill what the last conversation taught Mari about HOW to be with
    them ("ease off the questions") into her operating-notes slot. Idle + cooldown gated; internal
    (no surface push). Sibling to PersonaEditJob: that one revises who she is, this one how she acts."""
    name = "self_notes"
    meta_key = LAST_SELFNOTES_KEY

    async def act(self) -> None:
        await self.companion.update_self_notes()


LAST_PERSONA_EDIT_KEY = "last_persona_edit_at"


class PersonaEditJob(CooldownJob):
    """During idle, let Mari rewrite her own self-description (the self-modifying persona).

    Slow and rare: needs a minimum amount of history first (`min_messages`), then a long
    cooldown between edits. `companion.edit_persona()` reads her thoughts + core memories and
    is gated by familiarity so a stranger doesn't rewrite herself into a best friend.
    """
    name = "persona_edit"
    meta_key = LAST_PERSONA_EDIT_KEY

    def __init__(self, companion, interval: float, min_idle: float, cooldown: float,
                 min_messages: int, clock=time.time):
        super().__init__(companion, interval, min_idle, cooldown, clock)
        self.min_messages = min_messages

    def _ready(self) -> bool:
        # Too early in the relationship to have a developed self.
        return self.companion.message_count() >= self.min_messages

    async def act(self) -> None:
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
