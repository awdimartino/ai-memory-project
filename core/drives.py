"""Internal drives (multi-drive proactivity — roadmap arc A1, "observe first" slice).

Slow-integrating scalars in [0,1] that rise while the user is away and are
discharged by the behavior they motivate. Where the tick's fixed idle thresholds
ask only "has it been N seconds?", a drive folds *how long* she's been alone
together with *how she feels* into one pressure — deterministic, but more lifelike
than a stopwatch. This is the generalization of the existing tick gates flagged in
V2_PLAN §2.9.

A Tier-1 autonomic substrate, sibling to EmotionManager:
  - updated every tick from **elapsed wall-time** (dt via an injected clock), so the
    dynamics are independent of TICK_INTERVAL (unlike mood's per-tick decay) and fully
    deterministic under a fake clock in tests
  - **persisted** to the MetaStore so a drive survives restarts

This first slice ships two drives and only *observes* them (the panel shows them
move); the behaviors stay on their existing idle gates until the drives are shown to
move sensibly against real use. Wiring a job onto a drive is then a one-line gate
change (`if drives.get("connection") < threshold: return`).

Drives:
  - connection   — the urge to reach out. Rises while away; warmth and melancholy in
                   the current mood make her miss the user faster, irritation slows it.
                   Reset by contact (a user message); a reach-out would discharge it.
  - restlessness — mental idleness that motivates an inner-life beat (reflection). Rises
                   faster than connection and is largely mood-independent, nudged up
                   when interest is low (boredom). Partly relieved by contact; a
                   reflection would discharge it.
"""
import asyncio
import logging
import time

logger = logging.getLogger(__name__)

DRIVE_STATE_KEY = "drive_state"

DRIVES = ("connection", "restlessness")

# Resting value each drive relaxes toward while the user is present.
BASELINE = {
    "connection": 0.0,
    "restlessness": 0.0,
}

# Rise per hour of continuous idle, under neutral mood. Tuned against the first live demo
# so each drive has a usable *gradient* up to and a little past its threshold, rather than
# pegging at the ceiling. connection ~0.6 at 15 min (≈ reach-out's gate, pulled earlier by a
# warm/sad mood); restlessness ~0.4 at ~5 min and saturates ~12 min (the demo showed the
# original 15/hr pegged it at 1.0 within 4 min — a step function, not a signal).
RISE_PER_HOUR = {
    "connection": 2.4,     # ~0.6 after 15 min idle
    "restlessness": 5.0,   # ~0.4 after 5 min idle, saturates ~12 min
}

# Fraction pulled back toward baseline per update while the user is present (idle below
# `away_after`), so a drive doesn't linger high mid-conversation.
PRESENT_RELAX = 0.5

# Mood modulation of the connection rise: multiplier = 1 + sum(weight * channel). Warmth
# and melancholy make her miss the user faster; irritation makes her less eager.
CONNECTION_MOOD_WEIGHTS = {"warmth": 0.8, "melancholy": 0.6, "irritation": -0.7}
# Boredom nudge for restlessness: low interest speeds it up (multiplier 1 + w*(1-interest)).
RESTLESS_BOREDOM_WEIGHT = 0.5

# Discharge subtracted when a drive's behavior fires (floored at baseline). Reserved for
# the rewire slice; exposed now via discharge() so a job can call it in one line.
DISCHARGE = {
    "connection": 1.0,     # a reach-out fully satisfies the urge
    "restlessness": 0.6,   # a single reflection takes the edge off
}

# How much a user message relieves each drive (0 = none, 1 = full reset to baseline).
CONTACT_RELIEF = {
    "connection": 1.0,     # talking to her fully satisfies the urge to connect
    "restlessness": 0.5,   # interaction is stimulating, but she can still be restless
}


def value_to_word(v: float) -> str:
    """Human phrasing of a drive level (for legibility, mirrors emotion's value_to_word)."""
    if v < 0.15: return "quiet"
    if v < 0.35: return "stirring"
    if v < 0.55: return "noticeable"
    if v < 0.75: return "strong"
    if v < 0.90: return "pressing"
    return "urgent"


class DriveManager:
    """Holds the drives and integrates them over wall-time. Persistence via MetaStore.

    `away_after` is the idle threshold past which the user counts as "away" and drives
    start rising; below it they relax toward baseline. `clock` is injectable for tests.
    """

    def __init__(self, meta, away_after: float, clock=time.monotonic):
        self.meta = meta
        self.away_after = away_after
        self.clock = clock
        saved = (meta.get_json(DRIVE_STATE_KEY) if meta else None) or {}
        self.state = {d: float(saved.get(d, BASELINE[d])) for d in DRIVES}
        self._last = self.clock()

    async def update(self, idle_seconds: float, mood: dict | None = None) -> dict:
        """Integrate the drives over elapsed wall-time and persist. Called each tick.

        While the user is away (idle >= away_after) each drive rises at its rate —
        connection scaled by mood, restlessness by boredom; while present, drives relax
        toward baseline. Returns the new state.
        """
        now = self.clock()
        dt = now - self._last
        self._last = now
        if dt <= 0:
            return dict(self.state)
        hours = dt / 3600.0

        if idle_seconds >= self.away_after:
            self.state["connection"] += (
                RISE_PER_HOUR["connection"] * self._connection_mult(mood) * hours)
            self.state["restlessness"] += (
                RISE_PER_HOUR["restlessness"] * self._restless_mult(mood) * hours)
        else:  # user present — settle back toward baseline
            for d in DRIVES:
                self.state[d] += PRESENT_RELAX * (BASELINE[d] - self.state[d])

        self._clamp()
        await self._persist()
        return dict(self.state)

    def _connection_mult(self, mood: dict | None) -> float:
        if not mood:
            return 1.0
        m = 1.0 + sum(w * mood.get(ch, 0.0) for ch, w in CONNECTION_MOOD_WEIGHTS.items())
        return max(0.0, m)  # a foul mood can zero the urge, never reverse it

    def _restless_mult(self, mood: dict | None) -> float:
        if not mood:
            return 1.0
        return 1.0 + RESTLESS_BOREDOM_WEIGHT * (1.0 - mood.get("interest", 0.0))

    async def discharge(self, drive: str) -> None:
        """A drive's behavior fired: subtract its discharge, floor at baseline, persist.

        Not called yet in this "observe first" slice — the behaviors stay on their idle
        gates — but exposed so the rewire is a one-liner in each job.
        """
        self.state[drive] = max(BASELINE[drive], self.state[drive] - DISCHARGE[drive])
        self._clamp()
        await self._persist()

    async def on_user_message(self) -> None:
        """The user spoke: contact relieves the drives (connection fully, restlessness part)."""
        for d in DRIVES:
            self.state[d] += CONTACT_RELIEF[d] * (BASELINE[d] - self.state[d])
        self._clamp()
        await self._persist()

    def get(self, drive: str) -> float:
        return self.state[drive]

    def snapshot(self) -> dict:
        return dict(self.state)

    def _clamp(self) -> None:
        for d in DRIVES:
            self.state[d] = min(max(self.state[d], 0.0), 1.0)

    async def _persist(self) -> None:
        if self.meta is not None:
            await asyncio.to_thread(self.meta.set_json, DRIVE_STATE_KEY, self.state)
