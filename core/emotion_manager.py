"""Emotion (pillar 2): the 6-channel mood model.

A Tier-1 autonomic stage (like recall): every user message is scored by the
classifier and folds into six mood channels, each decaying toward a baseline at
its own rate. The mood colors the persona's tone via the system prompt; it is
never referenced literally.

Ported from v1's EmotionManager, with two v2 changes (per V2_PLAN §2.3):
  - **persisted** to the MetaStore so mood survives restarts (v1 reset every launch)
  - **async + off the event loop**: the blocking classifier call runs in a threadpool

The 28->6 mapping, decay rates, baseline, and pull strength are the v1-tuned
values; a behavioral eval (scripts/emotion_eval.py) is the tuning surface.
"""
import asyncio
import logging

logger = logging.getLogger(__name__)

MOOD_STATE_KEY = "mood_state"

# The six channels the 28 GoEmotions labels fold into.
CHANNELS = ("irritation", "warmth", "amusement", "melancholy", "unease", "interest")

# label -> {channel: weight}. Neutral (and any unmapped label) contributes nothing.
CHANNEL_MAP = {
    "anger":          {"irritation": 1.0},
    "annoyance":      {"irritation": 0.8},
    "disapproval":    {"irritation": 0.6},
    "disgust":        {"irritation": 0.7},
    "love":           {"warmth": 1.0},
    "caring":         {"warmth": 0.9},
    "gratitude":      {"warmth": 0.7},
    "admiration":     {"warmth": 0.6},
    "approval":       {"warmth": 0.4},
    "relief":         {"warmth": 0.3},
    "amusement":      {"amusement": 1.0},
    "joy":            {"amusement": 0.6},
    "excitement":     {"amusement": 0.5},
    "pride":          {"amusement": 0.4},
    "sadness":        {"melancholy": 1.0},
    "grief":          {"melancholy": 1.0},
    "disappointment": {"melancholy": 0.7},
    "remorse":        {"melancholy": 0.6},
    "embarrassment":  {"melancholy": 0.4},
    "fear":           {"unease": 1.0},
    "nervousness":    {"unease": 0.8},
    "confusion":      {"unease": 0.5},
    "surprise":       {"unease": 0.3},
    "curiosity":      {"interest": 1.0},
    "realization":    {"interest": 0.7},
    "desire":         {"interest": 0.6},
    "optimism":       {"interest": 0.5},
    "neutral":        {},
}

# Per-channel pull toward baseline each turn. Warmth/melancholy are sticky (slow);
# amusement/interest are of-the-moment (fast).
DECAY_RATES = {
    "irritation": 0.15,
    "warmth":     0.03,
    "amusement":  0.25,
    "melancholy": 0.02,
    "unease":     0.10,
    "interest":   0.20,
}

BASELINE_STATE = {
    "irritation": 0.2,
    "warmth":     0.05,
    "amusement":  0.0,
    "melancholy": 0.1,
    "unease":     0.0,
    "interest":   0.15,
}


def value_to_word(value: float) -> str:
    if value < 0.05: return "absent"
    if value < 0.15: return "barely present"
    if value < 0.25: return "faint"
    if value < 0.35: return "mild"
    if value < 0.45: return "noticeable"
    if value < 0.55: return "moderate"
    if value < 0.65: return "pronounced"
    if value < 0.75: return "strong"
    if value < 0.85: return "intense"
    if value < 0.95: return "overwhelming"
    return "all-consuming"


class EmotionManager:
    def __init__(self, classifier, meta, pull_strength: float = 0.4, noise_floor: float = 0.05):
        self.classifier = classifier
        self.meta = meta
        self.pull = pull_strength
        self.noise = noise_floor
        # Load persisted mood, tolerating a missing/partial saved state.
        saved = (meta.get_json(MOOD_STATE_KEY) if meta else None) or {}
        self.state = {c: float(saved.get(c, BASELINE_STATE[c])) for c in CHANNELS}

    async def react(self, text: str) -> dict:
        """Score a message, shift + decay mood, persist, and report what happened.

        Returns {"detected": [{"label","score"}...], "mood": {channel: value}} for
        display. The classifier is blocking (torch), so it runs in a threadpool.
        """
        detected = await asyncio.to_thread(self.classifier.classify, text)

        for label, score in detected:
            if score < self.noise:
                continue
            for channel, weight in CHANNEL_MAP.get(label, {}).items():
                self.state[channel] += weight * score * self.pull

        for c in CHANNELS:  # clamp after summing all contributions
            self.state[c] = min(max(self.state[c], 0.0), 1.0)

        self._decay()

        if self.meta is not None:
            await asyncio.to_thread(self.meta.set_json, MOOD_STATE_KEY, self.state)

        significant = [{"label": l, "score": s} for l, s in detected
                       if s >= self.noise and l != "neutral"]
        return {"detected": significant[:6], "mood": dict(self.state)}

    def arousal(self) -> float:
        """How far the current mood sits from baseline, summed across channels.

        A cheap salience signal: near 0 during small talk, high after a conversation
        that actually moved her. Used to consolidate memory early on emotionally
        loaded windows instead of on a flat message count.
        """
        return sum(abs(self.state[c] - BASELINE_STATE[c]) for c in CHANNELS)

    async def drift(self) -> dict:
        """Decay mood one step toward baseline without any new input, and persist.

        Called by the tick loop while the user is away, so mood settles over time
        instead of freezing wherever the last message left it. Returns the new mood.
        """
        self._decay()
        if self.meta is not None:
            await asyncio.to_thread(self.meta.set_json, MOOD_STATE_KEY, self.state)
        return dict(self.state)

    async def reset(self) -> None:
        """Reset mood to baseline and persist (admin full-reset)."""
        self.state = {c: BASELINE_STATE[c] for c in CHANNELS}
        if self.meta is not None:
            await asyncio.to_thread(self.meta.set_json, MOOD_STATE_KEY, self.state)

    def _decay(self) -> None:
        for c in CHANNELS:
            self.state[c] += DECAY_RATES[c] * (BASELINE_STATE[c] - self.state[c])

    def as_prompt(self) -> str:
        """Fold the current mood into the system prompt (never referenced literally)."""
        body = "\n".join(f"- {c}: {value_to_word(self.state[c])}" for c in CHANNELS)
        return (
            f"Your current emotional state:\n{body}\n\n"
            f"Let this shape how you come across and how you act right now, not just your "
            f"word choice. Never reference these values or name your emotions directly."
        )
