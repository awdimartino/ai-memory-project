"""Behavioral eval for the emotion pillar (V2_PLAN §2.3 — never done in v1).

Loads the REAL RoBERTa GoEmotions classifier (CPU) and drives a scripted set of
utterances through the EmotionManager, printing the detected emotions and the
resulting 6-channel mood after each turn. Eyeball whether the channels move
sensibly; it's the surface for tuning PULL_STRENGTH / decay / the 28->6 mapping.

No LM Studio needed (this is the emotion model only). Downloads the classifier
on first run.

Run:  python scripts/emotion_eval.py
"""
import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.emotion_manager import CHANNELS, EmotionManager, value_to_word
from infrastructure.emotion_classifier import EmotionClassifier


class InMemoryMeta:
    """Ephemeral meta so the eval never touches companion.db."""

    def __init__(self):
        self._d = {}

    def get_json(self, key, default=None):
        return self._d.get(key, default)

    def set_json(self, key, value):
        self._d[key] = value


# Each line is aimed at one channel; neutral lines let mood decay back.
SCRIPT = [
    ("warmth",     "You're honestly the best, thank you so much for being here for me."),
    ("amusement",  "hahaha that is hilarious, I actually can't stop laughing"),
    ("neutral",    "ok. sure. anyway."),
    ("irritation", "This is so frustrating, I'm really annoyed that it broke again."),
    ("melancholy", "I've been feeling really down and kind of empty lately."),
    ("unease",     "I'm nervous and a little scared about the results tomorrow."),
    ("interest",   "wait, how does that actually work? I'm so curious now."),
    ("neutral",    "mm. okay then."),
    ("neutral",    "nothing much going on."),
]


def mood_line(mood: dict) -> str:
    return "  ".join(f"{c[:4]}={mood[c]:.2f}" for c in CHANNELS)


async def main() -> int:
    print(f"loading {config.EMOTION_MODEL} on CPU (first run downloads it)...")
    classifier = await asyncio.to_thread(EmotionClassifier, config.EMOTION_MODEL)
    mgr = EmotionManager(classifier, InMemoryMeta(),
                         pull_strength=config.EMOTION_PULL_STRENGTH,
                         noise_floor=config.EMOTION_NOISE_FLOOR)

    print(f"\nbaseline: {mood_line(mgr.state)}\n")
    for aim, text in SCRIPT:
        info = await mgr.react(text)
        felt = ", ".join(f"{d['label']} {d['score']:.2f}" for d in info["detected"][:5])
        print(f"[{aim:>10}] \"{text}\"")
        print(f"   detected: {felt or 'neutral'}")
        print(f"   mood:     {mood_line(info['mood'])}")
        dom = max(info["mood"], key=info["mood"].get)
        print(f"   dominant: {dom} ({value_to_word(info['mood'][dom])})\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
