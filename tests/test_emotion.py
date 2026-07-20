"""Offline coverage for the 6-channel mood logic.

Uses the real EmotionManager + real SqliteMetaStore (persistence) with a FAKE
classifier (scripted (label, score) lists). No torch / model download needed —
the payoff of splitting the classifier (infrastructure) from the mood dynamics
(core). Each case gets its own fresh DB.

Verifies: channel mapping raises the right mood, contributions clamp to [0,1],
mood decays toward baseline over neutral turns, the noise floor drops weak labels,
`detected` is filtered/capped, and mood persists across manager instances.

Run:  python tests/test_emotion.py
"""
import os
import sys
import tempfile

from _harness import case, run  # also puts the repo root on sys.path

from core.emotion_manager import BASELINE_STATE, CHANNELS, EmotionManager
from infrastructure.db import connect
from infrastructure.meta_store import SqliteMetaStore


class FakeClassifier:
    """Returns scripted (label, score) lists, one per react() call."""

    def __init__(self, script):
        self.script = list(script)

    def classify(self, text):
        return self.script.pop(0) if self.script else []


def _mgr(script):
    path = os.path.join(tempfile.mkdtemp(), "emo.db")
    conn = connect(path)
    meta = SqliteMetaStore(conn)
    mgr = EmotionManager(FakeClassifier(script), meta, pull_strength=0.4, noise_floor=0.05)
    return mgr, meta, conn, path


@case
async def warmth_rises_on_gratitude():
    mgr, meta, conn, path = _mgr([[("gratitude", 0.9), ("neutral", 0.1)]])
    before = mgr.state["warmth"]
    info = await mgr.react("thanks, that means a lot")
    assert mgr.state["warmth"] > before, "warmth should rise on gratitude"
    labels = [d["label"] for d in info["detected"]]
    assert "gratitude" in labels and "neutral" not in labels, labels
    conn.close(); os.remove(path)


@case
async def contributions_clamp_to_one():
    mgr, meta, conn, path = _mgr([[
        ("anger", 1.0), ("annoyance", 1.0), ("disapproval", 1.0), ("disgust", 1.0)]])
    await mgr.react("i am furious")
    assert 0.0 <= mgr.state["irritation"] <= 1.0, mgr.state["irritation"]
    assert mgr.state["irritation"] > BASELINE_STATE["irritation"], "irritation should spike"
    conn.close(); os.remove(path)


@case
async def neutral_leaves_mood_at_baseline():
    mgr, meta, conn, path = _mgr([[("neutral", 1.0)]])
    await mgr.react("ok")
    for c in CHANNELS:  # no contribution + decay from baseline == baseline
        assert abs(mgr.state[c] - BASELINE_STATE[c]) < 1e-9, (c, mgr.state[c])
    conn.close(); os.remove(path)


@case
async def mood_decays_toward_baseline():
    mgr, meta, conn, path = _mgr([[("joy", 0.9)]] + [[("neutral", 1.0)]] * 40)
    await mgr.react("yesss")
    spiked = mgr.state["amusement"]
    assert spiked > 0.1, "joy should raise amusement"
    for _ in range(40):
        await mgr.react("mm")
    settled = mgr.state["amusement"]
    assert settled < spiked, "amusement should decay"
    assert abs(settled - BASELINE_STATE["amusement"]) < 0.02, f"not near baseline: {settled}"
    conn.close(); os.remove(path)


@case
async def noise_floor_drops_weak_labels():
    mgr, meta, conn, path = _mgr([[("anger", 0.04)]])  # below the 0.05 floor
    info = await mgr.react("meh")
    assert abs(mgr.state["irritation"] - BASELINE_STATE["irritation"]) < 1e-9, "weak label moved mood"
    assert info["detected"] == [], f"weak label surfaced: {info['detected']}"
    conn.close(); os.remove(path)


@case
async def detected_is_capped_at_six():
    labels = [("curiosity", 0.9), ("joy", 0.8), ("excitement", 0.7), ("pride", 0.6),
              ("amusement", 0.5), ("optimism", 0.4), ("desire", 0.3), ("relief", 0.2)]
    mgr, meta, conn, path = _mgr([labels])
    info = await mgr.react("tell me everything")
    assert len(info["detected"]) == 6, f"expected 6, got {len(info['detected'])}"
    conn.close(); os.remove(path)


@case
async def mood_persists_across_instances():
    mgr, meta, conn, path = _mgr([[("love", 1.0)]])
    await mgr.react("i love this")
    saved = dict(mgr.state)
    # A fresh manager on the same store must resume the saved mood, not baseline.
    mgr2 = EmotionManager(FakeClassifier([]), meta)
    for c in CHANNELS:
        assert abs(mgr2.state[c] - saved[c]) < 1e-9, (c, mgr2.state[c], saved[c])
    assert mgr2.state["warmth"] > BASELINE_STATE["warmth"], "warmth should have persisted"
    conn.close(); os.remove(path)


if __name__ == "__main__":
    raise SystemExit(run())
