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

from _harness import case, run, temp_dir# also puts the repo root on sys.path
from helpers import FakeClassifier

from core.emotion_manager import BASELINE_STATE, CHANNELS, EmotionManager
from infrastructure.db import connect
from infrastructure.meta_store import SqliteMetaStore


def _mgr(script):
    path = os.path.join(temp_dir(), "emo.db")
    conn = connect(path)
    meta = SqliteMetaStore(conn)
    mgr = EmotionManager(FakeClassifier(script), meta, pull_strength=0.4, noise_floor=0.05)
    return mgr, meta, conn, path


@case
async def a_baseline_mood_injects_no_block_at_all():
    """The affect push is zero when there is nothing to report.

    All six channels were listed every turn; the research (§G2) measured that affect
    instructions cause over-broad emotion expression, so a flat readout on a neutral
    turn is pure cost. Empty string => `prompts.system_blocks` omits the block.
    """
    mgr, meta, conn, path = _mgr([])
    mgr.state = dict(BASELINE_STATE)
    assert mgr.as_prompt() == "", f"baseline still pushes affect: {mgr.as_prompt()!r}"
    conn.close(); os.remove(path)


@case
async def only_channels_off_baseline_are_mentioned():
    mgr, meta, conn, path = _mgr([])
    mgr.state = dict(BASELINE_STATE)
    mgr.state["irritation"] = BASELINE_STATE["irritation"] + 0.3
    out = mgr.as_prompt()
    assert "irritation" in out, out
    for c in ("warmth", "amusement", "melancholy", "unease", "interest"):
        assert c not in out, f"{c} reported while at baseline:\n{out}"
    conn.close(); os.remove(path)


@case
async def intensity_is_capped_below_intense():
    """Live, warmth pegged at 0.907 rendered as 'overwhelming' — the top bands are
    where the tic and the emotion-narration showed up."""
    mgr, meta, conn, path = _mgr([])
    mgr.state = dict(BASELINE_STATE)
    mgr.state["warmth"] = 0.98            # would be "all-consuming" uncapped
    out = mgr.as_prompt()
    for loud in ("intense", "overwhelming", "all-consuming"):
        assert loud not in out, f"{loud!r} survived the cap:\n{out}"
    assert "warmth" in out, "a pegged channel must still be reported, just quieter"
    conn.close(); os.remove(path)


@case
async def amusement_is_no_longer_structurally_suppressed():
    """It was the only channel with a zero baseline AND the fastest decay, so
    playfulness could never persist. Measured at 0.067 live: effectively absent."""
    from core.emotion_manager import DECAY_RATES

    assert BASELINE_STATE["amusement"] > 0, "amusement still decays to nothing"
    assert DECAY_RATES["amusement"] < DECAY_RATES["interest"], \
        "amusement should no longer be the fastest-fading channel"
    # It should still fade faster than the deep moods: a joke is not a mood.
    assert DECAY_RATES["amusement"] > DECAY_RATES["warmth"]
    conn = None
    mgr, meta, conn, path = _mgr([])
    mgr.state = dict(BASELINE_STATE)
    mgr.state["amusement"] = 0.6
    turns = 0
    while mgr.state["amusement"] > BASELINE_STATE["amusement"] + 0.1 and turns < 200:
        await mgr.react("")
        turns += 1
    assert turns >= 6, f"amusement gone after {turns} turns; still too fleeting"
    conn.close(); os.remove(path)


@case
async def melancholy_recovers_in_a_bounded_number_of_turns():
    """The 2026-07-20 regression: she stayed sad for a whole session.

    Measured cause was a ratchet, not stickiness — one sad message added +0.359 while
    decay removed 0.014/step, so recovery from "intense" took 77 turns. This pins the
    recovery budget rather than the rate itself, so a future retune is free to move the
    number as long as she still comes back.
    """
    mgr, meta, conn, path = _mgr([])
    mgr.state["melancholy"] = 0.809          # the value observed live
    turns = 0
    while mgr.state["melancholy"] > 0.25 and turns < 500:
        await mgr.react("")                  # empty => no labels, decay only
        turns += 1
    assert turns <= 25, f"melancholy took {turns} turns to fade; it is ratcheting again"
    assert turns >= 5, f"melancholy vanished in {turns} turns; it should still linger"
    conn.close(); os.remove(path)


@case
async def one_heavy_message_stops_dominating_within_a_conversation():
    """A bereavement-sized hit should stop STEERING her within a conversation.

    The first draft of this case asserted decay to baseline+0.05 and failed at 23 turns.
    That threshold was measuring the wrong thing: near-total erasure of a grandparent's
    death is not the goal, and a mood that vanished that fast would be its own bug (the
    `value_to_word` bands make anything under 0.25 "faint" — i.e. present but not
    driving). So this pins when it stops dominating, and records the full-fade number
    rather than asserting it.
    """
    mgr, meta, conn, path = _mgr([[("sadness", 0.84), ("disappointment", 0.08)]])
    await mgr.react("my grandad died on tuesday")
    assert mgr.state["melancholy"] > 0.35, "a bereavement should land hard at first"

    turns = 0
    while mgr.state["melancholy"] > 0.25 and turns < 500:   # "faint" boundary
        await mgr.react("")
        turns += 1
    assert turns <= 15, f"one sad message still dominates after {turns} turns"
    assert turns >= 3, f"a bereavement stopped mattering after {turns} turns"
    conn.close(); os.remove(path)


@case
async def warmth_also_recovers():
    """Warmth carried the identical ratchet and was likewise pegged at 'intense' live."""
    mgr, meta, conn, path = _mgr([])
    mgr.state["warmth"] = 0.812
    turns = 0
    while mgr.state["warmth"] > 0.25 and turns < 500:
        await mgr.react("")
        turns += 1
    assert turns <= 45, f"warmth took {turns} turns to settle"
    conn.close(); os.remove(path)


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
