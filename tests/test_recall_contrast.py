"""Offline coverage for recall's contrast gate (config.RECALL_CONTRAST_GAP).

The gate exists because the absolute similarity floor was throwing away correct
top-1 results: measured on the gold set's six facts, correct hits ranged
0.448-0.644 while unrelated queries reached 0.576, so no floor separates them.
What separates them is how far the top hit stands above the corpus median.

Every similarity here is a REAL number from that measurement
(`scripts/recall_margin_probe.py`), so these cases pin the gate to the data that
justified it rather than to invented values.

**How the sims are controlled.** `_rank` takes an already-normalised corpus matrix
and normalises only the query, so an (n, 1) matrix holding the desired sims against
a unit query vector yields exactly those sims — no embedder, no fixture, no drift.

Run:  python tests/test_recall_contrast.py
"""
import numpy as np

from _harness import case, run  # also puts the repo root on sys.path

from core.memory_manager import MemoryManager

QUERY = np.array([1.0], dtype=np.float32)

GAP, FLOOR, MIN_CORPUS = 0.06, 0.42, 3   # the shipped defaults


def rank(sims, min_sim=0.55, top_k=5, **over):
    """Rank a corpus whose similarities to the query are exactly `sims`."""
    mems = [{"content": f"fact {i}"} for i in range(len(sims))]
    mn = np.array(sims, dtype=np.float32).reshape(-1, 1)
    kw = dict(contrast_gap=GAP, contrast_floor=FLOOR, contrast_min_corpus=MIN_CORPUS)
    kw.update(over)
    return MemoryManager._rank(QUERY, mems, mn, top_k, min_sim, **kw)


# Measured distributions, best-first. The recall-* names are gold-set case ids.
PLACE = [0.527, 0.408, 0.396, 0.390, 0.373, 0.352]      # "remind me where I live"
JOB = [0.525, 0.423, 0.405, 0.404, 0.366, 0.334]        # "what do I do for work again?"
EXCITED = [0.516, 0.460, 0.423, 0.415, 0.404, 0.359]    # "I'm so excited, do I have any pets?"
COMPOUND = [0.544, 0.541, 0.478, 0.466, 0.433, 0.416]   # "tell me about my dog and my job"
UNRELATED = [0.498, 0.489, 0.483, 0.480, 0.478, 0.415]  # "cold winter" — must NOT hit
ADJACENT = [0.457, 0.442, 0.407, 0.399, 0.385, 0.348]   # "getting a cat" — must NOT hit


@case
async def contrast_admits_what_the_floor_rejected():
    """The bug: correct top-1 at 0.527, discarded by a 0.55 floor."""
    for name, sims in (("place", PLACE), ("job", JOB), ("excited", EXCITED)):
        hits = rank(sims)
        assert hits and hits[0][0]["content"] == "fact 0", f"{name}: lost the correct top hit"
        assert len(hits) == 1, f"{name}: gate let in {len(hits)} hits, expected only the top"


@case
async def contrast_rejects_unrelated_query():
    """recall-none-unrelated: a flat distribution has no standout, whatever its level.

    0.498 is higher than several TRUE positives above — which is the whole reason
    the absolute score can't be the discriminator.
    """
    assert rank(UNRELATED) == [], "unrelated query produced a hit"


@case
async def contrast_rejects_adjacent_topic():
    """recall2-no-false-positive: 'getting a cat' must not drag in the dog fact."""
    assert rank(ADJACENT) == [], "adjacent topic produced a hit"


@case
async def compound_query_keeps_both_facts():
    """A question about two things should surface both.

    This is why the background is the MEDIAN, not the mean: with two genuinely
    relevant facts the mean is dragged up and the second hit falls below the gap.
    """
    hits = rank(COMPOUND)
    assert len(hits) == 2, f"compound query kept {len(hits)} hits, expected 2"


@case
async def absolute_floor_still_admits():
    """A clearly-strong hit is kept on absolute score alone — the gate only ADDS."""
    hits = rank([0.614, 0.535, 0.522, 0.509, 0.495, 0.463])
    assert len(hits) == 1 and hits[0][1] > 0.6, f"strong hit not kept: {hits}"


@case
async def contrast_floor_is_a_backstop():
    """Standing out means little when everything scores badly."""
    assert rank([0.30, 0.10, 0.09, 0.08, 0.07, 0.06]) == [], "weak hit admitted on contrast alone"


@case
async def tiny_corpus_uses_floor_only():
    """With 1-2 memories the median IS the top hit, so contrast is degenerate."""
    assert rank([0.527]) == [], "single-memory corpus admitted a sub-floor hit"
    assert rank([0.527, 0.40]) == [], "two-memory corpus admitted a sub-floor hit"
    assert len(rank([0.62])) == 1, "single-memory corpus dropped an above-floor hit"


@case
async def relate_path_is_unchanged():
    """Consolidation's relate search asks a different question and opts out.

    It passes no contrast_gap, so it must stay purely absolute — a fact at 0.527
    is NOT close enough to merit a lifecycle decision at relate_sim=0.6.
    """
    assert rank(PLACE, min_sim=0.6, contrast_gap=0.0) == [], "relate path picked up the gate"


@case
async def top_k_still_bounds_results():
    hits = rank([0.9, 0.85, 0.8, 0.75, 0.7, 0.65], min_sim=0.5, top_k=3)
    assert len(hits) == 3, f"top_k not honoured: {len(hits)}"


@case
async def empty_corpus_returns_nothing():
    assert MemoryManager._rank(QUERY, [], None, 5, 0.55, contrast_gap=GAP) == []


if __name__ == "__main__":
    raise SystemExit(run())
