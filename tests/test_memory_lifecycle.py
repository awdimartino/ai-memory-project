"""Deterministic, offline test of memory lifecycle.

Uses the real SQLite MemoryStore + MemoryManager, with a fake embedder (topic =
keyword -> one-hot vector, so "same topic" facts are similar) and a fake LLM
(scripted extraction + decisions). No LM Studio required. This is the payoff of
defining the store as a Protocol: model-free tests of the logic.

Distinct from test_memory_edge.py, which exercises each lifecycle action in
isolation: this drives the canonical FOUR-consolidation *sequence* against one
accumulating store (new -> update -> duplicate -> unrelated-new), so it catches
ordering and supersede-link bugs that per-case tests can't see.

Run:  python tests/test_memory_lifecycle.py
"""
import contextlib

import numpy as np

from _harness import case, run, temp_db  # also puts the repo root on sys.path
from helpers import OneHotEmbedder, ScriptedLLM
from helpers import fact as _fact

import config
from core.memory_manager import MemoryManager
from infrastructure.memory_store import SqliteMemoryStore

_TOPICS = ["live", "dog", "work", "food", "music", "name", "hik"]

_FACT_LISTS = [
    [_fact("The user lives in New York")],       # 1: new (empty store)
    [_fact("The user lives in Boston")],         # 2: update -> supersedes #1
    [_fact("The user lives in Boston")],         # 3: duplicate -> skip
    [_fact("The user owns a dog named Rufus")],  # 4: new (different topic)
]
# One BATCHED decision per consolidate that has a fact needing a decision (calls 2 and 3;
# call 1 is an empty store and call 4 is a different topic, so both insert directly).
_DECISIONS = [
    {"decisions": [{"candidate": 1, "action": "update", "target": 1}]},
    {"decisions": [{"candidate": 1, "action": "duplicate", "target": 0}]},
]


@contextlib.contextmanager
def _lifecycle():
    """Yield `(manager, store, conn)` wired for the canonical sequence."""
    with temp_db("lifecycle.db") as (conn, _path):
        store = SqliteMemoryStore(conn)
        yield (MemoryManager(OneHotEmbedder(_TOPICS), store,
                             ScriptedLLM(_FACT_LISTS, _DECISIONS),
                             brain_model="fake", top_k=5, min_sim=0.55,
                             relate_top_k=5, relate_sim=0.6),
               store, conn)


async def _run_sequence(mm):
    for _ in range(len(_FACT_LISTS)):
        await mm.consolidate([{"role": "user", "content": "x"}], session_id=1)


@case
async def sequence_leaves_exactly_the_two_true_facts():
    with _lifecycle() as (mm, store, conn):
        await _run_sequence(mm)
        contents = sorted(m["content"] for m in store.active())
        assert store.count() == 2, f"expected 2 active, got {store.count()}"
        assert contents == ["The user lives in Boston",
                            "The user owns a dog named Rufus"], contents


@case
async def update_soft_deletes_and_links_supersede():
    with _lifecycle() as (mm, store, conn):
        await _run_sequence(mm)
        rows = conn.execute(
            "SELECT id, content, active, superseded_by FROM memories ORDER BY id").fetchall()
        ny = next(r for r in rows if r["content"] == "The user lives in New York")
        boston = next(r for r in rows if r["content"] == "The user lives in Boston")

        assert ny["active"] == 0, "New York should be soft-deleted"
        assert boston["active"] == 1, "Boston should be active"
        assert ny["superseded_by"] == boston["id"], \
            f"New York should link to Boston ({boston['id']}), got {ny['superseded_by']}"


@case
async def duplicate_is_skipped_not_inserted():
    with _lifecycle() as (mm, store, conn):
        await _run_sequence(mm)
        rows = conn.execute("SELECT content FROM memories").fetchall()
        n = sum(1 for r in rows if r["content"] == "The user lives in Boston")
        assert n == 1, f"duplicate should not insert; found {n} Boston rows"


@case
async def history_is_kept_not_deleted():
    with _lifecycle() as (mm, store, conn):
        await _run_sequence(mm)
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        assert total == 3, f"expected 3 total rows (NY, Boston, dog), got {total}"


# --- the duplicate-verdict guard -------------------------------------------------
# A "duplicate" decision is the only one that DISCARDS information, and the model
# over-applies it to narrowing facts: measured 3 of 8 refinements called duplicates,
# e.g. "The user is a nurse" against "The user works in healthcare"
# (scripts/duplicate_guard_probe.py). Each was a fact silently lost.
#
# The embeddings separate the classes where the model doesn't — refinements 0.808 to
# 0.906, real restatements 0.924 to 0.986 — so the verdict is honoured only above
# MEMORY_DUPLICATE_MIN_SIMILARITY. These cases drive that boundary directly, with an
# embedder whose angle (and so cosine) is set per fact.

import math  # noqa: E402


class AngleEmbedder:
    """Maps each text to a unit vector at a chosen angle, so cosine is exact.

    The one-hot embedder used elsewhere only produces 0.0 or 1.0, which cannot
    express "similar but not identical" — the entire range this guard lives in.
    """

    def __init__(self, angles: dict[str, float]):
        self.angles = angles

    def _vec(self, text: str):
        theta = math.radians(self.angles.get(text.split("\n")[0].strip(), 0.0))
        return [math.cos(theta), math.sin(theta)]

    async def embed_query(self, text):
        return self._vec(text)

    async def embed_document(self, text):
        return self._vec(text)

    async def embed_documents(self, texts):
        return [self._vec(t) for t in texts]


async def _consolidate_with(sim: float, seed: str, candidate: str):
    """Seed one memory, consolidate one candidate at `sim` to it, with a scripted
    "duplicate" verdict. Returns the active contents afterwards."""
    angle = math.degrees(math.acos(sim))
    with temp_db("dupguard.db") as (conn, _path):
        store = SqliteMemoryStore(conn)
        mm = MemoryManager(
            AngleEmbedder({seed: 0.0, candidate: angle}), store,
            ScriptedLLM([[_fact(candidate)]],
                        [{"decisions": [{"candidate": 1, "action": "duplicate", "target": 0}]}]),
            brain_model="fake", top_k=5, min_sim=0.55, relate_top_k=5, relate_sim=0.6,
            dup_verdict_sim=config.MEMORY_DUPLICATE_MIN_SIMILARITY)
        vec = await mm.embedder.embed_document(seed)
        store.add(seed, None, np.asarray(vec, dtype=np.float32).tobytes(), None, core=False)
        await mm.consolidate([{"role": "user", "content": candidate}], session_id=None)
        return [m["content"] for m in store.active()]


@case
async def duplicate_verdict_is_overridden_for_a_refinement():
    """0.844 is the real measured similarity of the case that fails in every gold run."""
    active = await _consolidate_with(0.844, "The user works in healthcare", "The user is a nurse")
    assert any("nurse" in a for a in active), f"refinement still lost: {active}"


@case
async def duplicate_verdict_is_honoured_for_a_real_restatement():
    """0.943: "I'm a welder" against "works as a welder" — genuinely nothing new."""
    active = await _consolidate_with(0.943, "The user works as a welder", "The user is a welder")
    assert len(active) == 1, f"a true duplicate created a second row: {active}"


@case
async def the_guard_boundary_is_where_the_measurement_put_it():
    """Refinements top out at 0.906 and restatements start at 0.924; 0.92 sits between."""
    just_below = await _consolidate_with(0.906, "The user drinks coffee", "The user drinks it black")
    assert len(just_below) == 2, f"0.906 should be stored, got {just_below}"
    just_above = await _consolidate_with(0.924, "The user is a nurse", "The user works as a nurse")
    assert len(just_above) == 1, f"0.924 should be skipped, got {just_above}"


@case
async def an_unrelated_candidate_is_unaffected():
    """Below relate_sim there's no decision at all; the guard must not change that."""
    active = await _consolidate_with(0.30, "The user owns a dog", "The user is learning guitar")
    assert len(active) == 2, f"unrelated fact mishandled: {active}"


if __name__ == "__main__":
    raise SystemExit(run())
