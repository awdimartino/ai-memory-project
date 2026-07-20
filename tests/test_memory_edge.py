"""Offline edge-case coverage for memory recall + lifecycle logic.

Uses the real SQLite store with a fake embedder (topic keyword -> one-hot, so
same-topic facts are cosine-similar) and a fake LLM (scripted extraction +
decisions). No LM Studio needed. Each case gets its own fresh DB.

Run:  python tests/test_memory_edge.py
"""
import contextlib

from _harness import case, run, temp_db  # also puts the repo root on sys.path
from helpers import OneHotEmbedder, ScriptedLLM
from helpers import fact as _fact

from core.memory_manager import MemoryManager
from infrastructure.memory_store import SqliteMemoryStore

_TOPICS = ["live", "dog", "cat", "work", "food", "music", "tea"]


@contextlib.contextmanager
def _mm(fact_lists, decisions):
    """Yield `(manager, store, conn)` on a throwaway DB, torn down even on failure."""
    with temp_db("edge.db") as (conn, _path):
        store = SqliteMemoryStore(conn)
        yield (MemoryManager(OneHotEmbedder(_TOPICS), store,
                             ScriptedLLM(fact_lists, decisions),
                             brain_model="fake", top_k=5, min_sim=0.55,
                             relate_top_k=5, relate_sim=0.6),
               store, conn)


async def _consume(mm, n):
    for _ in range(n):
        await mm.consolidate([{"role": "user", "content": "x"}], session_id=1)


@case
async def recall_excludes_superseded():
    with _mm(
        [[_fact("The user lives in New York")], [_fact("The user lives in Boston")]],
        [{"decisions": [{"candidate": 1, "action": "update", "target": 1}]}],
    ) as (mm, store, conn):
        await _consume(mm, 2)
        hits = await mm.recall("where does the user live")
        contents = [c for c, _ in hits]
        assert contents == ["The user lives in Boston"], f"recall leaked superseded: {contents}"


@case
async def recall_threshold_and_ordering():
    with _mm(
        [[_fact("The user owns a dog")], [_fact("The user drinks tea")]],
        [],  # different topics -> never related, both inserted new
    ) as (mm, store, conn):
        await _consume(mm, 2)
        assert store.count() == 2
        hits = await mm.recall("tell me about their dog")
        contents = [c for c, _ in hits]
        assert contents == ["The user owns a dog"], f"threshold/order wrong: {contents}"


@case
async def within_batch_duplicate():
    # two identical facts in ONE consolidation: collapsed as near-verbatim (no model call)
    with _mm(
        [[_fact("The user owns a dog"), _fact("The user owns a dog")]],
        [],  # no decision needed — the collapse happens before any decision
    ) as (mm, store, conn):
        await _consume(mm, 1)
        assert store.count() == 1, f"within-batch dup not deduped: count={store.count()}"


@case
async def coexist_when_decision_new():
    # related fact but decision says "new" -> BOTH kept (must not wrongly supersede)
    with _mm(
        [[_fact("The user owns a dog named Rufus")], [_fact("The user owns a dog named Lucy")]],
        [{"decisions": [{"candidate": 1, "action": "new", "target": 0}]}],
    ) as (mm, store, conn):
        await _consume(mm, 2)
        assert store.count() == 2, f"coexist failed, count={store.count()}"


@case
async def bad_target_falls_back_to_new():
    # "update" with an out-of-range target must not crash or corrupt; insert as new
    with _mm(
        [[_fact("The user lives in New York")], [_fact("The user lives in Boston")]],
        [{"decisions": [{"candidate": 1, "action": "update", "target": 99}]}],
    ) as (mm, store, conn):
        await _consume(mm, 2)
        assert store.count() == 2, f"bad target mishandled: count={store.count()}"
        # nothing should be superseded
        retired = conn.execute("SELECT COUNT(*) FROM memories WHERE active=0").fetchone()[0]
        assert retired == 0, f"bad target wrongly superseded {retired}"


@case
async def garbage_decision_defaults_new():
    with _mm(
        [[_fact("The user lives in New York")], [_fact("The user lives in Boston")]],
        [{}],  # empty/garbage decision -> default "new"
    ) as (mm, store, conn):
        await _consume(mm, 2)
        assert store.count() == 2, f"garbage decision mishandled: count={store.count()}"


@case
async def empty_and_blank_facts_skipped():
    with _mm(
        [[_fact(""), _fact("   "), _fact("The user drinks tea")]],
        [],
    ) as (mm, store, conn):
        await _consume(mm, 1)
        assert store.count() == 1, f"blank facts not skipped: count={store.count()}"


@case
async def empty_extraction_no_write():
    with _mm([[]], []) as (mm, store, conn):
        await _consume(mm, 1)
        assert store.count() == 0, f"empty extraction wrote something: {store.count()}"


if __name__ == "__main__":
    raise SystemExit(run())
