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

from _harness import case, run, temp_db  # also puts the repo root on sys.path
from helpers import OneHotEmbedder, ScriptedLLM
from helpers import fact as _fact

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


if __name__ == "__main__":
    raise SystemExit(run())
