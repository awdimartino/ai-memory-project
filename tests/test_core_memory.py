"""Offline coverage for core memory (always-injected identity facts).

Real SQLite MemoryStore + MemoryManager with a fake embedder (topic -> one-hot) and
a fake LLM (scripted extraction + rerank). Verifies: the extractor's `core` flag lands
on the row, `store.core()` returns only core facts, `build_system` injects a core block,
and the cap is enforced by the brain re-rank (demoting the rest to regular).

Run:  python tests/test_core_memory.py
"""
import contextlib

from _harness import case, run, temp_db  # also puts the repo root on sys.path
from helpers import OneHotEmbedder, ScriptedLLM
from helpers import fact as _fact

from core.memory_manager import MemoryManager
from core.prompts import build_system
from infrastructure.memory_store import SqliteMemoryStore

_TOPICS = ["live", "dog", "cat", "work", "food", "music", "name"]


@contextlib.contextmanager
def _mm(fact_lists, decisions, core_max=12):
    """Yield `(manager, store)` on a throwaway DB, torn down even on failure.

    Core tests exercise the core-rerank structured_json ({"keep": [...]}); the
    ScriptedLLM decisions queue serves both, and its {"decisions": []} default is
    harmless (no lifecycle decision fires against an empty store).
    """
    with temp_db("core.db") as (conn, _path):
        store = SqliteMemoryStore(conn)
        yield (MemoryManager(OneHotEmbedder(_TOPICS), store,
                             ScriptedLLM(fact_lists, decisions),
                             brain_model="fake", top_k=5, min_sim=0.55,
                             relate_top_k=5, relate_sim=0.6, core_max=core_max),
               store)


@case
async def core_flag_from_extraction():
    with _mm(
        [[_fact("The user's name is Alex", core=True),
          _fact("The user likes spicy food", core=False)]],
        [],
    ) as (mm, store):
        await mm.consolidate([{"role": "user", "content": "x"}], session_id=1)
        core = [m["content"] for m in store.core()]
        assert core == ["The user's name is Alex"], core
        assert store.count() == 2 and store.count_core() == 1


@case
async def build_system_injects_core_block():
    sys_msg = build_system(
        ["The user enjoys sailing"], mood=None, core=["The user's name is Alex"])
    low = sys_msg.lower()
    assert "always know" in low, "core block header missing"
    assert "The user's name is Alex" in sys_msg, "core fact missing"
    assert "The user enjoys sailing" in sys_msg, "recalled fact missing"
    # the core block comes before the recalled block
    assert low.index("always know") < low.index("might be relevant"), "core not before recalled"


@case
async def core_cap_reranks_and_demotes():
    # 5 distinct-topic core facts, cap 3 -> brain keeps 1,2,3 -> 2 demoted.
    with _mm(
        [[_fact("The user lives in Denver", core=True),
          _fact("The user owns a dog", core=True),
          _fact("The user has a cat", core=True),
          _fact("The user works as a nurse", core=True),
          _fact("The user loves spicy food", core=True)]],
        [{"keep": [1, 2, 3]}],   # the rerank call
        core_max=3,
    ) as (mm, store):
        await mm.consolidate([{"role": "user", "content": "x"}], session_id=1)
        assert store.count() == 5, "all facts still stored"
        assert store.count_core() == 3, f"cap not enforced: {store.count_core()} core"
        kept = sorted(m["content"] for m in store.core())
        assert kept == sorted(["The user lives in Denver", "The user owns a dog",
                               "The user has a cat"]), kept


@case
async def core_cap_untouched_when_under_limit():
    with _mm(
        [[_fact("The user lives in Denver", core=True),
          _fact("The user owns a dog", core=True)]],
        [],   # no rerank call expected (2 <= cap 12)
        core_max=12,
    ) as (mm, store):
        await mm.consolidate([{"role": "user", "content": "x"}], session_id=1)
        assert store.count_core() == 2


if __name__ == "__main__":
    raise SystemExit(run())
