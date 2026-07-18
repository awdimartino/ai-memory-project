"""Offline coverage for core memory (always-injected identity facts).

Real SQLite MemoryStore + MemoryManager with a fake embedder (topic -> one-hot) and
a fake LLM (scripted extraction + rerank). Verifies: the extractor's `core` flag lands
on the row, `store.core()` returns only core facts, `build_system` injects a core block,
and the cap is enforced by the brain re-rank (demoting the rest to regular).

Run:  python tests/test_core_memory.py
"""
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.memory_manager import MemoryManager
from core.prompts import build_system
from infrastructure.db import connect
from infrastructure.memory_store import SqliteMemoryStore

_TOPICS = ["live", "dog", "cat", "work", "food", "music", "name"]


class FakeEmbedder:
    def _vec(self, text):
        v = np.zeros(len(_TOPICS) + 1, dtype=np.float32)
        low = text.lower()
        idx = len(_TOPICS)
        for i, k in enumerate(_TOPICS):
            if k in low:
                idx = i
                break
        v[idx] = 1.0
        return v.tolist()

    async def embed_query(self, text):
        return self._vec(text)

    async def embed_document(self, text):
        return self._vec(text)


class FakeLLM:
    def __init__(self, fact_lists, decisions):
        self.fact_lists = list(fact_lists)
        self.decisions = list(decisions)

    async def structured(self, messages, schema, model=None):
        return self.fact_lists.pop(0) if self.fact_lists else []

    async def structured_json(self, messages, schema, model=None):
        return self.decisions.pop(0) if self.decisions else {"action": "new", "target": 0}


def _fact(content, core=False, category="user"):
    return {"content": content, "category": category, "core": core}


def _mm(fact_lists, decisions, core_max=12):
    path = os.path.join(tempfile.mkdtemp(), "core.db")
    conn = connect(path)
    store = SqliteMemoryStore(conn)
    mm = MemoryManager(FakeEmbedder(), store, FakeLLM(fact_lists, decisions),
                       brain_model="fake", top_k=5, min_sim=0.55,
                       relate_top_k=5, relate_sim=0.6, core_max=core_max)
    return mm, store, conn, path


CASES = []


def case(fn):
    CASES.append(fn)
    return fn


@case
async def core_flag_from_extraction():
    mm, store, conn, path = _mm(
        [[_fact("The user's name is Alex", core=True),
          _fact("The user likes spicy food", core=False)]],
        [],
    )
    await mm.consolidate([{"role": "user", "content": "x"}], session_id=1)
    core = [m["content"] for m in store.core()]
    assert core == ["The user's name is Alex"], core
    assert store.count() == 2 and store.count_core() == 1
    conn.close(); os.remove(path)


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
    mm, store, conn, path = _mm(
        [[_fact("The user lives in Denver", core=True),
          _fact("The user owns a dog", core=True),
          _fact("The user has a cat", core=True),
          _fact("The user works as a nurse", core=True),
          _fact("The user loves spicy food", core=True)]],
        [{"keep": [1, 2, 3]}],   # the rerank call
        core_max=3,
    )
    await mm.consolidate([{"role": "user", "content": "x"}], session_id=1)
    assert store.count() == 5, "all facts still stored"
    assert store.count_core() == 3, f"cap not enforced: {store.count_core()} core"
    kept = sorted(m["content"] for m in store.core())
    assert kept == sorted(["The user lives in Denver", "The user owns a dog",
                           "The user has a cat"]), kept
    conn.close(); os.remove(path)


@case
async def core_cap_untouched_when_under_limit():
    mm, store, conn, path = _mm(
        [[_fact("The user lives in Denver", core=True),
          _fact("The user owns a dog", core=True)]],
        [],   # no rerank call expected (2 <= cap 12)
        core_max=12,
    )
    await mm.consolidate([{"role": "user", "content": "x"}], session_id=1)
    assert store.count_core() == 2
    conn.close(); os.remove(path)


async def main() -> int:
    failed = 0
    for fn in CASES:
        try:
            await fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(CASES) - failed}/{len(CASES)} passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
