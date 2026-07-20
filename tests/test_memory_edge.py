"""Offline edge-case coverage for memory recall + lifecycle logic.

Uses the real SQLite store with a fake embedder (topic keyword -> one-hot, so
same-topic facts are cosine-similar) and a fake LLM (scripted extraction +
decisions). No LM Studio needed. Each case gets its own fresh DB.

Run:  python tests/test_memory_edge.py
"""
import os
import sys
import tempfile

from _harness import case, run  # also puts the repo root on sys.path

import numpy as np

from core.memory_manager import MemoryManager
from infrastructure.db import connect
from infrastructure.memory_store import SqliteMemoryStore

_TOPICS = ["live", "dog", "cat", "work", "food", "music", "tea"]


class FakeEmbedder:
    def _vec(self, text: str):
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

    async def embed_documents(self, texts):
        return [self._vec(t) for t in texts]


class FakeLLM:
    """Scripted extraction + BATCHED decisions (one dict per consolidate that needs a decision)."""

    def __init__(self, fact_lists, decisions):
        self.fact_lists = list(fact_lists)
        self.decisions = list(decisions)

    async def structured(self, messages, schema, model=None):
        return self.fact_lists.pop(0) if self.fact_lists else []

    async def structured_json(self, messages, schema, model=None):
        return self.decisions.pop(0) if self.decisions else {"decisions": []}


def _fact(content, category="user"):
    return {"content": content, "category": category}


def _mm(fact_lists, decisions):
    path = os.path.join(tempfile.mkdtemp(), "edge.db")
    conn = connect(path)
    store = SqliteMemoryStore(conn)
    mm = MemoryManager(FakeEmbedder(), store, FakeLLM(fact_lists, decisions),
                       brain_model="fake", top_k=5, min_sim=0.55,
                       relate_top_k=5, relate_sim=0.6)
    return mm, store, conn, path


async def _consume(mm, n):
    for _ in range(n):
        await mm.consolidate([{"role": "user", "content": "x"}], session_id=1)


@case
async def recall_excludes_superseded(mm=None):
    mm, store, conn, path = _mm(
        [[_fact("The user lives in New York")], [_fact("The user lives in Boston")]],
        [{"decisions": [{"candidate": 1, "action": "update", "target": 1}]}],
    )
    await _consume(mm, 2)
    hits = await mm.recall("where does the user live")
    contents = [c for c, _ in hits]
    assert contents == ["The user lives in Boston"], f"recall leaked superseded: {contents}"
    conn.close(); os.remove(path)


@case
async def recall_threshold_and_ordering(mm=None):
    mm, store, conn, path = _mm(
        [[_fact("The user owns a dog")], [_fact("The user drinks tea")]],
        [],  # different topics -> never related, both inserted new
    )
    await _consume(mm, 2)
    assert store.count() == 2
    hits = await mm.recall("tell me about their dog")
    contents = [c for c, _ in hits]
    assert contents == ["The user owns a dog"], f"threshold/order wrong: {contents}"
    conn.close(); os.remove(path)


@case
async def within_batch_duplicate(mm=None):
    # two identical facts in ONE consolidation: collapsed as near-verbatim (no model call)
    mm, store, conn, path = _mm(
        [[_fact("The user owns a dog"), _fact("The user owns a dog")]],
        [],  # no decision needed — the collapse happens before any decision
    )
    await _consume(mm, 1)
    assert store.count() == 1, f"within-batch dup not deduped: count={store.count()}"
    conn.close(); os.remove(path)


@case
async def coexist_when_decision_new(mm=None):
    # related fact but decision says "new" -> BOTH kept (must not wrongly supersede)
    mm, store, conn, path = _mm(
        [[_fact("The user owns a dog named Rufus")], [_fact("The user owns a dog named Lucy")]],
        [{"decisions": [{"candidate": 1, "action": "new", "target": 0}]}],
    )
    await _consume(mm, 2)
    assert store.count() == 2, f"coexist failed, count={store.count()}"
    conn.close(); os.remove(path)


@case
async def bad_target_falls_back_to_new(mm=None):
    # "update" with an out-of-range target must not crash or corrupt; insert as new
    mm, store, conn, path = _mm(
        [[_fact("The user lives in New York")], [_fact("The user lives in Boston")]],
        [{"decisions": [{"candidate": 1, "action": "update", "target": 99}]}],
    )
    await _consume(mm, 2)
    assert store.count() == 2, f"bad target mishandled: count={store.count()}"
    # nothing should be superseded
    retired = conn.execute("SELECT COUNT(*) FROM memories WHERE active=0").fetchone()[0]
    assert retired == 0, f"bad target wrongly superseded {retired}"
    conn.close(); os.remove(path)


@case
async def garbage_decision_defaults_new(mm=None):
    mm, store, conn, path = _mm(
        [[_fact("The user lives in New York")], [_fact("The user lives in Boston")]],
        [{}],  # empty/garbage decision -> default "new"
    )
    await _consume(mm, 2)
    assert store.count() == 2, f"garbage decision mishandled: count={store.count()}"
    conn.close(); os.remove(path)


@case
async def empty_and_blank_facts_skipped(mm=None):
    mm, store, conn, path = _mm(
        [[_fact(""), _fact("   "), _fact("The user drinks tea")]],
        [],
    )
    await _consume(mm, 1)
    assert store.count() == 1, f"blank facts not skipped: count={store.count()}"
    conn.close(); os.remove(path)


@case
async def empty_extraction_no_write(mm=None):
    mm, store, conn, path = _mm([[]], [])
    await _consume(mm, 1)
    assert store.count() == 0, f"empty extraction wrote something: {store.count()}"
    conn.close(); os.remove(path)


if __name__ == "__main__":
    raise SystemExit(run())
