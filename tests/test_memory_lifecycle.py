"""Deterministic, offline test of memory lifecycle.

Uses the real SQLite MemoryStore + MemoryManager, with a fake embedder (topic =
keyword -> one-hot vector, so "same topic" facts are similar) and a fake LLM
(scripted extraction + decisions). No LM Studio required. This is the payoff of
defining the store as a Protocol: model-free tests of the logic.

Run:  python tests/test_memory_lifecycle.py
"""
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from core.memory_manager import MemoryManager
from infrastructure.db import connect
from infrastructure.memory_store import SqliteMemoryStore

_TOPICS = ["live", "dog", "work", "food", "music", "name", "hik"]


class FakeEmbedder:
    """One-hot vector by topic keyword, so same-topic facts are cosine-similar."""

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(len(_TOPICS) + 1, dtype=np.float32)
        low = text.lower()
        idx = len(_TOPICS)  # "misc" bucket
        for i, k in enumerate(_TOPICS):
            if k in low:
                idx = i
                break
        v[idx] = 1.0
        return v.tolist()

    async def embed_query(self, text: str):
        return self._vec(text)

    async def embed_document(self, text: str):
        return self._vec(text)


class FakeLLM:
    """Scripted extraction (fact lists) and lifecycle decisions."""

    def __init__(self, fact_lists, decisions):
        self.fact_lists = list(fact_lists)
        self.decisions = list(decisions)

    async def structured(self, messages, schema, model=None):
        return self.fact_lists.pop(0) if self.fact_lists else []

    async def structured_json(self, messages, schema, model=None):
        return self.decisions.pop(0) if self.decisions else {"action": "new", "target": 0}


def _fact(content, category="user"):
    return {"content": content, "category": category}


async def run() -> None:
    tmp = os.path.join(tempfile.mkdtemp(), "lifecycle.db")
    conn = connect(tmp)
    store = SqliteMemoryStore(conn)

    fact_lists = [
        [_fact("The user lives in New York")],   # 1: new (empty store)
        [_fact("The user lives in Boston")],      # 2: update -> supersedes #1
        [_fact("The user lives in Boston")],      # 3: duplicate -> skip
        [_fact("The user owns a dog named Rufus")],  # 4: new (different topic)
    ]
    # decisions are only requested when related memories exist (calls 2 and 3)
    decisions = [
        {"action": "update", "target": 1},
        {"action": "duplicate", "target": 0},
    ]

    mm = MemoryManager(
        FakeEmbedder(), store, FakeLLM(fact_lists, decisions),
        brain_model="fake", top_k=5, min_sim=0.55,
        relate_top_k=5, relate_sim=0.6,
    )

    for i in range(4):
        await mm.consolidate([{"role": "user", "content": "x"}], session_id=1)

    active = store.active()
    active_contents = sorted(m["content"] for m in active)
    rows = conn.execute(
        "SELECT content, active, superseded_by FROM memories ORDER BY id"
    ).fetchall()

    # --- assertions ---
    assert store.count() == 2, f"expected 2 active, got {store.count()}"
    assert active_contents == ["The user lives in Boston", "The user owns a dog named Rufus"], \
        active_contents

    ny = next(r for r in rows if r["content"] == "The user lives in New York")
    boston = next(r for r in rows if r["content"] == "The user lives in Boston")
    boston_id = next(r["id"] for r in conn.execute("SELECT id, content FROM memories") if r["content"] == "The user lives in Boston")

    assert ny["active"] == 0, "New York should be soft-deleted"
    assert ny["superseded_by"] == boston_id, \
        f"New York should link to Boston ({boston_id}), got {ny['superseded_by']}"
    assert boston["active"] == 1, "Boston should be active"

    # only ONE Boston row (the duplicate was skipped, not inserted)
    boston_count = sum(1 for r in rows if r["content"] == "The user lives in Boston")
    assert boston_count == 1, f"duplicate should not insert; found {boston_count} Boston rows"

    total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
    assert total == 3, f"expected 3 total rows (NY, Boston, dog), got {total}"

    print("PASS: new / update+supersede / duplicate / unrelated-new all correct")
    print(f"  active: {active_contents}")
    print(f"  NY soft-deleted, superseded_by -> Boston(id={boston_id})")

    conn.close()
    os.remove(tmp)


if __name__ == "__main__":
    asyncio.run(run())
