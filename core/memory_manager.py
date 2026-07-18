"""Semantic memory: recall (per turn), consolidation + lifecycle (per window).

Recall is autonomic (Tier-1): embed the incoming message, brute-force KNN over
stored facts, return the closest.

Consolidation is Tier-2 structured output run in the background at the end of a
context window. Each extracted fact goes through a lifecycle decision against
related existing memories: duplicate (skip), update (soft-delete the old, keep
history), or new (insert).
"""
import logging

import numpy as np

from core.prompts import (
    MEMORY_DECISION_SCHEMA,
    MEMORY_DECISION_SYSTEM,
    MEMORY_EXTRACTION_SYSTEM,
    MEMORY_SCHEMA,
    build_decision_user,
)

logger = logging.getLogger(__name__)


def _to_vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


class MemoryManager:
    def __init__(self, embedder, store, llm, brain_model: str,
                 top_k: int, min_sim: float,
                 relate_top_k: int, relate_sim: float):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.brain_model = brain_model
        self.top_k = top_k
        self.min_sim = min_sim
        self.relate_top_k = relate_top_k
        self.relate_sim = relate_sim

    def _search(self, vec: np.ndarray, top_k: int, min_sim: float) -> list[tuple[dict, float]]:
        """Cosine KNN over active memories. Returns (memory, similarity), best first."""
        mems = self.store.active()
        if not mems:
            return []
        matrix = np.stack([_to_vec(m["embedding"]) for m in mems])
        qn = vec / (np.linalg.norm(vec) + 1e-9)
        mn = matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)
        sims = mn @ qn
        order = np.argsort(-sims)[:top_k]
        return [(mems[i], float(sims[i])) for i in order if sims[i] >= min_sim]

    async def recall(self, text: str) -> list[tuple[str, float]]:
        """Return up to top_k relevant memories as (content, similarity), best first."""
        if self.store.count() == 0:
            return []
        qvec = np.asarray(await self.embedder.embed_query(text), dtype=np.float32)
        hits = self._search(qvec, self.top_k, self.min_sim)
        if hits:
            logger.info("recall: %d hit(s), top sim %.3f", len(hits), hits[0][1])
        return [(m["content"], s) for m, s in hits]

    async def consolidate(self, messages: list[dict], session_id: int | None) -> int:
        """Extract durable facts from a window and apply lifecycle. Returns facts written."""
        transcript = "\n".join(
            f"{'User' if m['role'] == 'user' else 'Mari'}: {m['content']}" for m in messages
        )
        facts = await self.llm.structured(
            [
                {"role": "system", "content": MEMORY_EXTRACTION_SYSTEM},
                {"role": "user", "content": transcript},
            ],
            MEMORY_SCHEMA,
            self.brain_model,
        )

        new = updated = duplicate = 0
        for f in facts:
            content = (f.get("content") or "").strip()
            if not content:
                continue
            vec = np.asarray(await self.embedder.embed_document(content), dtype=np.float32)
            blob = vec.tobytes()
            category = f.get("category")

            related = self._search(vec, self.relate_top_k, self.relate_sim)
            if not related:
                self.store.add(content, category, blob, session_id)
                new += 1
                continue

            decision = await self.llm.structured_json(
                [
                    {"role": "system", "content": MEMORY_DECISION_SYSTEM},
                    {"role": "user", "content": build_decision_user(
                        content, [m["content"] for m, _ in related])},
                ],
                MEMORY_DECISION_SCHEMA,
                self.brain_model,
            )
            action = decision.get("action", "new")
            target = decision.get("target", 0)

            if action == "duplicate":
                duplicate += 1
                continue
            if action == "update" and isinstance(target, int) and 1 <= target <= len(related):
                new_id = self.store.add(content, category, blob, session_id)
                self.store.deactivate(related[target - 1][0]["id"], superseded_by=new_id)
                updated += 1
                continue
            # "new" (or an out-of-range target we don't trust)
            self.store.add(content, category, blob, session_id)
            new += 1

        logger.info(
            "consolidated %d msg(s): %d new, %d updated, %d duplicate",
            len(messages), new, updated, duplicate,
        )
        return new + updated
