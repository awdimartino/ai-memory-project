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
    CORE_RERANK_SCHEMA,
    CORE_RERANK_SYSTEM,
    MEMORY_DECISION_SCHEMA,
    MEMORY_DECISION_SYSTEM,
    MEMORY_EXTRACTION_SYSTEM,
    MEMORY_SCHEMA,
    build_core_rerank_user,
    build_decision_user,
)

logger = logging.getLogger(__name__)


def _to_vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


class MemoryManager:
    def __init__(self, embedder, store, llm, brain_model: str,
                 top_k: int, min_sim: float,
                 relate_top_k: int, relate_sim: float,
                 core_max: int = 12):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.brain_model = brain_model
        self.top_k = top_k
        self.min_sim = min_sim
        self.relate_top_k = relate_top_k
        self.relate_sim = relate_sim
        self.core_max = core_max

    def core_memories(self) -> list[str]:
        """Active core facts, always injected into the prompt (identity-defining)."""
        return [m["content"] for m in self.store.core()]

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
            core = bool(f.get("core"))
            vec = np.asarray(await self.embedder.embed_document(content), dtype=np.float32)
            blob = vec.tobytes()
            category = f.get("category")

            related = self._search(vec, self.relate_top_k, self.relate_sim)
            if not related:
                self.store.add(content, category, blob, session_id, core=core)
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
                new_id = self.store.add(content, category, blob, session_id, core=core)
                self.store.deactivate(related[target - 1][0]["id"], superseded_by=new_id)
                updated += 1
                continue
            # "new" (or an out-of-range target we don't trust)
            self.store.add(content, category, blob, session_id, core=core)
            new += 1

        logger.info(
            "consolidated %d msg(s): %d new, %d updated, %d duplicate",
            len(messages), new, updated, duplicate,
        )
        if new or updated:
            await self._enforce_core_cap()
        return new + updated

    async def _enforce_core_cap(self) -> None:
        """If the core set exceeds the cap, ask the brain to keep the most essential ones
        and demote the rest to regular (still remembered, just not always injected)."""
        core = self.store.core()
        if len(core) <= self.core_max:
            return
        decision = await self.llm.structured_json(
            [
                {"role": "system", "content": CORE_RERANK_SYSTEM},
                {"role": "user", "content": build_core_rerank_user(
                    [m["content"] for m in core], self.core_max)},
            ],
            CORE_RERANK_SCHEMA,
            self.brain_model,
        )
        keep_nums = decision.get("keep") or []
        # 1-based indices into `core`; fall back to keeping the first core_max if unusable.
        keep_ids = {core[i - 1]["id"] for i in keep_nums
                    if isinstance(i, int) and 1 <= i <= len(core)}
        if not keep_ids:
            keep_ids = {m["id"] for m in core[:self.core_max]}
        demoted = 0
        for m in core:
            if m["id"] not in keep_ids:
                self.store.set_core(m["id"], False)
                demoted += 1
        logger.info("core cap: %d core -> demoted %d (cap %d)", len(core), demoted, self.core_max)
