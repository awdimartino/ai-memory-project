"""Semantic memory: recall (per turn), consolidation + lifecycle (per window).

Recall is autonomic (Tier-1): embed the incoming message, brute-force KNN over
stored facts, return the closest.

Consolidation is Tier-2 structured output run in the background at the end of a
context window. Each extracted fact goes through a lifecycle decision against
related existing memories: duplicate (skip), update (soft-delete the old, keep
history), or new (insert).
"""
import asyncio
import logging

import numpy as np

from core.prompts import (
    CORE_RERANK_SCHEMA,
    CORE_RERANK_SYSTEM,
    MEMORY_BATCH_DECISION_SCHEMA,
    MEMORY_BATCH_DECISION_SYSTEM,
    MEMORY_EXTRACTION_SYSTEM,
    MEMORY_SCHEMA,
    build_batch_decision_user,
    build_core_rerank_user,
)

logger = logging.getLogger(__name__)


def _to_vec(b: bytes) -> np.ndarray:
    return np.frombuffer(b, dtype=np.float32)


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float((a / (np.linalg.norm(a) + 1e-9)) @ (b / (np.linalg.norm(b) + 1e-9)))


class MemoryManager:
    def __init__(self, embedder, store, llm, brain_model: str,
                 top_k: int, min_sim: float,
                 relate_top_k: int, relate_sim: float,
                 core_max: int = 12, dup_sim: float = 0.97):
        self.embedder = embedder
        self.store = store
        self.llm = llm
        self.brain_model = brain_model
        self.top_k = top_k
        self.min_sim = min_sim
        self.relate_top_k = relate_top_k
        self.relate_sim = relate_sim
        self.core_max = core_max
        self.dup_sim = dup_sim  # within-window near-verbatim collapse threshold

    def core_memories(self) -> list[str]:
        """Active core facts, always injected into the prompt (identity-defining)."""
        return [m["content"] for m in self.store.core()]

    async def edit_memory(self, memory_id: int, content: str) -> None:
        """Replace a memory's text and re-embed it so recall still matches (inspector edit)."""
        content = content.strip()
        vec = np.asarray(await self.embedder.embed_document(content), dtype=np.float32)
        self.store.update_content(memory_id, content, vec.tobytes())
        logger.info("edited memory %d", memory_id)

    async def _corpus(self) -> tuple[list[dict], np.ndarray | None]:
        """Load active memories once as `(rows, L2-normalized matrix)`.

        This is the expensive part of a search — a full table read pulling every
        embedding blob, plus a stack and a normalize. Split out so a caller scoring
        MANY vectors (consolidation) pays it once instead of per vector, and run off
        the event loop since it's the one store read that isn't small.
        """
        mems = await asyncio.to_thread(self.store.active)
        if not mems:
            return [], None
        matrix = np.stack([_to_vec(m["embedding"]) for m in mems])
        return mems, matrix / (np.linalg.norm(matrix, axis=1, keepdims=True) + 1e-9)

    @staticmethod
    def _rank(vec: np.ndarray, mems: list[dict], mn: np.ndarray | None,
              top_k: int, min_sim: float) -> list[tuple[dict, float]]:
        """Cosine KNN of `vec` against a preloaded corpus. Returns (memory, sim), best first."""
        if mn is None:
            return []
        qn = vec / (np.linalg.norm(vec) + 1e-9)
        sims = mn @ qn
        order = np.argsort(-sims)[:top_k]
        return [(mems[i], float(sims[i])) for i in order if sims[i] >= min_sim]

    async def _search(self, vec: np.ndarray, top_k: int, min_sim: float) -> list[tuple[dict, float]]:
        """Cosine KNN over active memories, loading the corpus for this one query."""
        mems, mn = await self._corpus()
        return self._rank(vec, mems, mn, top_k, min_sim)

    async def recall(self, text: str) -> list[tuple[str, float]]:
        """Return up to top_k relevant memories as (content, similarity), best first."""
        # No count() pre-check: _corpus already short-circuits on an empty store, and
        # that guard cost an extra query on every single turn to learn nothing.
        qvec = np.asarray(await self.embedder.embed_query(text), dtype=np.float32)
        hits = await self._search(qvec, self.top_k, self.min_sim)
        if hits:
            logger.info("recall: %d hit(s), top sim %.3f", len(hits), hits[0][1])
        return [(m["content"], s) for m, s in hits]

    async def consolidate(self, messages: list[dict], session_id: int | None) -> int:
        """Extract durable facts from a window and apply lifecycle. Returns facts written.

        Batched for speed: ONE extraction call, ONE embeddings call for all facts, and ONE
        lifecycle-decision call covering every fact that resembles an existing memory (instead
        of one call per fact). Near-verbatim duplicates within the same window are collapsed
        without a model call. The lifecycle semantics are unchanged — supersede-links, coexist,
        and bad/garbage-decision fallbacks all still hold.
        """
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

        cands = []
        for f in facts:
            content = (f.get("content") or "").strip()
            if content:
                cands.append({"content": content, "category": f.get("category"),
                              "core": bool(f.get("core"))})
        if not cands:
            logger.info("consolidated %d msg(s): nothing durable", len(messages))
            return 0

        # One embeddings call for every candidate.
        vecs = await self.embedder.embed_documents([c["content"] for c in cands])
        for c, v in zip(cands, vecs):
            c["vec"] = np.asarray(v, dtype=np.float32)

        # Collapse near-verbatim duplicates within THIS window (no model call). Only near-
        # identical text merges, so a genuine second item of the same kind still coexists.
        kept: list[dict] = []
        for c in cands:
            twin = next((k for k in kept if _cosine(c["vec"], k["vec"]) >= self.dup_sim), None)
            if twin is not None:
                twin["core"] = twin["core"] or c["core"]
            else:
                kept.append(c)

        # Relate each survivor to EXISTING memories (pre-batch state). Those with a related
        # memory need a lifecycle decision; the rest are straight inserts.
        # ONE corpus load for all candidates — this used to re-read the whole memories
        # table and rebuild the matrix once per candidate.
        mems, mn = await self._corpus()
        for c in kept:
            c["related"] = self._rank(c["vec"], mems, mn, self.relate_top_k, self.relate_sim)
        need = [c for c in kept if c["related"]]
        direct = [c for c in kept if not c["related"]]

        decisions = await self._batch_decisions(need)

        new = updated = duplicate = 0
        superseded: set[int] = set()
        for i, c in enumerate(need, start=1):  # candidate numbers are 1-based
            d = decisions.get(i, {})
            action = d.get("action", "new")
            target = d.get("target", 0)
            blob = c["vec"].tobytes()
            if action == "duplicate":
                duplicate += 1
                continue
            if (action == "update" and isinstance(target, int)
                    and 1 <= target <= len(c["related"])):
                old_id = c["related"][target - 1][0]["id"]
                if old_id not in superseded:  # don't double-retire within one batch
                    new_id = self.store.add(c["content"], c["category"], blob, session_id, core=c["core"])
                    self.store.deactivate(old_id, superseded_by=new_id)
                    superseded.add(old_id)
                    updated += 1
                    continue
            # "new", an out-of-range/garbage target, or an already-retired target: insert
            self.store.add(c["content"], c["category"], blob, session_id, core=c["core"])
            new += 1

        for c in direct:
            self.store.add(c["content"], c["category"], c["vec"].tobytes(), session_id, core=c["core"])
            new += 1

        logger.info(
            "consolidated %d msg(s): %d new, %d updated, %d duplicate (%d decided in 1 call)",
            len(messages), new, updated, duplicate, len(need),
        )
        if new or updated:
            await self._enforce_core_cap()
        return new + updated

    async def _batch_decisions(self, need: list[dict]) -> dict[int, dict]:
        """One model call deciding every candidate that has related memories. Returns a
        {candidate_number: decision} map (empty when nothing needs deciding)."""
        if not need:
            return {}
        raw = await self.llm.structured_json(
            [
                {"role": "system", "content": MEMORY_BATCH_DECISION_SYSTEM},
                {"role": "user", "content": build_batch_decision_user(
                    [(c["content"], [m["content"] for m, _ in c["related"]]) for c in need])},
            ],
            MEMORY_BATCH_DECISION_SCHEMA,
            self.brain_model,
        )
        out: dict[int, dict] = {}
        for d in (raw.get("decisions") or []):
            if isinstance(d, dict) and isinstance(d.get("candidate"), int):
                out[d["candidate"]] = d
        return out

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
