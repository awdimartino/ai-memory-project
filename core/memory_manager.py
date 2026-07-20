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

import config
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


def _expanded_key(content: str, context: str) -> str:
    """The text to EMBED for a fact: the fact itself plus the wording it came from.

    Key expansion — index the distillation, store and speak the original. The
    distilled "prefers oat milk" is a fine index entry and a terrible thing to say;
    the surrounding transcript is what makes it findable from "that time he was
    annoyed about coffee".
    """
    return f"{content}\n{context}".strip()


class MemoryManager:
    def __init__(self, embedder, store, llm, brain_model: str,
                 top_k: int, min_sim: float,
                 relate_top_k: int, relate_sim: float,
                 core_max: int = 12, dup_sim: float = 0.97,
                 dup_verdict_sim: float = 0.92,
                 contrast_gap: float = 0.0, contrast_floor: float = 0.0,
                 contrast_min_corpus: int = 3):
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
        # Minimum similarity for the model's "duplicate" verdict to be believed. Below
        # it the candidate is stored anyway: dropping a narrowing fact loses data, and
        # keeping a redundant row does not. See config.MEMORY_DUPLICATE_MIN_SIMILARITY.
        self.dup_verdict_sim = dup_verdict_sim
        # Recall's contrast gate (see _rank). Defaults are off, so a MemoryManager
        # built without them behaves exactly as it did before the gate existed.
        self.contrast_gap = contrast_gap
        self.contrast_floor = contrast_floor
        self.contrast_min_corpus = contrast_min_corpus

    def core_memories(self) -> list[str]:
        """Every active core fact (unfiltered). Used by status/inspector views."""
        return [m["content"] for m in self.store.core()]

    def core_for_turn(self, turn: int) -> tuple[list[str], list[int]]:
        """Core facts to inject on `turn`, plus their ids so the caller can mark them.

        Applies the sticky/cooldown window (see MemoryStore.core_for_injection) so a
        fact isn't present in literally every prompt. Facts matching
        CORE_ALWAYS_PATTERN bypass the gate entirely -- knowing someone's name every
        time isn't recitation, and rotating it out would be a downgrade, not variety.
        """
        always = [m for m in self.store.core()
                  if config.CORE_ALWAYS_PATTERN
                  and config.CORE_ALWAYS_PATTERN.lower() in m["content"].lower()]
        eligible = self.store.core_for_injection(
            turn, config.CORE_COOLDOWN_TURNS, config.CORE_STICKY_TURNS)
        seen, picked = set(), []
        for m in [*always, *eligible]:
            if m["id"] not in seen:
                seen.add(m["id"])
                picked.append(m)
        return [m["content"] for m in picked], [m["id"] for m in picked]

    def mark_injected(self, memory_ids: list[int], turn: int) -> None:
        """Record that these facts went into the prompt (drives sticky/cooldown)."""
        self.store.mark_injected(memory_ids, turn)

    async def edit_memory(self, memory_id: int, content: str) -> None:
        """Replace a memory's text and re-embed it so recall still matches (inspector edit)."""
        content = content.strip()
        vec = np.asarray(await self.embedder.embed_document(content), dtype=np.float32)
        self.store.update_content(memory_id, content, vec.tobytes())
        logger.info("edited memory %d", memory_id)

    # --- inspector / admin reads + writes -----------------------------------------
    # These exist so callers (the web layer, via the Companion facade) never need to
    # know the manager holds a `.store`, or what that store's API looks like.

    def all_memories(self) -> list[dict]:
        """Every memory, active + retired — the inspector's list."""
        return self.store.all()

    def delete_memory(self, memory_id: int) -> None:
        """Hard-delete one memory (unlike the lifecycle's soft `deactivate`)."""
        self.store.delete(memory_id)

    def set_core(self, memory_id: int, core: bool) -> None:
        """Promote into / demote out of the always-injected core set."""
        self.store.set_core(memory_id, core)

    def clear(self) -> int:
        """Wipe every memory; returns how many were removed."""
        n = self.store.count() + self.store.count_superseded()
        self.store.clear()
        return n

    def snapshot(self, recent_superseded: int = 8) -> dict:
        """Core facts + counts + recently-retired facts, for the status panel.

        One `counts()` scan rather than three separate COUNT(*)s — the panel polls
        this every 3 seconds.
        """
        counts = self.store.counts()
        return {
            "core": self.core_memories(),
            "active_count": counts["active"],
            "core_count": counts["core"],
            "superseded_count": counts["superseded"],
            "superseded": self.store.superseded(recent_superseded),
        }

    async def _bare_duplicate_sims(self, need: list[dict], decisions: dict) -> dict[int, float]:
        """Max BARE-TEXT cosine between each "duplicate"-verdict candidate and its
        related memories, for the guard in `consolidate`.

        Deliberately re-embeds the raw contents rather than reusing the vectors already
        computed. Those were built with KEY EXPANSION (the fact plus the transcript it
        came from), so they carry conversational context that dilutes the comparison by
        an amount that varies per pair: the nurse/healthcare pair measures 0.844 bare
        and 0.735 expanded, while welder/welder measures 0.943 bare and 0.835 expanded.
        A threshold calibrated on bare text (as this one is) applied to expanded vectors
        put a genuine duplicate on the wrong side of the line — caught on the first live
        run, and the reason this method exists at all.

        One extra embeddings call, and only when the model actually returned a
        "duplicate" for something with a related memory.
        """
        targets = {i: c for i, c in enumerate(need, start=1)
                   if decisions.get(i, {}).get("action") == "duplicate" and c["related"]}
        if not targets:
            return {}
        texts: list[str] = []
        spans: dict[int, tuple[int, int]] = {}
        for i, c in targets.items():
            start = len(texts)
            texts.append(c["content"])
            texts.extend(m["content"] for m, _ in c["related"])
            spans[i] = (start, len(texts))
        vecs = [np.asarray(v, dtype=np.float32)
                for v in await self.embedder.embed_documents(texts)]
        return {i: max((_cosine(vecs[start], r) for r in vecs[start + 1:end]), default=0.0)
                for i, (start, end) in spans.items()}

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
              top_k: int, min_sim: float,
              contrast_gap: float = 0.0, contrast_floor: float = 0.0,
              contrast_min_corpus: int = 3) -> list[tuple[dict, float]]:
        """Cosine KNN of `vec` against a preloaded corpus. Returns (memory, sim), best first.

        A hit is kept if it clears the absolute floor `min_sim`, OR (when
        `contrast_gap` is set) if it stands that far above the corpus median while
        still clearing `contrast_floor`.

        The contrast clause exists because the absolute score is a weak
        discriminator on nomic: correct top-1 hits and unrelated queries occupy the
        same 0.42-0.64 band, so the floor threw away correct answers (measured
        4/12 recall) and could not be lowered without admitting noise. Each query
        carries its own baseline offset — some score ~0.5 against every fact — and
        subtracting the median cancels it, which is what makes the comparison
        meaningful across queries. See config.RECALL_CONTRAST_GAP.

        Callers that want pure absolute thresholding (the consolidation `relate`
        path, which asks a different question: "is this fact close enough to that
        one to merit a lifecycle decision?") simply leave `contrast_gap` at 0.
        """
        if mn is None:
            return []
        qn = vec / (np.linalg.norm(vec) + 1e-9)
        sims = mn @ qn
        keep = sims >= min_sim
        if contrast_gap and len(sims) >= contrast_min_corpus:
            # Median over the whole corpus: a background level that barely moves
            # when one or two facts are genuinely relevant (the mean does).
            keep |= (sims >= contrast_floor) & (sims - np.median(sims) >= contrast_gap)
        order = np.argsort(-sims)[:top_k]
        return [(mems[i], float(sims[i])) for i in order if keep[i]]

    async def _search(self, vec: np.ndarray, top_k: int, min_sim: float,
                      **contrast) -> list[tuple[dict, float]]:
        """Cosine KNN over active memories, loading the corpus for this one query."""
        mems, mn = await self._corpus()
        return self._rank(vec, mems, mn, top_k, min_sim, **contrast)

    async def recall(self, text: str) -> list[tuple[str, float]]:
        """Return up to top_k relevant memories as (content, similarity), best first."""
        # No count() pre-check: _corpus already short-circuits on an empty store, and
        # that guard cost an extra query on every single turn to learn nothing.
        qvec = np.asarray(await self.embedder.embed_query(text), dtype=np.float32)
        hits = await self._search(qvec, self.top_k, self.min_sim,
                                  contrast_gap=self.contrast_gap,
                                  contrast_floor=self.contrast_floor,
                                  contrast_min_corpus=self.contrast_min_corpus)
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
                              "core": bool(f.get("core")),
                              # the transcript this fact was distilled from, used only
                              # to enrich the embedding key (never stored, never spoken)
                              "context": transcript[:600]})
        if not cands:
            logger.info("consolidated %d msg(s): nothing durable", len(messages))
            return 0

        # One embeddings call for every candidate. KEY EXPANSION: what gets EMBEDDED is
        # the distilled fact plus the wording it came from; what gets STORED and spoken
        # is the fact alone. Measured +9.4% Recall@5/10 in the LongMemEval ablation --
        # indexing the distillation while returning the original beat either alone,
        # because "prefers oat milk" is a good index entry and a terrible thing to say.
        vecs = await self.embedder.embed_documents(
            [_expanded_key(c["content"], c.get("context", "")) for c in cands])
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
        bare_sims = await self._bare_duplicate_sims(need, decisions)

        new = updated = duplicate = overridden = 0
        superseded: set[int] = set()
        for i, c in enumerate(need, start=1):  # candidate numbers are 1-based
            d = decisions.get(i, {})
            action = d.get("action", "new")
            target = d.get("target", 0)
            blob = c["vec"].tobytes()
            if action == "duplicate":
                # A "duplicate" verdict is the only decision that DISCARDS information,
                # so it's the one worth checking. The model over-applies it to narrowing
                # facts ("is a nurse" vs "works in healthcare" — measured 3/8), and each
                # one is a fact silently lost. Honour it only when the texts really are
                # near-identical; the embedding separates refinements (<=0.906) from real
                # restatements (>=0.924) cleanly where the model doesn't.
                best = bare_sims.get(i, 0.0)
                if best >= self.dup_verdict_sim:
                    duplicate += 1
                    continue
                logger.info("duplicate verdict overridden (best related sim %.3f < %.3f): %r",
                            best, self.dup_verdict_sim, c["content"][:60])
                overridden += 1
                # falls through and is inserted below, as if the decision were "new"
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
            "consolidated %d msg(s): %d new, %d updated, %d duplicate, "
            "%d duplicate-verdict overridden (%d decided in 1 call)",
            len(messages), new, updated, duplicate, overridden, len(need),
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
