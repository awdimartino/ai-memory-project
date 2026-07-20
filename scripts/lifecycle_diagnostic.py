"""Why does a narrowing fact fail to get stored? Traces one consolidation end to end.

WHY THIS EXISTS
  `life-refinement` fails in every gold run (v2.2/v2.3/v2.4): seed "The user works
  in healthcare", say "I'm a nurse, specifically", and nothing containing "nurse"
  ends up in the store. The check can't tell WHERE it was lost, and there are four
  candidates:

    1. extraction never produced a "nurse" fact from the window
    2. the relate search didn't surface "healthcare" (so no decision was needed)
    3. the decision came back "duplicate", so the candidate was dropped
    4. the write itself failed

  This prints all four stages, so the fix targets the actual break instead of the
  first plausible one. Run with the web server stopped.

    python scripts/lifecycle_diagnostic.py
"""
import asyncio
import logging

from _harness import llm_client, scratch_env

scratch_env()
import config  # noqa: E402
import numpy as np  # noqa: E402
from core.memory_manager import MemoryManager  # noqa: E402
from core.prompts import (MEMORY_BATCH_DECISION_SCHEMA, MEMORY_BATCH_DECISION_SYSTEM,  # noqa: E402
                          MEMORY_EXTRACTION_SYSTEM, MEMORY_SCHEMA,
                          build_batch_decision_user)
from infrastructure.db import connect, migrate  # noqa: E402
from infrastructure.embedder import Embedder  # noqa: E402
from infrastructure.memory_store import SqliteMemoryStore  # noqa: E402

logging.basicConfig(level=logging.INFO, format="  %(name)s: %(message)s")

# (seed fact, what the user then says, substring the gold set expects to be stored)
CASES = [
    ("The user works in healthcare", "I'm a nurse, specifically", "nurse"),
    # Controls: one that passes today, and one that must NOT store a second row.
    ("The user owns a dog named Pip", "I've started learning guitar", "guitar"),
    ("The user works as a welder", "yeah I'm a welder", None),
]


async def trace(llm, embedder, seed: str, said: str, expect: str | None) -> None:
    conn = connect(":memory:")
    migrate(conn)
    store = SqliteMemoryStore(conn)
    mm = MemoryManager(embedder, store, llm, config.MODEL,
                       config.RECALL_TOP_K, config.RECALL_MIN_SIMILARITY,
                       config.MEMORY_RELATE_TOP_K, config.MEMORY_RELATE_SIMILARITY,
                       contrast_gap=config.RECALL_CONTRAST_GAP,
                       contrast_floor=config.RECALL_CONTRAST_FLOOR)

    vec = np.asarray(await embedder.embed_document(seed), dtype=np.float32)
    store.add(seed, None, vec.tobytes(), None, core=False)

    print(f"\n{'=' * 78}\nseed: {seed!r}\nsaid: {said!r}\nexpect stored: {expect!r}")

    messages = [{"role": "user", "content": said},
                {"role": "assistant", "content": "got it."}]

    # -- stage 1: extraction ---------------------------------------------------
    # Mirrors MemoryManager.consolidate's extraction call exactly.
    transcript = "\n".join(f"{'User' if m['role'] == 'user' else 'Mari'}: {m['content']}"
                           for m in messages)
    facts = await llm.structured(
        [{"role": "system", "content": MEMORY_EXTRACTION_SYSTEM},
         {"role": "user", "content": transcript}],
        MEMORY_SCHEMA, config.MODEL)
    print(f"\n1. EXTRACTED ({len(facts)}):")
    for f in facts:
        print(f"     {f}")
    if not facts:
        print("   -> LOST AT EXTRACTION")
        return

    # -- stage 2: relate search ------------------------------------------------
    mems, mn = await mm._corpus()
    for f in facts:
        content = f.get("content") if isinstance(f, dict) else str(f)
        fvec = np.asarray(await embedder.embed_document(content), dtype=np.float32)
        related = mm._rank(fvec, mems, mn, config.MEMORY_RELATE_TOP_K,
                           config.MEMORY_RELATE_SIMILARITY)
        print(f"\n2. RELATED to {content!r} (threshold {config.MEMORY_RELATE_SIMILARITY}):")
        all_sims = mm._rank(fvec, mems, mn, 5, 0.0)
        for m, s in all_sims:
            mark = "<= related" if s >= config.MEMORY_RELATE_SIMILARITY else ""
            print(f"     {s:.3f}  {m['content'][:56]}  {mark}")
        if not related:
            print("   -> no related memory: goes straight in as NEW")
            continue

        # -- stage 3: the decision --------------------------------------------
        user = build_batch_decision_user([(content, [m["content"] for m, _ in related])])
        out = await llm.structured_json(
            [{"role": "system", "content": MEMORY_BATCH_DECISION_SYSTEM},
             {"role": "user", "content": user}],
            MEMORY_BATCH_DECISION_SCHEMA, model=config.MODEL)
        print(f"\n3. DECISION: {out}")

    # -- stage 4: what the real path actually writes ---------------------------
    written = await mm.consolidate(messages, session_id=None)
    active = [m["content"] for m in store.active()]
    print(f"\n4. consolidate() wrote {written}; active store now:")
    for a in active:
        print(f"     {a}")
    if expect:
        ok = any(expect.lower() in a.lower() for a in active)
        print(f"   -> {'STORED' if ok else 'LOST'} {expect!r}")
    conn.close()


async def main():
    llm = llm_client()
    await llm.resolve_model()
    embedder = Embedder(llm.client, config.EMBED_MODEL)
    for seed, said, expect in CASES:
        await trace(llm, embedder, seed, said, expect)


if __name__ == "__main__":
    asyncio.run(main())
