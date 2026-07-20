"""Calibrate a guard on the lifecycle's "duplicate" verdict.

WHY THIS EXISTS
  `life-refinement` fails in every gold run. Traced with lifecycle_diagnostic.py:
  extraction and the relate search both work, and the model then calls "The user is
  a nurse" a DUPLICATE of "The user works in healthcare". consolidate() drops
  duplicates silently, so a strictly more informative fact is lost.

  A narrowing refinement is not a duplicate, and the embedding geometry knows it:
  the wrongly-dropped pair sits at 0.844 while a genuine restatement ("I'm a
  welder" vs "works as a welder") sits at 0.943. So a "duplicate" verdict can be
  checked against similarity and rejected when the geometry doesn't support it.

  Two data points is not a calibration. This measures the two distributions
  properly and sweeps the threshold, the same way scripts/recall_margin_probe.py
  settled the recall gate.

  The asymmetry that sets the direction of the error: LOSING a real fact is worse
  than keeping a slightly redundant row, in a project whose whole premise is
  remembering. When in doubt, store.

    python scripts/duplicate_guard_probe.py            # embeddings only (fast)
    python scripts/duplicate_guard_probe.py --decide   # also ask the model
"""
import argparse
import asyncio

from _harness import llm_client, scratch_env

scratch_env()
import config  # noqa: E402
import numpy as np  # noqa: E402
from core.prompts import (MEMORY_BATCH_DECISION_SCHEMA, MEMORY_BATCH_DECISION_SYSTEM,  # noqa: E402
                          build_batch_decision_user)
from infrastructure.embedder import Embedder  # noqa: E402

# (existing memory, newly extracted candidate). MUST be stored: each adds real
# information the existing row does not carry.
REFINEMENTS = [
    ("The user works in healthcare", "The user is a nurse."),
    ("The user has a dog", "The user's dog is a border collie named Pip."),
    ("The user plays an instrument", "The user is learning guitar."),
    ("The user lives on the west coast", "The user lives in Portland."),
    ("The user has a sister", "The user's sister is called Kate."),
    ("The user drinks coffee", "The user drinks his coffee black."),
    ("The user is training for a race", "The user is training for a half marathon."),
    ("The user works nights", "The user works the 7pm to 7am shift."),
]

# (existing memory, newly extracted candidate). Must NOT create a second row: the
# candidate restates what is already stored.
DUPLICATES = [
    ("The user works as a welder", "The user is a welder."),
    ("The user's name is Alex", "The user is called Alex."),
    ("The user lives in Portland", "The user lives in Portland."),
    ("The user owns a border collie named Pip", "The user has a border collie called Pip."),
    ("The user is learning to play guitar", "The user is learning guitar."),
    ("The user's sister is called Kate", "The user has a sister named Kate."),
    ("The user is a nurse", "The user works as a nurse."),
    ("The user enjoys hiking", "The user likes to hike."),
]


async def main(decide: bool) -> None:
    llm = llm_client()
    await llm.resolve_model()
    emb = Embedder(llm.client, config.EMBED_MODEL)

    async def sim(a: str, b: str) -> float:
        va, vb = await emb.embed_documents([a, b])
        va, vb = np.asarray(va), np.asarray(vb)
        return float(va @ vb / (np.linalg.norm(va) * np.linalg.norm(vb) + 1e-9))

    async def decision(existing: str, candidate: str) -> str:
        out = await llm.structured_json(
            [{"role": "system", "content": MEMORY_BATCH_DECISION_SYSTEM},
             {"role": "user", "content": build_batch_decision_user([(candidate, [existing])])}],
            MEMORY_BATCH_DECISION_SCHEMA, model=config.MODEL)
        ds = out.get("decisions") or [{}]
        return ds[0].get("action", "?")

    # TWO measures, because the obvious one is wrong.
    #
    # `embed` is cosine between the bare fact texts. That is NOT what consolidate
    # compares: candidates are embedded with KEY EXPANSION (the fact plus the
    # transcript it came from), so at runtime the same pairs scored 0.735 and 0.835
    # rather than 0.844 and 0.943 — the context dilutes them, by different amounts
    # per pair. Calibrating the guard on bare cosine put the threshold in the wrong
    # place and let a genuine duplicate through on the first live run.
    #
    # `lex` is Jaccard over content words (core/textsim.py), which compares exactly
    # the two stored strings, ignores the surrounding transcript entirely, and does
    # not depend on the embedder at all. That is the quantity the guard should use.
    from core.textsim import similarity as lexsim

    rows = []
    for kind, pairs in (("refinement", REFINEMENTS), ("duplicate", DUPLICATES)):
        for existing, candidate in pairs:
            e, x = await sim(existing, candidate), lexsim(existing, candidate)
            act = await decision(existing, candidate) if decide else "-"
            rows.append((kind, existing, candidate, e, x, act))
            # A refinement the model calls "duplicate" is a fact that gets silently lost.
            lost = kind == "refinement" and act == "duplicate"
            print(f"  {kind:<10} embed {e:.3f}  lex {x:.3f}  {act:<9} "
                  f"{candidate[:40]:<42}{'  <-- LOST' if lost else ''}")

    for idx, name in ((3, "embed (bare cosine — NOT what runtime compares)"), (4, "lex (Jaccard)")):
        ref = sorted(r[idx] for r in rows if r[0] == "refinement")
        dup = sorted(r[idx] for r in rows if r[0] == "duplicate")
        print(f"\n{name}")
        print(f"  refinements (must store):  {ref[0]:.3f} .. {ref[-1]:.3f}")
        print(f"  duplicates  (must skip):   {dup[0]:.3f} .. {dup[-1]:.3f}")
        clean = ref[-1] < dup[0]
        print(f"  overlap: {'none — gap %.3f to %.3f' % (ref[-1], dup[0]) if clean else 'YES'}")
        print(f"  sweep (honour \"duplicate\" only when >= T):")
        for t in [round(0.1 * i, 2) for i in range(2, 10)]:
            saved = sum(1 for s in ref if s < t)      # rescued from a wrong "duplicate"
            skipped = sum(1 for s in dup if s >= t)   # correctly still collapsed
            flag = "  <-- clean" if saved == len(ref) and skipped == len(dup) else ""
            print(f"    T={t:.2f}  refinements saved {saved}/{len(ref)}   "
                  f"duplicates skipped {skipped}/{len(dup)}{flag}")

    if decide:
        wrong = [r for r in rows if r[0] == "refinement" and r[5] == "duplicate"]
        print(f"\nthe model called {len(wrong)}/{len(REFINEMENTS)} refinements a duplicate "
              f"(each one a silently lost fact)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--decide", action="store_true", help="also ask the model for its verdict")
    asyncio.run(main(ap.parse_args().decide))
