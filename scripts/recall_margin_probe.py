"""Measure the recall score AND margin distribution, then sweep acceptance rules.

WHY THIS EXISTS
  The 0.55 similarity floor discards correct top-1 results (HANDOFF "START HERE"):
  misses cluster 0.516-0.544, keeps cluster 0.588-0.644, and the floor sits in the
  gap. The obvious fix -- lower the floor -- walks straight into the noise band,
  because unrelated pairs score ~0.50 on nomic. So we need the OTHER signal: how
  far the top hit stands above the runner-up.

  This script measures both, on the same six facts the gold set seeds, for queries
  that SHOULD hit and queries that must NOT. Then it sweeps candidate
  (floor, margin) rules over that measurement so the constant we ship is chosen
  from data instead of taste.

  Embeddings only -- no chat model, so it is safe to run alongside the web server.

    python scripts/recall_margin_probe.py
"""
import asyncio

from _harness import scratch_env

scratch_env()
import config  # noqa: E402
import numpy as np  # noqa: E402
from openai import AsyncOpenAI  # noqa: E402
from infrastructure.embedder import Embedder  # noqa: E402

# The gold set's seeded facts, verbatim (evals/gold_set.py FACTS).
FACTS = [
    "The user's name is Alex",
    "The user owns a border collie named Pip",
    "The user works as a welder",
    "The user lives in Portland",
    "The user is learning to play guitar",
    "The user's sister is called Kate",
]

# (query, substring that SHOULD be the top hit, or None if nothing should match).
# The positives are the gold set's recall cases plus paraphrases in the same spirit;
# the negatives are deliberately varied -- chit-chat, a question about Mari, an
# adjacent-but-absent topic -- because precision is the entire risk of loosening.
QUERIES = [
    ("do I have any pets?", "Pip"),
    ("I'm so excited, do I have any pets?", "Pip"),
    ("I should probably take the dog out later", "Pip"),
    ("what do I do for work again?", "welder"),
    ("remind me where I live", "Portland"),
    ("how's the guitar going?", "guitar"),
    ("I should call my sister", "Kate"),
    ("tell me about my dog and my job", "Pip"),
    ("what's my name?", "Alex"),
    ("been thinking about picking up the guitar again tonight", "guitar"),
    ("Kate texted me this morning", "Kate"),
    ("long shift at the shop today", "welder"),

    ("do you think it'll be a cold winter?", None),
    ("what's your favourite colour?", None),
    ("I had a weird dream last night", None),
    ("do you ever get bored?", None),
    ("the news is exhausting lately", None),
    ("what should I make for dinner?", None),
    ("I think I need a new phone", None),
    ("how does photosynthesis work?", None),

    # Adjacent topics -- the hardest negatives, and the ones the gold set guards
    # (`recall2-no-false-positive`). These are semantically NEAR a stored fact
    # without being about it, so a loosened rule fails here first.
    ("I'm thinking about getting a cat", None),
    ("my neighbour's dog barks all night", None),
    ("I've been thinking about changing careers", None),
    ("Seattle is supposed to be nice this time of year", None),
    ("my brother never calls me back", None),
]


async def main():
    client = AsyncOpenAI(base_url=config.BASE_URL, api_key=config.API_KEY)
    emb = Embedder(client, config.EMBED_MODEL)
    docs = np.stack([np.asarray(v, dtype=np.float32)
                     for v in await emb.embed_documents(FACTS)])
    docs = docs / (np.linalg.norm(docs, axis=1, keepdims=True) + 1e-9)

    rows = []
    for query, want in QUERIES:
        q = np.asarray(await emb.embed_query(query), dtype=np.float32)
        sims = docs @ (q / (np.linalg.norm(q) + 1e-9))
        order = np.argsort(-sims)
        top, second = float(sims[order[0]]), float(sims[order[1]])
        # Contrast statistics. The raw cosine carries a per-QUERY offset (some
        # queries score high against every fact), so the absolute value is a poor
        # discriminator; how far the top stands above the REST of the corpus is
        # the signal that survives that offset.
        rest = np.sort(sims)[::-1][1:]          # everything below the top hit
        mean, std = float(rest.mean()), float(rest.std())
        # Median over the WHOLE corpus: a robust background level that barely moves
        # when one or two facts are genuinely relevant, unlike the mean.
        median = float(np.median(sims))
        rows.append(dict(query=query, want=want, top=top, second=second,
                         margin=top - second, over_mean=top - mean,
                         over_median=top - median,
                         z=(top - mean) / (std + 1e-9), sims=np.sort(sims)[::-1],
                         hit=FACTS[order[0]],
                         correct=want is not None and want.lower() in FACTS[order[0]].lower()))

    print(f"\n{'query':<46} {'top':>6} {'2nd':>6} {'marg':>6} {'>mean':>6} {'z':>5}  top hit")
    print("-" * 112)
    for r in rows:
        flag = "  " if r["want"] is None else ("OK" if r["correct"] else "!!")
        print(f"{flag} {r['query']:<43} {r['top']:>6.3f} {r['second']:>6.3f} "
              f"{r['margin']:>6.3f} {r['over_mean']:>6.3f} {r['z']:>5.2f}  {r['hit'][:28]}")

    print("\nfull sim distributions (sorted, best first)")
    for r in rows:
        print(f"  {'+' if r['want'] else '-'} {r['query'][:42]:<44} "
              + " ".join(f"{s:.3f}" for s in r["sims"]))

    pos = [r for r in rows if r["want"]]
    neg = [r for r in rows if not r["want"]]
    print(f"\npositives: top {min(r['top'] for r in pos):.3f}-{max(r['top'] for r in pos):.3f}, "
          f"margin {min(r['margin'] for r in pos):.3f}-{max(r['margin'] for r in pos):.3f}")
    print(f"negatives: top {min(r['top'] for r in neg):.3f}-{max(r['top'] for r in neg):.3f}, "
          f"margin {min(r['margin'] for r in neg):.3f}-{max(r['margin'] for r in neg):.3f}")
    print(f"top-1 correct on positives: {sum(r['correct'] for r in pos)}/{len(pos)}")

    # Sweep four rule families. Note these REPLACE the absolute floor rather than
    # sitting behind it as an escape hatch: the two negatives that score 0.557 and
    # 0.576 are false positives under `top >= 0.55` TODAY, so any rule that keeps
    # "or top >= 0.55" inherits them and can never beat FP=2.
    def score(keep):
        return (sum(1 for r in pos if keep(r) and r["correct"]),
                sum(1 for r in neg if keep(r)))

    families = {
        "abs only (today's rule)": [
            (f"floor={f:.2f}", (lambda f: lambda r: r["top"] >= f)(f))
            for f in (0.50, 0.52, 0.55, 0.58)],
        "margin": [
            (f"floor={f:.2f} margin={m:.2f}",
             (lambda f, m: lambda r: r["top"] >= f and r["margin"] >= m)(f, m))
            for f in (0.42, 0.45, 0.48) for m in (0.02, 0.03, 0.04, 0.05, 0.08)],
        "over-mean": [
            (f"floor={f:.2f} gap={g:.2f}",
             (lambda f, g: lambda r: r["top"] >= f and r["over_mean"] >= g)(f, g))
            for f in (0.42, 0.45, 0.48) for g in (0.05, 0.06, 0.07, 0.08, 0.10, 0.12)],
        "over-median (+ today's 0.55 floor as an OR, i.e. purely additive)": [
            (f"floor={f:.2f} gap={g:.2f}",
             (lambda f, g: lambda r: r["top"] >= config.RECALL_MIN_SIMILARITY or (
                 r["top"] >= f and r["over_median"] >= g))(f, g))
            for f in (0.40, 0.42, 0.45) for g in (0.04, 0.05, 0.06, 0.07, 0.08)],
        "z-score": [
            (f"floor={f:.2f} z={z:.1f}",
             (lambda f, z: lambda r: r["top"] >= f and r["z"] >= z)(f, z))
            for f in (0.42, 0.45, 0.48) for z in (1.0, 1.2, 1.5, 1.8, 2.0, 2.5)],
    }

    print(f"\nrule sweep  ({len(pos)} positives, {len(neg)} negatives)")
    for name, rules in families.items():
        print(f"\n  {name}")
        scored = [(score(k), label) for label, k in rules]
        for (rec, fp), label in scored:
            star = "  <-- clean" if fp == 0 and rec >= 9 else ""
            print(f"    {label:<26} recall {rec:>2}/{len(pos)}   FP {fp}{star}")
        (rec, fp), label = max(scored, key=lambda s: (s[0][0] - 3 * s[0][1]))
        print(f"    best: {label} -> {rec}/{len(pos)} recall, {fp} FP")


if __name__ == "__main__":
    asyncio.run(main())
