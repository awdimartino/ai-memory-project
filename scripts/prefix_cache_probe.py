"""Does LM Studio reuse the KV cache for a shared prompt PREFIX, and what does it cost?

WHY THIS EXISTS
  The prompt is ~1500 tokens of mostly-static persona, and every restructuring
  decision hangs on one question nobody had measured: if the top of the prompt
  changes, does the whole thing get reprocessed?

  That decides two live arguments:
    - moving the constraint rules to the BOTTOM (better position, but they land
      after per-turn content, so they may stop being cacheable), and
    - injecting a relationship stage at the TOP (free if the cache tolerates a rare
      change, permanently expensive if it doesn't).

WHAT IT MEASURES
  Time-to-first-token, which is dominated by prompt processing. Four conditions:

    identical    same prompt twice in a row            -> is caching on at all?
    tail-change  only the LAST block differs           -> long prefix reusable
    top-change   the FIRST character differs           -> nothing reusable
    cold         a fully distinct prompt               -> reference for "no reuse"

  If prefix caching is active: identical ~= tail-change << top-change ~= cold.
  If TTFT is flat across all four, there is no prefix reuse and position is free.

  PROTOCOL (the part that's easy to get wrong): every measurement is preceded by a
  call sending the BASELINE prompt, so the cache holds a known prefix; the variant
  is then measured against it. Priming with the variant itself instead makes every
  measurement a self-repeat -- the best case in all four conditions -- and the probe
  silently reports "no caching" no matter what is true. (It did, on the first run.)

  Run with the web server STOPPED: a background tick generation lands in the middle
  of a measurement and shows up as a huge outlier.

    python scripts/prefix_cache_probe.py [--rounds 3]
"""
import argparse
import asyncio
import statistics

from _harness import llm_client, scratch_env

scratch_env()
import config  # noqa: E402
from core.prompts import build_system  # noqa: E402

HISTORY = [
    {"role": "user", "content": "long day"},
    {"role": "assistant", "content": "yeah, those stack up."},
]
CUE = "anyway, what's new with you"


def variants() -> dict[str, str]:
    """Four system prompts differing only in WHERE they diverge from the baseline."""
    base = build_system(["The user owns a border collie named Pip"], None,
                        core=["The user's name is Alex"],
                        persona="I've gotten more relaxed with them.",
                        allow_silence=True)
    return {
        "identical": base,
        # A one-word change inside the final block: everything above is byte-identical.
        "tail-change": base[: base.rfind("(For this one")] + "(For this instance: reply in ONE short "
                       "sentence, and don't end on a question.)",
        # A change at character 0: nothing above it, so nothing is reusable.
        "top-change": "Note: this conversation is being reviewed.\n\n" + base,
        # Different content at the SAME length, so "different" is separated from
        # "shorter" — otherwise a flat result can't distinguish no-reuse from
        # length-not-mattering.
        "cold-samelen": ("You are a terse assistant who answers plainly. " * (len(base) // 46))[:len(base)],
        # A much SHORTER prompt: if this is no faster, prompt length itself is free
        # at these sizes and the whole cache question is moot.
        "cold-short": "You are a terse assistant. Answer in one short sentence.",
    }


async def ttft(llm, system: str) -> float:
    async def sink(_t):
        pass

    messages = [{"role": "system", "content": system}, *HISTORY,
                {"role": "user", "content": CUE}]
    _, stats = await llm.stream(messages, sink)
    return float(stats.get("ttft") or 0.0)


async def main(rounds: int) -> None:
    llm = llm_client()
    await llm.resolve_model()
    v = variants()
    print(f"model={config.MODEL}  system prompt ~{len(v['identical'])} chars "
          f"(~{len(v['identical']) // 4} tokens)\n")

    results: dict[str, list[float]] = {k: [] for k in v}
    for r in range(rounds):
        for name, system in v.items():
            # Prime with the BASELINE, then measure the variant against that cache.
            # Priming with the variant would measure a self-repeat every time.
            await ttft(llm, v["identical"])
            t = await ttft(llm, system)
            results[name].append(t)
            print(f"  round {r + 1}  {name:<14} ttft {t:.3f}s")
        print()

    print(f"{'condition':<14} {'median ttft':>12}   vs identical")
    baseline = statistics.median(results["identical"])
    for name, ts in results.items():
        med = statistics.median(ts)
        ratio = med / baseline if baseline else float("nan")
        print(f"{name:<14} {med:>11.3f}s   {ratio:>5.2f}x")

    top = statistics.median(results["top-change"])
    tail = statistics.median(results["tail-change"])
    short = statistics.median(results["cold-short"])
    print("\nreading:")
    if top > tail * 1.3:
        print(f"  PREFIX CACHING IS ACTIVE: a top-of-prompt change costs {top / tail:.2f}x a tail\n"
              f"  change, so static bulk belongs EARLY and per-turn content LATE.")
    elif baseline > short * 1.3:
        print(f"  NO PREFIX REUSE, but prompt length DOES cost: the full prompt is "
              f"{baseline / short:.2f}x\n  the short one. Shorter is better; position is free.")
    else:
        print(f"  NEITHER position NOR length measurably affects TTFT at this size — a "
              f"{len(v['identical']) // 4}-token\n  prompt costs the same as a "
              f"{len(v['cold-short']) // 4}-token one. Order the prompt purely for BEHAVIOUR.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    asyncio.run(main(ap.parse_args().rounds))
