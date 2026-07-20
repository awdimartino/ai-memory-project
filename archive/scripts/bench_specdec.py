"""Speculative-decoding A/B bench. Measures tok/s on the CURRENTLY loaded model,
without touching its config (so it respects whatever draft-model setting LM Studio
has right now). Run it once with the draft model OFF (baseline) and once with it ON,
then compare.

Two prompts on purpose:
  - PREDICTABLE (count 1..120): near-deterministic -> high draft acceptance -> the
    BEST case for speculative decoding. If this one doesn't speed up, the draft model
    is incompatible/too big (the usual "no luck" cause).
  - CREATIVE (a short story): unpredictable -> lower acceptance -> the WORST case;
    spec decoding can even be a wash here.

Run:  python bench_specdec.py
"""
import os
import sys
import time

sys.path.insert(0, os.path.abspath("."))
import config  # noqa: E402
from openai import OpenAI  # noqa: E402

client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY)

PROMPTS = {
    "predictable (count 1..120)":
        "List the whole numbers from 1 to 120, one per line, and nothing else.",
    "creative (short story)":
        "Write a vivid 180-word story about a lighthouse keeper who finds a locked door in the sea.",
}


def run(model, prompt, max_tokens):
    start = time.perf_counter()
    ttft = None
    ctoks = None
    n = 0
    stream = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": prompt}],
        temperature=0.7, stream=True, max_tokens=max_tokens,
        stream_options={"include_usage": True},
    )
    for ch in stream:
        if getattr(ch, "usage", None):
            ctoks = ch.usage.completion_tokens
        if ch.choices and ch.choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - start
            n += 1
    end = time.perf_counter()
    total = max(end - start, 1e-6)
    toks = ctoks or n
    # Robust overall throughput (works whether the server streams incrementally or buffers).
    return {"ttft": ttft or total, "total": total, "tps": toks / total, "toks": toks}


def main():
    model = config.MODEL or client.models.list().data[0].id
    print(f"=== spec-decoding bench on {model} ===")
    print("(run this with the draft model OFF, then ON, and compare tok/s)\n")
    for label, prompt in PROMPTS.items():
        run(model, prompt, 8)                       # warmup
        r = run(model, prompt, 256)                 # measured
        print(f"  {label:28}  {r['tps']:6.1f} tok/s   ({r['toks']} tok in {r['total']:.1f}s)")


if __name__ == "__main__":
    main()
