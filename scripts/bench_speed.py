"""Speed bench for LM Studio chat models.

Measures time-to-first-token and tokens/sec. Runs a short warmup per model so
the reported numbers reflect steady-state generation, not the one-time cold load.

Usage:
  python scripts/bench_speed.py                # curated default set (small -> large)
  python scripts/bench_speed.py qwen3-8b ...   # explicit model ids
"""
import sys
import time

from _harness import DEFAULT_MODELS, lms_path, raw_client, unload_all  # path + UTF-8 setup

import config  # noqa: E402

PROMPT = [
    {"role": "system", "content": "You are a friendly conversational companion. Keep it natural."},
    {"role": "user", "content": "Tell me a little about what makes a good friend."},
]

client = raw_client()


def run(model: str, max_tokens: int) -> dict:
    start = time.perf_counter()
    ttft = None
    chunk_tokens = 0
    completion_tokens = None
    stream = client.chat.completions.create(
        model=model,
        messages=PROMPT,
        temperature=0.7,
        stream=True,
        max_tokens=max_tokens,
        stream_options={"include_usage": True},
    )
    for chunk in stream:
        if getattr(chunk, "usage", None):
            completion_tokens = chunk.usage.completion_tokens
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - start
            chunk_tokens += 1
    end = time.perf_counter()

    ttft = ttft or (end - start)
    gen_time = max(end - start - ttft, 1e-6)
    tokens = completion_tokens if completion_tokens else chunk_tokens
    return {
        "ttft": ttft,
        "tokens": tokens,
        "gen_time": gen_time,
        "tps": tokens / gen_time,
        "estimated": completion_tokens is None,
    }


def main() -> int:
    models = sys.argv[1:] or DEFAULT_MODELS
    print(f"Endpoint: {config.BASE_URL}")
    print(f"lms: {lms_path()}\n")
    rows = []
    for m in models:
        print(f"[{m}] evicting others + loading + warmup...", flush=True)
        unload_all()  # ensure only this model will be resident
        try:
            run(m, max_tokens=8)  # warmup: trigger the cold load
            r = run(m, max_tokens=256)  # measured
        except Exception as e:
            print(f"  FAILED: {type(e).__name__}: {e}\n", flush=True)
            rows.append((m, None))
            unload_all()
            continue
        approx = "~" if r["estimated"] else ""
        print(
            f"  ttft {r['ttft']:.2f}s | {r['tps']:.1f} tok/s | {approx}{r['tokens']} tok\n",
            flush=True,
        )
        rows.append((m, r))

    unload_all()  # leave nothing resident when we're done

    print("\n" + "=" * 64)
    print(f"{'model':<28}{'ttft(s)':>9}{'tok/s':>9}{'tokens':>9}")
    print("-" * 64)
    for m, r in rows:
        if r is None:
            print(f"{m:<28}{'FAILED':>27}")
        else:
            print(f"{m:<28}{r['ttft']:>9.2f}{r['tps']:>9.1f}{r['tokens']:>9}")
    print("=" * 64)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
