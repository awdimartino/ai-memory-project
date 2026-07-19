"""Probe whether LM Studio honors reasoning-shortening params for qwen3.5.
REQUIRES THINKING ON (template line set to true / removed), else there's no <think>
block to shorten and every row reads 0.

Sends the same reasoning problem several ways and reports how many reasoning tokens
each produced + time. Whichever variant shrinks reasoning is one LM Studio honors.
"""
import sys, os, time  # noqa: E401
sys.path.insert(0, os.path.abspath("."))
import config  # noqa: E402
from openai import OpenAI  # noqa: E402

client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY)
PROMPT = ("A train goes 60 mph for 2.5 hours, then 40 mph for 1.5 hours. "
          "What is its average speed over the whole trip? Show your work.")


def run(model, label, **kw):
    body = kw.pop("extra_body", None)
    t = time.perf_counter()
    try:
        r = client.chat.completions.create(
            model=model, messages=[{"role": "user", "content": PROMPT}],
            temperature=0.2, extra_body=body or {}, **kw)
    except Exception as e:  # noqa: BLE001
        print(f"  {label:34} ERROR: {str(e)[:80]}")
        return
    dt = time.perf_counter() - t
    m = r.choices[0].message
    rc = len(getattr(m, "reasoning_content", None) or "")
    print(f"  {label:34} {dt:5.1f}s  reasoning={rc:6} chars  completion_toks={r.usage.completion_tokens}")


def main():
    model = config.MODEL or client.models.list().data[0].id
    print(f"=== reasoning-control probe on {model} (needs thinking ON) ===\n")
    run(model, "baseline (no control)")
    run(model, "reasoning_effort=low", reasoning_effort="low")
    run(model, "reasoning_effort=minimal", reasoning_effort="minimal")
    run(model, "extra_body reasoning_budget=200", extra_body={"reasoning_budget": 200})
    run(model, "extra_body max_thinking_tokens=200", extra_body={"max_thinking_tokens": 200})
    run(model, "extra_body thinking_budget=200", extra_body={"thinking_budget": 200})
    # llama.cpp's native per-request reasoning-budget field (added 2026-03; injects the
    # end-of-thought token once the budget is hit). Works only if LM Studio's bundled
    # engine surfaces it AND --reasoning-budget wasn't pinned at server launch.
    run(model, "extra_body thinking_budget_tokens=64", extra_body={"thinking_budget_tokens": 64})
    run(model, "extra_body thinking_budget_tokens=256", extra_body={"thinking_budget_tokens": 256})
    print("\n(if any row's reasoning chars << baseline, LM Studio honors that knob)")
    print("(if baseline itself reads ~0, thinking is OFF — revert the template line and rerun)")


if __name__ == "__main__":
    main()
