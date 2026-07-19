"""Try one reasoning model (e.g. gpt-oss-20b) through the SAME 11 bake-off questions
we used for qwen, at one or more reasoning-effort levels.

The point: qwen-reasoning-ON is the quality bar and its only problem is speed, so this
captures each answer (content) separately from hidden reasoning (reasoning_content) PLUS
the total wall time — to see whether bounded reasoning hits the quality bar fast enough.

Reuses bakeoff.py's TESTS / SYSTEM_PROMPT / flags so the questions are identical. Does
NOT touch bakeoff/results.md (the qwen baseline); writes bakeoff/<model>_results.md.

Usage:
  python scripts/model_tryout.py openai/gpt-oss-20b            # efforts: low medium
  python scripts/model_tryout.py openai/gpt-oss-20b low        # just one effort
  python scripts/model_tryout.py <model-id> low medium high
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # repo root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))                   # scripts/

import bakeoff as bo  # noqa: E402  (reuse TESTS, SYSTEM_PROMPT, flags, unload_all, client, config)

client = bo.client
MAX_TOKENS = 1024
# Optional persona override (e.g. a model-specific variant) so we don't touch the live prompt.
_PERSONA_FILE = os.environ.get("PERSONA_FILE")
SYSTEM_PROMPT = (open(_PERSONA_FILE, encoding="utf-8").read()
                 if _PERSONA_FILE else bo.SYSTEM_PROMPT)


def ask(model: str, question: str, effort: str) -> dict:
    """One-shot: persona + question, reasoning on at `effort`. Splits answer from
    hidden reasoning and measures total wall time (the thing qwen loses on)."""
    start = time.perf_counter()
    first_answer_at = None
    answer = ""
    reasoning = ""
    completion_tokens = None
    stream = client.chat.completions.create(
        model=model,
        temperature=0.8,
        max_tokens=MAX_TOKENS,
        stream=True,
        stream_options={"include_usage": True},
        extra_body={"reasoning_effort": effort},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    for chunk in stream:
        if getattr(chunk, "usage", None):
            completion_tokens = chunk.usage.completion_tokens
        if not chunk.choices:
            continue
        delta = chunk.choices[0].delta
        rc = getattr(delta, "reasoning_content", None) or getattr(delta, "reasoning", None)
        if rc:
            reasoning += rc
        if getattr(delta, "content", None):
            if first_answer_at is None:
                first_answer_at = time.perf_counter()
            answer += delta.content
    end = time.perf_counter()

    answer = answer.strip()
    ttfa = (first_answer_at - start) if first_answer_at else (end - start)
    gen_time = max(end - (first_answer_at or start), 1e-6)
    toks = completion_tokens if completion_tokens else max(1, round(len(answer) / 4))
    return {
        "answer": answer,
        "reasoning_chars": len(reasoning),
        "total": end - start,
        "ttfa": ttfa,           # time to first ANSWER token (reasoning happens before it)
        "tokens": toks,
        "tps": toks / gen_time,
    }


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: python scripts/model_tryout.py <model-id> [effort ...]")
        return 1
    model = sys.argv[1]
    efforts = sys.argv[2:] or ["low", "medium"]
    tag = model.split("/")[-1].replace(":", "_")
    out_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                            "bakeoff", f"{tag}_results.md")

    print(f"Endpoint: {bo.config.BASE_URL}")
    print(f"Model: {model} | efforts: {efforts} | tests: {len(bo.TESTS)}")
    print(f"Persona: {_PERSONA_FILE or 'core.prompts.SYSTEM_PROMPT (live)'}")
    print(f"Writing: {out_path}\n", flush=True)

    lines = [f"# tryout — {model}", "",
             f"Efforts: {', '.join(efforts)} | persona: Mari SYSTEM_PROMPT | temp 0.8 | "
             f"max_tokens {MAX_TOKENS}", ""]

    try:
        ask(model, "hi", efforts[0])  # warmup / cold load
    except Exception as e:  # noqa: BLE001
        print(f"FAILED to load {model}: {type(e).__name__}: {e}")
        return 1

    for effort in efforts:
        print(f"===== reasoning_effort={effort} =====", flush=True)
        lines.append(f"\n## reasoning_effort = {effort}\n")
        totals = []
        for tid, cat, q in bo.TESTS:
            try:
                r = ask(model, q, effort)
            except Exception as e:  # noqa: BLE001
                print(f"  [{tid}] error: {type(e).__name__}: {e}", flush=True)
                lines.append(f"### [{cat}] {tid}\n**Q:** {q}\n**A:** _(error: {e})_\n")
                continue
            fl = bo.flags(tid, r["answer"])
            totals.append(r["total"])
            print(f"  [{tid}] {r['total']:5.1f}s total  reason={r['reasoning_chars']:5}c  flags={fl}",
                  flush=True)
            lines.append(f"### [{cat}] {tid}")
            lines.append(f"**Q:** {q}")
            lines.append(f"**A:** {r['answer']}")
            lines.append(f"\n`{r['total']:.1f}s total | ttfa {r['ttfa']:.1f}s | {r['tokens']} tok | "
                         f"{r['tps']:.0f} tok/s | reasoning {r['reasoning_chars']} chars | flags={fl}`\n")
        if totals:
            avg = sum(totals) / len(totals)
            print(f"  -> avg {avg:.1f}s/response over {len(totals)} tests\n", flush=True)
            lines.append(f"\n_avg {avg:.1f}s per response at effort={effort}_\n")

    with open(out_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Full transcript: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
