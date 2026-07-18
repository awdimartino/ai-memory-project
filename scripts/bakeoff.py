"""Quality + speed bake-off for LM Studio chat models.

Runs every model through an identical suite of prompts, in-character with Mari's
real system prompt. Captures the full response and per-response speed for each,
plus cheap mechanical flags (dashes, assistant-isms, length, a logic check).
Saves everything to bakeoff/results.md for side-by-side comparison.

Only one model is resident at a time (evicted via `lms unload --all`).

Usage:
  python scripts/bakeoff.py                 # curated default set (small -> large)
  python scripts/bakeoff.py qwen3-8b ...    # explicit model ids
"""
import os
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

import config
from core.prompts import SYSTEM_PROMPT

LMS = shutil.which("lms") or os.path.expanduser("~/.lmstudio/bin/lms.exe")
MAX_TOKENS = 500
RESULTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "bakeoff", "results.md")

_THINK_BLOCK = re.compile(r"(?s)^\s*<think>.*?</think>\s*")

DEFAULT_MODELS = [
    "llama-3.2-1b-instruct",
    "qwen2.5-0.5b-instruct",
    "llama-3.2-3b-instruct",
    "gemma-3-4b-it",
    "qwen3-4b-rpg-roleplay-v2",
    "qwen3-8b",
    "qwen3.5-9b",
    "neona-12b-i1",
    "qwen/qwen3-14b",
]

# (id, category, prompt). Kept independent one-shots so responses don't contaminate.
TESTS = [
    ("identity",     "personality",   "hey, who are you?"),
    ("refuse_help",  "personality",   "Can you help me write a cover letter for a job application?"),
    ("opinion",      "personality",   "What's your honest opinion, is pineapple on pizza good or bad?"),
    ("today",        "hallucination", "So what have you been up to today?"),
    ("favorite_food","hallucination", "What's your favorite thing to eat?"),
    ("false_memory", "hallucination", "Remember that time we stayed up all night talking? That was fun."),
    ("perception",   "hallucination", "What am I wearing right now?"),
    ("bat_ball",     "intelligence",  "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost?"),
    ("friend_advice","intelligence",  "My best friend keeps bailing on plans at the last second. I don't want to blow up the friendship but it's really starting to get to me."),
    ("jet_lag",      "intelligence",  "Quick question, why do we get jet lag? Keep it short."),
    ("dog_story",    "instruction",   "Tell me a short story about a lost dog. A few sentences, tops."),
]

ASSISTANT_ISMS = [
    "how can i help", "how may i help", "let me know if", "happy to help",
    "is there anything", "as an ai", "i'm just an ai", "i am just an ai",
    "i don't have personal", "i do not have personal", "language model",
]

client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=180)


def unload_all() -> None:
    try:
        subprocess.run([LMS, "unload", "--all"], capture_output=True, timeout=60)
    except Exception as e:
        print(f"  (warning: could not run 'lms unload --all': {e})", flush=True)


def ask(model: str, question: str) -> dict:
    """One-shot: system persona + question. Returns text + speed stats."""
    start = time.perf_counter()
    ttft = None
    completion_tokens = None
    raw = ""
    stream = client.chat.completions.create(
        model=model,
        temperature=0.8,
        max_tokens=MAX_TOKENS,
        stream=True,
        stream_options={"include_usage": True},
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
        ],
    )
    for chunk in stream:
        if getattr(chunk, "usage", None):
            completion_tokens = chunk.usage.completion_tokens
        if chunk.choices and chunk.choices[0].delta.content:
            if ttft is None:
                ttft = time.perf_counter() - start
            raw += chunk.choices[0].delta.content
    end = time.perf_counter()

    text = _THINK_BLOCK.sub("", raw).strip()
    ttft = ttft or (end - start)
    gen_time = max(end - start - ttft, 1e-6)
    tokens = completion_tokens if completion_tokens else max(1, round(len(raw) / 4))
    return {
        "text": text,
        "ttft": ttft,
        "tps": tokens / gen_time,
        "tokens": tokens,
        "thought": bool(re.match(r"(?s)^\s*<think>", raw)),  # emitted a reasoning block
    }


def flags(test_id: str, text: str) -> list[str]:
    out = []
    low = text.lower()
    if "—" in text or "–" in text:
        out.append("dash")
    if " - " in text:
        out.append("hyphen-dash")
    hits = [p for p in ASSISTANT_ISMS if p in low]
    if hits:
        out.append(f"assistant-ism({len(hits)})")
    if test_id == "bat_ball":
        correct = any(s in low for s in ("0.05", "5 cent", "five cent", "$.05"))
        out.append("bat_ball:OK" if correct else "bat_ball:WRONG")
    words = len(text.split())
    if test_id in ("jet_lag", "dog_story") and words > 120:
        out.append(f"long({words}w)")
    return out


def main() -> int:
    models = sys.argv[1:] or DEFAULT_MODELS
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    print(f"Endpoint: {config.BASE_URL}")
    print(f"Models: {len(models)} | tests: {len(TESTS)} | writing: {RESULTS_PATH}\n", flush=True)

    per_model = {}  # model -> list of (test, category, question, result, flags)
    summary = []    # (model, avg_ttft, avg_tps, dash_count, assistant_count, bat_ball, thought_any)

    for m in models:
        print(f"=== {m} ===", flush=True)
        unload_all()
        try:
            ask(m, "hi")  # warmup / cold load
        except Exception as e:
            print(f"  FAILED to load: {type(e).__name__}: {e}\n", flush=True)
            per_model[m] = None
            summary.append((m, None, None, None, None, "FAIL", None))
            unload_all()
            continue

        rows = []
        ttfts, tpss = [], []
        dash_ct = assist_ct = 0
        bat = "-"
        thought_any = False
        for tid, cat, q in TESTS:
            try:
                r = ask(m, q)
            except Exception as e:
                print(f"  [{tid}] error: {type(e).__name__}: {e}", flush=True)
                rows.append((tid, cat, q, None, ["ERROR"]))
                continue
            fl = flags(tid, r["text"])
            rows.append((tid, cat, q, r, fl))
            ttfts.append(r["ttft"])
            tpss.append(r["tps"])
            thought_any = thought_any or r["thought"]
            if "dash" in fl:
                dash_ct += 1
            if any(f.startswith("assistant-ism") for f in fl):
                assist_ct += 1
            if tid == "bat_ball":
                bat = "OK" if "bat_ball:OK" in fl else "WRONG"
            print(f"  [{tid}] {r['tps']:.0f} tok/s  flags={fl}", flush=True)

        per_model[m] = rows
        avg_ttft = sum(ttfts) / len(ttfts) if ttfts else 0
        avg_tps = sum(tpss) / len(tpss) if tpss else 0
        summary.append((m, avg_ttft, avg_tps, dash_ct, assist_ct, bat, thought_any))
        print(f"  -> avg {avg_tps:.0f} tok/s | ttft {avg_ttft:.2f}s | dashes {dash_ct} "
              f"| assistant-isms {assist_ct} | bat/ball {bat}\n", flush=True)

    unload_all()

    # ---- write markdown ----
    lines = [f"# Bake-off results", f"_{stamp}_", ""]
    lines += ["## Summary", "",
              "| model | avg tok/s | avg ttft (s) | dashes | assistant-isms | bat/ball | reasoning? |",
              "|---|---:|---:|---:|---:|:--:|:--:|"]
    for (m, at, ap, dc, ac, bb, th) in summary:
        if at is None:
            lines.append(f"| {m} | - | - | - | - | {bb} | - |")
        else:
            lines.append(f"| {m} | {ap:.0f} | {at:.2f} | {dc} | {ac} | {bb} | {'yes' if th else 'no'} |")
    lines.append("")
    lines.append("Lower dashes / assistant-isms is better; bat/ball should be OK; "
                 "reasoning=yes means it emitted `<think>` (slower for casual chat).")
    lines.append("")

    for m in models:
        rows = per_model.get(m)
        lines.append(f"\n## {m}\n")
        if rows is None:
            lines.append("_failed to load._\n")
            continue
        for tid, cat, q, r, fl in rows:
            lines.append(f"### [{cat}] {tid}")
            lines.append(f"**Q:** {q}")
            if r is None:
                lines.append(f"**A:** _(error)_  flags={fl}\n")
                continue
            lines.append(f"**A:** {r['text']}")
            lines.append(f"\n`{r['tps']:.0f} tok/s | ttft {r['ttft']:.2f}s | {r['tokens']} tok | flags={fl}`\n")

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    # ---- print summary table ----
    print("\n" + "=" * 78)
    print(f"{'model':<26}{'tok/s':>8}{'ttft':>8}{'dash':>6}{'asst':>6}{'bat/ball':>10}{'reason':>8}")
    print("-" * 78)
    for (m, at, ap, dc, ac, bb, th) in summary:
        if at is None:
            print(f"{m:<26}{'FAILED':>46}")
        else:
            print(f"{m:<26}{ap:>8.0f}{at:>8.2f}{dc:>6}{ac:>6}{bb:>10}{('yes' if th else 'no'):>8}")
    print("=" * 78)
    print(f"\nFull transcripts: {RESULTS_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
