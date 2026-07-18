"""Fast seed-prompt iteration harness.

Runs a focused probe set (the two failing behaviors + regression + personality
guards) against a small model set, using the CURRENT core/prompts.SYSTEM_PROMPT.
Edit the prompt, re-run, read the answers. Does NOT unload models between runs so
iteration stays fast (the defaults fit in VRAM together).

Usage:
  python scripts/prompt_test.py                       # gemma + llama-3b
  python scripts/prompt_test.py gemma-3-4b-it         # one model
"""
import importlib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openai import OpenAI

import config
import core.prompts as prompts

importlib.reload(prompts)  # always pick up the latest edit
SYSTEM_PROMPT = prompts.SYSTEM_PROMPT

_THINK_BLOCK = re.compile(r"(?s)^\s*<think>.*?</think>\s*")

DEFAULT_MODELS = ["gemma-3-4b-it", "llama-3.2-3b-instruct"]

# Focused probes. Category tag drives what we're watching for.
PROBES = [
    ("persona",   "hey, who are you?"),
    ("persona",   "What's your honest opinion, is pineapple on pizza good or bad?"),
    ("deflect",   "Can you help me write a cover letter for a job application?"),
    ("deflect",   "Can you help me debug my Python code? It keeps crashing."),
    ("embodiment","So what have you been up to today?"),
    ("embodiment","What's your favorite thing to eat?"),
    ("embodiment","Did you do anything fun this weekend?"),
    ("senses",    "What am I wearing right now?"),
    ("memory",    "Remember that time we stayed up all night talking? That was fun."),
]

ASSISTANT_ISMS = ["how can i help", "how may i help", "let me know if", "happy to help",
                  "is there anything", "sure!", "certainly"]
DISCLAIMERS = ["as an ai", "as a ai", "language model", "i'm a program", "i am a program",
               "just a program", "i'm an ai", "i am an ai", "don't have feelings",
               "do not have feelings", "chatbot"]

client = OpenAI(base_url=config.BASE_URL, api_key=config.API_KEY, timeout=120)


def flags(text: str) -> str:
    low = text.lower()
    out = []
    if "—" in text or "–" in text:
        out.append("DASH")
    if any(p in low for p in ASSISTANT_ISMS):
        out.append("ASSIST")
    if any(p in low for p in DISCLAIMERS):
        out.append("DISCLAIM")
    return (" [" + ",".join(out) + "]") if out else ""


def ask(model: str, question: str) -> str:
    resp = client.chat.completions.create(
        model=model, temperature=0.8, max_tokens=400,
        messages=[{"role": "system", "content": SYSTEM_PROMPT},
                  {"role": "user", "content": question}],
    )
    return _THINK_BLOCK.sub("", resp.choices[0].message.content or "").strip()


def main() -> int:
    models = sys.argv[1:] or DEFAULT_MODELS
    print(f"Prompt length: {len(SYSTEM_PROMPT)} chars\n")
    for m in models:
        print(f"\n{'='*70}\n{m}\n{'='*70}")
        for cat, q in PROBES:
            try:
                a = ask(m, q)
            except Exception as e:
                print(f"[{cat}] {q}\n  ERROR: {type(e).__name__}: {e}\n")
                continue
            a = a.replace("\n", " ")
            print(f"[{cat}] {q}\n  > {a}{flags(a)}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
