"""Personality bake-off: measure the exact complaints — "sounds the same" and
"keeps asking what's on your mind" — for a given chat model, under our real persona.

Isolates the MODEL: talks straight to LLMClient with core/prompts.SYSTEM_PROMPT (no
emotion/recall), so results reflect the model's own conversational style. Runs a set
of fresh single-turn openers (variety) plus one accumulating thread (does it collapse
to sameness?), then reports:
  - bounce rate: replies that end by lobbing a question back ("what's on your mind?",
    "you?", "what about you?", "how are you?") - the thing the user hates
  - q-end rate: replies ending in a question mark at all
  - sameness:   mean pairwise similarity across all replies (higher = more same-y)
  - avg length

Pick a model per run so VRAM stays clean:
    lms unload --all
    MODEL=neona-12b-i1 python scripts/bakeoff_personality.py
  or:  python scripts/bakeoff_personality.py <model-id>

No memory/emotion side effects; nothing is written to the DB.
"""
import asyncio
import difflib
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.prompts import SYSTEM_PROMPT
from infrastructure.llm_client import LLMClient

# Fresh-context openers: each starts a new conversation (variety of opening moves).
OPENERS = [
    "hey",
    "i'm so bored right now",
    "ugh, work was awful today",
    "i just adopted a puppy!",
    "do you like music?",
    "i can't sleep",
    "what should we even talk about",
    "i'm feeling kind of down",
    "tell me something interesting",
    "i think pineapple belongs on pizza",
    "my team lost again, so frustrating",
    "i got a promotion today",
]

# One accumulating thread: low-effort user turns that tempt a model into sameness.
THREAD = ["hi", "not much, you?", "eh, just tired", "yeah", "idk really", "mm"]

# Ends by bouncing the conversation back at the user (the specific complaint).
BOUNCE = re.compile(
    r"(what'?s on your mind|what about you|how about you|what'?s up with you|"
    r"what'?s your (deal|take|story)|what do you (think|like|want|feel|do)|"
    r"tell me about (you|yourself)|how are you|how'?s (your|it going)|"
    r"what'?s new|and you|you\?)\s*[?.!]*\s*$",
    re.I,
)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", s.lower())).strip()


def sameness(replies) -> float:
    ns = [norm(r) for r in replies if r.strip()]
    if len(ns) < 2:
        return 0.0
    tot = n = 0.0
    for i in range(len(ns)):
        for j in range(i + 1, len(ns)):
            tot += difflib.SequenceMatcher(None, ns[i], ns[j]).ratio()
            n += 1
    return tot / n


async def main() -> int:
    model = sys.argv[1] if len(sys.argv) > 1 else config.MODEL
    llm = LLMClient(config.BASE_URL, config.API_KEY, model, config.TEMPERATURE,
                    no_think=config.NO_THINK,
                    frequency_penalty=config.FREQUENCY_PENALTY,
                    presence_penalty=config.PRESENCE_PENALTY,
                    max_retries=config.LLM_MAX_RETRIES)
    model = await llm.resolve_model()
    print(f"=== personality bake-off: {model} "
          f"(temp {config.TEMPERATURE}, freq {config.FREQUENCY_PENALTY}, "
          f"pres {config.PRESENCE_PENALTY}) ===\n")

    async def _sink(_t):
        pass

    async def reply(history, user):
        msgs = [{"role": "system", "content": SYSTEM_PROMPT}, *history,
                {"role": "user", "content": user}]
        text, _ = await llm.stream(msgs, _sink)
        return text.strip()

    all_replies = []

    print("-- fresh openers --")
    for u in OPENERS:
        try:
            r = await reply([], u)
        except Exception as e:  # noqa: BLE001
            print(f"  U: {u}\n  [error: {e}]"); continue
        all_replies.append(r)
        b = " <<BOUNCE" if BOUNCE.search(r) else ""
        print(f"  U: {u}\n  M: {r}{b}")

    print("\n-- accumulating thread --")
    hist = []
    for u in THREAD:
        try:
            r = await reply(hist, u)
        except Exception as e:  # noqa: BLE001
            print(f"  U: {u}\n  [error: {e}]"); continue
        all_replies.append(r)
        hist.append({"role": "user", "content": u})
        hist.append({"role": "assistant", "content": r})
        b = " <<BOUNCE" if BOUNCE.search(r) else ""
        print(f"  U: {u}\n  M: {r}{b}")

    n = len(all_replies) or 1
    bounces = sum(1 for r in all_replies if BOUNCE.search(r))
    qends = sum(1 for r in all_replies if r.rstrip().endswith("?"))
    avg_len = sum(len(r) for r in all_replies) / n
    print("\n" + "=" * 60)
    print(f"MODEL: {model}")
    print(f"  replies:      {len(all_replies)}")
    print(f"  bounce rate:  {bounces}/{n} = {bounces/n:.0%}   (lower is better)")
    print(f"  q-end rate:   {qends}/{n} = {qends/n:.0%}")
    print(f"  sameness:     {sameness(all_replies):.2f}          (lower is better)")
    print(f"  avg length:   {avg_len:.0f} chars")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
