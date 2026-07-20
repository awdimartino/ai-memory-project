"""Focused live check: reminisce recalls a past conversational episode end-to-end.

reminisce is for episodic references ("remember when we talked about X") — distinct
from facts about the user (name/job/pet), which the core + semantic-memory tiers
handle via autonomic recall. So this drives the episodic path: seed a topic, scroll
it out of the tiny history window, disable the other memory tiers (so ONLY reminisce
can recover it), then ask with a "remember when..." phrasing and confirm the tool
is called, retrieves the log, and the answer uses it.

Run:  python scripts/reminisce_smoke.py
"""
import asyncio

from _harness import scratch_env  # repo-root path setup + UTF-8 stdout

scratch_env("rem.db",
            HISTORY_TURNS="2",            # tiny window: the seed scrolls out of context fast
            CONSOLIDATE_WINDOW="999",     # don't let consolidation absorb it into semantic recall
            RECALL_MIN_SIMILARITY="0.99")  # neutralize autonomic recall, so reminisce is the ONLY path

import bootstrap  # noqa: E402


async def say(companion, text):
    parts = []

    async def on_token(t):
        parts.append(t)

    r = await companion.send(text, on_token)
    return ("".join(parts).strip() or r.text), (r.tools or [])


async def main() -> int:
    bootstrap.configure_logging()
    try:
        companion, model = await bootstrap.build()
    except Exception as e:  # noqa: BLE001
        print(f"could not build (is LM Studio up?): {e}")
        return 0
    print(f"reminisce smoke on: {model}  (HISTORY_TURNS=2)")

    await say(companion, "I have to tell you about my dog Biscuit, he's a corgi and he's the best.")
    # push it out of the 2-message window with unrelated chatter
    await say(companion, "Anyway, what's a good simple dinner idea?")
    await say(companion, "Cool, and a good drink to go with it?")
    reply, tools = await say(companion, "hey, remember when I told you about my dog earlier?")

    print(f"\n  Mari: {reply}")
    for t in tools:
        print(f"  🔧 {t['name']}({t['args']}) -> {t['result']!r}")
    print("\n--- scorecard ---")
    print(f"  {'PASS' if any(t['name'] == 'reminisce' for t in tools) else 'MISS'}  reminisce was called")
    print(f"  {'PASS' if 'biscuit' in reply.lower() else 'MISS'}  answer recovers 'Biscuit'")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
