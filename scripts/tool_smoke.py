"""Live end-to-end smoke for the tool loop, through the REAL stack.

Builds the companion via bootstrap (Mari's real persona + system prompt) against
a throwaway DB and drives a few turns to confirm tool-calling holds up in-persona
(the probe used a neutral "assistant" prompt; this checks the companion voice):
  1. seed a fact to reminisce about later
  2. ask the time            -> expect get_current_time, answer names the time
  3. ask to recall the trip  -> expect reminisce, answer references Denver
  4. plain chit-chat         -> expect NO tool call

Needs LM Studio up with the chat model loaded. Emotion/tick/sleep are disabled so
this is fast and side-effect-free. Prints each turn's reply + tools invoked.

Run:  python scripts/tool_smoke.py
"""
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure BEFORE importing config/bootstrap (config reads env at import time).
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "smoke.db")
os.environ["EMOTION_ENABLED"] = "false"   # skip the torch load; persona still drives the turn
os.environ["TICK_ENABLED"] = "false"      # no background jobs during the smoke
os.environ["SLEEP_ENABLED"] = "false"     # no lms CLI interaction
os.environ.setdefault("TOOLS_ENABLED", "true")

import bootstrap  # noqa: E402


async def turn(companion, label, text):
    parts = []

    async def on_token(t):
        parts.append(t)

    result = await companion.send(text, on_token)
    reply = "".join(parts).strip() or result.text
    tools = result.tools or []
    print(f"\n[{label}] you: {text}")
    print(f"  Mari: {reply}")
    if tools:
        for t in tools:
            print(f"  🔧 {t['name']}({t['args']}) -> {t['result']!r}")
    else:
        print("  (no tool used)")
    return reply, tools


async def main() -> int:
    bootstrap.configure_logging()
    try:
        companion, model = await bootstrap.build()
    except Exception as e:  # noqa: BLE001 - most likely LM Studio isn't up
        print(f"could not build companion (is LM Studio running with a model loaded?): {e}")
        return 0

    print(f"tool smoke on: {model}  |  tools: {companion.tools.names() if companion.tools else 'OFF'}")

    _, seed_tools = await turn(companion, "seed",
                               "By the way, I went to Denver last week for a paramedic conference.")
    _, time_tools = await turn(companion, "time", "hey, what time is it right now?")
    recall_reply, recall_tools = await turn(companion, "recall",
                                            "do you remember where I traveled recently?")
    _, chat_tools = await turn(companion, "chat", "anyway, just say hi back to me, nothing fancy")

    # Lightweight scorecard (informational — live model behavior is probabilistic).
    print("\n--- scorecard ---")
    called = lambda ts, n: any(t["name"] == n for t in ts)  # noqa: E731
    checks = [
        ("time -> get_current_time", called(time_tools, "get_current_time")),
        ("recall -> reminisce", called(recall_tools, "reminisce")),
        ("recall answer mentions Denver", "denver" in recall_reply.lower()),
        ("plain chat -> no tool", not chat_tools),
    ]
    for name, ok in checks:
        print(f"  {'PASS' if ok else 'MISS'}  {name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
