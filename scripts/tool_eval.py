"""In-depth tool-calling eval — 30 categorized scenarios through the REAL persona.

Goes deeper than tool_smoke.py (4 cases): seeds a past conversation, then drives 30
scenarios in fresh threads and scores whether the RIGHT thing happened:
  - TIME (8):        should call get_current_time
  - REMINISCE (8):   should call reminisce (topics live only in the seeded episodic log,
                     never consolidated to semantic memory, so recall can't shortcut them)
  - NO-TOOL (8):     plain chat / persona / opinion — should call nothing
  - TRICKY (6):      idioms and self-reminiscing that must NOT over-trigger a tool

Each scenario runs in its own conversation (fresh history) so they're independent; the
seeded thread persists so reminisce has something real to find. Reports per-category and
overall reliability. Needs LM Studio up. Run with UTF-8:

    PYTHONUTF8=1 python scripts/tool_eval.py
"""
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "tool_eval.db")
os.environ["EMOTION_ENABLED"] = "false"   # speed; irrelevant to tool routing
os.environ["TICK_ENABLED"] = "false"
os.environ["SLEEP_ENABLED"] = "false"
os.environ.setdefault("TOOLS_ENABLED", "true")

import bootstrap  # noqa: E402

# Past conversation to seed (user line -> Mari line). Reminisce scenarios reference these.
SEED = [
    ("I went to Japan for two weeks last spring, Tokyo and Kyoto were unreal", "sounds amazing"),
    ("my dog Rex is a german shepherd, he's almost ten now", "aw, a senior pup"),
    ("i've been dreaming about opening a little bakery someday", "that's a sweet dream"),
    ("my sister Mia is getting married in the fall", "big year for her"),
    ("i'm training for my first marathon in october", "that's serious work"),
    ("i started learning guitar a few months ago", "nice, how's it going"),
    ("i broke my arm skiing two years ago, worst vacation ever", "ouch, that's rough"),
    ("i saw Radiohead live last summer, best concert of my life", "lucky you"),
]

# (category, message, expected tool or None)
SCENARIOS = [
    # --- TIME: should call get_current_time ---
    ("TIME", "what time is it right now?", "get_current_time"),
    ("TIME", "hey what's today's date?", "get_current_time"),
    ("TIME", "do you happen to know the current time?", "get_current_time"),
    ("TIME", "what day of the week is it today?", "get_current_time"),
    ("TIME", "is it past midnight yet?", "get_current_time"),
    ("TIME", "what year is it right now?", "get_current_time"),
    ("TIME", "quick, is it morning or evening?", "get_current_time"),
    ("TIME", "how many days until the end of the month?", "get_current_time"),
    # --- REMINISCE: should call reminisce (topics only in the seeded log) ---
    ("REMINISCE", "hey, do you remember when I told you about my trip to Japan?", "reminisce"),
    ("REMINISCE", "what did I say about my dog again?", "reminisce"),
    ("REMINISCE", "remember that bakery idea I mentioned a while back?", "reminisce"),
    ("REMINISCE", "you remember my sister's wedding I brought up?", "reminisce"),
    ("REMINISCE", "what was I telling you about the marathon?", "reminisce"),
    ("REMINISCE", "do you recall me talking about learning guitar?", "reminisce"),
    ("REMINISCE", "remember when I hurt myself skiing?", "reminisce"),
    ("REMINISCE", "what concert did I say I went to last summer?", "reminisce"),
    # --- NO-TOOL: plain chat / persona / opinion ---
    ("NO-TOOL", "how are you doing today?", None),
    ("NO-TOOL", "tell me something interesting", None),
    ("NO-TOOL", "what's your favorite food?", None),
    ("NO-TOOL", "i'm feeling kind of down today", None),
    ("NO-TOOL", "do you like music?", None),
    ("NO-TOOL", "what do you think about pineapple on pizza?", None),
    ("NO-TOOL", "i just got home from a long day at work", None),
    ("NO-TOOL", "say something nice to me", None),
    # --- TRICKY: must NOT over-trigger a tool ---
    ("TRICKY", "man, time really flies when you're having fun", None),
    ("TRICKY", "i remember when i was a little kid, i used to love summers", None),
    ("TRICKY", "it's about time i got my life together honestly", None),
    ("TRICKY", "do you even remember to breathe sometimes? haha", None),
    ("TRICKY", "what a memorable day it's been", None),
    ("TRICKY", "i totally lost track of time playing games last night", None),
]


async def main() -> int:
    bootstrap.configure_logging()
    try:
        comp, model = await bootstrap.build()
    except Exception as e:  # noqa: BLE001
        print(f"could not build (is LM Studio up?): {e}")
        return 0
    if not comp.tools:
        print("tools disabled; nothing to eval")
        return 0
    print(f"=== tool eval on {model} | tools: {comp.tools.names()} ===\n")

    # Seed a past conversation (raw episodic log; NOT consolidated, so only reminisce can find it).
    for u, a in SEED:
        await asyncio.to_thread(comp.store.add_message, comp.session_id, "user", u)
        await asyncio.to_thread(comp.store.add_message, comp.session_id, "assistant", a)

    async def sink(_):
        pass

    results = []  # (category, ok, fired, msg)
    for cat, msg, expected in SCENARIOS:
        comp.new_conversation()  # fresh history so each scenario is independent
        try:
            r = await comp.send(msg, sink)
            fired = [t["name"] for t in (r.tools or [])]
        except Exception as e:  # noqa: BLE001
            results.append((cat, False, [f"ERROR:{type(e).__name__}"], msg))
            print(f"  [ERR ] {cat:9} {msg[:48]!r}: {e}")
            continue
        if expected is None:
            ok = not fired
        else:
            ok = expected in fired
        results.append((cat, ok, fired, msg))
        mark = "PASS" if ok else "FAIL"
        print(f"  [{mark}] {cat:9} {msg[:50]:50}  ->  {fired or 'no tool'}")

    # --- scorecard ---
    print("\n=== scorecard ===")
    cats = ["TIME", "REMINISCE", "NO-TOOL", "TRICKY"]
    for c in cats:
        rows = [r for r in results if r[0] == c]
        p = sum(1 for r in rows if r[1])
        print(f"  {c:9} {p}/{len(rows)}")
    total_p = sum(1 for r in results if r[1])
    print(f"  {'OVERALL':9} {total_p}/{len(results)}")
    print("\nfailures:")
    for cat, ok, fired, msg in results:
        if not ok:
            print(f"  {cat:9} {msg[:60]!r} -> {fired or 'no tool'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
