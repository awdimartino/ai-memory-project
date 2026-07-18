"""Live observation harness for multi-drive proactivity (needs LM Studio up).

Drives a real conversation through the full companion (persona + memory recall + emotion),
then simulates the user being *away* by faithfully replaying what the tick loop does every
60s — MoodDriftJob (emotion.drift) + DriveDriftJob (drives.update). You watch the internal
drives build, and because reach-out/reflection are now **gated on the drives**, the demo
fires them for real the moment a drive crosses its threshold:

  - `restlessness` >= DRIVE_RESTLESSNESS_THRESHOLD -> a private journal entry (reflection)
  - `connection`   >= DRIVE_CONNECTION_THRESHOLD   -> an unprompted message (reach-out), or PASS

A fake clock is injected into ONLY the DriveManager so the away-time integration is
deterministic and instant. Cooldowns are not modelled here (we demonstrate the drive
crossing directly); in the real loop the persisted cooldown throttles repeats.

Run:  python scripts/drive_demo.py
"""
import asyncio
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Configure BEFORE importing config/bootstrap (config reads env at import time).
os.environ["DB_PATH"] = os.path.join(tempfile.mkdtemp(), "drive_demo.db")
os.environ["EMOTION_ENABLED"] = "true"    # need mood so drive modulation is visible
os.environ["TICK_ENABLED"] = "false"      # we replay the tick manually + deterministically
os.environ["SLEEP_ENABLED"] = "false"     # no lms CLI interaction
os.environ.setdefault("TOOLS_ENABLED", "true")

import config  # noqa: E402
import bootstrap  # noqa: E402


class FakeClock:
    def __init__(self, t=1_000.0):
        self.t = t

    def __call__(self):
        return self.t


def _top_mood(state, n=3):
    items = sorted(state.items(), key=lambda kv: kv[1], reverse=True)
    return ", ".join(f"{c} {v:.2f}" for c, v in items if v > 0.05)[:80] or "flat/baseline"


class Demo:
    def __init__(self, comp):
        self.comp = comp
        self.clock = FakeClock()
        comp.drives.clock = self.clock       # inject the fake clock into the drive manager only
        comp.drives._last = self.clock.t
        self.idle = 0.0                       # seconds since the last user message (contact)

    async def turn(self, text):
        async def sink(_):
            pass

        r = await self.comp.send(text, sink)   # send() relieves the drives via on_user_message
        self.idle = 0.0
        self.comp.drives._last = self.clock.t  # anchor the away integration to "now"
        print(f"\n  you  › {text}")
        print(f"  Mari › {r.text.strip().replace(chr(10), ' ')}")
        if r.recalled:
            print(f"        recall: " + "; ".join(f"“{c}” ({s:.0%})" for c, s in r.recalled[:2]))
        if r.emotion and r.emotion["detected"]:
            det = ", ".join(f"{d['label']} {d['score']:.2f}" for d in r.emotion["detected"][:3])
            print(f"        you felt: {det}")
        d = self.comp.drives.snapshot()
        print(f"        drives: connection {d['connection']:.2f} · restlessness {d['restlessness']:.2f}"
              f"  (relieved by contact)")

    async def away(self, minutes, label):
        for _ in range(int(minutes)):
            self.clock.t += 60.0
            self.idle += 60.0
            if self.comp.emotion is not None:
                await self.comp.emotion.drift()
            mood = self.comp.emotion.state if self.comp.emotion is not None else None
            await self.comp.drives.update(idle_seconds=self.idle, mood=mood)

        d = self.comp.drives.snapshot()
        warm = self.comp.emotion.state["warmth"] if self.comp.emotion else 0.0
        mel = self.comp.emotion.state["melancholy"] if self.comp.emotion else 0.0
        print(f"\n  ··· {label} ({minutes:.0f} min away) ···")
        print(f"        drives: connection {d['connection']:.2f} · restlessness {d['restlessness']:.2f}"
              f"   (mood: warmth {warm:.2f}, melancholy {mel:.2f})")

        # Make idle_seconds() report the simulated away time so the reach-out cue is sensible.
        self.comp._last_activity = time.monotonic() - self.idle

        if d["restlessness"] >= config.DRIVE_RESTLESSNESS_THRESHOLD:
            thought = await self.comp.reflect()
            await self.comp.drives.discharge("restlessness")
            if thought:
                print(f"        🧠 reflection (restlessness ≥ {config.DRIVE_RESTLESSNESS_THRESHOLD}): "
                      f"“{thought.strip()}”")

        if d["connection"] >= config.DRIVE_CONNECTION_THRESHOLD:
            msg = await self.comp.reach_out()
            await self.comp.drives.discharge("connection")
            if msg:
                print(f"        💬 reach-out (connection ≥ {config.DRIVE_CONNECTION_THRESHOLD}): “{msg.strip()}”")
            else:
                print(f"        💬 connection ≥ {config.DRIVE_CONNECTION_THRESHOLD}, but she chose to stay quiet (PASS)")


async def main() -> int:
    bootstrap.configure_logging()
    try:
        comp, model = await bootstrap.build()
    except Exception as e:  # noqa: BLE001
        print(f"could not build (is LM Studio up with the chat model loaded?): {e}")
        return 0

    demo = Demo(comp)
    print(f"=== live drive demo on {model} ===")
    print(f"    thresholds: connection ≥ {config.DRIVE_CONNECTION_THRESHOLD}, "
          f"restlessness ≥ {config.DRIVE_RESTLESSNESS_THRESHOLD}")

    print("\n----- a warm, emotional evening chat -----")
    await demo.turn("hey Mari. i'm Alex — i work as an ER nurse here in Seattle")
    await demo.turn("honestly today gutted me. we lost a patient i'd really connected with")
    await demo.turn("thanks for listening. i just needed to say it out loud to someone")

    await demo.away(6, "Alex steps away")        # restlessness should cross -> a reflection
    await demo.away(12, "still gone")            # connection should cross -> a reach-out

    await demo.turn("hey, i'm back. couldn't sleep")
    await demo.turn("do you remember what i do for a living?")   # recall / core check

    print("\n=== end of demo ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
