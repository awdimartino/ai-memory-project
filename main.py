"""Terminal REPL entry point.

Run:  python main.py
Requires LM Studio running with a model loaded (or MODEL set in .env).
Conversation is logged to the SQLite DB and carried across restarts.

Commands:
  /exit            quit
  /reset           clear in-memory context (the log on disk is kept)
  /model [name]    show or switch the model
  /temp [value]    show or set the sampling temperature
"""
import asyncio
import sys

import config
from bootstrap import build, configure_logging


async def amain() -> int:
    configure_logging()
    try:
        companion, model = await build()
    except Exception as e:
        print(f"Could not start: {e}", file=sys.stderr)
        print("Is LM Studio running with a model loaded?", file=sys.stderr)
        return 1

    if companion.tick is not None:
        companion.tick.start()  # proactivity heartbeat (internal jobs for now)

    print(f"Connected to {config.BASE_URL}")
    print(f"Model: {model}  |  temp: {companion.llm.temperature}")
    if companion.history:
        print(f"(carried {len(companion.history)} messages of context from earlier)")
    print("Commands: /exit, /reset, /thoughts, /model [name], /temp [value]")
    print(f"\nYou're talking to {config.BOT_NAME}. Say hi.\n")

    async def on_token(t: str) -> None:
        print(t, end="", flush=True)

    while True:
        user = (await asyncio.to_thread(input, "You: ")).strip()
        if not user:
            continue
        if user in ("/exit", "/quit"):
            break
        if user == "/reset":
            companion.reset()
            print("(context cleared)\n")
            continue
        if user == "/thoughts":
            recent = companion.thoughts.recent(10) if companion.thoughts else []
            if not recent:
                print("(no thoughts yet — Mari reflects while you're away)\n")
            else:
                print("(recent private thoughts, newest first)")
                for t in recent:
                    tag = f" [{t['mood']}]" if t.get("mood") else ""
                    print(f"  -{tag} {t['content']}")
                print()
            continue
        if user.startswith("/model"):
            parts = user.split(maxsplit=1)
            if len(parts) == 2:
                companion.llm.model = parts[1].strip()
                print(f"(model -> {companion.llm.model})\n")
            else:
                print(f"(current model: {companion.llm.model})\n")
            continue
        if user.startswith("/temp"):
            parts = user.split(maxsplit=1)
            if len(parts) == 2:
                try:
                    companion.llm.temperature = float(parts[1])
                    print(f"(temp -> {companion.llm.temperature})\n")
                except ValueError:
                    print("(usage: /temp 0.8)\n")
            else:
                print(f"(current temp: {companion.llm.temperature})\n")
            continue

        print(f"{config.BOT_NAME}: ", end="", flush=True)
        try:
            result = await companion.send(user, on_token)
        except Exception as e:
            print(f"\n[error: {e}]\n")
            continue

        stats = result.stats
        approx = "~" if stats["estimated"] else ""
        print(
            f"\n  [ttft {stats['ttft']:.2f}s | {stats['tok_per_s']:.1f} tok/s "
            f"| {approx}{stats['tokens']} tok]"
        )
        if result.recalled:
            mems = ", ".join(f"{c} ({s:.2f})" for c, s in result.recalled)
            print(f"  [recalled: {mems}]")
        if result.emotion:
            mood = result.emotion["mood"]
            top = ", ".join(
                f"{ch} {mood[ch]:.2f}"
                for ch in sorted(mood, key=mood.get, reverse=True)[:3]
            )
            detected = result.emotion["detected"]
            felt = ", ".join(f"{d['label']} {d['score']:.2f}" for d in detected[:3])
            print(f"  [detected: {felt or 'neutral'} | mood: {top}]")
        print()

    # Stop the heartbeat, then consolidate anything that didn't fill a window.
    if companion.tick is not None:
        await companion.tick.stop()
    print("(consolidating memory...)")
    await companion.flush()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(amain()))
