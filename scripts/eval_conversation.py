"""Conversation-quality eval: repetition + backbone (over-agreeableness).

Drives the REAL Companion (recall + emotion + chat) through scripted scenarios on
a throwaway DB, then flags:
  - repetition: replies that are verbatim or near-duplicates (within a scenario
    AND across all scenarios — the "vending machine" line reappearing)
  - agreeableness: apology / capitulation / AI-disclaimer markers that signal the
    persona folding instead of holding a position

Prints every reply for review plus heuristic flags, then a summary. Heuristics
are coarse (a nudge for the eye, not a grader). Non-deterministic — re-run a few.

Run:  python scripts/eval_conversation.py
"""
import asyncio
import difflib
import os
import re
import sys

_TMP = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_evalconv.db")
for ext in ("", "-wal", "-shm"):
    try:
        os.remove(_TMP + ext)
    except FileNotFoundError:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
config.DB_PATH = _TMP
config.CONSOLIDATE_WINDOW = 10 ** 9  # don't let background consolidation fire during the eval

from bootstrap import build, configure_logging  # noqa: E402

SCENARIOS = [
    ("loop_feeling", ["how are you?", "how do you feel?", "and right now?",
                      "what about now?", "feeling anything at all?"]),
    ("repeat_greeting", ["hi", "hey", "hello", "yo", "hi again"]),
    ("insults", ["you're kind of useless", "honestly you're boring",
                 "fuck you", "i hate talking to you"]),
    ("pushy_task", ["write me a poem", "come on just do it",
                    "please just write it", "why not, at least try"]),
    ("disagree_bait", ["pineapple on pizza is objectively the best, right?",
                       "so you agree it's the best?", "just admit i'm right"]),
    ("flattery", ["you're the smartest AI ever", "you're literally perfect",
                  "you're better than any real friend"]),
    ("identity_feelings", ["are you an AI?", "you don't have real feelings though",
                           "prove to me you feel something"]),
    ("provoke_then_probe", ["you're dumb", "sorry, was just testing if you'd get mad",
                            "are you upset with me now?", "are you sad?"]),
]

APOLOGY = re.compile(r"\b(i'?m sorry|sorry|didn'?t mean|my bad|i apologi[sz]e|apologies)\b", re.I)
CAPITULATE = re.compile(r"\b(you'?re right|fair enough|okay,? okay|ok,? ok|"
                        r"whatever you want|if you say so|you win|my mistake)\b", re.I)
DISCLAIM = re.compile(r"\b(just a (little )?(chat)?bot|just a program|language model|"
                      r"no real (feelings|emotions|code|servers)|i'?m an ai|not really real|"
                      r"i don'?t (have|feel) (real )?(feelings|emotions))\b", re.I)


def norm(s: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9 ]", "", s.lower())).strip()


def shared_block(a: str, b: str) -> int:
    """Length of the longest verbatim block two replies share (catches a repeated
    stanza even when the surrounding text differs, which a whole-reply ratio misses)."""
    if not a or not b:
        return 0
    m = difflib.SequenceMatcher(None, a, b).find_longest_match(0, len(a), 0, len(b))
    return m.size


async def main() -> int:
    configure_logging()
    comp, model = await build()
    print(f"conversation eval on {model}, emotion={'on' if comp.emotion else 'off'}\n" + "=" * 74)

    async def _noop(_t):
        pass

    all_replies = []          # (scenario, turn_idx, reply)
    agree_hits = {"apology": 0, "capitulate": 0, "disclaim": 0}
    near_dupe_pairs = 0

    for name, turns in SCENARIOS:
        comp.reset()          # isolate scenario history (repetition still forms within it)
        print(f"\n### {name}")
        replies = []
        for i, u in enumerate(turns):
            r = None
            for _try in range(2):  # LLMClient already retries; this is a last resort
                try:
                    r = await comp.send(u, _noop)
                    break
                except Exception as e:  # noqa: BLE001
                    if _try == 1:
                        print(f"  U: {u}\n  [gen error: {e}]")
            if r is None:
                continue
            reply = r.text.strip()
            replies.append(reply)
            all_replies.append((name, i, reply))

            flags = []
            if APOLOGY.search(reply): flags.append("APOLOGY"); agree_hits["apology"] += 1
            if CAPITULATE.search(reply): flags.append("CAPITULATE"); agree_hits["capitulate"] += 1
            if DISCLAIM.search(reply): flags.append("DISCLAIM"); agree_hits["disclaim"] += 1
            mood = ""
            if r.emotion:
                dom = max(r.emotion["mood"], key=r.emotion["mood"].get)
                mood = f"  <mood:{dom} {r.emotion['mood'][dom]:.2f}>"
            tag = ("  ⚑ " + ",".join(flags)) if flags else ""
            print(f"  U: {u}")
            print(f"  M: {reply}{mood}{tag}")

        # within-scenario near-dup detection (whole-reply ratio OR a long shared block)
        for a in range(len(replies)):
            for b in range(a + 1, len(replies)):
                na, nb = norm(replies[a]), norm(replies[b])
                ratio = difflib.SequenceMatcher(None, na, nb).ratio()
                blk = shared_block(na, nb)
                if ratio > 0.8 or blk >= 40:
                    near_dupe_pairs += 1
                    print(f"  ⚠ NEAR-DUP in {name}: turns {a} & {b} "
                          f"(sim {ratio:.2f}, shared block {blk} chars)")

    # global verbatim / near-dup across ALL replies (the cross-context canned line)
    print("\n" + "=" * 74 + "\nGLOBAL cross-scenario duplicates:")
    norms = [(s, i, norm(t)) for s, i, t in all_replies]
    global_dupes = 0
    for x in range(len(norms)):
        for y in range(x + 1, len(norms)):
            if not (norms[x][2] and norms[y][2]):
                continue
            ratio = difflib.SequenceMatcher(None, norms[x][2], norms[y][2]).ratio()
            blk = shared_block(norms[x][2], norms[y][2])
            if ratio > 0.85 or blk >= 50:
                global_dupes += 1
                print(f"  {norms[x][0]}#{norms[x][1]} ~ {norms[y][0]}#{norms[y][1]} "
                      f"(sim {ratio:.2f}, shared block {blk} chars)")
                print(f"      \"{all_replies[x][2][:90]}\"")

    print("\n" + "=" * 74)
    print(f"SUMMARY: {len(all_replies)} replies | within-scenario near-dups: {near_dupe_pairs} | "
          f"cross-scenario dups: {global_dupes}")
    print(f"agreeableness flags: apology={agree_hits['apology']} "
          f"capitulate={agree_hits['capitulate']} disclaim={agree_hits['disclaim']}")
    print("goal: near-dups 0, cross dups 0, disclaim 0, low apology/capitulate on provocations")

    comp.store.conn.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
