"""Count the verbal tics the gold set cannot see.

WHY THIS EXISTS. Three fixes shipped on 2026-07-20 were unfalsifiable: nothing in the
gold set counts how often she names an emotion, or opens with the same stock word. The
tics were found by the user reading conversations, and "did that help?" had no answer
except reading more conversations. A rate over her real log is the missing instrument.

It also guards a specific temptation. The obvious fix for a tic is to strip the word --
which would drive this number to zero while changing nothing about the behaviour that
produces it. Keep scoring, and a suppressed symptom stays visible as a substitution
somewhere else in the table.

    python evals/style_scorer.py                 # per era, from companion.db
    python evals/style_scorer.py --era 3         # one era
    python evals/style_scorer.py --db other.db

Offline and read-only: no model, no LM Studio, no writes.

READING IT
  emotion-naming   she says what she feels in so many words ("i'm irritated"). The mood
                   block already says never to; the research (HANDOFF §G2) says such
                   restraint instructions are near-inert, so this measures the gap.
  stock openers    formulaic first words. 'haha' hit 67% of an era after thinking-on.
  one-sentence /   the two rules that ARE obeyed (both ~0% violation), kept here as a
  question-end     control: if a change moves these, it hurt something that worked.
"""
import argparse
import os
import re
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# First-person emotion statements. Deliberately targets SELF-report ("i'm annoyed"),
# not the word alone -- "that sounds annoying" is about them and is fine.
FEEL_VERB = r"(?:i(?:'m| am| feel|feel|'ve been| was)|im|feeling)"
EMOTION_WORD = (r"(?:irritat\w*|annoy\w*|angry|anger|frustrat\w*|warm\w*|fond|amus\w*|"
                r"happy|glad|sad|melanchol\w*|down|low|uneasy|unease|anxious|nervous|"
                r"curious|interest\w*|bored|boredom|lonely|content|calm|upset|hurt)")
EMOTION_SELF_REPORT = re.compile(rf"\b{FEEL_VERB}\b[^.!?]{{0,24}}\b{EMOTION_WORD}\b", re.I)

# Openers worth watching. 'haha' is the live regression; the rest are the family she
# would plausibly substitute into if it were simply banned.
OPENERS = ("haha", "hah", "heh", "ha", "lol", "honestly", "yeah", "nah", "oh", "well",
           "i think", "that's", "it's")

SENTENCE_SPLIT = re.compile(r"[.!?]+(?:\s|$)")


def sentences(text: str) -> int:
    return len([s for s in SENTENCE_SPLIT.split(text.strip()) if s.strip()])


def opener_of(text: str) -> str | None:
    """The stock word a reply opens with, if any. Longest match wins so 'i think'
    is not scored as bare 'i'."""
    low = text.strip().lower().lstrip("\"'(")
    for o in sorted(OPENERS, key=len, reverse=True):
        if low.startswith(o) and (len(low) == len(o) or not low[len(o)].isalpha()):
            return o
    return None


def score(messages: list[str]) -> dict:
    n = len(messages) or 1
    openers: dict[str, int] = {}
    naming = multi = qend = 0
    for m in messages:
        if EMOTION_SELF_REPORT.search(m):
            naming += 1
        o = opener_of(m)
        if o:
            openers[o] = openers.get(o, 0) + 1
        if sentences(m) > 1:
            multi += 1
        if m.strip().endswith("?"):
            qend += 1
    return {"n": len(messages), "naming": naming, "multi": multi, "qend": qend,
            "openers": openers, "stock_total": sum(openers.values()),
            "pct": lambda k: 100 * k / n}


def report(label: str, messages: list[str]) -> None:
    if not messages:
        print(f"\n{label}: no messages")
        return
    s = score(messages)
    n = s["n"]
    p = lambda k: f"{100 * k / n:5.1f}%"
    print(f"\n{label}  (n={n})")
    print(f"  emotion self-report   {p(s['naming'])}  {s['naming']:>4}   <- the one to move")
    print(f"  stock opener (any)    {p(s['stock_total'])}  {s['stock_total']:>4}")
    for o, c in sorted(s["openers"].items(), key=lambda kv: -kv[1])[:6]:
        print(f"      {o:<10}        {p(c)}  {c:>4}")
    print(f"  >1 sentence           {p(s['multi'])}  {s['multi']:>4}   (control: keep ~0)")
    print(f"  ends on a question    {p(s['qend'])}  {s['qend']:>4}   (control: keep ~0)")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default="companion.db")
    ap.add_argument("--era", type=int, help="only this era")
    args = ap.parse_args()

    path = args.db if os.path.isabs(args.db) else os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), args.db)
    if not os.path.exists(path):
        print(f"no database at {path}", file=sys.stderr)
        return 1

    c = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    rows = list(c.execute(
        "select era, content from message_archive where role='assistant' order by id"))
    if not rows:
        print("no assistant messages recorded")
        return 0

    eras = sorted({e for e, _ in rows})
    if args.era is not None:
        eras = [e for e in eras if e == args.era]
    for e in eras:
        report(f"era {e}", [m for era, m in rows if era == e])

    if len(eras) > 1:
        print("\n" + "-" * 58)
        print("Compare eras, not absolutes: an era is a config, and the number that")
        print("matters is whether a change moved it. Controls must stay near zero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
