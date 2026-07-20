"""Reflection Repetition Rate — a health check on Mari's self-generated content.

Reflexive agents store confident-but-wrong self-diagnoses and then reuse them. The
measured signal is RRR: how often a new reflection duplicates an earlier one. In the
source study it correlated **0.808 with trials-to-solve** — i.e. repetition predicts
that a self-generated note is WRONG, not that it's important. Two environments
solvable in 1 trial *without* memory took 7-8 trials *with* confabulated reflections.
(arXiv:2605.29463)

Why it matters here: her journal feeds the persona edit, and her self-notes steer
every user-facing prompt. A confabulated operating-rule changes her behavior every
turn and then generates the evidence that confirms it.

Deliberately OFFLINE — lexical similarity only, no LM Studio, no embeddings. It reads
the real companion.db read-only so it can be run any time.

Run:  python scripts/rrr_diagnostic.py
      python scripts/rrr_diagnostic.py --db some-other.db --threshold 0.5
"""
import argparse
import re
import sqlite3
import sys
from itertools import combinations

from _harness import repo_path  # path + UTF-8 setup

from core.textsim import REPEAT_THRESHOLD, similarity  # ONE definition of "same thought"

DUPLICATE_THRESHOLD = REPEAT_THRESHOLD  # same threshold the live guard uses
NEAR_VERBATIM = 0.85



def report(label: str, entries: list[tuple[int, str]], threshold: float) -> float:
    """Print an RRR report for one stream. Returns the rate."""
    print(f"\n{'=' * 74}\n{label}  ({len(entries)} entries)\n{'=' * 74}")
    if len(entries) < 2:
        print("  too few entries to score")
        return 0.0

    # An entry is a repeat if it duplicates ANY earlier entry.
    repeats, pairs = set(), []
    for (i, ti), (j, tj) in combinations(entries, 2):
        sim = similarity(ti, tj)
        if sim >= threshold:
            repeats.add(j)                     # the later one is the repeat
            pairs.append((sim, i, j, ti, tj))

    rrr = len(repeats) / (len(entries) - 1)
    verdict = ("HEALTHY" if rrr < 0.15 else
               "ELEVATED — worth a prompt look" if rrr < 0.35 else
               "*** FROZEN — these are very likely confabulated ***")
    print(f"  RRR = {rrr:.2f}  ({len(repeats)} of {len(entries) - 1} could-be-repeats)   {verdict}")

    exact = sum(1 for s, *_ in pairs if s >= NEAR_VERBATIM)
    if exact:
        print(f"  {exact} near-verbatim pair(s) (>={NEAR_VERBATIM:.0%} overlap)")

    for sim, i, j, ti, tj in sorted(pairs, reverse=True)[:5]:
        print(f"\n  {sim:.0%} overlap — #{i} vs #{j}")
        print(f"    #{i}: {ti[:110]}")
        print(f"    #{j}: {tj[:110]}")
    return rrr


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=repo_path("companion.db"))
    ap.add_argument("--threshold", type=float, default=DUPLICATE_THRESHOLD)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row

    thoughts = [(r["id"], r["content"])
                for r in conn.execute("SELECT id, content FROM thoughts ORDER BY id")]
    t_rrr = report("THOUGHT JOURNAL (feeds the persona edit)", thoughts, args.threshold)

    # Self-notes live in a single overwritten MetaStore slot, so there is no history
    # to score unless revision logging is on (self_notes_log).
    try:
        notes = [(r["id"], r["content"])
                 for r in conn.execute("SELECT id, content FROM self_notes_log ORDER BY id")]
    except sqlite3.OperationalError:
        notes = []
    if notes:
        report("SELF-NOTES REVISIONS (steer every user-facing prompt)", notes, args.threshold)
    else:
        print(f"\n{'=' * 74}\nSELF-NOTES REVISIONS\n{'=' * 74}")
        print("  no revision history yet — the slot is overwritten wholesale, so RRR")
        print("  is uncomputable until a few edits have been logged.")

    print(f"\n{'=' * 74}")
    if t_rrr >= 0.35:
        print("The reflection prompt already injects recent thoughts to avoid repeats.")
        print("If RRR is still high, prompting is not the fix -- the study's mitigation")
        print("was PROGRAMMATIC extraction (parse real signals from the log) over")
        print("free-form self-diagnosis: correct-mention 0% -> 86%, RRR 0.64 -> 0.10.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
