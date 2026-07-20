"""Which gold cases are NOISE, and which failures are real? Compares saved runs.

WHY THIS EXISTS
  The gold set is stochastic — three runs of *unchanged* code scored 104/107/106 —
  so a single run cannot tell "my change fixed this" from "this case flips". Every
  measurement in this project is read against that ambiguity, and the only way out
  is to look across runs.

  A case that has flipped in BOTH directions across runs is proven noise: no single
  run's verdict on it means anything. A case that moved once and stayed is a
  candidate signal — not proof, but worth attributing. A case that fails in every
  run is real.

  Run it after any gold run, before you claim a change did something:

    python evals/flaky.py                  # every saved version, oldest first
    python evals/flaky.py v2.2 v2.3 v2.4   # specific ones, in order
"""
import json
import sys
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

RESULTS = Path(__file__).resolve().parent / "results"
# "review" is a manual case that was never auto-scored, and "known" is a gap we chose
# not to fix. Neither is a failure, and lumping them in with FAIL overstates how much
# is broken -- so they're tracked separately rather than counted as "not passing".
PASSED = {"pass", "fixed!"}
UNSCORED = {"review", "known"}


def load(versions):
    runs = {}
    for v in versions:
        path = RESULTS / f"{v}.json"
        if not path.exists():
            print(f"  (no results for {v})")
            continue
        d = json.loads(path.read_text(encoding="utf-8"))
        runs[v] = {c["id"]: c["status"] for c in d["records"]}
        print(f"  {v}: {d['rate']:.1%}  {d['counts']}")
    return runs


def main(argv):
    versions = argv or sorted(p.stem for p in RESULTS.glob("*.json"))
    print("runs:")
    runs = load(versions)
    if len(runs) < 2:
        print("\nneed at least two saved runs to say anything about flakiness.")
        return 1

    names = list(runs)
    shared = [i for i in runs[names[-1]] if all(i in r for r in runs.values())]

    both_ways, moved_once, always_failing, unscored = [], [], [], []
    for i in shared:
        seq = [runs[v][i] for v in names]
        verdicts = [s in PASSED for s in seq]
        if all(s in UNSCORED for s in seq):
            unscored.append(i)
        elif not any(verdicts):
            always_failing.append(i)
        elif len(set(verdicts)) > 1:
            # Count TRANSITIONS, not endpoints. Comparing first to last called
            # reg-rambling (FAIL->pass->FAIL->pass) a candidate signal, when two
            # reversals are exactly what noise looks like. One transition is a case
            # that moved and stayed; more than one is a case that cannot hold still.
            flips = sum(1 for a, b in zip(verdicts, verdicts[1:]) if a != b)
            (both_ways if flips > 1 else moved_once).append(i)

    def show(title, ids, note=""):
        print(f"\n{title} ({len(ids)}){note}")
        for i in sorted(ids):
            print(f"  {i:<34} " + " -> ".join(f"{runs[v][i]}" for v in names))

    show("PROVEN NOISE — flipped both ways", both_ways,
         "\n  Ignore single-run verdicts on these; they say nothing about a change.")
    show("MOVED AND HELD — candidate signal", moved_once,
         "\n  Attributable only if a mechanism explains them. Otherwise: noise that got lucky.")
    show("REAL FAILURES — failed every run", always_failing)
    print(f"\nawaiting human review in every run: {len(unscored)} "
          f"(never auto-scored; not failures)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
