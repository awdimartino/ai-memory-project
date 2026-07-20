"""Run every offline test file and exit non-zero if any of them fail.

The suite is deliberately offline — no LM Studio, no network, no HuggingFace. Each
`test_*.py` is a standalone script that reports its own pass count, so this runner
just discovers them and shells out.

**One process per file, on purpose.** A few files mutate `config` globals for a case
(and now restore them), but process isolation is the belt to that suspenders: a file
that forgets to restore still can't leak into its siblings. It also means a hard crash
in one file can't take the rest of the run down. Discovery is a glob, so a new
`tests/test_*.py` is picked up with no edit here — the failure mode that let
`test_intentions.py` and `test_self_notes.py` sit outside the documented suite.

Run:  python tests/run_all.py
      python tests/run_all.py -q          # only show failures
      python tests/run_all.py drives tick # only files matching these substrings
"""
import re
import subprocess
import sys
import time
from pathlib import Path

TESTS_DIR = Path(__file__).resolve().parent

# Files report their tally in one of two shapes, and several log expected errors
# (deliberate fault injection, retry warnings) *after* it — so scan for the tally
# rather than trusting the last line.
_TALLY = re.compile(r"(\d+)/(\d+) passed|OK - (\d+) checks passed")


def tally(output: str) -> tuple[str, int]:
    """Return (summary line, checks counted) for one file's output."""
    for line in reversed(output.splitlines()):
        m = _TALLY.search(line)
        if m:
            return line.strip(), int(m.group(1) or m.group(3))
    return "(no tally reported)", 0


def discover(filters: list[str]) -> list[Path]:
    files = sorted(TESTS_DIR.glob("test_*.py"))
    if filters:
        files = [f for f in files if any(s in f.stem for s in filters)]
    return files


def main() -> int:
    args = [a for a in sys.argv[1:] if a != "-q"]
    quiet = "-q" in sys.argv[1:]

    files = discover(args)
    if not files:
        print(f"No test files matched {args or '*'} in {TESTS_DIR}")
        return 1

    failures: list[tuple[str, str]] = []
    checks = 0
    started = time.perf_counter()

    for path in files:
        proc = subprocess.run(
            [sys.executable, str(path)],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            cwd=TESTS_DIR.parent,
        )
        out = (proc.stdout or "") + (proc.stderr or "")
        summary, counted = tally(out)
        checks += counted

        if proc.returncode == 0:
            if not quiet:
                print(f"  PASS  {path.name:<32} {summary}")
        else:
            failures.append((path.name, out))
            print(f"  FAIL  {path.name:<32} (exit {proc.returncode})")

    elapsed = time.perf_counter() - started

    for name, out in failures:  # full output only for what broke
        print(f"\n{'=' * 70}\n{name}\n{'=' * 70}\n{out.rstrip()}")

    passed = len(files) - len(failures)
    print(f"\n{passed}/{len(files)} files passed "
          f"({checks} checks) in {elapsed:.1f}s")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
