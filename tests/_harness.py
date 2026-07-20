"""The shared offline-test harness: case registry, runner, and scoped fixtures.

Every `tests/test_*.py` was carrying a byte-identical copy of the registry + runner
(~23 lines each). This is that copy, once. Import it and the file shrinks to its cases:

    from _harness import case, run, temp_db

    @case
    async def my_case():
        with temp_db() as (conn, path):
            ...

    if __name__ == "__main__":
        raise SystemExit(run())

**Why `sys.path` is set up here.** Each test file is executed as a script
(`python tests/test_x.py`), so the repo root isn't importable by default and `tests/`
itself is. Importing `_harness` puts the repo root on the path as a side effect, so
test files no longer repeat the `sys.path.insert` dance either.

The fixtures below are context managers on purpose. The suite previously tore down
with bare `conn.close(); os.remove(path)` at the end of each case, which never ran
when a case failed — leaking handles and temp dirs exactly when a run goes bad.
"""
import asyncio
import contextlib
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CASES = []


def case(fn):
    """Register an async test function. Order of definition is order of execution."""
    CASES.append(fn)
    return fn


async def _run_cases(cases) -> int:
    failed = 0
    for fn in cases:
        try:
            await fn()
            print(f"  PASS  {fn.__name__}")
        except AssertionError as e:
            failed += 1
            print(f"  FAIL  {fn.__name__}: {e}")
        except Exception as e:  # noqa: BLE001
            failed += 1
            print(f"  ERROR {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{len(cases) - failed}/{len(cases)} passed")
    return 1 if failed else 0


def run(cases=None) -> int:
    """Run the registered cases and return a process exit code (0 ok / 1 any failure)."""
    return asyncio.run(_run_cases(CASES if cases is None else cases))


@contextlib.contextmanager
def temp_db(name: str = "t.db"):
    """Yield `(conn, path)` for a throwaway SQLite DB, torn down even if the case fails.

    Removes the whole temp *directory*, not just the file — the old per-file teardown
    deleted the db and left the `mkdtemp()` dir behind, leaking one per case.
    """
    from infrastructure.db import connect

    d = tempfile.mkdtemp()
    path = os.path.join(d, name)
    conn = connect(path)
    try:
        yield conn, path
    finally:
        with contextlib.suppress(Exception):
            conn.close()
        with contextlib.suppress(Exception):
            _rmtree(d)


def _rmtree(d: str) -> None:
    import shutil
    shutil.rmtree(d, ignore_errors=True)


@contextlib.contextmanager
def config_override(**overrides):
    """Temporarily set `config` globals, restoring them even if the case fails.

    Cases that tune a knob (`CONSOLIDATE_WINDOW`, `FAMILIARITY_MESSAGES`, …) used to
    assign it and move on. That's invisible today because `run_all.py` gives each file
    its own process, but it makes those cases order-dependent within a file and would
    turn into cross-file flakes under any single-process runner.
    """
    import config

    previous = {k: getattr(config, k) for k in overrides}
    for k, v in overrides.items():
        setattr(config, k, v)
    try:
        yield config
    finally:
        for k, v in previous.items():
            setattr(config, k, v)


class Clock:
    """A mutable fake clock; advance `.t` to simulate elapsed wall-time.

    Injected wherever production reads `time.monotonic()`, so time-dependent logic
    (drive integration, cooldowns, idle gates) is deterministic and instant.
    """

    def __init__(self, t: float = 1_000.0):
        self.t = t

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> float:
        self.t += seconds
        return self.t
