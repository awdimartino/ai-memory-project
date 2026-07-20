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
import atexit
import contextlib
import os
import shutil
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

CASES = []

_TEMP_DIRS: list[str] = []
_TEMP_CONNS: list = []


def track_conn(conn):
    """Register a connection to be closed at exit; returns it for chaining.

    Windows won't remove a directory holding an open SQLite file, so a case that
    never closes its connection defeats the cleanup below. Wrap the connect call
    (`track_conn(connect(path))`) when a case has no natural place to close it.
    """
    _TEMP_CONNS.append(conn)
    return conn


def temp_dir() -> str:
    """A throwaway directory, removed when the process exits.

    Cases used to call `tempfile.mkdtemp()` directly and clean up with
    `os.remove(path)` — which deletes the *db file* and leaves the directory behind,
    one per case, forever. Routing them through here means the cleanup happens even
    when a case fails partway (the common reason teardown never ran).

    Prefer `temp_db()` for the single-connection case; this is for cases that open
    several connections to one path (crash/restart simulations) where a scoped
    context manager doesn't fit.
    """
    d = tempfile.mkdtemp()
    _TEMP_DIRS.append(d)
    return d


@atexit.register
def _cleanup_temp_dirs() -> None:
    for conn in _TEMP_CONNS:  # close first — an open handle blocks rmtree on Windows
        with contextlib.suppress(Exception):
            conn.close()
    for d in _TEMP_DIRS:
        shutil.rmtree(d, ignore_errors=True)
    _TEMP_CONNS.clear()
    _TEMP_DIRS.clear()


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

    d = temp_dir()  # also tracked for exit-time cleanup, in case the rmtree below fails
    path = os.path.join(d, name)
    conn = connect(path)
    try:
        yield conn, path
    finally:
        with contextlib.suppress(Exception):
            conn.close()
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
