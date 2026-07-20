"""Offline coverage for the gold-set runner's concurrency plumbing.

The gold set runs its cases concurrently. That is only sound because a case touches
no process-global state, and the two things that used to break it are pinned here:

  1. `bootstrap.build()` took its database from `config.DB_PATH`, so the runner
     ASSIGNED that global before each build. Two concurrent builds would then race and
     land on one database — which is the exact contamination the per-case DB exists to
     prevent, reintroduced silently. Results would still look plausible.
  2. The seed-embedding cache is shared across cases.

Neither failure is visible in a run's output, which is why they're worth a test rather
than a careful reading. No LM Studio needed — the LLM seam is never reached.

Run:  python tests/test_gold_runner.py
"""
import asyncio
import os
import sys

from _harness import case, run, temp_dir  # also puts the repo root on sys.path


def _run_gold():
    """Import `evals.run_gold`, working around the two modules named `_harness`.

    `tests/_harness.py` and `scripts/_harness.py` are different modules with the same
    name. This test file imports the first, so it owns `sys.modules["_harness"]`; when
    `run_gold` then does `from _harness import scratch_env` it gets the tests one and
    dies on ImportError. Stashing the entry for the duration of the import lets each
    file resolve the sibling it means.

    Deliberately a local helper rather than a fix to either harness: renaming one is a
    wide rename across ~20 call sites for the benefit of this single import.
    """
    stashed = sys.modules.pop("_harness", None)
    sys.path.insert(0, os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
    try:
        import evals.run_gold as rg
        return rg
    finally:
        sys.path.pop(0)
        if stashed is not None:
            sys.modules["_harness"] = stashed


# --- 1. the db_path seam ----------------------------------------------------------

@case
async def build_honours_an_explicit_db_path():
    """`build(db_path=...)` uses it, and leaves config.DB_PATH alone.

    Both halves matter: using it is what gives each case its own store, and not
    writing the global is what stops concurrent cases clobbering each other.
    """
    import config
    import bootstrap

    seen = []

    def fake_connect(path):
        seen.append(path)
        raise _Stop()          # everything after connect() wants a live LM Studio

    original, before = bootstrap.connect, config.DB_PATH
    bootstrap.connect = fake_connect
    try:
        wanted = os.path.join(temp_dir(), "case.db")
        try:
            await bootstrap.build(db_path=wanted)
        except _Stop:
            pass
        assert seen == [wanted], f"build used {seen}, not the db_path it was given"
        assert config.DB_PATH == before, "build mutated the global config.DB_PATH"
    finally:
        bootstrap.connect = original


@case
async def build_falls_back_to_config_when_not_given():
    """The REPL and web app pass nothing; they must keep getting config.DB_PATH."""
    import config
    import bootstrap

    seen = []

    def fake_connect(path):
        seen.append(path)
        raise _Stop()

    original = bootstrap.connect
    bootstrap.connect = fake_connect
    try:
        try:
            await bootstrap.build()
        except _Stop:
            pass
        assert seen == [config.DB_PATH], f"default build used {seen}"
    finally:
        bootstrap.connect = original


@case
async def concurrent_builds_get_distinct_databases():
    """The property the whole change rests on, asserted under real concurrency.

    Against the old global-assignment approach this fails: interleaved tasks each
    overwrite config.DB_PATH, so several builds read the same value.
    """
    import bootstrap

    seen = []

    def fake_connect(path):
        seen.append(path)
        raise _Stop()

    async def one(path):
        try:
            await bootstrap.build(db_path=path)
        except _Stop:
            pass

    original = bootstrap.connect
    bootstrap.connect = fake_connect
    try:
        paths = [os.path.join(temp_dir(), f"case{i}.db") for i in range(8)]
        await asyncio.gather(*(one(p) for p in paths))
        assert sorted(seen) == sorted(paths), "concurrent builds did not get distinct DBs"
        assert len(set(seen)) == 8, f"only {len(set(seen))} distinct databases"
    finally:
        bootstrap.connect = original


# --- 2. the shared seed cache -----------------------------------------------------

@case
async def seed_cache_embeds_each_fact_once_under_concurrency():
    """Concurrent cases seeded with the same fact must embed it once, not once each.

    Without the lock both tasks miss the cache, both await the embedder, and both
    write — correct results, wasted calls. With many cases sharing a handful of seed
    facts that is a real chunk of a run.
    """
    _seed = _run_gold()._seed

    calls = []

    class Embedder:
        async def embed_document(self, text):
            calls.append(text)
            await asyncio.sleep(0.01)      # the window the race needs
            return [1.0, 0.0]

    class Store:
        def __init__(self):
            self.rows = []

        def add(self, content, cat, vec, sess, core=False):
            self.rows.append(content)

    class Comp:
        def __init__(self):
            self.memory = type("M", (), {"embedder": Embedder(), "store": Store()})()

    cache, lock = {}, asyncio.Lock()
    comps = [Comp() for _ in range(5)]
    await asyncio.gather(*(_seed(c, ["my dog is called Pip"], cache, lock) for c in comps))

    assert calls == ["my dog is called Pip"], f"embedded {len(calls)} times, expected 1"
    for c in comps:                        # every case still got the fact planted
        assert c.memory.store.rows == ["my dog is called Pip"]


@case
async def seed_marks_name_facts_core():
    """Unchanged behaviour, pinned because the seeding path was edited around it."""
    _seed = _run_gold()._seed

    class Embedder:
        async def embed_document(self, text):
            return [1.0, 0.0]

    class Store:
        def __init__(self):
            self.rows = []

        def add(self, content, cat, vec, sess, core=False):
            self.rows.append((content, core))

    comp = type("C", (), {"memory": type("M", (), {"embedder": Embedder(), "store": Store()})()})()
    await _seed(comp, ["his name is Alex", "he likes coffee"], {}, asyncio.Lock())
    assert comp.memory.store.rows == [("his name is Alex", True), ("he likes coffee", False)]


# --- 3. the consolidation wait ----------------------------------------------------

@case
async def wait_for_returns_as_soon_as_the_predicate_holds():
    """The generous CONSOL_TIMEOUT must be a ceiling, not a sleep.

    If this ever became a fixed delay, a 300s budget would add ~11 hours to a run.
    """
    _wait_for = _run_gold()._wait_for

    flag = []
    loop = asyncio.get_running_loop()
    loop.call_later(0.05, lambda: flag.append(1))

    started = loop.time()
    assert await _wait_for(lambda: bool(flag), timeout=30.0) is True
    assert loop.time() - started < 1.0, "waited far longer than the predicate took"


@case
async def wait_for_gives_up_and_reports_false():
    """A case whose consolidation never lands scores the store anyway (and fails)."""
    _wait_for = _run_gold()._wait_for

    assert await _wait_for(lambda: False, timeout=0.05, step=0.01) is False


class _Stop(Exception):
    """Raised by the fake `connect` to stop build() before it needs a live server."""


if __name__ == "__main__":
    raise SystemExit(run())
