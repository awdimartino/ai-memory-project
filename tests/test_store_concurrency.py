"""Offline coverage for the store write lock.

All five stores are handed the SAME connection by bootstrap, and different stores
are written from different `asyncio.to_thread` executor threads — a chat turn
logging a message via `add_message` while a tick job persists drives via
`meta.set_json`. Each store used to create its own `threading.Lock`, which
serialized a store against itself and against nothing else.

That lost data. Measured before the fix, two stores hammering one connection from
four threads: 594/600 and 592/600 rows survived, with 173 errors including CPython
`SystemError: error return without exception set`. These cases pin the fix — one
lock per *connection*, shared by every store on it.

Run:  python tests/test_store_concurrency.py
"""
import threading

from _harness import case, run, temp_db  # also puts the repo root on sys.path

from infrastructure.conversation_store import SqliteConversationStore
from infrastructure.db import connection_lock
from infrastructure.intention_store import SqliteIntentionStore
from infrastructure.memory_store import SqliteMemoryStore
from infrastructure.meta_store import SqliteMetaStore
from infrastructure.thought_store import SqliteThoughtStore


def _all_stores(conn):
    return (SqliteConversationStore(conn), SqliteMemoryStore(conn), SqliteMetaStore(conn),
            SqliteThoughtStore(conn), SqliteIntentionStore(conn))


def _hammer(jobs, per_thread):
    """Run each job `per_thread` times on its own thread; return the errors seen."""
    errors = []

    def worker(fn):
        for i in range(per_thread):
            try:
                fn(i)
            except Exception as e:  # noqa: BLE001 - collecting, not swallowing
                errors.append(f"{type(e).__name__}: {e}")

    threads = [threading.Thread(target=worker, args=(j,)) for j in jobs]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    return errors


@case
async def every_store_on_one_connection_shares_one_lock():
    with temp_db("locks.db") as (conn, _path):
        stores = _all_stores(conn)
        locks = {id(s._lock) for s in stores}
        assert len(locks) == 1, f"expected one shared lock, got {len(locks)} distinct"
        assert stores[0]._lock is connection_lock(conn), "must be the connection's lock"


@case
async def separate_connections_get_separate_locks():
    # The lock is per connection, not a global: two DBs must not serialize together.
    with temp_db("a.db") as (conn_a, _a), temp_db("b.db") as (conn_b, _b):
        assert connection_lock(conn_a) is not connection_lock(conn_b)


@case
async def concurrent_writes_across_stores_lose_nothing():
    with temp_db("race.db") as (conn, _path):
        conv, _mem, meta, thoughts, intentions = _all_stores(conn)
        sid = conv.create_session()
        n = 120

        errors = _hammer([
            lambda i: thoughts.add(f"t{i}", "warmth"),
            lambda i: intentions.add(f"i{i}"),
            lambda i: conv.add_message(sid, "user", f"m{i}"),
            lambda i: meta.set_json(f"k{i}", {"v": i}),
            lambda i: conv.add_message(sid, "assistant", f"a{i}"),
        ], n)

        assert not errors, f"concurrent writes raised: {errors[:3]}"
        counts = {t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
                  for t in ("thoughts", "intentions", "messages")}
        assert counts["thoughts"] == n, f"lost thoughts: {counts['thoughts']}/{n}"
        assert counts["intentions"] == n, f"lost intentions: {counts['intentions']}/{n}"
        assert counts["messages"] == n * 2, f"lost messages: {counts['messages']}/{n * 2}"
        assert conn.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


@case
async def row_ids_stay_unique_under_concurrency():
    # add_message returns lastrowid, and the durability watermark is built on those
    # ids — a torn cursor handing back a duplicate would corrupt the checkpoint.
    with temp_db("ids.db") as (conn, _path):
        conv = SqliteConversationStore(conn)
        sid = conv.create_session()
        seen, guard = [], threading.Lock()

        def add(i):
            rid = conv.add_message(sid, "user", f"m{i}")
            with guard:
                seen.append(rid)

        errors = _hammer([add, add, add], 80)
        assert not errors, errors[:3]
        assert len(seen) == 240
        assert len(set(seen)) == len(seen), "duplicate row ids handed back"


@case
async def wal_and_synchronous_pragmas_are_set():
    with temp_db("pragma.db") as (conn, _path):
        mode = conn.execute("PRAGMA journal_mode").fetchone()[0]
        sync = conn.execute("PRAGMA synchronous").fetchone()[0]
        assert mode.lower() == "wal", f"expected WAL, got {mode}"
        # 1 == NORMAL, the standard WAL pairing: no fsync per commit (every
        # add_message commits) while a power loss still costs only recent writes.
        assert sync == 1, f"expected synchronous=NORMAL (1), got {sync}"


if __name__ == "__main__":
    raise SystemExit(run())
