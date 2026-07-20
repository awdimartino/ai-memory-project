"""Offline coverage for consolidation durability (hard-kill recovery).

Exercises the real SqliteConversationStore + SqliteMetaStore + migration runner
against the real Companion, with a fake LLM (canned reply) and a fake memory
manager (records the chunks it's asked to consolidate, can be told to fail). No
LM Studio needed. Each case gets its own fresh DB.

What's verified:
  - a successful window-triggered consolidation advances the persisted watermark
  - a hard kill (no flush) leaves the tail recoverable; a restart re-seeds it and
    a later flush consolidates exactly those messages, then advances the watermark
  - a failed consolidation re-queues the chunk and does NOT advance the watermark
  - upgrading an existing (pre-v4) DB seeds the watermark to its max message id,
    so the whole backlog isn't re-consolidated on first run

Run:  python tests/test_durability.py
"""
import asyncio
import os
import sys
import tempfile

from _harness import case, config_override, run  # also puts the repo root on sys.path

import config
import infrastructure.db as db
from core.companion import CONSOLIDATED_WATERMARK_KEY, Companion
from infrastructure.conversation_store import SqliteConversationStore
from infrastructure.db import connect
from infrastructure.meta_store import SqliteMetaStore

KEY = CONSOLIDATED_WATERMARK_KEY


class FakeLLM:
    async def stream(self, messages, on_token):
        for t in ("ok",):
            await on_token(t)
        return "ok", {"ttft": 0.0, "tok_per_s": 0.0, "tokens": 1, "estimated": True}


class FakeMemory:
    """Records consolidation chunks; optionally fails to simulate an error."""

    def __init__(self, fail: bool = False):
        self.fail = fail
        self.chunks: list[list[dict]] = []

    async def recall(self, text):
        return []

    def core_memories(self):
        return []

    async def consolidate(self, chunk, session_id):
        if self.fail:
            raise RuntimeError("boom")
        self.chunks.append(list(chunk))
        return len(chunk)


async def _noop(_t):
    pass


def _build(path: str, memory):
    """Mimic bootstrap's wiring + tail recovery against the real stores."""
    conn = connect(path)
    conv = SqliteConversationStore(conn)
    meta = SqliteMetaStore(conn)
    watermark = meta.get_int(KEY, 0)
    unconsolidated = conv.messages_after(watermark)
    session_id = conv.create_session()
    history = conv.recent_messages(config.HISTORY_TURNS)
    comp = Companion(FakeLLM(), conv, memory, meta, session_id, history, unconsolidated)
    return comp, conn, conv, meta


async def _wait(pred, tries: int = 1000) -> bool:
    """Poll until pred() is true (lets background create_task / to_thread finish)."""
    for _ in range(tries):
        if pred():
            return True
        await asyncio.sleep(0.001)
    return False


def _ids(chunk):
    return [m["id"] for m in chunk]


@case
async def watermark_advances_on_window_consolidation():
    with config_override(CONSOLIDATE_WINDOW=4):  # 2 exchanges
        path = os.path.join(tempfile.mkdtemp(), "d.db")
        mem = FakeMemory()
        comp, conn, conv, meta = _build(path, mem)

        await comp.send("hi", _noop)   # ids 1,2
        await comp.send("yo", _noop)   # ids 3,4 -> triggers consolidation

        ok = await _wait(lambda: meta.get_int(KEY, 0) == 4 and not comp._unconsolidated)
        assert ok, f"watermark/buffer wrong: wm={meta.get_int(KEY,0)}, buf={len(comp._unconsolidated)}"
        assert mem.chunks and _ids(mem.chunks[0]) == [1, 2, 3, 4], _ids(mem.chunks[0])
        conn.close(); os.remove(path)


@case
async def hard_kill_tail_is_recovered_and_flushed():
    with config_override(CONSOLIDATE_WINDOW=100):  # never window-triggers here
        path = os.path.join(tempfile.mkdtemp(), "d.db")

        # First run: chat, then "crash" (drop the companion WITHOUT flushing).
        comp, conn, conv, meta = _build(path, FakeMemory())
        for _ in range(3):
            await comp.send("x", _noop)          # 6 messages logged, ids 1..6
        assert meta.get_int(KEY, 0) == 0, "nothing consolidated yet"
        assert len(comp._unconsolidated) == 6
        conn.close()  # simulate hard kill: no flush

        # Restart: the tail must be recovered from the store.
        mem2 = FakeMemory()
        comp2, conn2, conv2, meta2 = _build(path, mem2)
        assert _ids(comp2._unconsolidated) == [1, 2, 3, 4, 5, 6], _ids(comp2._unconsolidated)

        # A graceful flush now consolidates exactly the recovered tail + checkpoints.
        await comp2.flush()
        assert mem2.chunks and _ids(mem2.chunks[0]) == [1, 2, 3, 4, 5, 6]
        assert meta2.get_int(KEY, 0) == 6
        conn2.close()

        # Restart once more: nothing left to recover.
        comp3, conn3, conv3, meta3 = _build(path, FakeMemory())
        assert comp3._unconsolidated == [], comp3._unconsolidated
        conn3.close(); os.remove(path)


@case
async def failed_consolidation_requeues_and_holds_watermark():
    with config_override(CONSOLIDATE_WINDOW=4):
        path = os.path.join(tempfile.mkdtemp(), "d.db")
        mem = FakeMemory(fail=True)
        comp, conn, conv, meta = _build(path, mem)

        await comp.send("hi", _noop)   # ids 1,2
        await comp.send("yo", _noop)   # ids 3,4 -> consolidation fires, raises

        ok = await _wait(lambda: not comp._consol_lock.locked() and len(comp._unconsolidated) == 4)
        assert ok, f"requeue failed: buf={len(comp._unconsolidated)}"
        assert _ids(comp._unconsolidated) == [1, 2, 3, 4]
        assert meta.get_int(KEY, 0) == 0, "watermark must not advance on failure"

        # Recover: succeed on flush; the same messages consolidate and checkpoint.
        mem.fail = False
        await comp.flush()
        assert mem.chunks and _ids(mem.chunks[0]) == [1, 2, 3, 4]
        assert meta.get_int(KEY, 0) == 4
        conn.close(); os.remove(path)


@case
async def watermark_never_jumps_past_pending():
    # A newer chunk (ids 5,6) consolidates while an OLDER chunk (ids 3,4) is still pending
    # (e.g. a failed background pass re-queued it). The watermark must clamp below the pending
    # ids, or a crash would leave 3,4 below the watermark and unrecoverable.
    path = os.path.join(tempfile.mkdtemp(), "d.db")
    comp, conn, conv, meta = _build(path, FakeMemory())
    comp._unconsolidated = [{"id": 3, "role": "user", "content": "x"},
                            {"id": 4, "role": "assistant", "content": "y"}]
    await comp._advance_watermark([{"id": 5, "role": "user", "content": "a"},
                                   {"id": 6, "role": "assistant", "content": "b"}])
    assert meta.get_int(KEY, 0) == 2, f"must clamp below pending id 3, got {meta.get_int(KEY, 0)}"
    conn.close(); os.remove(path)


@case
async def watermark_never_regresses():
    path = os.path.join(tempfile.mkdtemp(), "d.db")
    comp, conn, conv, meta = _build(path, FakeMemory())
    meta.set_int(KEY, 10)
    await comp._advance_watermark([{"id": 5, "role": "user", "content": "a"}])  # lower than 10
    assert meta.get_int(KEY, 0) == 10, "watermark must not regress below its current value"
    conn.close(); os.remove(path)


@case
async def upgrade_seeds_watermark_to_max_message_id():
    # An existing DB (pre-v4) with logged messages must not re-consolidate its
    # whole backlog: the v4 migration seeds the watermark to the current max id.
    path = os.path.join(tempfile.mkdtemp(), "d.db")
    original = db.MIGRATIONS
    try:
        db.MIGRATIONS = original[:3]          # bring the DB up to v3 only
        conn = connect(path)
        conv = SqliteConversationStore(conn)
        # Insert a session via raw SQL — create_session() writes `title`, which the
        # v3 schema doesn't have yet (it arrives in v7); we're simulating an old DB.
        conn.execute("INSERT INTO sessions (started_at) VALUES ('t')")
        conn.commit()
        for i in range(5):
            conv.add_message(1, "user", f"m{i}")   # ids 1..5
        conn.close()

        db.MIGRATIONS = original               # v4 now available
        conn2 = connect(path)                  # applies v4 -> seeds watermark
        meta = SqliteMetaStore(conn2)
        assert meta.get_int(KEY, 0) == 5, f"seed wrong: {meta.get_int(KEY, 0)}"
        # And so a restart recovers nothing (backlog treated as already handled).
        assert SqliteConversationStore(conn2).messages_after(meta.get_int(KEY, 0)) == []
        conn2.close()
    finally:
        db.MIGRATIONS = original
    os.remove(path)


if __name__ == "__main__":
    raise SystemExit(run())
