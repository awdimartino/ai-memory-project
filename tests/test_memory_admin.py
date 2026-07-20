"""Offline coverage for the memory admin ops (inspector + clear + factory reset).

Real SQLite stores on a throwaway DB with a fake embedder — no LM Studio needed.
Verifies the new store methods (all / delete / clear / update_content), the
MemoryManager edit (re-embed), and the two destructive Companion ops:
clear_memories (memories only) and factory_reset (everything, in-memory too).

Run:  python tests/test_memory_admin.py
"""
import os

from _harness import case, run, temp_dir# also puts the repo root on sys.path
from helpers import FakeClassifier

import numpy as np

from core.companion import PERSONA_SELF_KEY, Companion
from core.drives import BASELINE as DRIVE_BASELINE, DriveManager
from core.emotion_manager import BASELINE_STATE, CHANNELS, EmotionManager
from core.memory_manager import MemoryManager
from infrastructure.conversation_store import SqliteConversationStore
from infrastructure.db import connect
from infrastructure.memory_store import SqliteMemoryStore
from infrastructure.meta_store import SqliteMetaStore
from infrastructure.thought_store import SqliteThoughtStore


def _emb(seed=0.1):
    return np.array([seed, seed + 0.1, seed + 0.2], dtype=np.float32).tobytes()


class FakeEmbedder:
    async def embed_document(self, text):
        return [0.9, 0.8, 0.7]

    async def embed_query(self, text):
        return [0.9, 0.8, 0.7]


def _stores():
    path = os.path.join(temp_dir(), "admin.db")
    conn = connect(path)
    return (SqliteConversationStore(conn), SqliteMemoryStore(conn),
            SqliteMetaStore(conn), SqliteThoughtStore(conn), conn, path)


@case
async def store_all_lists_active_and_retired():
    _, mem, _, _, conn, path = _stores()
    a = mem.add("fact A", "life", _emb(), None, core=True)
    b = mem.add("fact B", None, _emb(0.2), None)
    mem.deactivate(b, superseded_by=a)                 # retire B
    rows = mem.all()
    assert len(rows) == 2, rows
    by_id = {r["id"]: r for r in rows}
    assert by_id[a]["active"] and by_id[a]["core"]
    assert not by_id[b]["active"] and by_id[b]["superseded_by"] == a
    conn.close(); os.remove(path)


@case
async def store_delete_removes_one():
    _, mem, _, _, conn, path = _stores()
    a = mem.add("keep", None, _emb(), None)
    b = mem.add("drop", None, _emb(0.2), None)
    mem.delete(b)
    ids = [r["id"] for r in mem.all()]
    assert ids == [a], ids
    conn.close(); os.remove(path)


@case
async def store_clear_empties():
    _, mem, _, _, conn, path = _stores()
    mem.add("one", None, _emb(), None)
    mem.add("two", None, _emb(0.2), None)
    mem.clear()
    assert mem.all() == [] and mem.count() == 0
    conn.close(); os.remove(path)


@case
async def store_update_content_changes_text_and_embedding():
    _, mem, _, _, conn, path = _stores()
    mid = mem.add("old text", None, _emb(0.1), None)
    mem.update_content(mid, "new text", _emb(0.5))
    row = next(m for m in mem.active() if m["id"] == mid)
    assert row["content"] == "new text"
    assert np.frombuffer(row["embedding"], dtype=np.float32)[0] == np.float32(0.5)
    conn.close(); os.remove(path)


@case
async def manager_edit_reembeds():
    _, mem, _, _, conn, path = _stores()
    mid = mem.add("i like tea", None, _emb(0.1), None)
    mgr = MemoryManager(FakeEmbedder(), mem, None, "", 5, 0.5, 5, 0.6)
    await mgr.edit_memory(mid, "i like coffee")
    row = next(m for m in mem.active() if m["id"] == mid)
    assert row["content"] == "i like coffee"
    # embedding replaced with the fake embedder's document vector [0.9, ...]
    assert np.frombuffer(row["embedding"], dtype=np.float32)[0] == np.float32(0.9)
    conn.close(); os.remove(path)


def _companion(conv, mem, meta, thoughts):
    emotion = EmotionManager(FakeClassifier(), meta)
    drives = DriveManager(meta, away_after=90.0)
    memory = MemoryManager(FakeEmbedder(), mem, None, "", 5, 0.5, 5, 0.6)
    sid = conv.create_session("seed chat")
    return Companion(None, conv, memory, meta, sid, emotion=emotion,
                     thoughts=thoughts, drives=drives), emotion, drives, sid


@case
async def clear_memories_keeps_conversations():
    conv, mem, meta, thoughts, conn, path = _stores()
    comp, _, _, sid = _companion(conv, mem, meta, thoughts)
    mem.add("a fact", None, _emb(), None)
    mem.add("another", None, _emb(0.2), None)
    conv.add_message(sid, "user", "hello")
    n = comp.clear_memories()   # sync: it does no I/O worth awaiting
    assert n == 2, n
    assert mem.all() == [], "memories wiped"
    assert conv.message_count() == 1, "conversation kept"
    conn.close(); os.remove(path)


@case
async def factory_reset_wipes_everything():
    conv, mem, meta, thoughts, conn, path = _stores()
    comp, emotion, drives, sid = _companion(conv, mem, meta, thoughts)
    # seed every kind of state
    mem.add("a fact", None, _emb(), None)
    conv.add_message(sid, "user", "hello")
    thoughts.add("a private thought", "warmth")
    meta.set(PERSONA_SELF_KEY, "you have grown fond of them")
    meta.set_int("last_consolidated_msg_id", 5)
    emotion.state["warmth"] = 0.8
    drives.state["connection"] = 0.7
    comp.history = [{"role": "user", "content": "hello"}]
    comp._unconsolidated = [{"id": 1, "role": "user", "content": "hello"}]

    await comp.factory_reset()

    assert mem.all() == [], "memories wiped"
    assert conv.message_count() == 0, "messages wiped"
    assert thoughts.count() == 0, "thoughts wiped"
    assert meta.get(PERSONA_SELF_KEY) is None, "persona wiped"
    assert meta.get_int("last_consolidated_msg_id", 0) == 0, "watermark wiped"
    assert comp.history == [] and comp._unconsolidated == []
    for c in CHANNELS:
        assert abs(emotion.state[c] - BASELINE_STATE[c]) < 1e-9, (c, emotion.state[c])
    for d in DRIVE_BASELINE:
        assert abs(drives.state[d] - DRIVE_BASELINE[d]) < 1e-9, (d, drives.state[d])
    # landed in a fresh, empty conversation
    convs = conv.list_conversations()
    assert len(convs) == 1 and convs[0]["count"] == 0, convs
    assert comp.session_id == convs[0]["id"]
    conn.close(); os.remove(path)


if __name__ == "__main__":
    raise SystemExit(run())
