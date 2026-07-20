"""Offline coverage for intentions (the planning pillar): the store, form_intentions
(add / dedupe / cap), and reach-out consumption (pick oldest, fulfill on a real reply,
not on PASS). Uses a real SQLite IntentionStore + a real Companion wired to tiny fakes.

Run:  python tests/test_intentions.py
"""
import asyncio
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config  # noqa: E402
from core.companion import Companion  # noqa: E402
from infrastructure.db import connect  # noqa: E402
from infrastructure.intention_store import SqliteIntentionStore  # noqa: E402

_checks = 0


def check(cond, msg):
    global _checks
    assert cond, msg
    _checks += 1


def _store():
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    return SqliteIntentionStore(connect(path))  # connect() runs the migrations (creates v8 table)


class FakeLLM:
    def __init__(self, intentions=None, reply="sure thing"):
        self.model = "fake"
        self._intentions = intentions or []
        self._reply = reply

    async def structured_json(self, messages, schema, model=None):
        return {"intentions": self._intentions}

    async def stream(self, messages, on_token):
        await on_token(self._reply)
        return self._reply, {"ttft": 0, "tok_per_s": 0, "tokens": 1, "estimated": True}


class FakeMemory:
    async def recall(self, text):
        return []

    def core_memories(self):
        return []


class FakeConv:
    def __init__(self):
        self._id = 0

    def add_message(self, sid, role, content):
        self._id += 1
        return self._id


class FakeMeta:
    def get(self, k):
        return None


def _companion(store, llm, history):
    return Companion(llm, FakeConv(), FakeMemory(), FakeMeta(), 1, history=history, intentions=store)


async def main():
    # --- 1) store CRUD ---
    st = _store()
    a = st.add("ask how his jacket dye turned out")
    b = st.add("find out what he does for work")
    check(len(st.active()) == 2, "two active")
    check(st.active()[0]["id"] == a, "active is oldest-first (FIFO)")
    check(st.active(limit=1)[0]["id"] == a, "limit respected")
    st.fulfill(a)
    check(len(st.active()) == 1 and st.active()[0]["id"] == b, "fulfill retires the acted-on one")
    check(any(x["id"] == a and not x["active"] and x["fulfilled_at"] for x in st.all()),
          "fulfilled kept with a timestamp")
    st.drop(b)
    check(st.count_active() == 0, "drop retires it")
    check(all(x["fulfilled_at"] is None for x in st.all() if x["id"] == b), "dropped has no fulfilled_at")
    st.clear()
    check(len(st.all()) == 0, "clear wipes")

    # --- 2) form_intentions: add new, dedupe (case-insensitive), cap ---
    st = _store()
    st.add("ask how his jacket dye turned out")  # existing
    llm = FakeLLM(intentions=[
        "ask how his jacket dye turned out",   # dup of existing
        "ASK how his jacket dye turned out",   # case-dup
        "find out what he does for work",      # new
        "check in about the deadlock game",    # new
    ])
    added = await _companion(st, llm, [{"role": "user", "content": "hi"}]).form_intentions()
    check(set(added) == {"find out what he does for work", "check in about the deadlock game"},
          f"only genuinely-new added: {added}")
    check(st.count_active() == 3, "existing + 2 new; dups skipped")

    old_max = config.INTENTION_MAX_ACTIVE
    config.INTENTION_MAX_ACTIVE = 2
    await _companion(st, FakeLLM(["one brand new thing"]), [{"role": "user", "content": "hi"}]).form_intentions()
    check(st.count_active() == 2, f"capped to 2 (oldest dropped), got {st.count_active()}")
    config.INTENTION_MAX_ACTIVE = old_max

    check(await _companion(st, FakeLLM([]), history=[]).form_intentions() == [], "empty history is a no-op")

    # --- 3) reach_out consumes the oldest intention, fulfills on a real reply ---
    st = _store()
    i1 = st.add("ask how his jacket dye turned out")
    st.add("find out what he does for work")
    msg = await _companion(st, FakeLLM(reply="been wondering how that jacket dye came out"),
                           [{"role": "user", "content": "later"}]).reach_out()
    check(msg, "reach_out produced a message")
    check(st.count_active() == 1 and all(x["id"] != i1 for x in st.active()),
          "the oldest intention was the one fulfilled")

    # PASS => no fulfillment
    st = _store()
    st.add("ask about his weekend")
    check(await _companion(st, FakeLLM(reply="PASS"), [{"role": "user", "content": "later"}]).reach_out() is None,
          "PASS reach_out returns None")
    check(st.count_active() == 1, "PASS does not fulfill the intention")

    print(f"OK - {_checks} checks passed")


asyncio.run(main())
