"""Offline coverage for intentions (the planning pillar): the store, form_intentions
(expire / resolve / add / dedupe / cap), prompt injection (chat + follow-up), and reach-out
consumption (pick oldest, fulfill on a real reply, not on PASS).

Uses a real SQLite IntentionStore + a real Companion wired to tiny fakes.
Run:  python tests/test_intentions.py
"""
import asyncio
import os
from datetime import datetime, timedelta, timezone

from _harness import config_override, temp_dir, track_conn  # also puts the repo root on sys.path
from helpers import FakeConv, FakeMemory, InMemoryMeta

from core.companion import Companion  # noqa: E402
from core.prompts import build_followup_system, build_system  # noqa: E402
from infrastructure.db import connect  # noqa: E402
from infrastructure.intention_store import SqliteIntentionStore  # noqa: E402

_checks = 0


def check(cond, msg):
    global _checks
    assert cond, msg
    _checks += 1


def _store():
    # track_conn: this file never closes its stores, so let the harness do it at exit
    path = os.path.join(temp_dir(), "int.db")
    return SqliteIntentionStore(track_conn(connect(path)))


def _backdate(store, intention_id, days):
    store.conn.execute(
        "UPDATE intentions SET created_at = ? WHERE id = ?",
        ((datetime.now(timezone.utc) - timedelta(days=days)).isoformat(), intention_id))
    store.conn.commit()


class FakeLLM:
    def __init__(self, intentions=None, resolved=None, reply="sure thing"):
        self.model = "fake"
        self._intentions = intentions or []
        self._resolved = resolved or []
        self._reply = reply

    async def structured_json(self, messages, schema, model=None):
        return {"intentions": self._intentions, "resolved": self._resolved}

    async def stream(self, messages, on_token):
        await on_token(self._reply)
        return self._reply, {"ttft": 0, "tok_per_s": 0, "tokens": 1, "estimated": True}


def _companion(store, llm, history):
    return Companion(llm, FakeConv(), FakeMemory(), InMemoryMeta(), 1,
                     history=history, intentions=store)


HIST = [{"role": "user", "content": "hi"}]


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

    # --- 2) expiry ---
    st = _store()
    stale = st.add("stale thing")
    _backdate(st, stale, 30)
    fresh = st.add("fresh thing")
    n = st.drop_older_than((datetime.now(timezone.utc) - timedelta(days=7)).isoformat())
    check(n == 1, f"one stale dropped, got {n}")
    check([i["id"] for i in st.active()] == [fresh], "only the fresh one remains")
    check(all(x["fulfilled_at"] is None for x in st.all() if x["id"] == stale),
          "an expired intention is not recorded as 'fulfilled'")

    st = _store()
    ancient = st.add("ancient")
    _backdate(st, ancient, 30)
    await _companion(st, FakeLLM(), HIST).form_intentions()
    check(st.count_active() == 0, "form_intentions expires stale items")

    # --- 3) form_intentions: add new, dedupe (case-insensitive), cap ---
    st = _store()
    st.add("ask how his jacket dye turned out")
    llm = FakeLLM(intentions=[
        "ask how his jacket dye turned out",   # dup of existing
        "ASK how his jacket dye turned out",   # case-dup
        "find out what he does for work",      # new
        "check in about the deadlock game",    # new
    ])
    added = await _companion(st, llm, HIST).form_intentions()
    check(set(added) == {"find out what he does for work", "check in about the deadlock game"},
          f"only genuinely-new added: {added}")
    check(st.count_active() == 3, "existing + 2 new; dups skipped")

    with config_override(INTENTION_MAX_ACTIVE=2):
        await _companion(st, FakeLLM(["one brand new thing"]), HIST).form_intentions()
        check(st.count_active() == 2, f"capped to 2 (oldest dropped), got {st.count_active()}")

    check(await _companion(st, FakeLLM(), history=[]).form_intentions() == [], "empty history is a no-op")

    # --- 4) resolution: conversation-covered intentions get cleared ---
    st = _store()
    i1 = st.add("ask how the jacket turned out")
    i2 = st.add("find out what he does for work")
    await _companion(st, FakeLLM(resolved=[1]), HIST).form_intentions()
    check([i["id"] for i in st.active()] == [i2], "resolved #1 cleared, #2 untouched")
    check(any(x["id"] == i1 and x["fulfilled_at"] for x in st.all()), "resolved is marked fulfilled")

    st = _store()
    keep = st.add("only one")
    await _companion(st, FakeLLM(resolved=[0, 5, -1, "x"]), HIST).form_intentions()
    check([i["id"] for i in st.active()] == [keep], "out-of-range/garbage resolved indices ignored")

    # resolve + add in the same pass
    st = _store()
    r1 = st.add("ask how the jacket turned out")
    await _companion(st, FakeLLM(intentions=["ask about the new game"], resolved=[1]), HIST).form_intentions()
    names = [i["content"] for i in st.active()]
    check(names == ["ask about the new game"], f"resolved cleared and new added in one pass: {names}")
    check(any(x["id"] == r1 and x["fulfilled_at"] for x in st.all()), "the resolved one is retired")

    # --- 5) prompt injection ---
    s = build_system([], None, intentions=["ask how the jacket turned out"])
    check("ask how the jacket turned out" in s, "chat prompt carries the open agenda")
    check("never force one in" in s.lower(), "chat injection is framed as soft / non-pushy")
    check("ask how the jacket" not in build_system([], None, intentions=[]), "empty agenda injects nothing")
    check("ask how the jacket" not in build_system([], None), "no agenda arg injects nothing")

    fu = build_followup_system([], None, intention="ask how the jacket turned out")
    check("ask how the jacket turned out" in fu, "follow-up prompt can carry an intention")
    check("only fold it in" in fu.lower(), "follow-up intention is optional, not forced")
    check("been meaning to" not in build_followup_system([], None), "no intention -> no anchor block")

    # --- 6) reach_out consumes the oldest intention ---
    st = _store()
    i1 = st.add("ask how his jacket dye turned out")
    st.add("find out what he does for work")
    msg = await _companion(st, FakeLLM(reply="been wondering how that jacket dye came out"),
                           [{"role": "user", "content": "later"}]).reach_out()
    check(msg, "reach_out produced a message")
    check(st.count_active() == 1 and all(x["id"] != i1 for x in st.active()),
          "the oldest intention was the one fulfilled")

    st = _store()
    st.add("ask about his weekend")
    check(await _companion(st, FakeLLM(reply="PASS"), [{"role": "user", "content": "later"}]).reach_out() is None,
          "PASS reach_out returns None")
    check(st.count_active() == 1, "PASS does not fulfill the intention")

    # --- 7) degrades gracefully with no store ---
    c = Companion(FakeLLM(), FakeConv(), FakeMemory(), InMemoryMeta(), 1, history=HIST)
    check(await c.form_intentions() == [], "no IntentionStore -> form_intentions is a no-op")

    print(f"OK - {_checks} checks passed")


asyncio.run(main())
