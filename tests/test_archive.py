"""Offline coverage for the permanent message archive.

`messages` is the WORKING log — what she operates on, and what a factory reset
wipes. `message_archive` is the record of what was actually said, and **no admin
operation clears it**. The whole point is that testing can wipe her working state as
often as it needs without destroying the conversation history.

The load-bearing case here is `factory_reset_spares_the_archive`. If that ever
fails, real conversations are being lost.

Run:  python tests/test_archive.py
"""
from _harness import case, run, temp_db  # also puts the repo root on sys.path
from helpers import FakeMemory, InMemoryMeta

from core.companion import Companion
from infrastructure.conversation_store import SqliteConversationStore


def _store(conn):
    return SqliteConversationStore(conn)


@case
async def every_message_lands_in_both_logs():
    with temp_db("arch.db") as (conn, _p):
        st = _store(conn)
        sid = st.create_session("A chat")
        st.add_message(sid, "user", "hello there")
        st.add_message(sid, "assistant", "hey")
        assert st.message_count() == 2
        assert st.archive_count() == 2, "the archive must mirror every write"


@case
async def the_archive_records_which_conversation_it_came_from():
    with temp_db("arch.db") as (conn, _p):
        st = _store(conn)
        sid = st.create_session("Jacket talk")
        st.add_message(sid, "user", "the dye came out patchy")
        row = st.archived_messages(1)[0]
        assert row["session_id"] == sid
        assert row["session_title"] == "Jacket talk", row
        assert row["role"] == "user"


@case
async def clearing_conversations_leaves_the_archive_intact():
    with temp_db("arch.db") as (conn, _p):
        st = _store(conn)
        sid = st.create_session("A chat")
        st.add_message(sid, "user", "something worth keeping")
        st.clear()
        assert st.message_count() == 0, "working log should be empty"
        assert st.archive_count() == 1, "the archive must survive clear()"
        assert "worth keeping" in st.archived_messages(1)[0]["content"]


@case
async def factory_reset_spares_the_archive():
    """The case that matters: a full reset must not destroy real conversations."""
    with temp_db("arch.db") as (conn, _p):
        st = _store(conn)
        comp = Companion(llm=None, store=st, memory=FakeMemory(), meta=InMemoryMeta(),
                         session_id=st.create_session("Before"))
        st.add_message(comp.session_id, "user", "my dog is called Pip")
        st.add_message(comp.session_id, "assistant", "good name")

        await comp.factory_reset()

        assert st.message_count() == 0, "working log wiped, as intended"
        contents = " ".join(m["content"] for m in st.archived_messages(20))
        assert "Pip" in contents, "THE conversation record was destroyed by a reset"


@case
async def a_reset_starts_a_new_era():
    # So a future look-back can tell a discontinuity happened, rather than reading
    # across it as one continuous relationship.
    with temp_db("arch.db") as (conn, _p):
        st = _store(conn)
        comp = Companion(llm=None, store=st, memory=FakeMemory(), meta=InMemoryMeta(),
                         session_id=st.create_session("Before"))
        st.add_message(comp.session_id, "user", "era one message")
        assert st.current_era() == 1

        await comp.factory_reset()
        assert st.current_era() == 2, "a reset should open a new era"

        st.add_message(comp.session_id, "user", "era two message")
        eras = {e["era"]: e for e in st.archive_eras()}
        assert 1 in eras and 2 in eras, eras
        assert any("era one" in m["content"] for m in st.archived_messages(50, era=1))
        assert any("era two" in m["content"] for m in st.archived_messages(50, era=2))
        assert not any("era one" in m["content"] for m in st.archived_messages(50, era=2))


@case
async def the_era_counter_survives_meta_being_cleared():
    # It lives in the archive, not the meta table — a counter that reset alongside
    # the thing it counts would be useless.
    with temp_db("arch.db") as (conn, _p):
        st = _store(conn)
        meta = InMemoryMeta()
        comp = Companion(llm=None, store=st, memory=FakeMemory(), meta=meta,
                         session_id=st.create_session("x"))
        st.add_message(comp.session_id, "user", "one")
        await comp.factory_reset()
        await comp.factory_reset()
        await comp.factory_reset()
        assert st.current_era() == 4, f"eras should keep climbing, got {st.current_era()}"
        assert meta.data.get("archive_era") is None, "must not depend on meta"


@case
async def archive_search_finds_wiped_conversations():
    with temp_db("arch.db") as (conn, _p):
        st = _store(conn)
        comp = Companion(llm=None, store=st, memory=FakeMemory(), meta=InMemoryMeta(),
                         session_id=st.create_session("Old"))
        st.add_message(comp.session_id, "user", "I went to Japan last spring")
        await comp.factory_reset()

        assert st.search_messages("Japan", 5) == [], "gone from the working log"
        hits = st.search_archive("Japan", 5)
        assert hits and "Japan" in hits[0]["content"], "should still be findable in the archive"


@case
async def archive_search_ignores_the_era_markers():
    with temp_db("arch.db") as (conn, _p):
        st = _store(conn)
        comp = Companion(llm=None, store=st, memory=FakeMemory(), meta=InMemoryMeta(),
                         session_id=st.create_session("x"))
        st.add_message(comp.session_id, "user", "a real message about memory")
        await comp.factory_reset()
        hits = st.search_archive("memory cleared fresh start", 5)
        assert all(h["role"] != "system" for h in hits), "markers are not conversation"


@case
async def companion_exposes_the_archive_through_the_facade():
    # Callers shouldn't reach past Companion into the store (V2_PLAN §1).
    with temp_db("arch.db") as (conn, _p):
        st = _store(conn)
        comp = Companion(llm=None, store=st, memory=FakeMemory(), meta=InMemoryMeta(),
                         session_id=st.create_session("x"))
        st.add_message(comp.session_id, "user", "hello archive")
        stats = comp.archive_stats()
        assert stats["total"] == 1 and stats["current_era"] == 1, stats
        assert comp.search_archive("archive")[0]["content"] == "hello archive"


@case
async def existing_history_is_backfilled_by_the_migration():
    # v10 seeds the archive from whatever messages were already on disk, so turning
    # this on doesn't start the record from today.
    with temp_db("arch.db") as (conn, _p):
        st = _store(conn)
        sid = st.create_session("Seeded")
        st.add_message(sid, "user", "already here")
        # Simulate a pre-v10 DB: the row exists in messages but not the archive.
        conn.execute("DELETE FROM message_archive")
        conn.commit()
        conn.executescript(
            "INSERT INTO message_archive (era, session_id, session_title, role, content, created_at)"
            " SELECT 1, m.session_id, s.title, m.role, m.content, m.created_at"
            " FROM messages m LEFT JOIN sessions s ON s.id = m.session_id ORDER BY m.id;")
        assert st.archive_count() == 1
        assert st.archived_messages(1)[0]["session_title"] == "Seeded"


if __name__ == "__main__":
    raise SystemExit(run())
