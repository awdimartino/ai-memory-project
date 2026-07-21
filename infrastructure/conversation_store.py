"""SQLite implementation of the ConversationStore contract.

Pure persistence: SQL in, plain dicts out. Timestamps are always UTC ISO-8601
(a v1 rule). A lock serializes writes since the connection is shared across
asyncio's threadpool.
"""
import re

from infrastructure.db import SqliteStore, utcnow


class SqliteConversationStore(SqliteStore):
    def create_session(self, title: str | None = None) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO sessions (started_at, title) VALUES (?, ?)", (utcnow(), title)
            )
            self.conn.commit()
            return cur.lastrowid

    def session_title(self, session_id: int) -> str | None:
        row = self.conn.execute(
            "SELECT title FROM sessions WHERE id = ?", (session_id,)
        ).fetchone()
        return row["title"] if row else None

    def set_title(self, session_id: int, title: str) -> None:
        with self._lock:
            self.conn.execute(
                "UPDATE sessions SET title = ? WHERE id = ?", (title, session_id)
            )
            self.conn.commit()

    def latest_session(self) -> int | None:
        """The most recently active conversation (by last message, else start), to resume on boot."""
        row = self.conn.execute(
            "SELECT s.id FROM sessions s LEFT JOIN messages m ON m.session_id = s.id "
            "GROUP BY s.id ORDER BY COALESCE(MAX(m.created_at), s.started_at) DESC LIMIT 1"
        ).fetchone()
        return row["id"] if row else None

    def list_conversations(self) -> list[dict]:
        """All conversations, most-recently-active first: {id, title, count, last_at}."""
        rows = self.conn.execute(
            "SELECT s.id, s.title, COUNT(m.id) AS n, "
            "       COALESCE(MAX(m.created_at), s.started_at) AS last_at "
            "FROM sessions s LEFT JOIN messages m ON m.session_id = s.id "
            "GROUP BY s.id ORDER BY last_at DESC"
        ).fetchall()
        return [{"id": r["id"], "title": r["title"], "count": r["n"], "last_at": r["last_at"]}
                for r in rows]

    def session_messages(self, session_id: int, limit: int) -> list[dict]:
        """Up to `limit` most recent messages of one conversation (oldest first)."""
        rows = self.conn.execute(
            "SELECT role, content FROM messages WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (session_id, limit),
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def delete_session(self, session_id: int) -> None:
        """Delete a conversation and its messages (memories extracted from it are kept)."""
        with self._lock:
            self.conn.execute("DELETE FROM messages WHERE session_id = ?", (session_id,))
            self.conn.execute("DELETE FROM sessions WHERE id = ?", (session_id,))
            self.conn.commit()

    def log_self_notes(self, content: str) -> int:
        """Append one revision of the operating-notes slot (for RRR scoring)."""
        cur = self._write(
            "INSERT INTO self_notes_log (content, created_at) VALUES (?, ?)",
            (content, utcnow()))
        return cur.lastrowid

    def recent_self_notes(self, limit: int) -> list[str]:
        """Most recent operating-note revisions, newest first."""
        rows = self.conn.execute(
            "SELECT content FROM self_notes_log ORDER BY id DESC LIMIT ?", (limit,)).fetchall()
        return [r["content"] for r in rows]

    def clear(self) -> None:
        """Delete every conversation and message (full reset).

        Does NOT touch `message_archive`. That is the point of the archive: the
        working state can be wiped as often as testing needs without destroying the
        record of what was actually said.
        """
        with self._lock:
            self.conn.execute("DELETE FROM messages")
            self.conn.execute("DELETE FROM sessions")
            self.conn.commit()

    def add_message(self, session_id: int, role: str, content: str) -> int:
        """Append to the working log AND to the permanent archive, in one transaction.

        Both writes happen together so the archive can never miss a message that the
        working log has: they are the same commit, under the same lock.
        """
        now = utcnow()
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, role, content, now),
            )
            title = self.conn.execute(
                "SELECT title FROM sessions WHERE id = ?", (session_id,)).fetchone()
            self.conn.execute(
                "INSERT INTO message_archive "
                "(era, session_id, session_title, role, content, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (self._era_locked(), session_id, title["title"] if title else None,
                 role, content, now),
            )
            self.conn.commit()
            return cur.lastrowid

    # --- the permanent record -------------------------------------------------
    # `messages` is the working log and a factory reset wipes it. This never gets
    # wiped by any admin operation. Testing needs a clean slate often; the record of
    # what was actually said should outlive that.

    def _era_locked(self) -> int:
        """Current archive era. Caller must already hold the lock."""
        row = self.conn.execute(
            "SELECT COALESCE(MAX(era), 1) FROM message_archive").fetchone()
        return row[0]

    def current_era(self) -> int:
        """Which era new messages are being filed under."""
        with self._lock:
            return self._era_locked()

    def begin_new_era(self) -> int:
        """Mark a discontinuity (a factory reset). Returns the new era number.

        Stored in the archive itself rather than the meta table, because meta is one
        of the things a reset clears -- an era counter that resets with the thing it
        is counting would be useless.
        """
        with self._lock:
            era = self._era_locked() + 1
            # Nothing to write yet; the next archived message carries the new era.
            self.conn.execute(
                "INSERT INTO message_archive "
                "(era, session_id, session_title, role, content, created_at) "
                "VALUES (?, NULL, NULL, 'system', ?, ?)",
                (era, "-- memory cleared; everything below is a fresh start --", utcnow()),
            )
            self.conn.commit()
            return era

    def archive_count(self) -> int:
        """Total messages ever recorded, across every era."""
        return self.conn.execute("SELECT COUNT(*) FROM message_archive").fetchone()[0]

    def archive_eras(self) -> list[dict]:
        """One row per era: how much was said, and when it ran."""
        rows = self.conn.execute(
            "SELECT era, COUNT(*) AS n, MIN(created_at) AS started, MAX(created_at) AS ended "
            "FROM message_archive GROUP BY era ORDER BY era").fetchall()
        return [{"era": r["era"], "messages": r["n"],
                 "started": r["started"], "ended": r["ended"]} for r in rows]

    def archived_messages(self, limit: int, era: int | None = None) -> list[dict]:
        """Most recent archived messages, newest first; optionally one era only."""
        sql = ("SELECT era, session_id, session_title, role, content, created_at "
               "FROM message_archive")
        params: tuple = ()
        if era is not None:
            sql += " WHERE era = ?"
            params = (era,)
        sql += " ORDER BY id DESC LIMIT ?"
        rows = self.conn.execute(sql, (*params, limit)).fetchall()
        return [dict(r) for r in rows]

    def search_archive(self, query: str, limit: int) -> list[dict]:
        """Keyword search across EVERY era, including conversations since wiped.

        Same word-tokenized ranking as search_messages, but over the permanent
        record. Not wired into `reminisce` by default -- see REMINISCE_INCLUDE_ARCHIVE.
        """
        words = [w for w in re.findall(r"[\w']+", query.lower()) if len(w) > 2]
        if not words:
            return []
        rows = self.conn.execute(
            "SELECT era, session_title, role, content, created_at FROM message_archive "
            "WHERE role != 'system' ORDER BY id DESC LIMIT 4000").fetchall()
        scored = []
        for r in rows:
            low = r["content"].lower()
            hits = sum(1 for w in words if w in low)
            if hits:
                scored.append((hits, dict(r)))
        scored.sort(key=lambda t: -t[0])
        return [d for _, d in scored[:limit]]

    def recent_messages(self, limit: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def recent_messages_with_ids(self, limit: int) -> list[dict]:
        """Like `recent_messages`, but keeps the row id — the citation surface for
        A3's grounded open-question mining (§A4): a mined question must point back
        to a real message here, or it's rejected."""
        rows = self.conn.execute(
            "SELECT id, role, content FROM messages ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [{"id": r["id"], "role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def messages_after(self, msg_id: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, role, content FROM messages WHERE id > ? ORDER BY id", (msg_id,)
        ).fetchall()
        return [
            {"id": r["id"], "role": r["role"], "content": r["content"]} for r in rows
        ]

    def message_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]

    def search_messages(self, query: str, limit: int) -> list[dict]:
        """Messages matching `query` across all conversations, for the reminisce tool.

        Keyword search: split the topic into words (>2 chars), gather messages
        containing any of them (LIKE is case-insensitive for ASCII), then rank by
        how many distinct words hit. Returns up to `limit`, oldest-first so the
        digest reads chronologically. A pragmatic v1 — semantic search over the log
        is a later upgrade; underscore-free word tokens keep LIKE wildcard-safe.
        """
        words = [w for w in re.findall(r"[^\W_]+", (query or "").lower()) if len(w) > 2]
        if not words:
            return []
        clauses = " OR ".join(["content LIKE ?"] * len(words))
        params = [f"%{w}%" for w in words]
        rows = self.conn.execute(
            f"SELECT id, role, content FROM messages WHERE {clauses} ORDER BY id DESC LIMIT 200",
            params,
        ).fetchall()
        scored = [(sum(w in r["content"].lower() for w in words), r["id"], r) for r in rows]
        scored.sort(key=lambda t: (-t[0], -t[1]))            # best match, then most recent
        top = sorted(scored[:limit], key=lambda t: t[1])     # oldest-first for reading
        return [{"id": t[2]["id"], "role": t[2]["role"], "content": t[2]["content"]} for t in top]
