"""SQLite implementation of the ConversationStore contract.

Pure persistence: SQL in, plain dicts out. Timestamps are always UTC ISO-8601
(a v1 rule). A lock serializes writes since the connection is shared across
asyncio's threadpool.
"""
import re
import sqlite3

from infrastructure.db import SqliteStore, utcnow
from datetime import datetime, timezone


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

    def clear(self) -> None:
        """Delete every conversation and message (the full-reset admin op)."""
        with self._lock:
            self.conn.execute("DELETE FROM messages")
            self.conn.execute("DELETE FROM sessions")
            self.conn.commit()

    def add_message(self, session_id: int, role: str, content: str) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, role, content, utcnow()),
            )
            self.conn.commit()
            return cur.lastrowid

    def recent_messages(self, limit: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

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
