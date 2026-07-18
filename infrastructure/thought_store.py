"""SQLite implementation of the ThoughtStore contract: Mari's private journal.

Self-reflections written during idle ticks — the bot's own thoughts, kept separate
from the conversation log (the `messages` table) so they never leak into chat or
recall. Pure persistence. A lock serializes writes since the connection is shared
across asyncio's threadpool.
"""
import sqlite3
import threading
from datetime import datetime, timezone


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class SqliteThoughtStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._lock = threading.Lock()

    def add(self, content: str, mood: str | None) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO thoughts (content, mood, created_at) VALUES (?, ?, ?)",
                (content, mood, _utcnow()),
            )
            self.conn.commit()
            return cur.lastrowid

    def recent(self, limit: int) -> list[dict]:
        """Most recent thoughts, newest first, as {content, mood, created_at}."""
        rows = self.conn.execute(
            "SELECT content, mood, created_at FROM thoughts ORDER BY id DESC LIMIT ?",
            (limit,),
        ).fetchall()
        return [{"content": r["content"], "mood": r["mood"], "created_at": r["created_at"]}
                for r in rows]

    def count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM thoughts").fetchone()[0]
