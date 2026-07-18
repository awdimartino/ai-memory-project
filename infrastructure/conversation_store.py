"""SQLite implementation of the ConversationStore contract.

Pure persistence: SQL in, plain dicts out. Timestamps are always UTC ISO-8601
(a v1 rule). A lock serializes writes since the connection is shared across
asyncio's threadpool.
"""
import sqlite3
import threading
from datetime import datetime, timezone


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class SqliteConversationStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._lock = threading.Lock()

    def create_session(self) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO sessions (started_at) VALUES (?)", (_utcnow(),)
            )
            self.conn.commit()
            return cur.lastrowid

    def add_message(self, session_id: int, role: str, content: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO messages (session_id, role, content, created_at) "
                "VALUES (?, ?, ?, ?)",
                (session_id, role, content, _utcnow()),
            )
            self.conn.commit()

    def recent_messages(self, limit: int) -> list[dict]:
        rows = self.conn.execute(
            "SELECT role, content FROM messages ORDER BY id DESC LIMIT ?", (limit,)
        ).fetchall()
        return [{"role": r["role"], "content": r["content"]} for r in reversed(rows)]

    def message_count(self) -> int:
        return self.conn.execute("SELECT COUNT(*) FROM messages").fetchone()[0]
