"""SQLite implementation of the MemoryStore contract (semantic tier).

Embeddings are stored as raw float32 bytes. Vector search itself lives in the
MemoryManager (brute-force numpy KNN, which the v1 retrospective confirmed is
plenty fast at personal scale). This store is pure persistence.
"""
import sqlite3
import threading
from datetime import datetime, timezone


def _utcnow() -> str:
    return datetime.now(timezone.utc).isoformat()


class SqliteMemoryStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._lock = threading.Lock()

    def add(self, content: str, category: str | None, embedding: bytes,
            source_session: int | None, core: bool = False) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO memories (content, category, embedding, created_at, source_session, core) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (content, category, embedding, _utcnow(), source_session, 1 if core else 0),
            )
            self.conn.commit()
            return cur.lastrowid

    def active(self) -> list[dict]:
        rows = self.conn.execute(
            "SELECT id, content, category, embedding, core FROM memories WHERE active = 1"
        ).fetchall()
        return [
            {"id": r["id"], "content": r["content"], "category": r["category"],
             "embedding": r["embedding"], "core": bool(r["core"])}
            for r in rows
        ]

    def core(self) -> list[dict]:
        """Active core memories (always injected into the prompt), oldest first."""
        rows = self.conn.execute(
            "SELECT id, content, category FROM memories WHERE active = 1 AND core = 1 ORDER BY id"
        ).fetchall()
        return [{"id": r["id"], "content": r["content"], "category": r["category"]} for r in rows]

    def set_core(self, memory_id: int, core: bool) -> None:
        """Promote a memory into (or demote it out of) the always-injected core set."""
        with self._lock:
            self.conn.execute(
                "UPDATE memories SET core = ? WHERE id = ?", (1 if core else 0, memory_id)
            )
            self.conn.commit()

    def deactivate(self, memory_id: int, superseded_by: int | None = None) -> None:
        """Soft-delete a memory (keeps history), optionally linking its replacement."""
        with self._lock:
            self.conn.execute(
                "UPDATE memories SET active = 0, superseded_by = ? WHERE id = ?",
                (superseded_by, memory_id),
            )
            self.conn.commit()

    def count(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE active = 1"
        ).fetchone()[0]

    def count_core(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM memories WHERE active = 1 AND core = 1"
        ).fetchone()[0]
