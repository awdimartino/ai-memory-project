"""SQLite implementation of the IntentionStore: Mari's private forward agenda.

Intentions are short first-person notes of things she means to bring up or find out
next time — the "planning" pillar of the Generative-Agents loop. Minted during idle
reflection, drawn on by reach-out to make proactive messages purposeful, and marked
fulfilled once she acts on one. Kept separate from memories (facts about the user) and
thoughts (private reflections). Pure persistence; a lock serializes writes on the
shared connection.
"""
import sqlite3

from infrastructure.db import SqliteStore, utcnow
from datetime import datetime, timezone


class SqliteIntentionStore(SqliteStore):
    def add(self, content: str) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO intentions (content, created_at) VALUES (?, ?)",
                (content, utcnow()),
            )
            self.conn.commit()
            return cur.lastrowid

    def active(self, limit: int | None = None) -> list[dict]:
        """Open intentions, oldest first (FIFO — act on the longest-waiting), as
        {id, content, created_at}."""
        sql = "SELECT id, content, created_at FROM intentions WHERE active = 1 ORDER BY id"
        if limit is not None:
            sql += f" LIMIT {int(limit)}"
        rows = self.conn.execute(sql).fetchall()
        return [{"id": r["id"], "content": r["content"], "created_at": r["created_at"]}
                for r in rows]

    def fulfill(self, intention_id: int) -> None:
        """Mark an intention acted-on: retired, timestamped, kept for history."""
        with self._lock:
            self.conn.execute(
                "UPDATE intentions SET active = 0, fulfilled_at = ? WHERE id = ?",
                (utcnow(), intention_id),
            )
            self.conn.commit()

    def drop(self, intention_id: int) -> None:
        """Retire an intention without acting on it (expiry / over-cap pruning)."""
        with self._lock:
            self.conn.execute(
                "UPDATE intentions SET active = 0 WHERE id = ?", (intention_id,))
            self.conn.commit()

    def drop_older_than(self, cutoff_iso: str) -> int:
        """Retire active intentions created before `cutoff_iso` (stale-agenda expiry).
        Returns how many were dropped."""
        with self._lock:
            cur = self.conn.execute(
                "UPDATE intentions SET active = 0 WHERE active = 1 AND created_at < ?", (cutoff_iso,))
            self.conn.commit()
            return cur.rowcount

    def all(self) -> list[dict]:
        """Every intention (active + retired), newest first — for inspection."""
        rows = self.conn.execute(
            "SELECT id, content, created_at, fulfilled_at, active FROM intentions ORDER BY id DESC"
        ).fetchall()
        return [{"id": r["id"], "content": r["content"], "created_at": r["created_at"],
                 "fulfilled_at": r["fulfilled_at"], "active": bool(r["active"])} for r in rows]

    def count_active(self) -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM intentions WHERE active = 1").fetchone()[0]

    def clear(self) -> None:
        """Delete every intention (the full-reset admin op)."""
        with self._lock:
            self.conn.execute("DELETE FROM intentions")
            self.conn.commit()
