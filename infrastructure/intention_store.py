"""SQLite implementation of the IntentionStore: Mari's private forward agenda.

Intentions are short first-person notes of things she means to bring up or find out
next time — the "planning" pillar of the Generative-Agents loop. Minted during idle
reflection, drawn on by reach-out to make proactive messages purposeful, and marked
fulfilled once she acts on one. Kept separate from memories (facts about the user) and
thoughts (private reflections). Pure persistence; a lock serializes writes on the
shared connection.

Also holds A3-mined "pursuits" — self-directed things she may step away to do (§A4) —
in the same table, discriminated by `kind`: 'agenda' (the raise-with-him items above,
the default everywhere) vs 'pursuit'. This keeps one lifecycle (add/active/fulfill/
drop/expire) for both without a second table, while every read defaults to 'agenda' so
no existing caller changes behavior.
"""
import json

from infrastructure.db import SqliteStore, utcnow


class SqliteIntentionStore(SqliteStore):
    def add(self, content: str, kind: str = "agenda", citations: list[int] | None = None) -> int:
        with self._lock:
            cur = self.conn.execute(
                "INSERT INTO intentions (content, created_at, kind, citations) VALUES (?, ?, ?, ?)",
                (content, utcnow(), kind, json.dumps(citations) if citations else None),
            )
            self.conn.commit()
            return cur.lastrowid

    def active(self, kind: str = "agenda", limit: int | None = None) -> list[dict]:
        """Open items of one kind, oldest first (FIFO — act on the longest-waiting), as
        {id, content, created_at, citations}."""
        sql = ("SELECT id, content, created_at, citations FROM intentions "
               "WHERE active = 1 AND kind = ? ORDER BY id")
        params: tuple = (kind,)
        if limit is not None:
            sql += " LIMIT ?"          # bound, like every sibling store
            params = (kind, int(limit))
        rows = self.conn.execute(sql, params).fetchall()
        return [{"id": r["id"], "content": r["content"], "created_at": r["created_at"],
                 "citations": json.loads(r["citations"]) if r["citations"] else []}
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

    def drop_older_than(self, cutoff_iso: str, kind: str = "agenda") -> int:
        """Retire active items of one kind created before `cutoff_iso` (stale-agenda
        expiry). Returns how many were dropped."""
        with self._lock:
            cur = self.conn.execute(
                "UPDATE intentions SET active = 0 WHERE active = 1 AND kind = ? AND created_at < ?",
                (kind, cutoff_iso))
            self.conn.commit()
            return cur.rowcount

    def all(self, kind: str | None = None) -> list[dict]:
        """Every item (active + retired), newest first — for inspection. `kind=None`
        returns both kinds; otherwise filters to one."""
        sql = ("SELECT id, content, created_at, fulfilled_at, active, kind, citations "
               "FROM intentions")
        params: tuple = ()
        if kind is not None:
            sql += " WHERE kind = ?"
            params = (kind,)
        sql += " ORDER BY id DESC"
        rows = self.conn.execute(sql, params).fetchall()
        return [{"id": r["id"], "content": r["content"], "created_at": r["created_at"],
                 "fulfilled_at": r["fulfilled_at"], "active": bool(r["active"]),
                 "kind": r["kind"],
                 "citations": json.loads(r["citations"]) if r["citations"] else []}
                for r in rows]

    def count_active(self, kind: str = "agenda") -> int:
        return self.conn.execute(
            "SELECT COUNT(*) FROM intentions WHERE active = 1 AND kind = ?", (kind,)
        ).fetchone()[0]

    def clear(self) -> None:
        """Delete every intention AND pursuit (the full-reset admin op)."""
        with self._lock:
            self.conn.execute("DELETE FROM intentions")
            self.conn.commit()
