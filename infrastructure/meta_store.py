"""SQLite implementation of the MetaStore contract: durable scalar state.

A tiny key/value table for state that must survive restarts but doesn't warrant
its own schema — the consolidation watermark today, mood channels (pillar 2) and
pending-tick bookkeeping later. Pure persistence: string values in and out, with
typed helpers layered on top. A lock serializes writes since the connection is
shared across asyncio's threadpool.
"""
import json
import sqlite3
import threading


class SqliteMetaStore:
    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._lock = threading.Lock()

    def get(self, key: str) -> str | None:
        row = self.conn.execute(
            "SELECT value FROM meta WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row is not None else None

    def set(self, key: str, value: str) -> None:
        with self._lock:
            self.conn.execute(
                "INSERT INTO meta (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (key, value),
            )
            self.conn.commit()

    def get_int(self, key: str, default: int = 0) -> int:
        value = self.get(key)
        if value is None:
            return default
        try:
            return int(value)
        except ValueError:
            return default

    def set_int(self, key: str, value: int) -> None:
        self.set(key, str(value))

    def get_json(self, key: str, default=None):
        value = self.get(key)
        if value is None:
            return default
        try:
            return json.loads(value)
        except (json.JSONDecodeError, TypeError):
            return default

    def set_json(self, key: str, value) -> None:
        self.set(key, json.dumps(value))
