import json
from datetime import datetime

import sqlite_vec

import infrastructure.config as config
from core.models import MemoryRecord, utc_now
from infrastructure.database import DatabaseConnection


class MemoryStore:
    """Stores and retrieves memories in SQLite + sqlite-vec.

    Metadata lives in a normal `memories` table; the embedding lives in a
    `vec_memories` vec0 virtual table keyed by the same rowid, so vector KNN
    and relational metadata join on `memories.id == vec_memories.rowid`.

    Similarity uses cosine distance: cosine_similarity = 1 - distance.
    """

    def __init__(self, database: DatabaseConnection):
        self.database = database

    def setup(self):
        """Create the memory tables if they don't exist.

        Columns:
            id               - unique memory id (INTEGER PRIMARY KEY)
            content          - the textual memory
            category         - fact | preference | goal | relation | event | belief
            origin_type      - message | tick_short | tick_long | reflection | tool
            origin_id        - optional id of the origin (e.g. a message id)
            conversation_id  - optional owning conversation
            emotion_snapshot - JSON of the bot's mood when the memory formed
            importance       - retrieval/consolidation weight
            timestamp        - ISO-8601 UTC creation time
        """
        self.database.executescript(f"""
        CREATE TABLE IF NOT EXISTS memories (
            id                INTEGER PRIMARY KEY AUTOINCREMENT,
            content           TEXT NOT NULL,
            category          TEXT NOT NULL,
            origin_type       TEXT NOT NULL,
            origin_id         TEXT,
            conversation_id   TEXT,
            emotion_snapshot  TEXT,
            importance        REAL DEFAULT 0.5,
            timestamp         TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS memories_origin_idx
            ON memories (origin_type, origin_id);
        CREATE INDEX IF NOT EXISTS memories_category_idx
            ON memories (category);

        CREATE VIRTUAL TABLE IF NOT EXISTS vec_memories USING vec0(
            embedding float[{config.EMBED_DIM}] distance_metric=cosine
        );
        """)

    def store_memory(self, record: MemoryRecord) -> bool:
        """Persist a memory and its embedding. Returns True on success."""
        timestamp = (record.timestamp or utc_now()).isoformat()
        memory_id = self.database.execute_returning_id(
            """
            INSERT INTO memories
                (content, category, origin_type, origin_id,
                 conversation_id, emotion_snapshot, importance, timestamp)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.content,
                record.category,
                record.origin_type,
                record.origin_id,
                record.conversation_id,
                json.dumps(record.emotion_snapshot or {}),
                record.importance,
                timestamp,
            ),
        )
        if memory_id is None:
            return False

        return self.database.execute(
            "INSERT INTO vec_memories (rowid, embedding) VALUES (?, ?)",
            (memory_id, sqlite_vec.serialize_float32(record.embedding)),
        )

    def memory_exists(self, query_embedding, threshold=config.MEMORY_DEDUP_THRESHOLD) -> bool:
        """True if a stored memory is at least `threshold` cosine-similar."""
        row = self.database.fetch_one(
            """
            SELECT distance
            FROM vec_memories
            WHERE embedding MATCH ? AND k = 1
            ORDER BY distance
            """,
            (sqlite_vec.serialize_float32(query_embedding),),
        )
        if row is None:
            return False
        similarity = 1 - row["distance"]
        return similarity >= threshold

    def fetch_memories(self, query_embedding, threshold=config.MEMORY_RECALL_THRESHOLD, limit=5):
        """Return the most similar memories above the similarity threshold."""
        max_distance = 1 - threshold
        # Isolate the vec0 KNN scan in a CTE, then join metadata and filter by
        # distance outside it (vec0 MATCH queries don't accept extra WHERE terms).
        rows = self.database.fetch_all(
            """
            WITH matches AS (
                SELECT rowid AS mid, distance
                FROM vec_memories
                WHERE embedding MATCH ? AND k = ?
            )
            SELECT
                m.id, m.content, m.category, m.origin_type, m.origin_id,
                m.conversation_id, m.emotion_snapshot, m.importance, m.timestamp,
                matches.distance
            FROM matches
            JOIN memories AS m ON m.id = matches.mid
            WHERE matches.distance <= ?
            ORDER BY matches.distance
            """,
            (sqlite_vec.serialize_float32(query_embedding), limit, max_distance),
        )
        return [
            MemoryRecord(
                id=row["id"],
                content=row["content"],
                # The stored vector isn't needed on the read path; skip rehydrating it.
                embedding=[],
                category=row["category"],
                origin_type=row["origin_type"],
                origin_id=row["origin_id"],
                conversation_id=row["conversation_id"],
                emotion_snapshot=json.loads(row["emotion_snapshot"] or "{}"),
                importance=row["importance"],
                timestamp=_parse_ts(row["timestamp"]),
            )
            for row in rows
        ]


def _parse_ts(value):
    """Parse an ISO-8601 timestamp string back into a datetime (best effort)."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return value
