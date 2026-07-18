import uuid
from datetime import datetime, timedelta, timezone

from core.models import ConversationRecord, MessageRecord, utc_now
from infrastructure.database import DatabaseConnection


class ConversationStore:
    """Stores conversations and messages in SQLite.

    Timestamps are persisted as ISO-8601 UTC strings (lexicographically
    sortable/comparable) and rehydrated into aware datetimes on read.
    """

    def __init__(self, database: DatabaseConnection):
        self.database = database

    def setup_conversations(self):
        self.database.executescript("""
        CREATE TABLE IF NOT EXISTS conversations (
            id                TEXT PRIMARY KEY,
            created_at        TEXT NOT NULL,
            user_last_active  TEXT NOT NULL,
            bot_last_active   TEXT NOT NULL
        );

        CREATE INDEX IF NOT EXISTS conversations_user_last_active_idx
            ON conversations (user_last_active);
        CREATE INDEX IF NOT EXISTS conversations_bot_last_active_idx
            ON conversations (bot_last_active);
        """)

    def setup_messages(self):
        self.database.executescript("""
        CREATE TABLE IF NOT EXISTS messages (
            id               INTEGER PRIMARY KEY AUTOINCREMENT,
            conversation_id  TEXT NOT NULL,
            role             TEXT NOT NULL,
            content          TEXT NOT NULL,
            timestamp        TEXT NOT NULL,
            FOREIGN KEY (conversation_id)
                REFERENCES conversations(id)
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS messages_conversation_id_idx
            ON messages (conversation_id);
        """)

    def new_conversation(self):
        """Create and persist a new conversation, returning the record."""
        record = ConversationRecord(id=str(uuid.uuid4()))
        self.database.execute(
            """
            INSERT INTO conversations
                (id, created_at, user_last_active, bot_last_active)
            VALUES (?, ?, ?, ?)
            """,
            (
                record.id,
                record.created_at.isoformat(),
                record.user_last_active.isoformat(),
                record.bot_last_active.isoformat(),
            ),
        )
        return record

    def get_conversation(self, conversation_id):
        row = self.database.fetch_one(
            """
            SELECT id, created_at, user_last_active, bot_last_active
            FROM conversations
            WHERE id = ?
            """,
            (conversation_id,),
        )
        return _row_to_conversation(row) if row else None

    def get_messages(self, conversation_record):
        """Return a conversation's messages in chronological order."""
        rows = self.database.fetch_all(
            """
            SELECT id, role, content, timestamp
            FROM messages
            WHERE conversation_id = ?
            ORDER BY timestamp ASC, id ASC
            """,
            (conversation_record.id,),
        )
        return [
            MessageRecord(
                id=row["id"],
                role=row["role"],
                content=row["content"],
                conversation_id=conversation_record.id,
                timestamp=_parse_ts(row["timestamp"]),
            )
            for row in rows
        ]

    def store_message(self, conversation, message):
        """Persist a message and return its new row id."""
        timestamp = (message.timestamp or utc_now()).isoformat()
        message_id = self.database.execute_returning_id(
            """
            INSERT INTO messages (conversation_id, role, content, timestamp)
            VALUES (?, ?, ?, ?)
            """,
            (conversation.id, message.role, message.content, timestamp),
        )

        # Update the relevant activity column.
        column = "user_last_active" if message.role == "user" else "bot_last_active"
        self.database.execute(
            f"UPDATE conversations SET {column} = ? WHERE id = ?",
            (timestamp, conversation.id),
        )
        return message_id

    def get_recent_conversations(self, hours=2, limit=10):
        """Return conversations active within the last `hours`, newest first."""
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=hours)).isoformat()
        rows = self.database.fetch_all(
            """
            SELECT id, created_at, user_last_active, bot_last_active,
                   MAX(user_last_active, bot_last_active) AS last_active
            FROM conversations
            WHERE MAX(user_last_active, bot_last_active) >= ?
            ORDER BY last_active DESC
            LIMIT ?
            """,
            (cutoff, limit),
        )
        return [_row_to_conversation(row) for row in rows]


def _row_to_conversation(row):
    return ConversationRecord(
        id=row["id"],
        created_at=_parse_ts(row["created_at"]),
        user_last_active=_parse_ts(row["user_last_active"]),
        bot_last_active=_parse_ts(row["bot_last_active"]),
    )


def _parse_ts(value):
    if not value:
        return None
    try:
        return datetime.fromisoformat(value)
    except (ValueError, TypeError):
        return value
