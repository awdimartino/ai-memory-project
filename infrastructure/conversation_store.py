from unittest import result
import uuid

import infrastructure.config as config
import json
from core.models import ConversationRecord, MessageRecord
from infrastructure.database import DatabaseConnection
from infrastructure.embedder import Embedder

from openai import OpenAI as oai

class ConversationStore:
    def __init__(self, database: DatabaseConnection):
        self.database = database

    def setup_conversations(self):
        """Set up the database schema for storing conversations."""

        sql = """
        CREATE TABLE IF NOT EXISTS conversations (
            id UUID PRIMARY KEY,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            user_last_active TIMESTAMPTZ DEFAULT NOW(),
            bot_last_active TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS conversations_user_last_active_idx
            ON conversations (user_last_active);

        CREATE INDEX IF NOT EXISTS conversations_bot_last_active_idx
            ON conversations (bot_last_active);
        """
        self.database.execute(sql)

    def setup_messages(self):
        """Set up the database schema for storing messages."""

        sql = """
        CREATE TABLE IF NOT EXISTS messages (
            id SERIAL PRIMARY KEY,
            conversation_id UUID NOT NULL,

            role TEXT NOT NULL,
            content TEXT NOT NULL,
            timestamp TIMESTAMPTZ DEFAULT NOW(),

            FOREIGN KEY (conversation_id)
                REFERENCES conversations(id)
                ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS messages_conversation_id_idx
            ON messages (conversation_id);
        """
        self.database.execute(sql)

    def new_conversation(self):
        """Create a new conversation record in the database and return it."""
        conversation_id = str(uuid.uuid4())

        self.database.execute(
            "INSERT INTO conversations (id) VALUES (%s)",
            (conversation_id,)
        )

        return self.get_conversation(conversation_id)

    def get_conversation(self, conversation_id):
        sql = """
        SELECT id, created_at, user_last_active, bot_last_active
        FROM conversations
        WHERE id = %s
        """

        result = self.database.fetch_all(sql, (conversation_id,))
        if not result:
            return None

        row = result[0]

        return ConversationRecord(
            id=row["id"],
            created_at=row["created_at"],
            user_last_active=row["user_last_active"],
            bot_last_active=row["bot_last_active"]
        )
    
    def get_messages(self, conversation_record):
        """Retrieve messages for a specific conversation from the database."""
        sql = """
        SELECT role, content, timestamp
        FROM messages
        WHERE conversation_id = %s
        ORDER BY timestamp ASC
        """
        results = self.database.fetch_all(sql, (conversation_record.id,))
        message_records = []
        for message_data in results:
            message_record = MessageRecord(
                role=message_data["role"],
                content=message_data["content"],
                conversation_id=conversation_record.id,
                timestamp=message_data["timestamp"]
            )
            message_records.append(message_record)

        return message_records

    def store_message(self, conversation, message):
        self.database.execute(
            """
            INSERT INTO messages (conversation_id, role, content)
            VALUES (%s, %s, %s)
            """,
            (conversation.id, message.role, message.content)
        )

        # update activity
        if message.role == "user":
            self.database.execute(
                """
                UPDATE conversations
                SET user_last_active = NOW()
                WHERE id = %s
                """,
                (conversation.id,)
            )

        else:
            self.database.execute(
                """
                UPDATE conversations
                SET bot_last_active = NOW()
                WHERE id = %s
                """,
                (conversation.id,)
            )

    def get_recent_conversations(self, hours=2, limit=10):
        sql = """
        SELECT 
            id,
            created_at,
            user_last_active,
            bot_last_active,
            GREATEST(user_last_active, bot_last_active) AS last_active
        FROM conversations
        WHERE GREATEST(user_last_active, bot_last_active)
              >= NOW() - (%s * INTERVAL '1 hour')
        ORDER BY last_active DESC
        LIMIT %s
        """

        rows = self.database.fetch_all(sql, (hours, limit))

        return [
            ConversationRecord(
                id=r["id"],
                created_at=r["created_at"],
                user_last_active=r["user_last_active"],
                bot_last_active=r["bot_last_active"]
            )
            for r in rows
        ]