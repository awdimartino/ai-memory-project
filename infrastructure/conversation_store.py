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