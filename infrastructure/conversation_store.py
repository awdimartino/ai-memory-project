from unittest import result
import uuid

import infrastructure.config as config
import json
from core.models import ConversationRecord, MessageRecord
from infrastructure.database import DatabaseConnection
from infrastructure.embedder import Embedder

from openai import OpenAI as oai

class ConversationStore:
    """Responsible for interfacing with the database to store and retrieve conversations."""
    def __init__(self, database: DatabaseConnection):
        self.database = database

    def setup_conversations(self):
        """Set up the database schema for storing conversations."""

        sql = """
        CREATE TABLE IF NOT EXISTS conversations (
            id UUID PRIMARY KEY,
            created_at TIMESTAMPTZ DEFAULT NOW(),
            last_active TIMESTAMPTZ DEFAULT NOW(),
        );

        CREATE INDEX IF NOT EXISTS conversations_last_active_idx
            ON conversations (last_active);
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
        """"Create a new conversation in the database and return it as a ConversationRecord."""
        conversation_id = str(uuid.uuid4())

        sql = """
        INSERT INTO conversations (id)
        VALUES (%s)
        """

        self.database.execute(sql, (conversation_id,))
        return self.get_conversation(conversation_id)

    def store_message(self, conversation_record, message_record):
        sql = """
        INSERT INTO messages (conversation_id, role, content)
        VALUES (%s, %s, %s)
        """
        self.database.execute(sql, (conversation_record.id, message_record.role, message_record.content))

    def get_conversation(self, conversation_id):
        """Retrieve a conversation from the database."""
        conversation_sql = """
        SELECT id, created_at, last_active
        FROM conversations
        WHERE id = %s
        """

        conversation_result = self.database.fetch_all(conversation_sql, (conversation_id,))
        if not conversation_result:
            return None

        conversation_data = conversation_result[0]
        conversation_record = ConversationRecord(
            id=conversation_data["id"],
            created_at=conversation_data["created_at"],
            last_active=conversation_data["last_active"]
        )
        return conversation_record

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
    
    def get_recent_conversations(self, hours=2, limit=10):
        """Get recent conversations active within the last specified hours, ordered by last active time."""
        sql = """
        SELECT id, created_at, last_active
        FROM conversations
        WHERE last_active >= NOW() - INTERVAL '%s hours'
        ORDER BY last_active DESC
        LIMIT %s
        """

        results = self.database.fetch_all(sql, (hours, limit))

        conversations = []
        for conversation_data in results:
            conversation_record = ConversationRecord(
                id=conversation_data["id"],
                created_at=conversation_data["created_at"],
                last_active=conversation_data["last_active"]
            )
            conversations.append(conversation_record)

        return conversations