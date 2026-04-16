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
        conversation_id = str(uuid.uuid4())

        sql = """
        INSERT INTO conversations (id)
        VALUES (%s)
        """

        self.database.execute(sql, (conversation_id,))
        return conversation_id

    def store_message(self, conversation_record, message_record):
        sql = """
        INSERT INTO messages (conversation_id, role, content)
        VALUES (%s, %s, %s)
        """
        self.database.execute(sql, (conversation_record.id, message_record.role, message_record.content))