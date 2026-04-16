from unittest import result

import infrastructure.config as config
import json
from core.models import MemoryRecord
from infrastructure.database import DatabaseConnection
from infrastructure.embedder import Embedder

from openai import OpenAI as oai

class ConversationStore:
    """Responsible for interfacing with the database to store and retrieve conversations."""
    def __init__(self, database: DatabaseConnection):
        self.database = database

    def setup(self):
        """Set up the database schema for storing conversations."""
        """
        
        """
        sql = """
        CREATE TABLE IF NOT EXISTS conversations (
            id SERIAL PRIMARY KEY,
            conversation_id TEXT NOT NULL,
            role TEXT NOT NULL,         -- 'user' or 'bot'
            content TEXT NOT NULL,
            timestamp TIMESTAMPTZ DEFAULT NOW()
        );
        CREATE INDEX IF NOT EXISTS conversations_conversation_id_idx
            ON conversations (conversation_id);
        """
        self.database.execute(sql)

    def store_message(self, conversation_id: str, role: str, content: str):