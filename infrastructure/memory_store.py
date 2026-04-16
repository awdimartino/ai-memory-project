from unittest import result

import infrastructure.config as config
import json
from core.models import MemoryRecord
from infrastructure.database import DatabaseConnection
from infrastructure.embedder import Embedder

from openai import OpenAI as oai

class MemoryStore:
    """Responsible for interfacing with the database to store and retrieve memories."""
    def __init__(self, database: DatabaseConnection):
        self.database = database

    def setup(self):
        """Set up the database schema for storing memories."""
        """
        id - Unique identifier for the memory (SERIAL PRIMARY KEY)
        content - The textual content of the memory (TEXT NOT NULL)
        memory_type - The type of memory 
            episode - a specific event or experience
            reflection - an internal thought or insight
            consolidation - summary of multiple episodes
            abstraction/fact - long term belief or fact about the world, self, or user
        origin_type - Where the memory came from
            message - a specific message in a conversation
            tick_short - a thought tick
            tick_long - a longer reflection
            reflection - internal thought process
            tool - output from an external tool
        origin_id - An optional identifier for the origin 
        category - The category of the memory
            fact - a factual statement about the world, self, or user
            preference - a stable preference or trait of the user or bot
            goal - a desired outcome or objective
            relation - a relationship between entities (e.g. "Alice is Bob's sister")
            event - a specific occurrence or experience
            belief - an internal belief that may not be objectively true but is held by the user or bot
        embedding - A vector representation of the memory content for similarity search
        emotion_snapshot - A snapshot of the bot's emotional state when the memory was formed (JSONB)
        importance - A score representing the importance of the memory for retrieval and consolidation
        timestamp - When the memory was created
        last_accessed - When the memory was last accessed
        """
        sql = """
        CREATE TABLE IF NOT EXISTS memories (
            id                SERIAL PRIMARY KEY,
            content           TEXT NOT NULL,
            memory_type       TEXT NOT NULL,
            category          TEXT NOT NULL,

            embedding         VECTOR(1024),

            origin_type       TEXT NOT NULL,
            origin_id         TEXT,

            emotion_snapshot  JSONB,
            importance        FLOAT DEFAULT 0.5,
            timestamp         TIMESTAMPTZ DEFAULT NOW()
        );

        CREATE INDEX IF NOT EXISTS memories_embedding_idx
            ON memories USING ivfflat (embedding vector_cosine_ops);

        CREATE INDEX IF NOT EXISTS memories_origin_idx
            ON memories (origin_type, origin_id);

        CREATE INDEX IF NOT EXISTS memories_category_idx
            ON memories (category);
        """
        self.database.execute(sql)
        
    def store_memory(self, record: MemoryRecord):
        """Create a new memory from a memory record"""
        sql = """
        INSERT INTO memories (content, memory_type, origin_type, origin_id, category, embedding, emotion_snapshot, importance)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
        """
        try:
            self.database.execute(sql, (
                record.content,
                record.memory_type,
                record.origin_type,
                record.origin_id,
                record.category,
                record.embedding,
                json.dumps(record.emotion_snapshot),
                record.importance
            ))
            return True
        except Exception as e:
            print(e)
            return False

    def memory_exists(self, query_embedding, origin_type=None, category=None, threshold=0.92):
        """Check if a memory exists in the database based on the embedding and optional filters."""
        conditions = ["1 - (embedding <=> %s) > %s"]
        params = [str(query_embedding), threshold]

        if origin_type:
            conditions.append("origin_type = %s")
            params.append(origin_type)
        if category:
            conditions.append("category = %s")
            params.append(category)

        where = " AND ".join(conditions)
        sql = f"SELECT EXISTS (SELECT 1 FROM memories WHERE {where});"
        row = self.database.fetch_one(sql, params)
        return row if row else False
        
    def fetch_memories(self, query_embedding, origin_type=None, category=None, threshold=0.92, limit=5):
        """Fetch memories from the database based on embedding similarity and optional filters."""
        conditions = ["1 - (embedding <=> %s) > %s"]
        params = [str(query_embedding), threshold]

        if origin_type:
            conditions.append("origin_type = %s")
            params.append(origin_type)
        if category:
            conditions.append("category = %s")
            params.append(category)

        where = " AND ".join(conditions)
        sql = f"""
        SELECT id, content, memory_type, origin_type, origin_id, category, embedding, emotion_snapshot, importance, timestamp
        FROM memories
        WHERE {where}
        ORDER BY embedding <=> %s
        LIMIT %s
        """
        params.append(str(query_embedding))
        params.append(limit)
        try:
            results = self.database.fetch_all(sql, params)
            return [
                MemoryRecord(
                    id=row["id"],
                    content=row["content"],
                    memory_type=row["memory_type"],
                    origin_type=row["origin_type"],
                    origin_id=row["origin_id"],
                    category=row["category"],
                    embedding=row["embedding"],
                    emotion_snapshot=row["emotion_snapshot"],
                    importance=row["importance"],
                    timestamp=row["timestamp"]
                )
                for row in results ]
        except Exception as e:
            print(e)
            return []
        