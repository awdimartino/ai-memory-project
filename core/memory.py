from infrastructure import embedder
from infrastructure import memory_store
from infrastructure.database import DatabaseConnection

class MemoryManager:
    def __init__(self, memory_store: memory_store.MemoryStore, embedder: embedder.Embedder):
        self.memory_store = memory_store
        self.embedder = embedder
        self.memory_store.setup()

    def save_memory(self, memory):
        """Save a memory to the memory store."""
        self.memory_store.store_memory(memory)

    def retrieve_memories(self, query, limit=5):
        """Retrieve relevant memories based on a query."""
        return self.memory_store.fetch_memories(self.embedder.get_embedding(query), limit=limit)