class MemoryService:
    def __init__(self, embedder, memory_store):
        self.embedder = embedder
        self.memory_store = memory_store

    def save_memory(self, memory):
        """Save a memory to the memory store."""
        self.memory_store.save(memory)

    def retrieve_memories(self, query, limit=5):
        """Retrieve relevant memories based on a query."""
        return self.memory_store.retrieve(query, limit=limit)