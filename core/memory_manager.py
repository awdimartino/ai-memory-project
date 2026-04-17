from core.prompt_builder import PromptBuilder
from infrastructure import embedder
from infrastructure import memory_store
from infrastructure.database import DatabaseConnection
from infrastructure.llm_client import LLMClient

class MemoryManager:
    """Responsible for interfacing with the memory database"""
    def __init__(self, memory_store: memory_store.MemoryStore, embedder: embedder.Embedder, llm_client: LLMClient, prompt_builder: PromptBuilder):
        self.memory_store = memory_store
        self.embedder = embedder
        self.llm_client = llm_client
        self.prompt_builder = prompt_builder
        self.memory_store.setup()

    def save_memory(self, memory):
        """Save a memory to the memory store."""
        self.memory_store.store_memory(memory)

    def retrieve_memories(self, query, limit=5):
        """Retrieve relevant memories based on a query."""
        return self.memory_store.fetch_memories(self.embedder.get_embedding(query), limit=limit)
    
    def memory_exists(self, query, threshold=0.92):
        """Check if a similar memory already exists based on the query."""
        return self.memory_store.memory_exists(self.embedder.get_embedding(query), threshold=threshold)
    
    def classify_query(self, query, conversation_history):
        """Classify a user query to determine if it should create a new memory or fetch existing ones."""
        prompt = self.prompt_builder.build_user_brain_prompt(query, conversation_history)
        response = self.llm_client.memory_classification(prompt)
        return response