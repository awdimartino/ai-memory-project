import logging

import infrastructure.config as config
from core.interfaces import MemoryStoreProtocol
from core.models import MemoryRecord
from core.prompt_builder import PromptBuilder
from infrastructure.embedder import Embedder
from infrastructure.llm_client import LLMClient

logger = logging.getLogger(__name__)


class MemoryManager:
    """Coordinates the memory store, the embedder, and the LLM classifier."""

    def __init__(self, memory_store: MemoryStoreProtocol, embedder: Embedder, llm_client: LLMClient):
        self.memory_store = memory_store
        self.embedder = embedder
        self.llm_client = llm_client
        self.memory_store.setup()

    def save_memory(self, memory):
        """Save a memory to the memory store."""
        self.memory_store.store_memory(memory)

    def retrieve_memories(self, query, limit=5):
        """Retrieve relevant memories based on a query."""
        return self.memory_store.fetch_memories(self.embedder.get_embedding(query), limit=limit)

    def memory_exists(self, query, threshold=config.MEMORY_DEDUP_THRESHOLD):
        """Check if a similar memory already exists based on the query."""
        return self.memory_store.memory_exists(self.embedder.get_embedding(query), threshold=threshold)

    def classify_memories(self, messages, conversation):
        """Extract long-term memories from a batch of recent messages and store the new ones."""
        if not messages:
            return []

        prompt = PromptBuilder.build_classify_prompt(messages)
        memories = self.llm_client.memory_classification(messages=prompt)
        if not memories:
            return []

        # Provenance points at the last message in the classified batch.
        origin_id = str(messages[-1].id) if messages[-1].id is not None else None

        saved = []
        for memory in memories:
            if self.memory_exists(memory["content"]):
                logger.debug("Skipping duplicate memory: %s", memory["content"])
                continue

            memory_record = MemoryRecord(
                content=memory["content"],
                embedding=self.embedder.get_embedding(memory["content"]),
                category=memory["category"],
                origin_type="message",
                origin_id=origin_id,
                conversation_id=conversation.id,
                emotion_snapshot={},
            )
            self.save_memory(memory_record)
            saved.append(memory_record)
            logger.debug("Saved memory: %s", memory["content"])

        return saved
