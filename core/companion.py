import logging

import infrastructure.config as config
from core.conversation_manager import ConversationManager
from core.emotion_manager import EmotionManager
from core.memory_manager import MemoryManager
from core.prompt_builder import PromptBuilder
from infrastructure.llm_client import LLMClient

logger = logging.getLogger(__name__)


class Companion:
    """The AI companion: orchestrates conversation, memory, and emotion.

    Callers (ChatLoop, TickSystem) go through this facade rather than reaching
    into the individual managers, so the managers stay encapsulated.
    """

    def __init__(self, llm_client: LLMClient, memory_manager: MemoryManager,
                 conversation_manager: ConversationManager, emotion_manager: EmotionManager):
        self.llm_client = llm_client
        self.memory_manager = memory_manager
        self.conversation_manager = conversation_manager
        self.emotion_manager = emotion_manager

    def respond(self, query: str):
        """Generate a response to a user query, streaming the LLM's reply."""
        conversation_history = self.conversation_manager.get_active_messages()
        user_message = self.conversation_manager.add_message(role="user", content=query)

        # React to the input, then gather memory + emotion context for the prompt.
        self.emotion_manager.react(query)
        memories = self.memory_manager.retrieve_memories(query)
        logger.debug("Retrieved %d memories for query", len(memories))

        messages = PromptBuilder.build_response_prompt(
            query=query,
            conversation=conversation_history,
            emotions=self.emotion_manager.as_prompt(),
            memories=memories,
        )
        response = self.llm_client.stream(messages)

        bot_message = self.conversation_manager.add_message("assistant", response)
        return user_message, bot_message

    def think(self):
        """Generate an internal thought, for use with ticks. (Phase 6)"""
        pass

    # --- Facade: intent methods so callers don't reach through to managers ---

    def ensure_conversation(self):
        """Return the active conversation, resuming a recent one or starting fresh."""
        conversation = self.conversation_manager.check_conversation()
        if not conversation:
            conversation = self.conversation_manager.start_conversation()
        return conversation

    def maybe_classify(self):
        """Classify the pending batch of messages once enough have accumulated.

        Classification doesn't need to run every turn — the active conversation
        already stays in the response prompt's context window regardless. No-op
        (returns []) until config.CLASSIFY_BATCH_SIZE messages have piled up.
        """
        batch = self.conversation_manager.pending_batch()
        if len(batch) < config.CLASSIFY_BATCH_SIZE:
            return []
        return self._classify_batch(batch)

    def flush_pending_classification(self):
        """Classify whatever has accumulated so far, regardless of batch size.

        Call on graceful shutdown so a session that ends mid-batch doesn't lose
        those messages — pending_batch is in-memory only.
        """
        batch = self.conversation_manager.pending_batch()
        if not batch:
            return []
        return self._classify_batch(batch)

    def _classify_batch(self, batch):
        conversation = self.conversation_manager.current_conversation
        saved = self.memory_manager.classify_memories(batch, conversation)
        self.conversation_manager.clear_pending_batch()
        return saved

    def decay(self):
        """Decay the emotional state one step (called by the tick system)."""
        self.emotion_manager.decay()

    def minutes_since_user_interaction(self):
        return self.conversation_manager.minutes_since_user_interaction()

    def minutes_since_any_interaction(self):
        return self.conversation_manager.minutes_since_any_interaction()
