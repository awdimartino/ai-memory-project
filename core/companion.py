
import os
from pyexpat.errors import messages

import infrastructure.config as config
from infrastructure.llm_client import LLMClient
from core.memory_manager import MemoryManager
from core.emotion_manager import EmotionManager
from core.conversation_manager import ConversationManager
from infrastructure.database import DatabaseConnection
from core.prompt_builder import PromptBuilder

from openai import OpenAI as oai

class Companion:
    """The main class representing the AI companion, responsible for managing conversations, memories, emotions, and interfacing with the LLM."""
    def __init__(self, llm_client: LLMClient, memory_manager: MemoryManager, conversation_manager: ConversationManager, emotion_manager: EmotionManager, prompt_builder: PromptBuilder):
        self.llm_client = llm_client
        self.memory_manager = memory_manager
        self.conversation_manager = conversation_manager
        self.emotion_manager = emotion_manager
        self.prompt_builder = prompt_builder

    def respond(self, query: str):
        """Generate a response to a user query by building a prompt and streaming the LLM's response."""
        # React to user input and build prompt
        self.emotion_manager.react(query)
            
        messages = self.prompt_builder.build_response_prompt(
            query=query,
            conversation=self.conversation_manager.get_active_messages(),
            emotions=self.emotion_manager.as_prompt()
        )
        # Stream the response
        response = self.llm_client.stream(messages)

        # Add the user message to the conversation
        self.conversation_manager.add_message("user", query)
        self.conversation_manager.add_message("assistant", response)
        return response

    def think(self):
        """Generate a thinking process, for use with short ticks"""
        # Placeholder for internal thought process logic
        pass
