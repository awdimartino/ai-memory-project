
import os
from pyexpat.errors import messages

import infrastructure.config as config
from infrastructure.llm_client import LLMClient
from core.memory import MemoryManager
from core.emotions import EmotionManager
from core.conversation import ConversationManager
from infrastructure.database import DatabaseConnection
from core.prompt_builder import PromptBuilder

from openai import OpenAI as oai

class Companion:
    def __init__(self, llm_client, memory_manager, conversation_manager, emotion_manager, prompt_builder):
        self.llm_client = llm_client
        self.memory_manager = memory_manager
        self.conversation_manager = conversation_manager
        self.emotion_manager = emotion_manager
        self.prompt_builder = prompt_builder

    def respond(self, query):
        messages = self.prompt_builder.build_response_prompt(
            query=query,
            conversation=self.conversation_manager.get_conversation(),
            emotions=self.emotion_manager.get_emotions(),
            memories=self.memory_manager.get_memories()
        )
        pass

    def think(self):
        # Placeholder for internal thought process logic
        pass
    
    def chat(self):
        while True:
            # Check for existing conversation or start a new one
            conversation = self.conversation_manager.resume_conversation()
            if not conversation:
                conversation = self.conversation_manager.start_conversation()
            messages = self.conversation_manager.get_active_messages()

            # Get user input
            query = input("You: ")
            
            if query.strip().lower() == "/exit":
                break

            # React to user input and build prompt
            self.emotion_manager.react(query)
            
            messages = self.prompt_builder.build_response_prompt(
                query=query,
                conversation=self.conversation_manager.get_active_messages(),
                emotions=self.emotion_manager.as_prompt()
            )

            for message in messages:
                print(f"{message}")

            response = self.llm_client.stream(messages)

            # Add the user message to the conversation
            self.conversation_manager.add_message("user", query)
            self.conversation_manager.add_message("assistant", response)