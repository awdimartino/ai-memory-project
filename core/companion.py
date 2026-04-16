
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
    def __init__(self, client):
        self.client = client
        self.database = DatabaseConnection()

        self.llm_client = LLMClient(self.client)
        self.memory_service = MemoryManager(self.client, self.database)
        self.conversation_manager = ConversationManager(self.database)
        self.emotion_service = EmotionManager()
        self.prompt_builder = PromptBuilder()

    def respond(self, query):
        messages = self.prompt_builder.build_response_prompt(
            query=query,
            conversation=self.conversation_manager.get_conversation(),
            emotions=self.emotion_service.get_emotions(),
            memories=self.memory_service.get_memories()
        )
        pass

    def think(self):
        # Placeholder for internal thought process logic
        pass
    
    def chat(self):
        while True:
            conversation = self.conversation_manager.resume_conversation()
            if not conversation:
                conversation = self.conversation_manager.start_conversation()
            messages = self.conversation_manager.get_active_messages()
            
            break
        print(conversation.id)
        # Placeholder for chat loop logic
client = oai(
            base_url=config.AI_BASE_URL,
            api_key=config.AI_API_KEY
        )
companion = Companion(client)
companion.chat()