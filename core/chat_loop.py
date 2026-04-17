from datetime import datetime, timedelta

from core.companion import Companion
from core.tick_system import TickSystem
import threading
from concurrent.futures import ThreadPoolExecutor

class ChatLoop():
    """Responsible for managing the main chat loop and tick system"""
    def __init__(self, companion: Companion, tick_system: TickSystem):
        self.companion = companion
        self.tick_system = tick_system

    def start(self):
        """Main loop for chatting with the user, managing tick system, and synchronizing access to shared resources."""        
        # Initialize background tick system
        self.tick_system.start()
        executor = ThreadPoolExecutor(max_workers=2)

        while True:
            # Check for existing conversation or start a new one
            conversation = self.companion.conversation_manager.check_conversation()
            if not conversation:
                conversation = self.companion.conversation_manager.start_conversation()

            # Get user input
            query = input("You: ")
            # Lock for response
            with self.tick_system.lock:
                user_message, bot_message = self.companion.respond(query)

            # Classify the memories asynchronously
            # TODO: CHANGE THIS TO RUN LESS FREQUENTLY
            # Conversation context can be relied on more
            executor.submit(
                self.companion.memory_manager.classify_memories,
                user_message,
                bot_message,
                conversation

            )


