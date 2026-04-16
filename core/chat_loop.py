from time import time

from core.companion import Companion
from core.ticks import TickSystem
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
            conversation = self.companion.conversation_manager.resume_conversation()
            if not conversation:
                conversation = self.companion.conversation_manager.start_conversation()
            messages = self.companion.conversation_manager.get_active_messages()

            # Get user input
            query = input("You: ")
            self.tick_system.last_user_interaction = time.time()
            self.tick_system.last_any_interaction = time.time()
            
            if query.strip().lower() == "/exit":
                break

            response = self.companion.respond(query)

