from infrastructure import conversation_store
from infrastructure.database import DatabaseConnection

class ConversationManager:
    def __init__(self, database: DatabaseConnection):
        self.database = database
        self.conversation_store = conversation_store.ConversationStore(database)

    def start_conversation(self):
        """Start a new conversation and return its ID."""
        return self.conversation_store.new_conversation()