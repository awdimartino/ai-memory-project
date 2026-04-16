from infrastructure import conversation_store
from infrastructure.database import DatabaseConnection
from core.models import ConversationRecord, MessageRecord

class ConversationManager:
    def __init__(self, conversation_store: conversation_store.ConversationStore):
        self.current_conversation = None
        self.conversation_store = conversation_store
        self.conversation_store.setup_conversations()
        self.conversation_store.setup_messages()

    def start_conversation(self):
        """Start a new conversation and set it as the current conversation."""
        self.current_conversation= self.conversation_store.new_conversation()
        return self.current_conversation

    def resume_conversation(self):
        """Resume most recent conversation, or return None if no conversations exist."""
        recent = self.conversation_store.get_recent_conversations()

        if not recent:
            return None

        self.current_conversation = recent[0]

        return self.current_conversation

    def get_active_messages(self):
        """Get the current active conversation messages, or None if no conversation is active."""
        if not self.current_conversation:
            return None

        messages = self.conversation_store.get_messages(self.current_conversation)

        return [
            {"role": m.role, "content": m.content}
            for m in messages
        ]
    
    def add_message(self, role, content):
        """Add a message to the current conversation."""
        if not self.current_conversation:
            raise ValueError("No active conversation to add messages to.")
        
        message_record = MessageRecord(
            role=role,
            content=content,
            conversation_id=self.current_conversation.id
        )
        self.conversation_store.store_message(self.current_conversation, message_record)