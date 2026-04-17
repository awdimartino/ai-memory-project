from datetime import datetime, timedelta, timezone

from infrastructure import conversation_store
from infrastructure.database import DatabaseConnection
from core.models import ConversationRecord, MessageRecord

class ConversationManager:
    """Responsible for managing conversations, including starting new conversations, resuming existing ones, and adding messages to conversations."""
    def __init__(self, conversation_store: conversation_store.ConversationStore):
        self.current_conversation = None
        self.conversation_store = conversation_store
        self.conversation_store.setup_conversations()
        self.conversation_store.setup_messages()

    def start_conversation(self):
        """Start a new conversation and set it as the current conversation."""
        self.current_conversation= self.conversation_store.new_conversation()
        return self.current_conversation

    def check_conversation(self, timeout_hours=2):
        """Returns active conversation or resumes recent one."""

        if self.current_conversation:
            cutoff = datetime.now(timezone.utc) - timedelta(hours=timeout_hours)

            # local cache check (fast path)
            if self.last_active() > cutoff:
                return self.current_conversation

        recent = self.conversation_store.get_recent_conversations(
            hours=timeout_hours,
            limit=1
        )

        if not recent:
            return None

        self.current_conversation = recent[0]
        return self.current_conversation

    def get_active_messages(self):
        """Get the current active conversation messages, or None if no conversation is active."""
        if not self.current_conversation:
            return []

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
        
        if role == "user":
            self.current_conversation.user_last_active = message_record.timestamp
        else:
            self.current_conversation.bot_last_active = message_record.timestamp

        self.conversation_store.store_message(self.current_conversation, message_record)
        return message_record

    def last_active(self):
        """Return the timestamp of the last activity in the current conversation."""
        if not self.current_conversation:
            return None
        return max(self.current_conversation.user_last_active, self.current_conversation.bot_last_active)

    def minutes_since_user_interaction(self):
        """Get the number of minutes since the last user interaction in the current conversation."""
        if not self.current_conversation:
            return None
        
        last_user_active = self.current_conversation.user_last_active
        if not last_user_active:
            return None
        
        now = datetime.now(timezone.utc)
        delta = now - last_user_active
        return delta.total_seconds() / 60
    
    def minutes_since_any_interaction(self):
        """Get the number of minutes since the last user or bot interaction in the current conversation."""
        if not self.current_conversation:
            return None
        
        last_active = self.last_active()
        if not last_active:
            return None
        
        now = datetime.now(timezone.utc)
        delta = now - last_active
        return delta.total_seconds() / 60
    