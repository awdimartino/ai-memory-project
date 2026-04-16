from dataclasses import dataclass, field
from datetime import datetime
import uuid

@dataclass
class MemoryRecord:
    content: str
    embedding: list[float]
    memory_type: str
    category: str

    origin_type: str
    origin_id: str | None = None

    conversation_id: str | None = None
    message_id: int | None = None

    emotion_snapshot: dict = field(default_factory=dict)
    importance: float = 0.5
    timestamp: datetime = field(default_factory=datetime.now)

    id: int | None = None

@dataclass
class MessageRecord:
    role: str
    content: str
    conversation_id: str
    timestamp: datetime = field(default_factory=datetime.now)
    id: int = None

@dataclass
class ConversationRecord:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: datetime = field(default_factory=datetime.now)
    user_last_active: datetime = field(default_factory=datetime.now)
    bot_last_active: datetime = field(default_factory=datetime.now)

@dataclass
class PromptConfig:
    include_identity: bool = True
    include_emotion: bool = True
    include_relational: bool = True
    include_temporal: bool = True
    include_history: bool = True
    include_format: bool = True
    history_limit: int = 10
    task_frame: str = ""