from dataclasses import dataclass, field
from datetime import datetime

@dataclass
class MemoryRecord:
    content: str
    memory_type: str
    source: str
    category: str
    embedding: list[float]
    emotion_snapshot: dict
    importance: float = 0.5
    access_count: int = 0
    timestamp: datetime = field(default_factory=datetime.now)
    id: int = None