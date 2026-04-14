class MemoryRecord:
    def __init__(self, content, memory_type, source, embedding, emotion_snapshot, importance, timestamps, id=None):
        self.id = id
        self.content = content
        self.embedding = embedding
        self.memory_type = memory_type
        self.source = source
        self.emotion_snapshot = emotion_snapshot
        self.importance = importance
        self.timestamps = timestamps

    