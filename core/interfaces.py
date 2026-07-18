"""Store contracts (the swap seams).

Defining these as Protocols from the start documents exactly what a persistence
layer must satisfy and lets us drop in an in-memory fake for deterministic tests
later (a v1 lesson: this was added too late last time).
"""
from typing import Protocol


class ConversationStore(Protocol):
    """Persistence for the episodic tier: full, verbatim conversation logs."""

    def create_session(self) -> int:
        """Start a new conversation session; return its id."""
        ...

    def add_message(self, session_id: int, role: str, content: str) -> None:
        """Append one message to a session."""
        ...

    def recent_messages(self, limit: int) -> list[dict]:
        """Return up to `limit` most recent messages (oldest first) as {role, content}."""
        ...


class MemoryStore(Protocol):
    """Persistence for the semantic tier: distilled, embedded facts."""

    def add(self, content: str, category: str | None, embedding: bytes,
            source_session: int | None) -> int:
        """Store one fact + its embedding (float32 bytes); return its id."""
        ...

    def active(self) -> list[dict]:
        """Return all active memories as {id, content, category, embedding(bytes)}."""
        ...

    def deactivate(self, memory_id: int, superseded_by: int | None) -> None:
        """Soft-delete a memory, optionally linking the memory that replaced it."""
        ...

    def count(self) -> int:
        """Number of active memories."""
        ...
