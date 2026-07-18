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

    def add_message(self, session_id: int, role: str, content: str) -> int:
        """Append one message to a session; return its row id."""
        ...

    def recent_messages(self, limit: int) -> list[dict]:
        """Return up to `limit` most recent messages (oldest first) as {role, content}."""
        ...

    def messages_after(self, msg_id: int) -> list[dict]:
        """Return messages with id > msg_id (oldest first) as {id, role, content}.

        Used on startup to recover the unconsolidated tail after a hard kill.
        """
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


class ThoughtStore(Protocol):
    """Persistence for Mari's private thought journal (self-reflections)."""

    def add(self, content: str, mood: str | None) -> int:
        """Store one reflection (optionally tagged with the dominant mood); return its id."""
        ...

    def recent(self, limit: int) -> list[dict]:
        """Return up to `limit` most recent thoughts (newest first) as {content, mood, created_at}."""
        ...


class MetaStore(Protocol):
    """Persistence for durable scalar state (a small key/value table)."""

    def get_int(self, key: str, default: int = 0) -> int:
        """Read an integer value, or `default` if unset/unparseable."""
        ...

    def set_int(self, key: str, value: int) -> None:
        """Write an integer value."""
        ...

    def get_json(self, key: str, default=None):
        """Read a JSON value, or `default` if unset/unparseable."""
        ...

    def set_json(self, key: str, value) -> None:
        """Write a JSON-serializable value."""
        ...
