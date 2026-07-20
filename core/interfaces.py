"""Store contracts (the swap seams).

Defining these as Protocols from the start documents exactly what a persistence
layer must satisfy and lets us drop in an in-memory fake for deterministic tests
(a v1 lesson: this was added too late last time). The offline suite in tests/ is
that payoff, so these contracts are load-bearing — a method a caller depends on
but that isn't declared here is a fake waiting to pass vacuously. Keep them in
sync with the implementations in infrastructure/.
"""
from typing import Protocol


class ConversationStore(Protocol):
    """Persistence for the episodic tier: full, verbatim conversation logs."""

    def create_session(self, title: str | None = None) -> int:
        """Start a new conversation session (optionally titled); return its id."""
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

    def message_count(self) -> int:
        """Total messages across all conversations (drives familiarity + persona gates)."""
        ...

    # --- conversation tabs (one shared brain, per-tab threads) ---
    def session_messages(self, session_id: int, limit: int) -> list[dict]:
        """Up to `limit` most recent messages of ONE conversation (oldest first)."""
        ...

    def latest_session(self) -> int | None:
        """Most recently active conversation id (resume-on-boot), or None if there are none."""
        ...

    def list_conversations(self) -> list[dict]:
        """All conversations, most-recent-first, as {id, title, count, last_at}."""
        ...

    def session_title(self, session_id: int) -> str | None:
        """A conversation's title, or None if untitled/missing."""
        ...

    def set_title(self, session_id: int, title: str) -> None:
        """Set a conversation's title."""
        ...

    def delete_session(self, session_id: int) -> None:
        """Delete a conversation and its messages."""
        ...

    def search_messages(self, query: str, limit: int) -> list[dict]:
        """Keyword search over all messages (the reminisce tool's episodic read)."""
        ...

    def clear(self) -> None:
        """Delete every conversation and message (the full-reset admin op)."""
        ...


class MemoryStore(Protocol):
    """Persistence for the semantic tier: distilled, embedded facts."""

    def add(self, content: str, category: str | None, embedding: bytes,
            source_session: int | None, core: bool = False) -> int:
        """Store one fact + its embedding (float32 bytes); return its id."""
        ...

    def active(self) -> list[dict]:
        """Return all active memories as {id, content, category, embedding(bytes), core(bool)}."""
        ...

    def core(self) -> list[dict]:
        """Return active core memories (always injected) as {id, content, category}."""
        ...

    def all(self) -> list[dict]:
        """Every memory (active + retired), newest first — for the inspector/admin UI."""
        ...

    def set_core(self, memory_id: int, core: bool) -> None:
        """Promote a memory into (or demote it out of) the always-injected core set."""
        ...

    def update_content(self, memory_id: int, content: str, embedding: bytes) -> None:
        """Replace a memory's text + embedding (the inspector's edit)."""
        ...

    def deactivate(self, memory_id: int, superseded_by: int | None = None) -> None:
        """Soft-delete a memory, optionally linking the memory that replaced it."""
        ...

    def delete(self, memory_id: int) -> None:
        """Hard-delete one memory (unlike deactivate, which keeps history)."""
        ...

    def clear(self) -> None:
        """Delete ALL memories (active + retired); admin-only."""
        ...

    def count(self) -> int:
        """Number of active memories."""
        ...

    def count_core(self) -> int:
        """Number of active core memories."""
        ...

    def count_superseded(self) -> int:
        """Number of retired (soft-deleted) memories — the status panel's history count."""
        ...

    def superseded(self, limit: int) -> list[dict]:
        """Up to `limit` most recently retired memories, for the inspector's history view."""
        ...

    def counts(self) -> dict:
        """{active, core, superseded} in one scan — the status panel polls this often."""
        ...


class ThoughtStore(Protocol):
    """Persistence for Mari's private thought journal (self-reflections)."""

    def add(self, content: str, mood: str | None) -> int:
        """Store one reflection (optionally tagged with the dominant mood); return its id."""
        ...

    def recent(self, limit: int) -> list[dict]:
        """Return up to `limit` most recent thoughts (newest first) as {content, mood, created_at}."""
        ...

    def count(self) -> int:
        """Total number of thoughts written (surfaced in the status panel)."""
        ...

    def clear(self) -> None:
        """Delete every private thought (the full-reset admin op)."""
        ...


class IntentionStore(Protocol):
    """Persistence for Mari's private forward agenda (the "planning" pillar): short
    notes of things she means to bring up or find out, minted during reflection and
    consumed by reach-out."""

    def add(self, content: str) -> int:
        """Store one intention; return its id."""
        ...

    def active(self, limit: int | None = None) -> list[dict]:
        """Open intentions, oldest first, as {id, content, created_at}."""
        ...

    def fulfill(self, intention_id: int) -> None:
        """Mark an intention acted-on (retired, timestamped, kept for history)."""
        ...

    def drop(self, intention_id: int) -> None:
        """Retire an intention without acting on it (expiry / over-cap pruning)."""
        ...

    def drop_older_than(self, cutoff_iso: str) -> int:
        """Retire active intentions created before `cutoff_iso`; return how many."""
        ...

    def all(self) -> list[dict]:
        """Every intention (open + retired), for the status panel's agenda history."""
        ...

    def count_active(self) -> int:
        """Number of open intentions (against which INTENTION_MAX_ACTIVE is enforced)."""
        ...

    def clear(self) -> None:
        """Delete every intention (the full-reset admin op)."""
        ...


class ModelManager(Protocol):
    """Loads/unloads LLMs from VRAM (the sleep/standby seam)."""

    async def unload_all(self) -> None:
        """Unload all resident models (free VRAM)."""
        ...

    async def load(self, models: list[str]) -> None:
        """Load the given models (on wake)."""
        ...


class MetaStore(Protocol):
    """Persistence for durable scalar state (a small key/value table)."""

    def get(self, key: str) -> str | None:
        """Read a raw string value, or None if unset (used for the persona-self slot)."""
        ...

    def set(self, key: str, value: str) -> None:
        """Write a raw string value."""
        ...

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

    def clear(self) -> None:
        """Delete every key (mood, drives, persona, watermark, cooldowns) — full-reset op."""
        ...
