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

    def recent_messages_with_ids(self, limit: int) -> list[dict]:
        """Like `recent_messages`, but keeps each message's id — the citation
        surface for A3's grounded open-question mining (§A4)."""
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
        """Delete every conversation and message (the full-reset admin op).

        Must NOT clear the permanent archive below.
        """
        ...

    # --- the permanent record (never cleared) ---
    def begin_new_era(self) -> int:
        """Mark a discontinuity in the archive (a factory reset); return the new era."""
        ...

    def current_era(self) -> int:
        """Which era new messages are filed under."""
        ...

    def archive_count(self) -> int:
        """Total messages ever recorded, across every era."""
        ...

    def archive_eras(self) -> list[dict]:
        """One row per era: {era, messages, started, ended}."""
        ...

    def archived_messages(self, limit: int, era: int | None = None) -> list[dict]:
        """Most recent archived messages, newest first; optionally one era only."""
        ...

    def search_archive(self, query: str, limit: int) -> list[dict]:
        """Keyword search across every era, including wiped conversations."""
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
    consumed by reach-out.

    Also holds A3-mined "pursuits" (self-directed things she may step away to do,
    §A4) in the same table, discriminated by `kind`: 'agenda' (default, the
    raise-with-him items above) vs 'pursuit'. Every method defaults to `kind="agenda"`
    so every pre-existing caller is unaffected; a pursuit only ever appears where a
    caller explicitly asks for `kind="pursuit"`.
    """

    def add(self, content: str, kind: str = "agenda", citations: list[int] | None = None) -> int:
        """Store one intention/pursuit; return its id. `citations` are message ids
        grounding a pursuit (ignored/None for agenda items)."""
        ...

    def active(self, kind: str = "agenda", limit: int | None = None) -> list[dict]:
        """Open items of one kind, oldest first, as {id, content, created_at, citations}."""
        ...

    def fulfill(self, intention_id: int) -> None:
        """Mark an item acted-on (retired, timestamped, kept for history)."""
        ...

    def drop(self, intention_id: int) -> None:
        """Retire an item without acting on it (expiry / over-cap pruning)."""
        ...

    def drop_older_than(self, cutoff_iso: str, kind: str = "agenda") -> int:
        """Retire active items of one kind created before `cutoff_iso`; return how many."""
        ...

    def all(self, kind: str | None = None) -> list[dict]:
        """Every item (open + retired), for the status panel's agenda history.
        `kind=None` returns both kinds; otherwise filters to one."""
        ...

    def count_active(self, kind: str = "agenda") -> int:
        """Number of open items of one kind (against which *_MAX_ACTIVE is enforced)."""
        ...

    def clear(self) -> None:
        """Delete every intention AND pursuit (the full-reset admin op)."""
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
