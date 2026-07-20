"""SQLite connection, a minimal versioned migration runner, and the store base.

Migrations use SQLite's built-in `PRAGMA user_version` as the schema version.
To evolve the schema, append a new statement to MIGRATIONS; never edit an
existing entry (that's what keeps upgrades deterministic across machines).

`SqliteStore` is the shared base for the five stores. Its most important job is
the write lock — see the class docstring for why that lock belongs to the
*connection* rather than to each store.
"""
import logging
import sqlite3
import threading
from datetime import datetime, timezone

logger = logging.getLogger(__name__)

# One write lock per connection, keyed by identity because sqlite3.Connection
# supports neither attribute assignment nor weak references. Connections live for
# the process lifetime here, so this never grows or goes stale.
_CONNECTION_LOCKS: dict[int, threading.Lock] = {}
_REGISTRY_GUARD = threading.Lock()


def connection_lock(conn: sqlite3.Connection) -> threading.Lock:
    """The one write lock shared by every store using `conn`."""
    with _REGISTRY_GUARD:
        return _CONNECTION_LOCKS.setdefault(id(conn), threading.Lock())


def utcnow() -> str:
    """Timestamp for stored rows: UTC, ISO-8601, no microseconds."""
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat()


class SqliteStore:
    """Base for the SQLite stores: shared connection, shared write lock, timestamps.

    **The lock belongs to the connection, not the store.** Every store here is
    handed the *same* connection by bootstrap, but each used to create its own
    `threading.Lock`. Those locks therefore serialized a store against itself and
    against nothing else — so two different stores writing concurrently (a chat
    turn logging a message via `add_message` while a tick job persists drives via
    `meta.set_json`, both on `asyncio.to_thread` executor threads) reached one
    connection unsynchronized.

    That is not theoretical. Measured with two stores hammering one connection from
    four threads: **594/600 and 592/600 rows survived**, with 173 errors including
    CPython `SystemError: error return without exception set`. The same test with a
    connection each was lossless. Sharing one lock per connection fixes it, and
    costs nothing at single-user write volumes.
    """

    def __init__(self, conn: sqlite3.Connection):
        self.conn = conn
        self._lock = connection_lock(conn)

    def _write(self, sql: str, params: tuple = ()) -> sqlite3.Cursor:
        """Run one INSERT/UPDATE/DELETE and commit, holding the connection lock."""
        with self._lock:
            cur = self.conn.execute(sql, params)
            self.conn.commit()
            return cur

MIGRATIONS = [
    # v1 — episodic conversation log
    """
    CREATE TABLE sessions (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        started_at TEXT NOT NULL
    );
    CREATE TABLE messages (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        session_id INTEGER NOT NULL REFERENCES sessions(id),
        role       TEXT NOT NULL,
        content    TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    CREATE INDEX idx_messages_session ON messages(session_id);
    """,
    # v2 — semantic memory tier (distilled, embedded facts)
    """
    CREATE TABLE memories (
        id             INTEGER PRIMARY KEY AUTOINCREMENT,
        content        TEXT NOT NULL,
        category       TEXT,
        embedding      BLOB NOT NULL,
        created_at     TEXT NOT NULL,
        source_session INTEGER,
        active         INTEGER NOT NULL DEFAULT 1
    );
    CREATE INDEX idx_memories_active ON memories(active);
    """,
    # v3 — lifecycle: link a superseded memory to the one that replaced it
    """
    ALTER TABLE memories ADD COLUMN superseded_by INTEGER;
    """,
    # v4 — durable scalar state (key/value). Seeds the consolidation watermark to
    # the current max message id so an existing DB doesn't re-consolidate its whole
    # backlog on first run with this feature (fresh DB: no messages => 0).
    """
    CREATE TABLE meta (
        key   TEXT PRIMARY KEY,
        value TEXT NOT NULL
    );
    INSERT INTO meta (key, value)
    VALUES ('last_consolidated_msg_id', (SELECT COALESCE(MAX(id), 0) FROM messages));
    """,
    # v5 — Mari's private thought journal (self-reflection during idle ticks)
    """
    CREATE TABLE thoughts (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        content    TEXT NOT NULL,
        mood       TEXT,
        created_at TEXT NOT NULL
    );
    """,
    # v6 — core memory: identity-defining facts always injected into the prompt
    """
    ALTER TABLE memories ADD COLUMN core INTEGER NOT NULL DEFAULT 0;
    CREATE INDEX idx_memories_core ON memories(core);
    """,
    # v7 — named conversations (tabs): each session is a separate message thread
    """
    ALTER TABLE sessions ADD COLUMN title TEXT;
    """,
    # v8 — intentions: Mari's private forward agenda (the Generative-Agents "planning"
    # pillar) — things she means to bring up or find out, minted during idle reflection
    # and drawn on by reach-out. fulfilled_at is set when she acts on one.
    """
    CREATE TABLE intentions (
        id           INTEGER PRIMARY KEY AUTOINCREMENT,
        content      TEXT NOT NULL,
        created_at   TEXT NOT NULL,
        fulfilled_at TEXT,
        active       INTEGER NOT NULL DEFAULT 1
    );
    CREATE INDEX idx_intentions_active ON intentions(active);
    """,
    # v9 — injection bookkeeping + self-note revision history.
    #
    # `last_injected_turn` / `inject_count` support sticky-and-cooldown injection:
    # core facts used to be injected on EVERY turn unconditionally, which is the
    # structural cause of a companion sounding like it's reading off a card. Counting
    # in TURNS (not wall time) matches how the effect is felt — it's about how many
    # messages ago she last said it, not how many minutes.
    #
    # `self_notes_log` keeps every revision of the operating-notes slot, which is
    # otherwise overwritten wholesale. Without history you cannot compute the
    # Reflection Repetition Rate, and repetition in self-generated rules is the
    # signal that they're confabulated (scripts/rrr_diagnostic.py).
    """
    ALTER TABLE memories ADD COLUMN last_injected_turn INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE memories ADD COLUMN inject_count INTEGER NOT NULL DEFAULT 0;
    CREATE TABLE self_notes_log (
        id         INTEGER PRIMARY KEY AUTOINCREMENT,
        content    TEXT NOT NULL,
        created_at TEXT NOT NULL
    );
    """,
    # v10 — the permanent record.
    #
    # `messages` is the WORKING log: what she operates on, and what a factory reset
    # wipes. This is the ARCHIVE: every message ever, append-only, never cleared by
    # any admin operation. Testing needs a clean slate often; the record of what was
    # actually said should outlive that.
    #
    # Deliberately denormalized and FK-free — sessions get deleted, and an archive
    # that breaks when its parent row goes is not an archive. `era` increments on
    # each factory reset, so a future look-back can tell that a discontinuity
    # happened rather than reading across it as one continuous relationship.
    #
    # Seeded from the current messages table so nothing already on disk is lost.
    """
    CREATE TABLE message_archive (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        era           INTEGER NOT NULL DEFAULT 1,
        session_id    INTEGER,
        session_title TEXT,
        role          TEXT NOT NULL,
        content       TEXT NOT NULL,
        created_at    TEXT NOT NULL
    );
    CREATE INDEX idx_archive_era ON message_archive(era);
    CREATE INDEX idx_archive_session ON message_archive(era, session_id);
    INSERT INTO message_archive (era, session_id, session_title, role, content, created_at)
        SELECT 1, m.session_id, s.title, m.role, m.content, m.created_at
        FROM messages m LEFT JOIN sessions s ON s.id = m.session_id
        ORDER BY m.id;
    """,
]


def migrate(conn: sqlite3.Connection) -> None:
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    if version >= len(MIGRATIONS):
        return
    for i in range(version, len(MIGRATIONS)):
        logger.info("applying migration v%d", i + 1)
        conn.executescript(MIGRATIONS[i])
        conn.execute(f"PRAGMA user_version = {i + 1}")
    conn.commit()


def connect(path: str) -> sqlite3.Connection:
    # check_same_thread=False so the connection can be used from asyncio's
    # threadpool executor; SqliteStore serializes writes with one lock per
    # connection (see its docstring — per-store locks were not enough).
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    # NORMAL is the standard pairing with WAL: it drops the per-commit fsync that
    # FULL forces (every add_message commits) while still being crash-safe — WAL
    # keeps the log, so at worst a power loss costs the last transactions, not the DB.
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA busy_timeout = 5000")  # wait out a writer instead of raising
    conn.execute("PRAGMA foreign_keys = ON")
    migrate(conn)
    return conn
