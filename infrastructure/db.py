"""SQLite connection + a minimal versioned migration runner.

Migrations use SQLite's built-in `PRAGMA user_version` as the schema version.
To evolve the schema, append a new statement to MIGRATIONS; never edit an
existing entry (that's what keeps upgrades deterministic across machines).
"""
import logging
import sqlite3

logger = logging.getLogger(__name__)

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
    # threadpool executor; the store serializes writes with a lock.
    conn = sqlite3.connect(path, check_same_thread=False)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA foreign_keys = ON")
    migrate(conn)
    return conn
