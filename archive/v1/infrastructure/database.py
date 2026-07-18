import sqlite3

import sqlite_vec

import infrastructure.config as config


class DatabaseConnection:
    """Embedded SQLite connection with the sqlite-vec extension loaded.

    There is no separate database server to launch — the whole store lives in a
    single local file (config.DB_PATH). Parameter placeholders are '?'.
    """

    def __init__(self, db_path=None):
        self.conn = sqlite3.connect(
            db_path or config.DB_PATH,
            check_same_thread=False,  # tick system runs on a background thread
        )
        self.conn.row_factory = sqlite3.Row

        # Load the sqlite-vec extension (provides vec0 virtual tables + KNN).
        self.conn.enable_load_extension(True)
        sqlite_vec.load(self.conn)
        self.conn.enable_load_extension(False)

        # Enforce foreign keys (messages -> conversations ON DELETE CASCADE).
        self.conn.execute("PRAGMA foreign_keys = ON")

    def execute(self, query, params=None):
        """Execute a write query (INSERT/UPDATE/DELETE) or DDL. Returns success."""
        try:
            self.conn.execute(query, params or ())
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print(e)
            return False

    def executescript(self, script):
        """Execute a multi-statement SQL script (used for schema setup)."""
        try:
            self.conn.executescript(script)
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print(e)
            return False

    def execute_returning_id(self, query, params=None):
        """Execute an INSERT and return the new row's integer id (or None)."""
        try:
            cursor = self.conn.execute(query, params or ())
            self.conn.commit()
            return cursor.lastrowid
        except Exception as e:
            self.conn.rollback()
            print(e)
            return None

    def fetch_all(self, query, params=None):
        """Execute a SELECT and return all rows (each row indexable by column)."""
        try:
            cursor = self.conn.execute(query, params or ())
            return cursor.fetchall()
        except Exception as e:
            print(e)
            return []

    def fetch_one(self, query, params=None):
        """Execute a SELECT and return a single row (or None)."""
        try:
            cursor = self.conn.execute(query, params or ())
            return cursor.fetchone()
        except Exception as e:
            print(e)
            return None

    def close(self):
        self.conn.close()
