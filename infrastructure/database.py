import psycopg2
import infrastructure.config as config
from pgvector.psycopg2 import register_vector
from psycopg2.extras import RealDictCursor


class DatabaseConnection:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=config.PG_HOST,
            database=config.PG_DATABASE,
            user=config.PG_USERNAME,
            password=config.PG_PASSWORD,
            port=config.PG_PORT
        )
        self.conn.autocommit = True
        register_vector(self.conn)
        self.conn.autocommit = False  # explicit control

    def execute(self, query, params=None):
        """Execute a write query (INSERT/UPDATE/DELETE)."""
        try:
            with self.conn.cursor() as cursor:
                cursor.execute(query, params)
            self.conn.commit()
            return True
        except Exception as e:
            self.conn.rollback()
            print(e)
            return False

    def fetch_all(self, query, params=None):
        """Execute a SELECT query and return all rows."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                return cursor.fetchall()
        except Exception as e:
            self.conn.rollback()
            print(e)
            return []

    def fetch_one(self, query, params=None):
        """Execute a SELECT query and return one row."""
        try:
            with self.conn.cursor(cursor_factory=RealDictCursor) as cursor:
                cursor.execute(query, params)
                return cursor.fetchone()
        except Exception as e:
            self.conn.rollback()
            print(e)
            return None

    def close(self):
        self.conn.close()