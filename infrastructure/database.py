import psycopg2
import infrastructure.config as config

class DatabaseConnection:
    def __init__(self):
        self.conn = psycopg2.connect(
            host=config.PG_HOST,
            database=config.PG_DATABASE,
            user=config.PG_USERNAME,
            password=config.PG_PASSWORD,
            port=config.PG_PORT
        )
        self.cursor = self.conn.cursor()

    def execute(self, query, params=None):
        """Execute a SQL query with optional parameters."""
        self.cursor.execute(query, params)

    def fetchall(self):
        """Fetch all results from the last executed query."""
        return self.cursor.fetchall()

    def commit(self):
        """Commit the current transaction."""
        self.conn.commit()

    def close(self):
        """Close the cursor and connection."""
        self.cursor.close()
        self.conn.close()