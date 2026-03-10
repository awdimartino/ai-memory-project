# Imports
import psycopg2
import datetime

# Files
from config import *

# Database Class for PostgreSQL interactions
class Database:
     def __init__(self):
          self.connection, self.cursor = self.create_connection()
     
     # Create a connection to the PostgreSQL database
     def create_connection(self):
          connection = psycopg2.connect(
               host=PG_HOST,
               database=PG_DATABASE,
               user=PG_USERNAME,
               password=PG_PASSWORD,
               port=PG_PORT
          )
          connection.autocommit=True
          cursor = connection.cursor()
          return connection, cursor
     
     def close_connection(self):
          self.cursor.close()
          self.connection.close()

     # Create the memory table
     def create_memory_table(self):
          sql = """
          CREATE TABLE IF NOT EXISTS memories (
               id              SERIAL PRIMARY KEY,
               owner           TEXT NOT NULL,
               category        TEXT NOT NULL,
               memory          TEXT NOT NULL,
               embedding       VECTOR(1024),
               importance      FLOAT DEFAULT 0.5,
               access_count    INT DEFAULT 0,
               timestamp       TIMESTAMPTZ DEFAULT NOW(),
               last_accessed   TIMESTAMPTZ
          );
          CREATE INDEX IF NOT EXISTS memories_embedding_idx
               ON memories USING ivfflat (embedding vector_cosine_ops);
          CREATE INDEX IF NOT EXISTS memories_owner_category_idx
               ON memories (owner, category);
          """
          try:
               self.cursor.execute(sql)
               return True
          except Exception as e:
               print(e)
               return False
     
     # Drop the memory table
     def drop_table(self):
          sql = "DROP TABLE IF EXISTS memories;"
          try:
               self.cursor.execute(sql)
               return True
          except Exception as e:
               print(e)
               return False
     
     # Create a new memory entry in the memories table
     def create_memory(self, owner, category, memory, embedding, importance=0.5):
          sql = """
          INSERT INTO memories (owner, category, memory, embedding, importance)
          VALUES (%s, %s, %s, %s, %s)
          """
          try:
               self.cursor.execute(sql, (owner, category, memory, str(embedding), importance))
               return True
          except Exception as e:
               print(e)
               return False
          
     # Search if a memory exists
     def memory_exists(self, query_embedding, owner=None, category=None, threshold=0.92):
          conditions = ["1 - (embedding <=> %s) > %s"]
          params = [str(query_embedding), threshold]

          if owner:
               conditions.append("owner = %s")
               params.append(owner)
          if category:
               conditions.append("category = %s")
               params.append(category)

          where = " AND ".join(conditions)
          sql = f"SELECT EXISTS (SELECT 1 FROM memories WHERE {where});"
          try:
               self.cursor.execute(sql, params)
               return self.cursor.fetchone()[0]
          except Exception as e:
               print(e)
               return False

     # Fetch memories from the memories table based on category and embedding similarity     
     def fetch_memory(self, query_embedding, owner=None, category=None, threshold=0.7, limit=3):
          conditions = ["1 - (embedding <=> %s) > %s"]
          params = [str(query_embedding), threshold]

          if owner:
               conditions.append("owner = %s")
               params.append(owner)
          if category:
               conditions.append("category = %s")
               params.append(category)

          where = " AND ".join(conditions)
          sql = f"""
          SELECT memory, owner, category
          FROM memories
          WHERE {where}
          ORDER BY embedding <=> %s
          LIMIT %s;
          """
          params += [str(query_embedding), limit]
          try:
               self.cursor.execute(sql, params)
               return self.cursor.fetchall()
          except Exception as e:
               print(e)
               return []
