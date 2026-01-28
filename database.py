# Imports
from openai import OpenAI as oai
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

     # Create a new memory table
     def create_memory_table(self, name):
          sql = f"""
          CREATE TABLE IF NOT EXISTS {name} (
          id SERIAL PRIMARY KEY,
          category TEXT,
          memory TEXT,
          timestamp TIMESTAMP,
          embedding VECTOR(1024)
          );
          """

          try:
               self.cursor.execute(sql)
               return True
          except Exception as e:
               print(e)
               return False
     
     # Drop a table by name
     def drop_table(self, name):
          sql = f"DROP TABLE IF EXISTS {name};"

          try:
               self.cursor.execute(sql)
               return True
          except Exception as e:
               print(e)
               return False
     
     # Create a new memory entry in the memories table
     def create_memory(self, category, memory, embedding):
          sql = f"""
          INSERT INTO memories (category, memory, timestamp, embedding) VALUES (%s, %s, %s, %s)"""
          try:
               self.cursor.execute(sql, (category, memory, datetime.datetime.now(), embedding))
               return True
          except Exception as e:
               print(e)
               return False

     # Fetch memories from the memories table based on category and embedding similarity     
     def fetch_memory(self, category, query_embedding):
          sql = f"""
          SELECT memory, timestamp
          FROM memories
          WHERE category = '{category}'
          ORDER BY embedding <=> %s
          LIMIT 10;
          """
          try:
               self.cursor.execute(sql, (str(query_embedding),))
               return self.cursor.fetchall()
          except Exception as e:
               print(e)
               return False