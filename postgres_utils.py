from openai import OpenAI as oai
import time
import pgvector
import psycopg2
import datetime

def create_connection():
     connection = psycopg2.connect(
          host="localhost",
          database="postgres",
          user="postgres",
          password="admin",
          port="5432"
     )
     connection.autocommit=True
     cursor = connection.cursor()

     return connection, cursor

def create_table(cursor, name):
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
          cursor.execute(sql)
          return True
     except Exception as e:
          print(e)
          return False
    
def create_memory(cursor, category, memory, embedding):
     sql = f"""
     INSERT INTO memories (category, memory, timestamp, embedding) VALUES (%s, %s, %s, %s)"""
     try:
          cursor.execute(sql, (category, memory, datetime.datetime.now(), embedding))
          return True
     except Exception as e:
          print(e)
          return False
     
def fetch_memory(cursor, category, query_embedding):
     sql = f"""
     SELECT memory, timestamp
     FROM memories
     WHERE category = '{category}'
     ORDER BY embedding <=> %s
     LIMIT 10;
     """
     try:
          cursor.execute(sql, (str(query_embedding),))
          return cursor.fetchall()
     except Exception as e:
          print(e)
          return False