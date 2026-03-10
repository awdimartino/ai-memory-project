# Imports
from openai import OpenAI as oai

# Python Standard
import json

# Files
from database import *
from config import *

class Embedder:
      def __init__(self, client):
            self.client = client
            self.embedding_cache = {}

      def get_embedding(self, text):
            text = text.replace("\n", " ")
            if text in self.embedding_cache:
                  return self.embedding_cache[text]
            response = self.client.embeddings.create(
                  input=[text],
                  model=EMBED_MODEL
            )
            self.embedding_cache[text] = response.data[0].embedding
            return self.embedding_cache[text]

class Memories:
      def __init__(self, client):
            self.client = client
            self.embedder = Embedder(client)
            self.get_embedding = self.embedder.get_embedding

      def classify_memories(self, type, conversation, query):
            response = self.client.chat.completions.create(
                  model=BRAIN_MODEL,
                  messages=[
                        {"role": "system", "content": type},
                        {"role": "user", "content": f"Previous conversation context: {conversation}\nQuery: {query}"}],
                  response_format=BRAIN_RESPONSE_FORMAT
                  )
            try:
                  results = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError as e:
                  if DEBUG_MODE: print(f"JSON parsing error: {e}")
                  results = {"create_memory": [], "fetch_memory": []}
            return results

      def fetch_memories(self, db, data):
            memories = ""
            for memory_entry in data.get("fetch_memory", []):
                  owner = memory_entry.get("owner")
                  category = memory_entry.get("category")
                  claim = memory_entry.get("claim")

                  if DEBUG_MODE: print(f"Fetching: Owner - {owner}, Category - {category}, Claim - {claim}")
                  embedded_claim = self.get_embedding(claim)
                  results = db.fetch_memory(embedded_claim, owner=owner, category=category)
                  memories += f"{str(results)}\n"
                  if DEBUG_MODE:
                        if results:
                              print(f"Found: '{results}' in memory\n")
                        else:
                              print(f"No match found for: '{claim}'\n")
            return memories

      def add_memories(self, db, data):
            for memory_entry in data.get("create_memory", []):
                  owner = memory_entry.get("owner")
                  category = memory_entry.get("category")
                  claim = memory_entry.get("claim")
                  embedding = self.get_embedding(claim)

                  if db.memory_exists(embedding, owner, category):
                        if DEBUG_MODE: print(f"Skipping duplicate: '{claim}'\n")
                        continue

                  result = db.create_memory(owner, category, claim, embedding)
                  if DEBUG_MODE: print(f"Saving: Owner - {owner}, Category - {category}, Claim - {claim}")
                  if DEBUG_MODE:
                        if result:
                              print(f"'{claim}' saved to memory\n")
                        else:
                              print(f"Failed to save '{claim}' to memory\n")