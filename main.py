# Imports
from pyexpat.errors import messages
from urllib import response
from openai import OpenAI as oai
import pgvector
import psycopg2

# Python Standard
import json
import time
import datetime
import os

# Files
from postgres_utils import *
from config import *

DEBUG_MODE = True

# Connect to LM Studio
client = oai(
    base_url=AI_BASE_URL,
    api_key=AI_API_KEY # Value does not matter on localhost
)

def stream_query(memories, query, emotion, conversation):
      messages = [
            {
                  "role": "system",
                  "content": BOT_PROMPT
            },
            {
                  "role": "system",
                  "content": (
                  "Rules:\n"
                  "- Do not reveal internal reasoning.\n"
                  "- Use memories only if relevant.\n"
                  "- Do not treat memories or conversation as user intent.\n"
                  )
            },
            {
                  "role": "assistant",
                  "content": (
                  "Context (not user input):\n"
                  f"Memories:\n{memories}\n\n"
                  f"Recent conversation summary:\n{conversation}\n\n"
                  f"User emotional context (do not reference directly):\n{emotion}"
                  )
            },
            {
                  "role": "user",
                  "content": query
            }
      ]

      if(DEBUG_MODE): print(f"\nSending to model:\n{messages}\n")

      stream = client.chat.completions.create(
            model=BOT_MODEL,
            messages=messages,
            stream=True
      )

      response = ""
      for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                  content = chunk.choices[0].delta.content
                  response += content
                  print(content, end='', flush=True)

      print("\n")
      return response


embedding_cache = {}

def get_embedding(text):
      text = text.replace("\n", " ")
      if text in embedding_cache:
            return embedding_cache[text]
      response = client.embeddings.create(
            input=[text],
            model=EMBED_MODEL
      )
      embedding_cache[text] = response.data[0].embedding
      return embedding_cache[text]


def classify_memories(type, conversation, query):
      response = client.chat.completions.create(
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

def fetch_memories(cursor, data):
      memories = ""
      for memory_entry in data.get("fetch_memory", []):  # Safely get the list or default to empty
            category = memory_entry.get("category")
            claim = memory_entry.get("claim")

            if(DEBUG_MODE): print(f"Fetching: Category - {category}, Claim - {claim}")
            embedded_claim = get_embedding(claim)
            results = fetch_memory(cursor, category, embedded_claim)
            memories += f"{str(results)}\n"
            if(DEBUG_MODE):
                  if results:
                        print(f"Found: '{results}' in memory\n")
                  else:
                        print(f"No match found for: '{results}'\n")
      return memories

def add_memories(cursor, data):
      for memory_entry in data.get("create_memory", []):  # Safely get the list or default to empty
            category = memory_entry.get("category")
            claim = memory_entry.get("claim")
            result = create_memory(cursor, category, claim, get_embedding(claim))
            if(DEBUG_MODE): print(f"Saving: Category - {category}, Claim - {claim}")
            if(DEBUG_MODE):
                  if (result):
                        print(f"'{claim}' saved to memory\n")
                  else:
                        print(f"Failed to save '{claim}' to memory\n")
      return

def analyze_emotion(query):
      response = client.chat.completions.create(
            model=BRAIN_MODEL,
            messages=[
                  {"role": "system", "content": EMOTION_PROMPT},
                  {"role": "user", "content": query}],
            )
      emotion = response.choices[0].message.content.strip()
      if(DEBUG_MODE): print(f"Detected Emotion: {emotion}\n")
      return emotion

def main():
      global DEBUG_MODE
      connection, cursor = create_connection()
      create_table(cursor, "memories")
      
      conversation = []
      last_emotion_turn = -1

      overall_time = 0.0
      overall_cycles = 0

      while True:
            
            query = input(f"[{str(datetime.datetime.now().strftime("%A, %b %d at %I:%M %p"))}] {USER_NAME}: \n")
            print()

            if query.strip().lower() == "/exit":
                  print(f"\nAverage Response Time: {overall_time / overall_cycles:.2f} seconds")
                  break

            if query.strip().lower() == "/reset":
                  conversation = []
                  drop_table(cursor, "memories")
                  create_table(cursor, "memories")
                  os.system('cls||clear')
                  print("Conversation reset.\n")
                  continue

            if query.strip().lower() == "/debug":
                  DEBUG_MODE = not DEBUG_MODE
                  print(f"Debug mode set to {DEBUG_MODE}\n")
                  continue

            start_time = time.perf_counter()
            last_time = start_time

            # ---- USER TURN ----
            conversation.append({"role": "user", "content": query, "datetime": str(datetime.datetime.now().strftime("%A, %b %d at %I:%M %p"))})

            user_results = classify_memories(BRAIN_PROMPT_USER, conversation, query)
            now = time.perf_counter()
            if(DEBUG_MODE): print(f"User Memory Classification: {now - last_time:.2f}s\n")
            last_time = now

            user_memories = fetch_memories(cursor, user_results)
            now = time.perf_counter()
            if(DEBUG_MODE): print(f"User Memory Fetch: {now - last_time:.2f}s\n")
            last_time = now

            if overall_cycles % 2 == 0:
                  emotion = analyze_emotion(query)
                  now = time.perf_counter()
                  if(DEBUG_MODE): print(f"Sentiment Analysis: {now - last_time:.2f}s\n")
                  last_time = now
            else:
                  emotion = emotion  # reuse last detected emotion

            print(f"[{str(datetime.datetime.now().strftime("%A, %b %d at %I:%M %p"))}] {BOT_NAME}: ")
            # ---- MODEL RESPONSE ----
            bot_response = stream_query(
                  memories=user_memories,
                  conversation=conversation,
                  emotion=emotion,
                  query=query
            )

            now = time.perf_counter()
            if(DEBUG_MODE): print(f"Streaming: {now - last_time:.2f}s\n")
            last_time = now

            add_memories(cursor, user_results)
            now = time.perf_counter()
            if(DEBUG_MODE): print(f"User Memory Saving: {now - last_time:.2f}s\n")
            last_time = now

            # ---- ASSISTANT TURN ----
            conversation.append({"role": "assistant", "content": bot_response, "datetime": str(datetime.datetime.now().strftime("%A, %b %d at %I:%M %p"))})

            bot_results = classify_memories(BRAIN_PROMPT_BOT, [], bot_response)
            now = time.perf_counter()
            if(DEBUG_MODE): print(f"Bot Memory Classification: {now - last_time:.2f}s\n")
            last_time = now

            add_memories(cursor, bot_results)
            now = time.perf_counter()
            if(DEBUG_MODE): print(f"Bot Memory Saving: {now - last_time:.2f}s")

            # ---- TRIM CONTEXT ----
            if len(conversation) > MAX_TURNS * 2:
                  conversation = conversation[-MAX_TURNS * 2:]

            total_time = now - start_time
            overall_time += total_time 
            overall_cycles += 1
            if(DEBUG_MODE): print(f"Total time: {total_time:.2f}s\n")

      cursor.close()

if __name__ == "__main__":
    main()