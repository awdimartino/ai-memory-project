# Imports
from openai import OpenAI as oai
import pgvector
import psycopg2

# Python Standard
import json
import time
import datetime

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
      stream = client.chat.completions.create(
            model=BOT_MODEL,
            messages=[
                  {"role": "system", "content": BOT_PROMPT}, 
                  {"role": "user", "content": ("/no_think\nCurrent Time: " + str(datetime.datetime.now()) + "\nRecent Conversation: " + str(conversation) + "\nMemories: " + str(memories) + "\n" + "Current Emotion: " + emotion + "\n" + USER_NAME + ": " + query)}], # FORMAT NEEDS TO BE CHANGED
            stream=True # Enable streaming
      )
      response = ""
      # Process the response chunks as they arrive
      for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content is not None:
                  content = chunk.choices[0].delta.content
                  response += content
                  for char in content:
                        print(char, end='', flush=True)
      print()
      print()
      return response

def get_embedding(text):
      text = text.replace("\n", " ")
      response = client.embeddings.create(
            input=[text],
            model=EMBED_MODEL
      )
      return response.data[0].embedding

def classify_memories(type, conversation, query):
      response = client.chat.completions.create(
            model=BRAIN_MODEL,
            messages=[
                  {"role": "system", "content": type},
                  {"role": "user", "content": f"Previous conversation context: {conversation}\nQuery: {query}"}],
            response_format=BRAIN_RESPONSE_FORMAT
            )
      results = json.loads(response.choices[0].message.content)
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
      connection, cursor = create_connection()
      create_table(cursor, "memories")
      
      conversation = []
      MAX_TURNS = 6

      overall_time = 0.0
      overall_cycles = 0

      while True:
            query = input(f"[{str(datetime.datetime.now())}] {USER_NAME}: \n")
            print()

            if query.strip().lower() == "/exit":
                  print(f"\nAverage Response Time: {overall_time / overall_cycles:.2f} seconds")
                  break

            if query.strip().lower() == "/reset":
                  conversation = []
                  drop_table(cursor, "memories")
                  create_table(cursor, "memories")
                  print("Conversation reset.\n")
                  continue

            if query.strip().lower() == "/debug":
                  global DEBUG_MODE
                  DEBUG_MODE = not DEBUG_MODE
                  print(f"Debug mode set to {DEBUG_MODE}\n")
                  continue

            start_time = time.perf_counter()
            last_time = start_time

            # ---- USER TURN ----
            conversation.append({"role": "user", "content": query, "datetime": str(datetime.datetime.now())})

            user_results = classify_memories(BRAIN_PROMPT_USER, conversation, f"{USER_NAME}: {query}")
            now = time.perf_counter()
            if(DEBUG_MODE): print(f"User Memory Classification: {now - last_time:.2f}s\n")
            last_time = now

            user_memories = fetch_memories(cursor, user_results)
            now = time.perf_counter()
            if(DEBUG_MODE): print(f"User Memory Fetch: {now - last_time:.2f}s\n")
            last_time = now

            emotion = analyze_emotion(query)
            now = time.perf_counter()
            if(DEBUG_MODE): print(f"Sentiment Analysis: {now - last_time:.2f}s\n")
            last_time = now

            print(f"[{str(datetime.datetime.now())}] {BOT_NAME}: ")
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
            conversation.append({"role": "assistant", "content": bot_response, "datetime": str(datetime.datetime.now())})

            bot_results = classify_memories(BRAIN_PROMPT_BOT, conversation, f"{BOT_NAME}: + {bot_response}")
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