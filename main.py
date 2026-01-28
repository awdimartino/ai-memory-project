# Imports
from pyexpat.errors import messages
from urllib import response
from openai import OpenAI as oai

# Python Standard
import json
import time
import datetime
import os

# Files
from database import *
from config import *
from speaker import *

DEBUG_MODE = True

class Chatbot:
      def __init__(self):
            self.speaker = Speaker()
            self.memory_manager = MemoryManager()
            self.embedding_manager = EmbeddingManager()
            self.sentiment_analyzer = SentimentAnalyzer()
            
            self.client = oai(
                  base_url=AI_BASE_URL,
                  api_key=AI_API_KEY # Value does not matter on localhost
            )


      def stream_query(self, memories, query, emotion, conversation):
            speech_buffer = SpeechBuffer()
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

            stream = self.client.chat.completions.create(
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
                        # Add tts buffer 
                        speech_buffer.add(content)
                        for sentence in speech_buffer.pop_ready():
                              self.speaker.speak(sentence + " ")  # space improves prosody

            # Flush leftovers
            for sentence in speech_buffer.flush():
                  self.speaker.speak(sentence)
            print("\n")
            return response

class EmbeddingManager:
      def __init__(self):
            self.client = oai(
                  base_url=AI_BASE_URL,
                  api_key=AI_API_KEY # Value does not matter on localhost
            )
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

class MemoryManager:
      def __init__(self):
            self.client = oai(
                  base_url=AI_BASE_URL,
                  api_key=AI_API_KEY #
            )
            embedding_manger = EmbeddingManager()
            self.get_embedding = embedding_manger.get_embedding

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
            for memory_entry in data.get("fetch_memory", []):  # Safely get the list or default to empty
                  category = memory_entry.get("category")
                  claim = memory_entry.get("claim")

                  if(DEBUG_MODE): print(f"Fetching: Category - {category}, Claim - {claim}")
                  embedded_claim = self.get_embedding(claim)
                  results = db.fetch_memory(category, embedded_claim)
                  memories += f"{str(results)}\n"
                  if(DEBUG_MODE):
                        if results:
                              print(f"Found: '{results}' in memory\n")
                        else:
                              print(f"No match found for: '{results}'\n")
            return memories

      def add_memories(self, db, data):
            for memory_entry in data.get("create_memory", []):  # Safely get the list or default to empty
                  category = memory_entry.get("category")
                  claim = memory_entry.get("claim")
                  result = db.create_memory(category, claim, self.get_embedding(claim))
                  if(DEBUG_MODE): print(f"Saving: Category - {category}, Claim - {claim}")
                  if(DEBUG_MODE):
                        if (result):
                              print(f"'{claim}' saved to memory\n")
                        else:
                              print(f"Failed to save '{claim}' to memory\n")
            return

class SentimentAnalyzer:
      def __init__(self):
            self.client = oai(
                  base_url=AI_BASE_URL,
                  api_key=AI_API_KEY #
            )

      def analyze_emotion(self, query):
            response = self.client.chat.completions.create(
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

      db = Database()
      db.create_memory_table("memories")

      chatbot = Chatbot()
      
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
                  db.drop_table("memories")
                  db.create_memory_table("memories")
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

            user_results = chatbot.memory_manager.classify_memories(BRAIN_PROMPT_USER, conversation, query)
            now = time.perf_counter()
            if(DEBUG_MODE): print(f"User Memory Classification: {now - last_time:.2f}s\n")
            last_time = now

            user_memories = chatbot.memory_manager.fetch_memories(db, user_results)
            now = time.perf_counter()
            if(DEBUG_MODE): print(f"User Memory Fetch: {now - last_time:.2f}s\n")
            last_time = now

            if overall_cycles % 2 == 0:
                  emotion = chatbot.sentiment_analyzer.analyze_emotion(query)
                  now = time.perf_counter()
                  if(DEBUG_MODE): print(f"Sentiment Analysis: {now - last_time:.2f}s\n")
                  last_time = now
            else:
                  emotion = emotion  # reuse last detected emotion

            print(f"[{str(datetime.datetime.now().strftime("%A, %b %d at %I:%M %p"))}] {BOT_NAME}: ")
            # ---- MODEL RESPONSE ----
            bot_response = chatbot.stream_query(
                  memories=user_memories,
                  conversation=conversation,
                  emotion=emotion,
                  query=query
            )

            now = time.perf_counter()
            if(DEBUG_MODE): print(f"Streaming: {now - last_time:.2f}s\n")
            last_time = now

            chatbot.memory_manager.add_memories(db, user_results)
            now = time.perf_counter()
            if(DEBUG_MODE): print(f"User Memory Saving: {now - last_time:.2f}s\n")
            last_time = now

            # ---- ASSISTANT TURN ----
            conversation.append({"role": "assistant", "content": bot_response, "datetime": str(datetime.datetime.now().strftime("%A, %b %d at %I:%M %p"))})

            bot_results = chatbot.memory_manager.classify_memories(BRAIN_PROMPT_BOT, [], bot_response)
            now = time.perf_counter()
            if(DEBUG_MODE): print(f"Bot Memory Classification: {now - last_time:.2f}s\n")
            last_time = now

            chatbot.memory_manager.add_memories(db, bot_results)
            now = time.perf_counter()
            if(DEBUG_MODE): print(f"Bot Memory Saving: {now - last_time:.2f}s")

            # ---- TRIM CONTEXT ----
            if len(conversation) > MAX_TURNS * 2:
                  conversation = conversation[-MAX_TURNS * 2:]

            total_time = now - start_time
            overall_time += total_time 
            overall_cycles += 1
            if(DEBUG_MODE): print(f"Total time: {total_time:.2f}s\n")

      db.close_connection()

if __name__ == "__main__":
    main()