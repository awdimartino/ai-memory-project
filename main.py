# Imports
import threading
from urllib import response

from openai import OpenAI as oai

# Python Standard
import datetime
import os
import time

# Files
from config import *
from emotions import Emotions
from memories import Memories
from database import Database
from ticks import TickSystem

DEBUG_MODE = True

class Chatbot:
      def __init__(self, client):
            self.client = client
            self.memory_manager = Memories(client)
            self.emotions = Emotions()

      def stream_query(self, query, conversation, memories=""):
            messages = [
            {
                  "role": "system",
                  "content": (
                        f"{BOT_PROMPT}\n\n"
                        f"Current date and time: {datetime.datetime.now().strftime('%A, %b %d at %I:%M %p')}\n\n"
                        f"{self.emotions.as_prompt()}"
                  )
            },
            *([{
                  "role": "assistant",
                  "content": f"CONFIRMED MEMORIES ONLY — do not reference anything about {USER_NAME} "
                              f"or your relationship that is not listed here. If the list is empty, you know nothing about them yet:\n{memories}"
            }] if memories else []),
            *conversation,
            {
                  "role": "user",
                  "content": query
            }
            ]

            if DEBUG_MODE:
                  print(f"\n{'='*60}")
                  print(f"OUTGOING MESSAGES ({len(messages)} total)")
                  print(f"{'='*60}")
                  for i, msg in enumerate(messages):
                        role = msg['role'].upper()
                        content = msg['content']
                        print(f"\n[{i}] {role}")
                        print(f"{'-'*40}")
                        print(content)
                  print(f"\n{'='*60}\n")

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

            print("\n")
            return response


def main():
      global DEBUG_MODE

      db = Database()
      db.create_memory_table()

      client = oai(
            base_url=AI_BASE_URL,
            api_key=AI_API_KEY
      )
      chatbot = Chatbot(client)
      conversation = []
      # Create a lock for synchronizing access to shared resources
      lock = threading.Lock()  
      # Initialize background tick system
      tick_system = TickSystem(chatbot, db, conversation, interval=30, lock=lock)
      tick_system.start()
      
      while True:
            query = input(f"[{datetime.datetime.now().strftime('%A, %b %d at %I:%M %p')}] {USER_NAME}: \n")
            # Update last_interaction whenever user sends a message
            tick_system.last_user_interaction = time.time()
            tick_system.last_any_interaction = time.time()
            print()

            if query.strip().lower() == "/exit":
                  break

            if query.strip().lower() == "/reset":
                  conversation = []
                  db.drop_table()
                  db.create_memory_table()
                  os.system('cls||clear')
                  print("Conversation reset.\n")
                  continue

            if query.strip().lower() == "/debug":
                  DEBUG_MODE = not DEBUG_MODE
                  print(f"Debug mode set to {DEBUG_MODE}\n")
                  continue

            # ---- USER TURN ----
            with lock:
                  user_results = chatbot.memory_manager.classify_memories(BRAIN_PROMPT_USER, conversation, query)
                  if DEBUG_MODE: print(f"User classification: {user_results}\n")

                  user_memories = chatbot.memory_manager.fetch_memories(db, user_results)

                  print(f"[{datetime.datetime.now().strftime('%A, %b %d at %I:%M %p')}] {BOT_NAME}: ")

                  # ---- MODEL RESPONSE ----
                  chatbot.emotions.react(query)  # Update emotional state based on user input
                  bot_response = chatbot.stream_query(query, conversation, memories=user_memories)
                  chatbot.emotions.react(bot_response)  # Update emotional state based on bot response
                  chatbot.memory_manager.add_memories(db, user_results)

                  bot_results = chatbot.memory_manager.classify_memories(BRAIN_PROMPT_BOT, [], bot_response)
                  if DEBUG_MODE: print(f"Bot classification: {bot_results}\n")

                  chatbot.memory_manager.add_memories(db, bot_results)

                  conversation.append({"role": "user", "content": query})
                  conversation.append({"role": "assistant", "content": bot_response})
                  tick_system.last_any_interaction = time.time()


if __name__ == "__main__":
      main()