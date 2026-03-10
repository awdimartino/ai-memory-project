# Imports
from concurrent.futures import ThreadPoolExecutor
import threading

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

import logging

logging.basicConfig(
      filename='debug.log',
      filemode='w',
      level=logging.DEBUG,
      format='%(asctime)s %(message)s',
      datefmt='%H:%M:%S'
)

def log(msg):
      if DEBUG_MODE:
            print(msg)
            logging.debug(msg)

class Chatbot:
      def __init__(self, client):
            self.client = client
            self.memory_manager = Memories(client)
            self.emotions = Emotions()

      def stream_query(self, query, conversation, memories="", last_thought="", display=True):
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
                  "content": f"What I was just thinking about (not said aloud):\n{last_thought}"
            }] if last_thought else []),
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
                  lines = []
                  lines.append(f"\n{'='*60}")
                  recent = messages[-3:]
                  lines.append(f"OUTGOING MESSAGES (showing {len(recent)} of {len(messages)} total)")
                  lines.append(f"{'='*60}")
                  for i, msg in enumerate(recent):
                        role = msg['role'].upper()
                        content = msg['content']
                        lines.append(f"\n[{i}] {role}")
                        lines.append(f"{'-'*40}")
                        lines.append(content)
                  lines.append(f"\n{'='*60}")
                  lines.append("EMOTIONAL STATE")
                  lines.append(f"{'-'*40}")
                  for channel, value in self.emotions.state.items():
                        lines.append(f"{channel:<12} {self.emotions.value_to_word(value)} ({value:.2f})")
                  lines.append(f"{'='*60}\n")
                  log("\n".join(lines))

            stream = self.client.chat.completions.create(
                  model=BOT_MODEL,
                  temperature=BOT_TEMPERATURE,
                  messages=messages,
                  stream=True
            )

            response = ""
            for chunk in stream:
                  if chunk.choices and chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        response += content
                        if display:
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
      lock = threading.Lock()
      tick_system = TickSystem(chatbot, db, conversation, interval=30, lock=lock)
      tick_system.start()
      executor = ThreadPoolExecutor(max_workers=2)
      
      while True:
            query = input(f"[{datetime.datetime.now().strftime('%A, %b %d at %I:%M %p')}] {USER_NAME}: \n")
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
                  user_results = chatbot.memory_manager.classify_memories(BRAIN_PROMPT_USER, conversation[-10:], query)
                  log(f"User classification: {user_results}\n")

                  user_memories = chatbot.memory_manager.fetch_memories(db, user_results)

                  print(f"[{datetime.datetime.now().strftime('%A, %b %d at %I:%M %p')}] {BOT_NAME}: ")

                  # ---- MODEL RESPONSE ----
                  chatbot.emotions.react(query)
                  bot_response = chatbot.stream_query(query, conversation, memories=user_memories, last_thought=tick_system.last_thought)
                  tick_system.last_thought = ""
                  chatbot.emotions.react(bot_response)
                  chatbot.memory_manager.add_memories(db, user_results, source_text=query)

                  executor.submit(
                        TickSystem._classify_and_save_bot,
                        chatbot, db, bot_response, conversation[:-10]
                  )

                  conversation.append({"role": "user", "content": query})
                  conversation.append({"role": "assistant", "content": bot_response})
                  tick_system.last_any_interaction = time.time()


if __name__ == "__main__":
      main()