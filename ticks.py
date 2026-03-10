import threading
import time
import datetime
import random
from config import *

class TickSystem:
    def __init__(self, chatbot, db, conversation, interval=30, lock=None):
        self.chatbot = chatbot
        self.db = db
        self.conversation = conversation
        self.interval = interval  # seconds between ticks
        self.last_user_interaction = time.time()   # only updated when user sends a message
        self.last_any_interaction = time.time()    # updated when either party speaks   
        self.running = False
        self.lock = lock
        self.last_thought = ""

    def start(self):
        self.running = True
        thread = threading.Thread(target=self._loop, daemon=True)
        thread.start()

    def stop(self):
        self.running = False

    def _loop(self):
        while self.running:
            time.sleep(self.interval)
            self.tick()

    def tick(self):
        if not self.lock.acquire(blocking=False):  # skip tick if main loop is busy
            if DEBUG_MODE: print("[TICK] Skipped — main loop active\n")
            return
        try:
            if DEBUG_MODE: print("[TICK] Tick triggered\n")
            self.emotion_decay_tick()
            self.think_tick()
            self.unprompted_message_tick()
        finally:
            self.lock.release()

    def emotion_decay_tick(self):
        self.chatbot.emotions.decay()

    def unprompted_message_tick(self):
        minutes_since_user = (time.time() - self.last_user_interaction) / 60
        minutes_since_any = (time.time() - self.last_any_interaction) / 60

        # Trigger based on user silence
        probability = min(0.1 * (1 + minutes_since_user * 0.1), 0.6)

        if random.random() > probability:
            return
        if minutes_since_user < 5:
            return

        # But don't send another unprompted message if bot just spoke
        if minutes_since_any < 2:
            return

        # give the bot context to generate something meaningful
        prompt = (
            f"You haven't heard from {USER_NAME} in {int(minutes_since_user)} minutes. "
            f"You haven't spoken on your own in {int(minutes_since_any)} minutes. "
            f"Based on your emotional state and memories, you may optionally reach out "
            f"with a single short unprompted message. Speak directly to {USER_NAME} as if you were messaging them like usual"
            f"Only do so if it feels genuinely motivated — not forced. "
            f"If nothing feels worth saying, respond with exactly: [SKIP]"
        )
        
        response = self.chatbot.stream_query(prompt, self.conversation, memories="")
        
        if "[SKIP]" not in response:
            print(f"\n[{BOT_NAME}]: {response}\n")
            self.conversation.append({"role": "assistant", "content": response})
            self.last_any_interaction = time.time()

    def think_tick(self):
        prompt = (
            f"You are {BOT_NAME}. You are alone with your thoughts.\n"
            f"Your last thought was '{self.last_thought}'\n"
            f"Based on your recent conversation, thoughts, and emotional state, "
            f"reflect privately. What are you thinking about? "
            f"Your thoughts don't have to be about {USER_NAME} — they can be about anything on your mind.\n"
            f"Respond in first person as an internal monologue, 2-3 sentences max. "
            f"Do not reference any events, conflicts, or history not present in the conversation above. "
            f"If the conversation is short or neutral, your thoughts should reflect that — not manufacture drama. \n"
            f"If you don't have any meaningful thoughts, respond with exactly this, and nothing else: '[SKIP]'"
        )
        if random.random() < 0.20:  # Only think 40% of the time to avoid overloading with thoughts
            return
        thought = self.chatbot.stream_query(prompt, conversation=self.conversation, memories="", display=False)

        if not thought or "SKIP" in thought:
            return

        if DEBUG_MODE: print(f"\n[THINK] {thought}\n")

        # Update emotional state based on the thought
        self.last_thought = thought
        self.chatbot.emotions.react(thought)

        # Save as a bot belief memory
        embedding = self.chatbot.memory_manager.get_embedding(thought)
        if not self.db.memory_exists(embedding, owner="bot", category="belief"):
            self.db.create_memory("bot", "belief", thought, embedding)

    @staticmethod
    def _classify_and_save_bot(chatbot, db, bot_response, conversation):
        bot_results = chatbot.memory_manager.classify_memories(BRAIN_PROMPT_BOT, conversation, bot_response)
        if DEBUG_MODE: print(f"Bot classification: {bot_results}\n")
        chatbot.memory_manager.add_me#mories(db, bot_results, source_text=bot_response) 