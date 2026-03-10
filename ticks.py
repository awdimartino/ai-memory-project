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
            f"with a single short unprompted message. "
            f"Only do so if it feels genuinely motivated — not forced. "
            f"If nothing feels worth saying, respond with exactly: [SKIP]"
        )
        
        response = self.chatbot.stream_query(prompt, self.conversation, memories="")
        
        if "[SKIP]" not in response:
            print(f"\n[{BOT_NAME}]: {response}\n")
            self.conversation.append({"role": "assistant", "content": response})
            self.last_interaction = time.time()