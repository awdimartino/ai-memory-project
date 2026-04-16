import threading
import time
import datetime
import random
from deprecated.config_old import *
from core.companion import Companion

class TickSystem:
    def __init__(self, companion: Companion, interval=30, lock=None):
        self.companion = companion
        self.interval = interval  # seconds between ticks
        self.last_user_interaction = time.time()   # only updated when user sends a message
        self.last_any_interaction = time.time()    # updated when either party speaks   
        self.running = False
        self.lock = lock
        self.last_thought = ""

    def start(self):
        """Start the background tick system in a separate thread."""
        self.running = True
        thread = threading.Thread(target=self._loop, daemon=True)
        thread.start()

    def stop(self):
        """Stop the background tick system."""
        self.running = False

    def _loop(self):
        """Main loop for the tick system."""
        while self.running:
            time.sleep(self.interval)
            self.tick()

    def tick(self):
        """Perform a tick, which may involve emotional decay, thinking, or sending an unprompted message based on timing and probabilities."""
        # If locked, skip this tick
        if not self.lock.acquire(blocking=False):
            return
        try:
            self.emotion_decay_tick()
            self.think_tick()
            self.unprompted_message_tick()
        finally:
            self.lock.release()

    def emotion_decay_tick(self):
        """Decay the bot's emotional state over time."""
        self.companion.emotion_manager.decay()

    def unprompted_message_tick(self):
        minutes_since_user = self.companion.conversation_manager.minutes_since_user_interaction()
        minutes_since_any = self.companion.conversation_manager.minutes_since_any_interaction()

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
        pass

    def think_tick(self):
        prompt = (
            f"What have you been thinking about lately, unrelated to {USER_NAME}? "
            f"What's something that's been on your mind today? "
            f"What do you actually think about the last conversation you had? "
            f"What are you curious about right now? "
            f"If you don't want to think about anything, just say [SKIP]. "
            f"YOU ARE NOT TALKING TO {USER_NAME} — this is just for you to reflect and process your own thoughts and feelings. "
        )
        pass