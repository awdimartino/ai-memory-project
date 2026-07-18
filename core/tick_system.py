import logging
import random
import threading
import time
from datetime import datetime, timezone

from core.companion import Companion
from core.prompt_builder import PromptBuilder

logger = logging.getLogger(__name__)


class TickSystem:
    """Background thread that decays emotion and (eventually) drives proactive
    thinking and unprompted messages during silence."""

    def __init__(self, companion: Companion, interval=30, lock=None):
        self.companion = companion
        self.interval = interval  # seconds between ticks
        self.last_user_interaction = datetime.now(timezone.utc)
        self.last_any_interaction = datetime.now(timezone.utc)
        self.running = False
        self.lock = lock or threading.Lock()
        self.last_thought = ""

    def start(self):
        """Start the background tick loop in a daemon thread."""
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
        """Run one tick unless the chat turn currently holds the lock."""
        if not self.lock.acquire(blocking=False):
            logger.debug("tick skipped (chat turn in progress)")
            return
        try:
            logger.debug("tick")
            self.emotion_decay_tick()
            self.think_tick()
            self.unprompted_message_tick()
        finally:
            self.lock.release()

    def emotion_decay_tick(self):
        """Decay the bot's emotional state over time."""
        self.companion.decay()

    def unprompted_message_tick(self):
        """Maybe reach out during user silence. (Phase 6 — not yet sending.)"""
        minutes_since_user = self.companion.minutes_since_user_interaction()
        minutes_since_any = self.companion.minutes_since_any_interaction()

        if minutes_since_user is None or minutes_since_any is None:
            return

        # More likely to reach out the longer the user has been silent.
        probability = min(0.1 * (1 + minutes_since_user * 0.1), 0.6)
        if random.random() > probability:
            return
        if minutes_since_user < 5:
            return
        # Don't pile on if the bot just spoke.
        if minutes_since_any < 2:
            return

        prompt = PromptBuilder.build_unprompted_prompt(
            minutes_since_user=int(minutes_since_user),
            minutes_since_any=int(minutes_since_any),
        )
        # TODO (Phase 6): send `prompt` to the LLM and deliver the reply.

    def think_tick(self):
        """Reflect internally during idle time. (Phase 6 — not yet sending.)"""
        prompt = PromptBuilder.build_thought_prompt(self.last_thought)
        # TODO (Phase 6): send `prompt` to the LLM and store the thought.
