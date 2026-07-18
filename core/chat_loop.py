from core.companion import Companion
from core.tick_system import TickSystem


class ChatLoop:
    """Runs the main chat REPL and coordinates with the background tick system."""

    def __init__(self, companion: Companion, tick_system: TickSystem):
        self.companion = companion
        self.tick_system = tick_system

    def start(self):
        """Main loop: read user input, respond, then classify memories."""
        self.tick_system.start()

        while True:
            self.companion.ensure_conversation()

            try:
                query = input("You: ")
            except (EOFError, KeyboardInterrupt):
                print("\nbye!")
                self.companion.flush_pending_classification()
                self.tick_system.stop()
                break

            if not query.strip():
                continue

            # Hold the tick lock so a background tick can't interleave the turn.
            with self.tick_system.lock:
                self.companion.respond(query)

            # Classification runs in batches (config.CLASSIFY_BATCH_SIZE), not
            # every turn — the active conversation already stays in the response
            # prompt's context window, so per-turn extraction is redundant.
            self.companion.maybe_classify()
