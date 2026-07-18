"""The Companion facade.

Ties together the pieces of one conversation turn:
- recall (autonomic, every turn): pull relevant memories, fold into the system prompt
- generate + stream the reply
- log both turns to the episodic store
- consolidation (backgrounded, at the end of a context window): distill durable
  facts without blocking the conversation

History is seeded from the store on startup so conversation survives restarts.

Durability: consolidation is checkpointed by a persisted watermark — the id of
the last message that has been consolidated. The episodic log already stores every
message durably, so the unconsolidated buffer is just "messages newer than the
watermark"; on startup it's recovered from the store (see bootstrap), and a hard
kill no longer drops in-flight facts. The watermark only ever advances on a
*successful* consolidation, and only one consolidation runs at a time, so it stays
strictly contiguous (a failed, re-queued chunk can't be jumped over).
"""
import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Awaitable, Callable

import config
from core.prompts import build_system

logger = logging.getLogger(__name__)

# Key for the consolidation checkpoint in the MetaStore. Shared with bootstrap,
# which reads it on startup to recover the unconsolidated tail.
CONSOLIDATED_WATERMARK_KEY = "last_consolidated_msg_id"


@dataclass
class TurnResult:
    """Everything a caller might want to show for one turn.

    `stats` are perf numbers; `recalled` are the memories injected this turn as
    (content, similarity); `emotion` is the classifier read + resulting mood
    ({"detected": [...], "mood": {...}}), or None when emotion is disabled.
    """
    text: str
    stats: dict
    recalled: list[tuple[str, float]]
    emotion: dict | None = None


class Companion:
    def __init__(self, llm, store, memory, meta, session_id: int,
                 history: list[dict] | None = None,
                 unconsolidated: list[dict] | None = None,
                 emotion=None):
        self.llm = llm
        self.store = store
        self.memory = memory
        self.meta = meta
        self.emotion = emotion  # EmotionManager, or None when disabled
        self.session_id = session_id
        self.history: list[dict] = history or []
        # Each entry carries its message id so a successful consolidation can
        # advance the durable watermark. Seeded on startup from the store.
        self._unconsolidated: list[dict] = unconsolidated or []
        # Serializes consolidation so watermark advancement stays contiguous and
        # a background pass can't overlap a shutdown flush.
        self._consol_lock = asyncio.Lock()
        # When the last turn finished, so the tick loop can tell if the user is
        # away. Set to "now" on startup (a fresh session isn't instantly idle).
        self._last_activity = time.monotonic()
        # True while a turn is being processed; the tick treats this as "not idle"
        # so autonomy jobs never fire mid-reply (even during a slow generation).
        self._busy = False
        # The proactivity heartbeat, attached by bootstrap; started by the entry point.
        self.tick = None

    def idle_seconds(self) -> float:
        """Seconds since the last turn finished (0 while a turn is in progress)."""
        if self._busy:
            return 0.0
        return time.monotonic() - self._last_activity

    def pending_count(self) -> int:
        """Messages awaiting consolidation (the tick may flush these while idle)."""
        return len(self._unconsolidated)

    async def send(self, user_text: str, on_token: Callable[[str], Awaitable]) -> TurnResult:
        """Recall + react, stream a reply, log, and maybe consolidate. Returns a TurnResult."""
        self._busy = True  # the tick treats an in-progress turn as "not idle"
        try:
            recalled = await self.memory.recall(user_text)

            # Emotion is autonomic (Tier-1): the user's message shifts mood, which
            # then colors this reply's tone via the system prompt.
            emotion_info = None
            mood_prompt = None
            if self.emotion is not None:
                emotion_info = await self.emotion.react(user_text)
                mood_prompt = self.emotion.as_prompt()

            system = build_system([content for content, _ in recalled], mood_prompt)

            messages = [{"role": "system", "content": system}]
            messages.extend(self.history[-config.HISTORY_TURNS:])
            messages.append({"role": "user", "content": user_text})

            text, stats = await self.llm.stream(messages, on_token)

            uid = await asyncio.to_thread(self.store.add_message, self.session_id, "user", user_text)
            aid = await asyncio.to_thread(self.store.add_message, self.session_id, "assistant", text)

            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": text})
            self._unconsolidated.append({"id": uid, "role": "user", "content": user_text})
            self._unconsolidated.append({"id": aid, "role": "assistant", "content": text})

            self._maybe_consolidate()
            return TurnResult(text, stats, recalled, emotion_info)
        finally:
            self._busy = False
            self._last_activity = time.monotonic()

    def _maybe_consolidate(self) -> None:
        """At the end of a context window, consolidate in the background (non-blocking).

        Skips while a consolidation is already running so only one chunk is in
        flight at a time; new messages just accumulate in the buffer until it's
        free. This keeps the watermark contiguous and re-queues collision-free.
        """
        if self._consol_lock.locked():
            return
        if len(self._unconsolidated) < config.CONSOLIDATE_WINDOW:
            return
        chunk = self._unconsolidated
        self._unconsolidated = []
        asyncio.create_task(self._consolidate(chunk))

    async def _consolidate(self, chunk: list[dict]) -> None:
        async with self._consol_lock:
            try:
                await self.memory.consolidate(chunk, self.session_id)
            except Exception:  # noqa: BLE001 - background task must not die silently
                # Don't lose the facts: put the chunk back so a later window retries
                # it. The watermark is untouched, so the messages are still recovered
                # on a hard kill.
                logger.exception("consolidation failed; re-queuing %d message(s)", len(chunk))
                self._unconsolidated[:0] = chunk
                return
            await self._advance_watermark(chunk)

    async def _advance_watermark(self, chunk: list[dict]) -> None:
        """Checkpoint the highest consolidated message id (best-effort, never fatal)."""
        ids = [m["id"] for m in chunk if m.get("id") is not None]
        if not ids:
            return
        try:
            await asyncio.to_thread(
                self.meta.set_int, CONSOLIDATED_WATERMARK_KEY, max(ids)
            )
        except Exception:  # noqa: BLE001 - a missed checkpoint only means a re-consolidation
            logger.exception("failed to persist consolidation watermark")

    async def flush(self) -> None:
        """Consolidate whatever is pending (e.g. on shutdown). Safe to await inline."""
        if not self._unconsolidated:
            return
        chunk = self._unconsolidated
        self._unconsolidated = []
        await self._consolidate(chunk)

    def reset(self) -> None:
        """Clear in-memory context only (logs and memories on disk are kept)."""
        self.history.clear()
