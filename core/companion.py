"""The Companion facade.

Ties together the pieces of one conversation turn:
- recall (autonomic, every turn): pull relevant memories, fold into the system prompt
- generate + stream the reply
- log both turns to the episodic store
- consolidation (backgrounded, at the end of a context window): distill durable
  facts without blocking the conversation

History is seeded from the store on startup so conversation survives restarts.
"""
import asyncio
import logging
from typing import Awaitable, Callable

import config
from core.prompts import build_system

logger = logging.getLogger(__name__)


class Companion:
    def __init__(self, llm, store, memory, session_id: int, history: list[dict] | None = None):
        self.llm = llm
        self.store = store
        self.memory = memory
        self.session_id = session_id
        self.history: list[dict] = history or []
        self._unconsolidated: list[dict] = []

    async def send(self, user_text: str, on_token: Callable[[str], Awaitable]):
        """Recall, stream a reply, log, and maybe kick off consolidation. Returns (text, stats)."""
        recalled = await self.memory.recall(user_text)
        system = build_system([content for content, _ in recalled])

        messages = [{"role": "system", "content": system}]
        messages.extend(self.history[-config.HISTORY_TURNS:])
        messages.append({"role": "user", "content": user_text})

        text, stats = await self.llm.stream(messages, on_token)

        await asyncio.to_thread(self.store.add_message, self.session_id, "user", user_text)
        await asyncio.to_thread(self.store.add_message, self.session_id, "assistant", text)

        for msg in ({"role": "user", "content": user_text},
                    {"role": "assistant", "content": text}):
            self.history.append(msg)
            self._unconsolidated.append(msg)

        self._maybe_consolidate()
        return text, stats

    def _maybe_consolidate(self) -> None:
        """At the end of a context window, consolidate in the background (non-blocking)."""
        if len(self._unconsolidated) < config.CONSOLIDATE_WINDOW:
            return
        chunk = self._unconsolidated
        self._unconsolidated = []
        asyncio.create_task(self._consolidate(chunk))

    async def _consolidate(self, chunk: list[dict]) -> None:
        try:
            await self.memory.consolidate(chunk, self.session_id)
        except Exception:  # noqa: BLE001 - background task must not die silently
            # Don't lose the facts: put the chunk back so a later window retries it.
            logger.exception("consolidation failed; re-queuing %d message(s)", len(chunk))
            self._unconsolidated[:0] = chunk

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
