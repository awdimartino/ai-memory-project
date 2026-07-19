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
from core.prompts import (
    FOLLOWUP_CUE,
    REFLECT_CUE,
    build_followup_system,
    build_persona_edit_system,
    build_persona_edit_user,
    build_reachout_cue,
    build_reachout_system,
    build_reflect_system,
    build_system,
)

logger = logging.getLogger(__name__)

# Key for the consolidation checkpoint in the MetaStore. Shared with bootstrap,
# which reads it on startup to recover the unconsolidated tail.
CONSOLIDATED_WATERMARK_KEY = "last_consolidated_msg_id"
# Mari's own evolving self-description (the self-modifying persona slot), in the MetaStore.
PERSONA_SELF_KEY = "persona_self"


def familiarity_label(f: float) -> str:
    """Human phrasing of the familiarity scalar (0..1), used to gate the persona edit."""
    if f < 0.15:
        return "a stranger you're only just starting to know"
    if f < 0.40:
        return "a new acquaintance"
    if f < 0.70:
        return "someone you're becoming familiar with"
    if f < 0.90:
        return "someone you know pretty well by now"
    return "someone you're genuinely close to"


def _humanize_away(seconds: float) -> str:
    """Rough, human phrasing for how long the user has been gone (for the reach-out cue)."""
    minutes = seconds / 60
    if minutes < 2:
        return "a couple minutes"
    if minutes < 60:
        return f"{round(minutes)} minutes"
    hours = minutes / 60
    if hours < 2:
        return "an hour"
    if hours < 24:
        return f"{round(hours)} hours"
    return "a while"


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
    core: list[str] | None = None   # always-injected core memories about the user
    tools: list[dict] | None = None  # tools invoked this turn: [{name, args, result}]


class Companion:
    def __init__(self, llm, store, memory, meta, session_id: int,
                 history: list[dict] | None = None,
                 unconsolidated: list[dict] | None = None,
                 emotion=None, thoughts=None, model_manager=None,
                 tools=None, tool_max_iters: int = 5, drives=None):
        self.llm = llm
        self.store = store
        self.memory = memory
        self.meta = meta
        self.emotion = emotion  # EmotionManager, or None when disabled
        self.thoughts = thoughts  # ThoughtStore for the private journal, or None
        # DriveManager (multi-drive proactivity), or None when disabled. Updated by the
        # tick's DriveDriftJob and relieved here on each user message (contact).
        self.drives = drives
        self.model_manager = model_manager  # unload/reload the LLM for sleep, or None
        # ToolRegistry (pillar 4), or None/empty to run plain streaming turns. When
        # non-empty, send() streams through the tool loop so Mari can call tools.
        self.tools = tools
        self.tool_max_iters = tool_max_iters
        self._asleep = False  # True while the LLM is unloaded from VRAM (§2.8)
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
        # Follow-up state: after a reply, how many spontaneous follow-ups she may still send
        # this turn, and when her last message went out (to bound the follow-up window).
        self._followups_left = 0
        self._last_reply_at = time.monotonic()
        # Title of the active conversation (None => untitled; set from the first user message).
        self._session_title: str | None = None
        # The proactivity heartbeat, attached by bootstrap; started by the entry point.
        self.tick = None

    def switch_conversation(self, session_id: int) -> None:
        """Make `session_id` the active conversation: rebind history + title. Everything
        else (memory, mood, thoughts, persona, the consolidation buffer) is the user's and
        stays shared across conversations."""
        self.session_id = session_id
        self.history = self.store.session_messages(session_id, config.HISTORY_TURNS)
        self._session_title = self.store.session_title(session_id)

    def new_conversation(self, title: str | None = None) -> int:
        """Start a fresh conversation thread and switch to it. Returns its id."""
        sid = self.store.create_session(title)
        self.switch_conversation(sid)
        return sid

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
            if self._asleep:
                await self.wake()  # reload the LLM before the first reply after standby
            recalled = await self.memory.recall(user_text)
            core = self.memory.core_memories()
            persona = self.meta.get(PERSONA_SELF_KEY)

            # Emotion is autonomic (Tier-1): the user's message shifts mood, which
            # then colors this reply's tone via the system prompt.
            emotion_info = None
            mood_prompt = None
            if self.emotion is not None:
                emotion_info = await self.emotion.react(user_text)
                mood_prompt = self.emotion.as_prompt()

            # Core facts are always injected; drop any recalled duplicates of them.
            extra = [content for content, _ in recalled if content not in core]
            tool_names = self.tools.names() if self.tools else None
            system = build_system(extra, mood_prompt, core=core, persona=persona, tools=tool_names)

            messages = [{"role": "system", "content": system}]
            messages.extend(self.history[-config.HISTORY_TURNS:])
            messages.append({"role": "user", "content": user_text})

            # With tools registered, stream through the tool loop (Mari may call one
            # mid-turn); otherwise a plain streaming turn with zero tool overhead.
            if self.tools:
                text, stats = await self.llm.stream_with_tools(
                    messages, on_token, self.tools, self.tool_max_iters)
            else:
                text, stats = await self.llm.stream(messages, on_token)

            uid = await asyncio.to_thread(self.store.add_message, self.session_id, "user", user_text)
            aid = await asyncio.to_thread(self.store.add_message, self.session_id, "assistant", text)

            self.history.append({"role": "user", "content": user_text})
            self.history.append({"role": "assistant", "content": text})
            self._unconsolidated.append({"id": uid, "role": "user", "content": user_text})
            self._unconsolidated.append({"id": aid, "role": "assistant", "content": text})

            if not self._session_title:  # name a fresh conversation by its first message
                title = " ".join(user_text.split())[:40] or "New chat"
                await asyncio.to_thread(self.store.set_title, self.session_id, title)
                self._session_title = title

            if self.drives is not None:
                await self.drives.on_user_message()  # contact relieves connection/restlessness

            # Arm a possible spontaneous follow-up to this reply (the web follow-up job
            # decides, a tick or a few later, and may PASS).
            self._followups_left = config.FOLLOWUP_MAX_PER_TURN
            self._last_reply_at = time.monotonic()

            self._maybe_consolidate()
            return TurnResult(text, stats, recalled, emotion_info, core, stats.get("tools"))
        finally:
            self._busy = False
            self._last_activity = time.monotonic()

    async def reach_out(self) -> str | None:
        """Generate an unprompted message from recent context, or None to stay quiet.

        Mari decides: the prompt lets her reply "PASS" when nothing feels natural. A real
        message is logged as an assistant turn (so it also shows up on reconnect) and
        folded into history/the consolidation buffer, just like a normal reply.
        """
        last_user = next((m["content"] for m in reversed(self.history) if m["role"] == "user"), None)
        recalled = await self.memory.recall(last_user) if last_user else []
        core = self.memory.core_memories()
        extra = [content for content, _ in recalled if content not in core]
        mood_prompt = self.emotion.as_prompt() if self.emotion is not None else None
        system = build_reachout_system(extra, mood_prompt, core=core,
                                       persona=self.meta.get(PERSONA_SELF_KEY))

        messages = [{"role": "system", "content": system}]
        messages.extend(self.history[-config.HISTORY_TURNS:])
        messages.append({"role": "user", "content": build_reachout_cue(_humanize_away(self.idle_seconds()))})

        async def _sink(_t):
            pass

        text, _ = await self.llm.stream(messages, _sink)
        text = text.strip()
        if not text or text.rstrip(".!").strip().upper() == "PASS":
            logger.info("reach-out: stayed quiet")
            return None

        aid = await asyncio.to_thread(self.store.add_message, self.session_id, "assistant", text)
        self.history.append({"role": "assistant", "content": text})
        self._unconsolidated.append({"id": aid, "role": "assistant", "content": text})
        logger.info("reach-out: %s", text[:80])
        return text

    def is_busy(self) -> bool:
        """True while a turn is being generated (the follow-up job skips mid-turn)."""
        return self._busy

    def followups_pending(self) -> int:
        """Spontaneous follow-ups Mari may still send for the current turn."""
        return self._followups_left

    def seconds_since_reply(self) -> float:
        """How long since her last message went out (bounds the follow-up window)."""
        return time.monotonic() - self._last_reply_at

    def cancel_followups(self) -> None:
        """Close the follow-up window for this turn (moment passed / she declined)."""
        self._followups_left = 0

    async def follow_up(self) -> str | None:
        """Maybe fire a spontaneous follow-up to her own last message (an afterthought).

        Only when her message is still the most recent one (the user hasn't replied yet). She
        may reply PASS to stay quiet. A real message decrements the per-turn budget and extends
        the window (so a rare second one can chain); a PASS closes follow-ups for this turn. Like
        reach-out, the text is logged as an assistant turn so it replays on reconnect.
        """
        if not self.history or self.history[-1]["role"] != "assistant":
            self._followups_left = 0  # she's no longer the last speaker; nothing to follow up
            return None
        last_user = next((m["content"] for m in reversed(self.history) if m["role"] == "user"), None)
        recalled = await self.memory.recall(last_user) if last_user else []
        core = self.memory.core_memories()
        extra = [content for content, _ in recalled if content not in core]
        mood_prompt = self.emotion.as_prompt() if self.emotion is not None else None
        system = build_followup_system(extra, mood_prompt, core=core,
                                       persona=self.meta.get(PERSONA_SELF_KEY))

        messages = [{"role": "system", "content": system}]
        messages.extend(self.history[-config.HISTORY_TURNS:])
        messages.append({"role": "user", "content": FOLLOWUP_CUE})

        async def _sink(_t):
            pass

        text, _ = await self.llm.stream(messages, _sink)
        text = text.strip()
        if not text or text.rstrip(".!").strip().upper() == "PASS":
            self._followups_left = 0
            logger.info("follow-up: stayed quiet")
            return None

        aid = await asyncio.to_thread(self.store.add_message, self.session_id, "assistant", text)
        self.history.append({"role": "assistant", "content": text})
        self._unconsolidated.append({"id": aid, "role": "assistant", "content": text})
        self._followups_left -= 1
        self._last_reply_at = time.monotonic()  # extend the window for a possible next one
        logger.info("follow-up: %s", text[:80])
        return text

    async def reflect(self) -> str | None:
        """Write a short private thought to the journal (not shown to the user).

        Draws on recent conversation, current mood, and recent thoughts (to avoid
        repeating itself). Stored via the ThoughtStore; returns the thought text.
        """
        if self.thoughts is None:
            return None
        last_user = next((m["content"] for m in reversed(self.history) if m["role"] == "user"), None)
        recalled = await self.memory.recall(last_user) if last_user else []
        core = self.memory.core_memories()
        extra = [content for content, _ in recalled if content not in core]
        mood_prompt = self.emotion.as_prompt() if self.emotion is not None else None
        recent = [t["content"] for t in self.thoughts.recent(5)]
        system = build_reflect_system(extra, mood_prompt, recent, core=core,
                                      persona=self.meta.get(PERSONA_SELF_KEY))

        messages = [{"role": "system", "content": system}]
        messages.extend(self.history[-config.HISTORY_TURNS:])
        messages.append({"role": "user", "content": REFLECT_CUE})

        async def _sink(_t):
            pass

        text, _ = await self.llm.stream(messages, _sink)
        text = text.strip()
        if not text:
            return None

        dom = None
        if self.emotion is not None:
            dom = max(self.emotion.state, key=self.emotion.state.get)
        await asyncio.to_thread(self.thoughts.add, text, dom)
        logger.info("reflected: %s", text[:80])
        return text

    def familiarity(self) -> float:
        """Relationship depth 0..1, from how much you two have talked (gates persona drift)."""
        return min(1.0, self.store.message_count() / max(1, config.FAMILIARITY_MESSAGES))

    def is_asleep(self) -> bool:
        return self._asleep

    async def sleep(self) -> None:
        """Go to standby: save pending work, then unload the LLM from VRAM (§2.8).

        The tick keeps running (mood still drifts), but model-using jobs pause until wake.
        """
        if self._asleep or self.model_manager is None:
            return
        await self.flush()  # don't lose the unconsolidated buffer before unloading
        self._asleep = True
        try:
            await self.model_manager.unload_all()
        except Exception:  # noqa: BLE001 - never let sleep crash the tick
            logger.exception("sleep: unload failed")
        logger.info("went to sleep (LLM unloaded)")

    async def wake(self) -> None:
        """Reload the models so the next reply can be generated (best-effort)."""
        if not self._asleep:
            return
        logger.info("waking up (reloading models)...")
        try:
            if self.model_manager is not None:
                await self.model_manager.load([self.llm.model, config.EMBED_MODEL])
        except Exception:  # noqa: BLE001 - if reload fails, the LLM call will JIT-load or error+retry
            logger.exception("wake: reload failed; relying on JIT / retry")
        self._asleep = False

    async def edit_persona(self) -> str | None:
        """Rewrite Mari's own self-description slot from her thoughts + core memories, gated by
        familiarity. Runs during idle ticks; the new text colors later turns via build_system."""
        current = self.meta.get(PERSONA_SELF_KEY) or ""
        thoughts = [t["content"] for t in self.thoughts.recent(8)] if self.thoughts else []
        core = self.memory.core_memories()
        fam = familiarity_label(self.familiarity())

        system = build_persona_edit_system(fam, config.PERSONA_MAX_CHARS)
        user = build_persona_edit_user(current, thoughts, core)

        async def _sink(_t):
            pass

        text, _ = await self.llm.stream(
            [{"role": "system", "content": system}, {"role": "user", "content": user}], _sink)
        text = text.strip()
        if not text or text.rstrip(".!").strip().upper() == "PASS":
            logger.info("persona edit: no change")
            return None
        text = text[:config.PERSONA_MAX_CHARS].strip()  # hard cap so the slot stays small
        await asyncio.to_thread(self.meta.set, PERSONA_SELF_KEY, text)
        logger.info("persona updated: %s", text[:80])
        return text

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
        # Take at most ONE window; a lone fact drowns in a huge, low-signal chunk (extraction
        # misses a single fact buried in 20+ noisy messages). The rest stays buffered.
        chunk = self._unconsolidated[:config.CONSOLIDATE_WINDOW]
        self._unconsolidated = self._unconsolidated[config.CONSOLIDATE_WINDOW:]
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
        """Checkpoint the consolidated *prefix* (best-effort, never fatal).

        The watermark means "every message up to here is consolidated", so it must never
        regress and must never jump past a message still awaiting consolidation. That second
        rule matters because chunks aren't guaranteed id-ordered: a background pass that *fails*
        re-queues its (lower-id) chunk while a concurrent flush of a newer chunk succeeds — if
        we advanced to the newer chunk's max id, the re-queued lower ids would sit below the
        watermark and be skipped by the crash-recovery in bootstrap (lost facts). So clamp to
        just below the lowest still-pending id, and take the max with the current value.
        """
        ids = [m["id"] for m in chunk if m.get("id") is not None]
        if not ids:
            return
        new_wm = max(ids)
        pending = [m["id"] for m in self._unconsolidated if m.get("id") is not None]
        if pending:
            new_wm = min(new_wm, min(pending) - 1)
        try:
            new_wm = max(self.meta.get_int(CONSOLIDATED_WATERMARK_KEY, 0), new_wm)
            await asyncio.to_thread(self.meta.set_int, CONSOLIDATED_WATERMARK_KEY, new_wm)
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

    async def clear_memories(self) -> int:
        """Admin: wipe the semantic memory store (keeps conversations, mood, persona,
        thoughts). Returns how many memories were removed."""
        n = len(self.memory.store.all())
        self.memory.store.clear()
        logger.info("admin: cleared %d memories", n)
        return n

    async def factory_reset(self) -> None:
        """Admin: wipe EVERYTHING back to a factory-fresh companion — memories,
        conversations, private thoughts, and all persisted scalar state (mood, drives,
        persona, consolidation watermark, reach-out/reflect cooldowns) — then start a
        fresh conversation. In-memory state is reset to match so no reload is required."""
        self.memory.store.clear()
        self.store.clear()                       # messages + sessions
        if self.thoughts is not None:
            self.thoughts.clear()
        self.meta.clear()                        # mood, drives, persona, watermark, cooldowns
        self.history = []
        self._unconsolidated = []
        self._session_title = None
        if self.emotion is not None:
            await self.emotion.reset()           # re-persist baseline mood
        if self.drives is not None:
            await self.drives.reset()            # re-persist baseline drives
        sid = self.store.create_session()        # a clean conversation to land in
        self.switch_conversation(sid)
        logger.info("admin: factory reset complete; fresh session %d", sid)
