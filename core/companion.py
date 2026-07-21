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
import re
import time
from collections import deque
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from typing import Awaitable, Callable

import config
from core.cues import farewell_cue
from core.embodiment import embodiment_claim
from core.textsim import is_repeat
from core.prompts import (
    FOLLOWUP_CUE,
    INTENTIONS_SCHEMA,
    INTENTIONS_SYSTEM,
    PURSUIT_QUESTIONS_SCHEMA,
    PURSUIT_QUESTIONS_SYSTEM,
    REFLECT_CUE,
    SLEEP_CHOICE_SCHEMA,
    build_intentions_user,
    build_persona_edit_system,
    build_persona_edit_user,
    build_pursuit_choice_schema,
    build_pursuit_choice_system,
    build_pursuit_choice_user,
    build_pursuit_questions_user,
    build_reachout_cue,
    build_self_notes_system,
    build_self_notes_user,
    build_sleep_choice_system,
    build_sleep_choice_user,
    followup_blocks,
    join_blocks,
    reachout_blocks,
    reflect_blocks,
    relationship_stage,
    system_blocks,
)

logger = logging.getLogger(__name__)

# Key for the consolidation checkpoint in the MetaStore. Shared with bootstrap,
# which reads it on startup to recover the unconsolidated tail.
CONSOLIDATED_WATERMARK_KEY = "last_consolidated_msg_id"
# Mari's own evolving self-description (the self-modifying persona slot), in the MetaStore.
PERSONA_SELF_KEY = "persona_self"
# Learned operating-notes: what she's worked out about HOW to be with this user (vs. who she is).
SELF_NOTES_KEY = "self_notes"


def familiarity_label(f: float) -> str:
    """Human phrasing of the familiarity scalar (0..1), used to gate the persona edit.

    Delegates to the stage table in `prompts`, which also supplies the relationship
    framing injected into every prompt — so the status panel's label and what she is
    actually told about the relationship can never drift apart.
    """
    return relationship_stage(f)[0]


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

    `stats` is the raw LLM-call record — perf numbers plus the `tools` list the
    turn invoked; `recalled` are the memories injected this turn as
    (content, similarity); `emotion` is the classifier read + resulting mood
    ({"detected": [...], "mood": {...}}), or None when emotion is disabled.
    """
    text: str
    stats: dict
    recalled: list[tuple[str, float]]
    emotion: dict | None = None
    core: list[str] | None = None   # always-injected core memories about the user
    tools: list[dict] | None = None  # tools invoked this turn: [{name, args, result}]
    silent: bool = False             # she chose to stay quiet (no reply shown or logged)
    slept: bool = False              # she chose to power down after a goodnight (§A2)


async def _sink(_token: str) -> None:
    """Swallow tokens. Internal generations (reach-out, reflection, persona edit,
    self-notes) still stream, but nobody is watching them arrive."""


def _is_pass(text: str) -> bool:
    """True when a generation is Mari declining to speak — the PASS convention.

    Shared by every path that offers her the choice (chat silence, reach-out,
    follow-up, persona edit, self-notes). Tolerates what the model actually returns:
    surrounding whitespace, a trailing period or exclamation, any casing. This used
    to be written inline six times in two different operator orders, one of which
    only worked because callers happened to `.strip()` first.
    """
    return text.strip().rstrip(".!").strip().upper() == "PASS"


class Companion:
    def __init__(self, llm, store, memory, meta, session_id: int,
                 history: list[dict] | None = None,
                 unconsolidated: list[dict] | None = None,
                 emotion=None, thoughts=None, model_manager=None,
                 tools=None, tool_max_iters: int = 5, drives=None, intentions=None,
                 session_title: str | None = None, pursuits=None,
                 clock: Callable[[], float] = time.monotonic):
        self.llm = llm
        self.store = store
        self.memory = memory
        self.meta = meta
        self.emotion = emotion  # EmotionManager, or None when disabled
        self.thoughts = thoughts  # ThoughtStore for the private journal, or None
        self.intentions = intentions  # IntentionStore (forward agenda / planning + pursuits), or None
        self.pursuits = pursuits  # PursuitRegistry (§A4), or None/empty when disabled
        # DriveManager (multi-drive proactivity), or None when disabled. Updated by the
        # tick's DriveDriftJob and relieved here on each user message (contact).
        self.drives = drives
        self.model_manager = model_manager  # unload/reload the LLM for sleep, or None
        # ToolRegistry (pillar 4), or None/empty to run plain streaming turns. When
        # non-empty, send() streams through the tool loop so Mari can call tools.
        self.tools = tools
        self.tool_max_iters = tool_max_iters
        self._asleep = False  # True while the LLM is unloaded from VRAM (§2.8)
        # §A4: a state distinct from sleep (different trigger, different UI) — she can
        # be awake-idle, asleep, or unavailable, never two at once (enforced in core/tick.py).
        # `clock` is injectable (tests use a fake) and used ONLY for these timers; it does
        # not touch idle_seconds()/_last_activity below.
        self._clock = clock
        self._unavailable_until: float | None = None
        self._unavailable_reason: str | None = None
        self._unavailable_model_unloaded = False
        self._pending_message: str | None = None  # held while unavailable; one slot, overwritten
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
        # Title of the active conversation (None => untitled; set from the first user
        # message). Passed in rather than read here — the constructor stays I/O-free.
        self._session_title: str | None = session_title
        # The proactivity heartbeat, attached by bootstrap; started by the entry point.
        self.tick = None
        # Recent prompts, newest last, for the prompt inspector. In-memory and bounded:
        # this is a debugging window on "why did she just say that", not a record —
        # persisting it would mean a schema migration and a growing table for data
        # whose value expires within a few turns.
        self._prompts: deque = deque(maxlen=config.PROMPT_LOG_MAX)

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

    def _record_prompt(self, kind: str, blocks: list[tuple[str, str]],
                       messages: list[dict]) -> dict:
        """Log one generation's prompt for the inspector. Returns the record, so the
        caller can attach the reply once it exists.

        `messages` is the array actually handed to the model, stored as-is — the
        inspector's whole value is being the ground truth, so nothing here
        reconstructs or summarises what was sent.
        """
        record = {
            "kind": kind,
            "at": time.time(),
            "blocks": [{"label": label, "text": text} for label, text in blocks],
            "messages": [dict(m) for m in messages],
            "reply": None,      # filled in after generation; None => she stayed quiet
            "reasoning": "",    # her hidden thinking, filled in alongside the reply
        }
        self._prompts.append(record)
        return record

    def prompt_log(self) -> list[dict]:
        """Recent prompts, newest first — for the prompt inspector."""
        return list(reversed(self._prompts))

    async def send(self, user_text: str, on_token: Callable[[str], Awaitable]) -> TurnResult:
        """Recall + react, stream a reply, log, and maybe consolidate. Returns a TurnResult."""
        # A live turn always cancels any dangling "reply once the unavailable window
        # ends" obligation — whichever path reaches send() first (this one, or
        # PursuitReturnJob) wins cleanly; the other becomes a no-op.
        self._pending_message = None
        self._busy = True  # the tick treats an in-progress turn as "not idle"
        try:
            if self._asleep:
                await self.wake()  # reload the LLM before the first reply after standby
            recalled = await self.memory.recall(user_text)
            # Turn-gated: sticky/cooldown so core facts aren't in literally every prompt.
            turn = self.store.message_count()
            core, core_ids = self.memory.core_for_turn(turn)
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
            # Her open agenda rides along softly — she may raise one if the talk gets there.
            agenda = [i["content"] for i in self.intentions.active(kind="agenda")] if self.intentions else None
            blocks = system_blocks(extra, mood_prompt, core=core, persona=persona,
                                   tools=tool_names, allow_silence=True, intentions=agenda,
                                   self_notes=self.meta.get(SELF_NOTES_KEY),
                                   familiarity=self.familiarity())
            system = join_blocks(blocks)

            messages = [{"role": "system", "content": system}]
            messages.extend(self.history[-config.HISTORY_TURNS:])
            messages.append({"role": "user", "content": user_text})
            prompt_record = self._record_prompt("chat", blocks, messages)

            # She may choose to stay silent by replying "PASS". Gate the token stream so the
            # word PASS never reaches the UI: hold any prefix that could still become "PASS",
            # and only start emitting once the reply clearly isn't it.
            gate = {"buf": "", "open": False}

            async def guarded(tok: str) -> None:
                if gate["open"]:
                    await on_token(tok)
                    return
                gate["buf"] += tok
                probe = gate["buf"].strip().rstrip(".!").upper()
                if not probe or "PASS".startswith(probe):
                    return  # whitespace-only, or still a possible "PASS" — keep holding
                gate["open"] = True
                await on_token(gate["buf"])

            # With tools registered, stream through the tool loop (Mari may call one
            # mid-turn); otherwise a plain streaming turn with zero tool overhead.
            if self.tools:
                text, stats = await self.llm.stream_with_tools(
                    messages, guarded, self.tools, self.tool_max_iters)
            else:
                text, stats = await self.llm.stream(messages, guarded)

            silent = _is_pass(text)
            prompt_record["reply"] = None if silent else text
            # Her hidden thinking, for the inspector. Especially worth seeing on a
            # SILENT turn, where the reply explains nothing about why she passed.
            prompt_record["reasoning"] = stats.get("reasoning") or ""

            uid = await asyncio.to_thread(self.store.add_message, self.session_id, "user", user_text)
            self.history.append({"role": "user", "content": user_text})
            self._unconsolidated.append({"id": uid, "role": "user", "content": user_text})
            if not silent:  # a silent turn logs the user's message but no reply of her own
                aid = await asyncio.to_thread(self.store.add_message, self.session_id, "assistant", text)
                self.history.append({"role": "assistant", "content": text})
                self._unconsolidated.append({"id": aid, "role": "assistant", "content": text})

            if not self._session_title:  # name a fresh conversation by its first message
                title = " ".join(user_text.split())[:40] or "New chat"
                await asyncio.to_thread(self.store.set_title, self.session_id, title)
                self._session_title = title

            if self.drives is not None:
                await self.drives.on_user_message()  # contact relieves connection/restlessness

            # Arm a spontaneous follow-up only if she actually spoke; a chosen silence
            # shouldn't be immediately undone by a double-text.
            if silent:
                self._followups_left = 0
            else:
                self._followups_left = config.FOLLOWUP_MAX_PER_TURN
                self._last_reply_at = time.monotonic()

            if core_ids:
                await asyncio.to_thread(self.memory.mark_injected, core_ids, turn)
            self._maybe_consolidate()
            # After the reply is stored, so her goodnight is on record either way, and
            # after _maybe_consolidate so the two don't both try to drain the buffer
            # (sleep() flushes, and the consolidation lock serialises them).
            slept = await self.maybe_sleep_on_farewell(user_text, "" if silent else text)
            return TurnResult("" if silent else text, stats, recalled, emotion_info,
                              core, stats.get("tools"), silent=silent, slept=slept)
        finally:
            self._busy = False
            self._last_activity = time.monotonic()

    # --- shared machinery for her unprompted generations ----------------------------
    # reach_out / follow_up / reflect are the same shape: gather context, build a
    # system prompt, generate against [system, ...history, cue], and either take what
    # came back or accept a PASS. Only the prompt, the cue, and what happens to a real
    # reply differ, so those three steps live here once.

    async def _recall_context(self) -> tuple[list[str], list[str], str | None]:
        """The context every unprompted generation wants: (recalled, core, mood prompt).

        Recall is anchored on the last thing the user actually said — she's reaching
        into the conversation, not responding to a turn.
        """
        last_user = next((m["content"] for m in reversed(self.history) if m["role"] == "user"), None)
        recalled = await self.memory.recall(last_user) if last_user else []
        core = self.memory.core_memories()
        extra = [content for content, _ in recalled if content not in core]
        mood_prompt = self.emotion.as_prompt() if self.emotion is not None else None
        return extra, core, mood_prompt

    def _recent_said(self, n: int = 6) -> list[str]:
        """Her own last `n` messages — what a new one must not simply restate."""
        return [m["content"] for m in self.history if m.get("role") == "assistant"][-n:]

    async def _aside(self, blocks: list[tuple[str, str]], cue: str, kind: str,
                     avoid_repeats: bool = False) -> str | None:
        """Generate one unprompted message. Returns None if she declines or says nothing.

        Also drops anything claiming a physical experience she didn't have. Staying
        quiet is already a valid outcome here, so a fabricated "I found a good spot to
        sit" (an actual logged follow-up) costs nothing to discard — whereas letting it
        through is a falsifiable lie, which is the thing that collapses trust. The
        prompt forbids this too; the filter exists because ~a quarter of the base
        model's dialogue prior is exactly these utterances.

        `avoid_repeats` applies the same programmatic guard `reflect()` needed. The
        asymmetry was never principled: reflection got its recent thoughts injected AND
        a mechanical check, while reach-out and follow-up got neither — so she could not
        see what she had just opened with. Measured in the real log: reach-outs
        clustering on "i was just thinking…" / "i was just wondering…", and one era where
        63% of messages opened with the same word. Staying quiet is a valid outcome, so a
        near-duplicate costs nothing to drop.

        Takes labelled `blocks` rather than an assembled system string so the prompt
        inspector can attribute an out-of-nowhere message to the block that caused it;
        `kind` names the path for the inspector's list.
        """
        messages = [{"role": "system", "content": join_blocks(blocks)}]
        messages.extend(self.history[-config.HISTORY_TURNS:])
        messages.append({"role": "user", "content": cue})
        record = self._record_prompt(kind, blocks, messages)
        text, stats = await self.llm.stream(messages, _sink)
        record["reasoning"] = stats.get("reasoning") or ""
        text = text.strip()
        if not text or _is_pass(text):
            return None
        claim = embodiment_claim(text)
        if claim:
            logger.info("aside discarded, claims an experience she didn't have (%r): %s",
                        claim, text[:80])
            # Recorded as discarded rather than dropped: a filtered aside is exactly
            # the kind of invisible event the inspector should still show.
            record["reply"] = f"(discarded — claimed an experience she didn't have: {text})"
            return None
        if avoid_repeats:
            recent = self._recent_said()
            if is_repeat(text, recent):
                logger.info("%s discarded, restates something she just said: %s", kind, text[:80])
                record["reply"] = f"(discarded — restates a recent message: {text})"
                return None
        record["reply"] = text
        return text

    async def _log_assistant_turn(self, text: str) -> None:
        """Record one of her unprompted messages as a real assistant turn.

        So it replays on reconnect and counts toward memory, exactly like a reply.
        """
        aid = await asyncio.to_thread(self.store.add_message, self.session_id, "assistant", text)
        self.history.append({"role": "assistant", "content": text})
        self._unconsolidated.append({"id": aid, "role": "assistant", "content": text})

    async def reach_out(self) -> str | None:
        """Generate an unprompted message from recent context, or None to stay quiet.

        Mari decides: the prompt lets her reply "PASS" when nothing feels natural. A real
        message is logged as an assistant turn (so it also shows up on reconnect) and
        folded into history/the consolidation buffer, just like a normal reply.
        """
        extra, core, mood_prompt = await self._recall_context()
        # Anchor the reach-out on the longest-waiting intention, if any — makes it purposeful
        # ("been meaning to ask...") rather than generic; fulfilled only if she actually sends it.
        intention = None
        if self.intentions is not None:
            pending = self.intentions.active(kind="agenda", limit=1)
            intention = pending[0] if pending else None
        blocks = reachout_blocks(extra, mood_prompt, core=core,
                                 persona=self.meta.get(PERSONA_SELF_KEY),
                                 intention=intention["content"] if intention else None,
                                 self_notes=self.meta.get(SELF_NOTES_KEY),
                                 familiarity=self.familiarity())

        text = await self._aside(blocks, build_reachout_cue(_humanize_away(self.idle_seconds())),
                                 "reach-out", avoid_repeats=True)
        if text is None:
            logger.info("reach-out: stayed quiet")
            return None

        await self._log_assistant_turn(text)
        if intention is not None:
            await asyncio.to_thread(self.intentions.fulfill, intention["id"])
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
        extra, core, mood_prompt = await self._recall_context()
        # ⚠️ NO carried intention here (removed 2026-07-20). Riding the agenda along was
        # what turned a double-text into an unrelated topic switch: measured in the real
        # log, both double-texts pivoted to a fresh subject a moment after her last
        # message ("i was just wondering how your interest in ai chatbots is going" —
        # an intention discharged nearly verbatim), and one asserted an outcome she
        # could not know. The prompt said "only fold it in if it genuinely connects";
        # she folded it in regardless.
        #
        # An afterthought is about what she JUST SAID. The agenda already has a proper
        # home — reach-out anchors on the longest-waiting intention and fulfils it — so
        # nothing is lost by keeping this path for actual afterthoughts.
        blocks = followup_blocks(extra, mood_prompt, core=core,
                                 persona=self.meta.get(PERSONA_SELF_KEY),
                                 self_notes=self.meta.get(SELF_NOTES_KEY),
                                 familiarity=self.familiarity())

        text = await self._aside(blocks, FOLLOWUP_CUE, "follow-up", avoid_repeats=True)
        if text is None:
            self._followups_left = 0
            logger.info("follow-up: stayed quiet")
            return None

        await self._log_assistant_turn(text)
        self._followups_left -= 1
        self._last_reply_at = time.monotonic()  # extend the window for a possible next one
        logger.info("follow-up: %s", text[:80])
        return text

    async def reflect(self, seed: str | None = None) -> str | None:
        """Write a short private thought to the journal (not shown to the user).

        Draws on recent conversation, current mood, and recent thoughts (to avoid
        repeating itself). Stored via the ThoughtStore; returns the thought text.

        `seed` gives this reflection a genuine subject (an A3-mined open question,
        via `sit_with_open_question`) instead of the unseeded "how are you doing?" —
        the documented cause of her journal repeating itself.
        """
        if self.thoughts is None:
            return None
        extra, core, mood_prompt = await self._recall_context()
        recent = [t["content"] for t in self.thoughts.recent(5)]
        blocks = reflect_blocks(extra, mood_prompt, recent, core=core,
                                persona=self.meta.get(PERSONA_SELF_KEY),
                                familiarity=self.familiarity(), seed=seed)

        text = await self._aside(blocks, REFLECT_CUE, "reflection")
        if text is None:
            return None
        # Programmatic repeat guard. The prompt above already shows her `recent` and
        # asks her not to repeat, and it does not hold: measured RRR 0.26 over the real
        # journal, with five near-verbatim pairs and three byte-identical entries. A
        # restated thought is not a new thought, and repetition in self-generated
        # content predicts it's confabulated rather than important.
        if is_repeat(text, recent):
            logger.info("reflection discarded, restates a recent thought: %s", text[:80])
            return None

        dom = None
        if self.emotion is not None:
            dom = max(self.emotion.state, key=self.emotion.state.get)
        await asyncio.to_thread(self.thoughts.add, text, dom)
        logger.info("reflected: %s", text[:80])
        return text

    async def form_intentions(self) -> list[str]:
        """Note forward intentions from the recent conversation — things to bring up or find out
        next time. Additive + deduped against the open agenda, and caps the active set; returns the
        new ones. Run during idle by IntentionJob; reach-out draws on them (the planning pillar)."""
        if self.intentions is None:
            return []
        # Expire stale items first, so they neither linger nor crowd out fresher ones at the cap.
        if config.INTENTION_MAX_AGE_DAYS > 0:
            cutoff = (datetime.now(timezone.utc)
                      - timedelta(days=config.INTENTION_MAX_AGE_DAYS)).isoformat()
            gone = await asyncio.to_thread(self.intentions.drop_older_than, cutoff, kind="agenda")
            if gone:
                logger.info("expired %d stale intention(s)", gone)
        recent = self.history[-config.HISTORY_TURNS:]
        if not recent:
            return []
        existing = self.intentions.active(kind="agenda")
        raw = await self.llm.structured_json(
            [{"role": "system", "content": INTENTIONS_SYSTEM},
             {"role": "user", "content": build_intentions_user(
                 recent, [i["content"] for i in existing])}],
            INTENTIONS_SCHEMA)
        # Clear any the conversation actually covered (1-based indices into `existing`). This is how
        # chat- and follow-up-raised intentions get retired without a second model call.
        resolved = 0
        for n in (raw.get("resolved") or []):
            if isinstance(n, int) and 1 <= n <= len(existing):
                await asyncio.to_thread(self.intentions.fulfill, existing[n - 1]["id"])
                resolved += 1
        proposed = [s.strip() for s in (raw.get("intentions") or [])
                    if isinstance(s, str) and s.strip()]
        seen = {i["content"].lower() for i in existing}
        added: list[str] = []
        for s in proposed:
            if s.lower() not in seen:  # cheap dedupe guard on top of the prompt's own
                await asyncio.to_thread(self.intentions.add, s, kind="agenda")
                seen.add(s.lower())
                added.append(s)
        # Cap the open agenda — retire the oldest beyond the cap.
        active = self.intentions.active(kind="agenda")
        for old in active[:max(0, len(active) - config.INTENTION_MAX_ACTIVE)]:
            await asyncio.to_thread(self.intentions.drop, old["id"])
        if added or resolved:
            logger.info("intentions: +%d new, %d resolved", len(added), resolved)
        return added

    async def mine_open_questions(self) -> list[str]:
        """A3: mine grounded open questions about the user from the recent window.

        A reframed "nightly deep pass" — NOT day-summaries (measured worse elsewhere:
        aggressive clustering loses retrieval accuracy), a small number of salient
        open questions, each cited to a real message id. This project has already
        been bitten three times by ungrounded generation on a thin window (memory
        extraction, self-notes, intentions all previously confabulated on a barren
        window), so citation-checking here is a structural guard, not a prompt
        request: anything uncited, or citing an id outside the fetched window, is
        rejected outright, no exceptions. Accepted questions become `kind="pursuit"`
        rows — the backlog `go_unavailable()`'s "sit_with_question" consumes.
        """
        if self.intentions is None:
            return []
        window = self.store.recent_messages_with_ids(config.PURSUIT_WINDOW_MESSAGES)
        if len(window) < config.PURSUIT_MIN_MESSAGES:
            return []
        core = self.memory.core_memories()
        existing = self.intentions.active(kind="pursuit")
        raw = await self.llm.structured_json(
            [{"role": "system", "content": PURSUIT_QUESTIONS_SYSTEM},
             {"role": "user", "content": build_pursuit_questions_user(
                 window, core, [i["content"] for i in existing])}],
            PURSUIT_QUESTIONS_SCHEMA)
        valid_ids = {m["id"] for m in window}
        seen = {i["content"].lower() for i in existing}
        added: list[str] = []
        for item in (raw.get("questions") or [])[:3]:
            if not isinstance(item, dict):
                continue
            q = (item.get("question") or "").strip()
            cites = [c for c in (item.get("citations") or []) if isinstance(c, int) and c in valid_ids]
            if not q or not cites:  # HARD REJECT — no citation, no question, no exceptions
                logger.info("pursuit question rejected (uncited): %r", q[:80])
                continue
            if q.lower() in seen:
                continue
            await asyncio.to_thread(self.intentions.add, q, kind="pursuit", citations=cites)
            seen.add(q.lower())
            added.append(q)
        active = self.intentions.active(kind="pursuit")
        for old in active[:max(0, len(active) - config.PURSUIT_MAX_ACTIVE)]:
            await asyncio.to_thread(self.intentions.drop, old["id"])
        if added:
            logger.info("pursuits: +%d new open question(s)", len(added))
        return added

    def message_count(self) -> int:
        """Total messages exchanged, ever (the persona job's "developed self" gate)."""
        return self.store.message_count()

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

    async def maybe_sleep_on_farewell(self, user_text: str, reply: str) -> bool:
        """He said goodnight — ask whether she wants to go quiet too. True if she slept.

        Gated by a lexical cue (`core.cues.farewell_cue`) so this costs one extra
        structured call on the handful of turns that end a night, not on every turn.
        Declining is free: `SleepJob`'s idle/energy trigger still fires later, so this
        can only ever make standby *earlier and chosen*, never prevent it.
        """
        if not config.SLEEP_ON_FAREWELL_ENABLED or self.model_manager is None:
            return False
        if self._asleep or self.is_unavailable():
            return False
        cue = farewell_cue(user_text)
        if cue is None:
            return False
        energy = self.drives.energy() if self.drives is not None else None
        # Several turns, not one. Insisting is the signal that he means it — but with a
        # single exchange the decision cannot SEE that he already asked twice and she
        # deflected, so it read repeated urging as "the conversation is still open" and
        # said STAY every time. Observed live 2026-07-21 across four turns of the user
        # telling her to sleep.
        recent = self.history[-config.SLEEP_CHOICE_CONTEXT_TURNS * 2:]
        lines = [f"{'Them' if m['role'] == 'user' else config.BOT_NAME}: {m['content']}"
                 for m in recent]
        lines.append(f"Them: {user_text}")
        lines.append(f"{config.BOT_NAME}: {reply or '(she stayed quiet)'}")
        exchange = "\n".join(lines)
        try:
            raw = await self.llm.structured_json(
                [{"role": "system", "content": build_sleep_choice_system()},
                 {"role": "user", "content": build_sleep_choice_user(energy, exchange)}],
                SLEEP_CHOICE_SCHEMA)
        except Exception:  # noqa: BLE001 - a failed decision must not break the turn
            logger.exception("sleep decision failed; staying up")
            return False
        if (raw.get("choice") or "").strip().upper() != "SLEEP":
            logger.info("farewell cue %r -> she's staying up", cue)
            return False
        logger.info("farewell cue %r -> going to sleep", cue)
        await self.sleep()
        return True

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

    # --- §A4: she can make herself unavailable ------------------------------------
    # A state distinct from sleep (different trigger — restlessness + a real pursuit
    # to do, not idle/energy — and a different UI treatment), but they can't overlap:
    # CooldownJob already refuses to fire while is_asleep(), and SleepJob refuses to
    # fire while is_unavailable() (core/tick.py). She's awake-idle, asleep, or
    # unavailable — never two at once.

    def is_unavailable(self) -> bool:
        return self._unavailable_until is not None and self._clock() < self._unavailable_until

    def unavailable_reason(self) -> str | None:
        return self._unavailable_reason if self.is_unavailable() else None

    def unavailable_eta_seconds(self) -> float | None:
        if not self.is_unavailable():
            return None
        return max(0.0, self._unavailable_until - self._clock())

    def has_pending_message(self) -> bool:
        return self._pending_message is not None

    def queue_pending_message(self, text: str) -> None:
        """A real message arrived while unavailable: hold exactly one — a later
        message overwrites it rather than queuing alongside it."""
        self._pending_message = text

    async def sit_with_open_question(self) -> str | None:
        """The A4 pursuit that consumes A3's mined backlog: pull the oldest open
        question and seed reflect() with it instead of firing it unseeded — this is
        also what fixes the documented journal-repetition bug (ReflectionJob has no
        subject on its own). Fulfilled only on a genuine artifact; a PASS/discarded
        attempt leaves the question open to try again later."""
        if self.intentions is None:
            return None
        pending = self.intentions.active(kind="pursuit", limit=1)
        if not pending:
            return None
        text = await self.reflect(seed=pending[0]["content"])
        if text is not None:
            await asyncio.to_thread(self.intentions.fulfill, pending[0]["id"])
        return text

    async def go_unavailable(self) -> bool:
        """Consider stepping away: a structured-output decision picks one pursuit (or
        PASS), the underlying real operation runs for real (model already loaded —
        she's awake), then the model is freed for the REMAINDER of a sampled window
        since nothing further needs it until she returns. Returns True iff she went
        unavailable."""
        if not self.pursuits:
            return False
        pending_pursuit = (self.intentions.active(kind="pursuit", limit=1)[0]
                           if self.intentions is not None
                           and self.intentions.active(kind="pursuit", limit=1) else None)
        choices = self.pursuits.available(has_open_pursuit=pending_pursuit is not None)
        if not choices:
            return False
        names = [t.name for t in choices]
        raw = await self.llm.structured_json(
            [{"role": "system", "content": build_pursuit_choice_system()},
             {"role": "user", "content": build_pursuit_choice_user(
                 choices, seed_question=pending_pursuit["content"] if pending_pursuit else None)}],
            build_pursuit_choice_schema(names))
        choice = (raw.get("choice") or "").strip()
        if choice not in names:  # PASS, garbage, or unlisted -> stay available
            logger.info("unavailable: declined")
            return False
        tool = self.pursuits.get(choice)
        await tool.handler()  # the real op, for real, while the model is still loaded
        duration = tool.sample_duration()
        reason = tool.description
        if choice == "sit_with_question" and pending_pursuit is not None:
            reason = f"sitting with something on her mind: {pending_pursuit['content']}"
        self._unavailable_until = self._clock() + duration
        self._unavailable_reason = reason
        if self.model_manager is not None:
            self._unavailable_model_unloaded = True
            try:
                await self.model_manager.unload_all()  # nothing else needs it for the remainder
            except Exception:  # noqa: BLE001 - never let this crash the tick
                logger.exception("go_unavailable: unload failed")
        logger.info("unavailable: %s for %.0fs", choice, duration)
        return True

    async def end_unavailable(self, on_token: Callable[[str], Awaitable] = _sink) -> "TurnResult | None":
        """The window elapsed. Reload the model if it was unloaded (also covers her
        having gone to sleep normally in the meantime — a redundant reload is a no-op
        cost, not a correctness bug), then deliver any pending message as a real turn.
        Returns None if nothing was pending."""
        self._unavailable_until = None
        self._unavailable_reason = None
        if self._asleep or self._unavailable_model_unloaded:
            try:
                if self.model_manager is not None:
                    await self.model_manager.load([self.llm.model, config.EMBED_MODEL])
            except Exception:  # noqa: BLE001 - if reload fails, the LLM call will JIT-load or error+retry
                logger.exception("end_unavailable: reload failed; relying on JIT / retry")
            self._asleep = False
            self._unavailable_model_unloaded = False
        pending, self._pending_message = self._pending_message, None
        if pending is None:
            return None
        return await self.send(pending, on_token)

    async def edit_persona(self) -> str | None:
        """Rewrite Mari's own self-description slot from her thoughts + core memories, gated by
        familiarity. Runs during idle ticks; the new text colors later turns via build_system.

        The "too early to have a developed self" gate used to live ONLY in
        `PersonaEditJob._ready()` — fine while that job was the only caller, but a
        pursuit (§A4) can now call this directly, bypassing that external gate. Self-
        contained here so a stranger-stage companion can't rewrite herself prematurely
        no matter what calls this.
        """
        if self.message_count() < config.PERSONA_MIN_MESSAGES:
            return None
        current = self.meta.get(PERSONA_SELF_KEY) or ""
        thoughts = [t["content"] for t in self.thoughts.recent(8)] if self.thoughts else []
        core = self.memory.core_memories()
        fam = familiarity_label(self.familiarity())

        system = build_persona_edit_system(fam, config.PERSONA_MAX_CHARS)
        user = build_persona_edit_user(current, thoughts, core)

        text, _ = await self.llm.stream(
            [{"role": "system", "content": system}, {"role": "user", "content": user}], _sink)
        text = text.strip()
        if not text or _is_pass(text):
            logger.info("persona edit: no change")
            return None
        text = text[:config.PERSONA_MAX_CHARS].strip()  # hard cap so the slot stays small
        await asyncio.to_thread(self.meta.set, PERSONA_SELF_KEY, text)
        logger.info("persona updated: %s", text[:80])
        return text

    async def update_self_notes(self) -> str | None:
        """Distill short operating-notes — how to BE with this user — from the recent conversation.

        The closed loop the persona edit doesn't cover: experience -> principle -> future behavior.
        edit_persona rewrites who she IS from her private thoughts; this rewrites how she ACTS from
        how the user actually responded to her. Rewritten wholesale (so revising and dropping notes
        need no dedupe pass), capped, and injected into every user-facing prompt via build_system.
        """
        recent = self.history[-config.HISTORY_TURNS:]
        if not recent:
            return None
        current = self.meta.get(SELF_NOTES_KEY) or ""
        system = build_self_notes_system(config.SELFNOTES_MAX_CHARS)
        user = build_self_notes_user(current, recent)

        text, _ = await self.llm.stream(
            [{"role": "system", "content": system}, {"role": "user", "content": user}], _sink)
        text = text.strip()
        if not text or _is_pass(text):
            logger.info("self-notes: no change")
            return None
        # These notes are addressed TO her, so a correct one never names her. When her name shows up
        # the model has written the note to the USER instead ("You get annoyed when Mari asks
        # questions"), which injects the lesson backwards — drop the pass and keep the old notes.
        if re.search(rf"\b{re.escape(config.BOT_NAME)}\b", text, re.IGNORECASE):
            logger.info("self-notes: discarded, written in the wrong voice: %s", text[:80])
            return None
        text = text[:config.SELFNOTES_MAX_CHARS].strip()
        await asyncio.to_thread(self.meta.set, SELF_NOTES_KEY, text)
        # Log the revision. The slot is overwritten wholesale, so without this there's
        # no history to score for repetition — and an operating-rule that keeps getting
        # re-derived is the signature of a WRONG one, not an important one.
        await asyncio.to_thread(self.store.log_self_notes, text)
        logger.info("self-notes updated: %s", text[:80].replace("\n", " | "))
        return text

    def _maybe_consolidate(self) -> None:
        """At the end of a context window, consolidate in the background (non-blocking).

        Skips while a consolidation is already running so only one chunk is in
        flight at a time; new messages just accumulate in the buffer until it's
        free. This keeps the watermark contiguous and re-queues collision-free.
        """
        if self._consol_lock.locked():
            return
        if len(self._unconsolidated) < config.CONSOLIDATE_WINDOW and not self._is_salient():
            return
        # Take at most ONE window; a lone fact drowns in a huge, low-signal chunk (extraction
        # misses a single fact buried in 20+ noisy messages). The rest stays buffered.
        chunk = self._unconsolidated[:config.CONSOLIDATE_WINDOW]
        self._unconsolidated = self._unconsolidated[config.CONSOLIDATE_WINDOW:]
        asyncio.create_task(self._consolidate(chunk))

    def _is_salient(self) -> bool:
        """Has enough emotionally-charged material accumulated to consolidate early?

        A fixed window spends identical effort on "morning / ok" and on the
        conversation where he says his dad is sick — which is exactly backwards, and
        fills memory with trivia. Generative Agents fires reflection on an accumulated
        importance threshold rather than a timer; the emotion classifier gives us that
        accumulator for free.

        This only lets a charged window fire EARLY. The window size remains a hard
        ceiling above, so a long flat stretch is still consolidated — salience can
        never make her save less, only sooner.
        """
        if self.emotion is None or config.CONSOLIDATE_SALIENCE <= 0:
            return False
        if len(self._unconsolidated) < config.SALIENCE_MIN_MESSAGES:
            return False
        # Distance of the current mood from its baseline: how far this conversation
        # has moved her, summed across channels.
        return self.emotion.arousal() >= config.CONSOLIDATE_SALIENCE

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

    # --- inspector / admin surface -------------------------------------------------
    # The facade exists so callers don't bind to the store layout (V2_PLAN §1: "no
    # reach-through coupling"). The web layer used to reach `c.memory.store.*` and
    # `companion.store.*` directly for these; each one now has a named seam.

    def memory_snapshot(self) -> dict:
        """Core facts + counts + recently-retired facts (the status panel's memory card)."""
        return self.memory.snapshot()

    def all_memories(self) -> list[dict]:
        """Every memory, active + retired — the inspector modal's list."""
        return self.memory.all_memories()

    def delete_memory(self, memory_id: int) -> None:
        """Hard-delete one memory."""
        self.memory.delete_memory(memory_id)

    def set_memory_core(self, memory_id: int, core: bool) -> None:
        """Toggle a memory's always-injected core flag."""
        self.memory.set_core(memory_id, core)

    def rename_conversation(self, session_id: int, title: str) -> None:
        """Retitle a conversation thread, keeping the cached title in sync if it's ours."""
        self.store.set_title(session_id, title)
        if session_id == self.session_id:
            self._session_title = title

    def delete_conversation(self, session_id: int) -> None:
        """Delete a conversation thread and its messages (memory/mood/persona are global)."""
        self.store.delete_session(session_id)

    def archive_stats(self) -> dict:
        """The permanent record: total messages and a row per era (survives resets)."""
        return {"total": self.store.archive_count(),
                "eras": self.store.archive_eras(),
                "current_era": self.store.current_era()}

    def search_archive(self, query: str, limit: int = 10) -> list[dict]:
        """Keyword search every conversation ever, including pre-reset ones."""
        return self.store.search_archive(query, limit)

    def list_conversations(self) -> list[dict]:
        """All conversation threads, for the sidebar."""
        return self.store.list_conversations()

    def latest_conversation(self) -> int | None:
        """Most recent thread's id, or None if there are none left."""
        return self.store.latest_session()

    def clear_memories(self) -> int:
        """Admin: wipe the semantic memory store (keeps conversations, mood, persona,
        thoughts). Returns how many memories were removed."""
        n = self.memory.clear()
        logger.info("admin: cleared %d memories", n)
        return n

    async def factory_reset(self) -> None:
        """Admin: wipe her back to factory-fresh — memories, working conversation log,
        private thoughts, intentions, and all persisted scalar state (mood, drives,
        persona, watermark, cooldowns) — then start a fresh conversation. In-memory
        state is reset to match, so no reload is needed.

        **The message archive is NOT wiped.** Every message ever said is kept in
        `message_archive`, which no admin operation clears — so the working state can
        be reset as often as testing needs without losing the record.
        """
        # The archive survives on purpose: this wipes what she WORKS on, not the
        # record of what was said. begin_new_era marks the discontinuity so a future
        # look-back doesn't read across it as one continuous relationship.
        self.store.begin_new_era()
        self.memory.clear()                      # via the manager, not through it
        self.store.clear()                       # messages + sessions (NOT the archive)
        if self.thoughts is not None:
            self.thoughts.clear()
        if self.intentions is not None:
            self.intentions.clear()
        self.meta.clear()                        # mood, drives, persona, watermark, cooldowns
        self.history = []
        self._unconsolidated = []
        self._session_title = None
        self._unavailable_until = None
        self._unavailable_reason = None
        self._unavailable_model_unloaded = False
        self._pending_message = None
        if self.emotion is not None:
            await self.emotion.reset()           # re-persist baseline mood
        if self.drives is not None:
            await self.drives.reset()            # re-persist baseline drives
        sid = self.store.create_session()        # a clean conversation to land in
        self.switch_conversation(sid)
        logger.info("admin: factory reset complete; fresh session %d", sid)
