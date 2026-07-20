"""Single composition root.

Both entry points (the REPL and the web app) build the Companion here so wiring
lives in exactly one place.
"""
import asyncio
import logging
import sys

import config
from core.builtin_tools import make_reminisce_tool, make_time_tool
from core.companion import CONSOLIDATED_WATERMARK_KEY, Companion
from core.drives import DriveManager
from core.emotion_manager import EmotionManager
from core.memory_manager import MemoryManager
from core.tools import ToolRegistry
from core.tick import (
    DriveDriftJob,
    IdleConsolidationJob,
    IntentionJob,
    MoodDriftJob,
    PersonaEditJob,
    ReflectionJob,
    SelfNotesJob,
    SleepJob,
    TickLoop,
)
from infrastructure.conversation_store import SqliteConversationStore
from infrastructure.db import connect
from infrastructure.embedder import Embedder
from infrastructure.intention_store import SqliteIntentionStore
from infrastructure.llm_client import LLMClient
from infrastructure.memory_store import SqliteMemoryStore
from infrastructure.meta_store import SqliteMetaStore
from infrastructure.model_manager import LmsModelManager
from infrastructure.thought_store import SqliteThoughtStore


def configure_logging() -> None:
    """Scoped logging: quiet by default, INFO for our own packages only."""
    logging.basicConfig(
        level=logging.WARNING,
        stream=sys.stderr,
        format="%(levelname)s %(name)s: %(message)s",
    )
    for name in ("core", "infrastructure", "web", "bootstrap"):
        logging.getLogger(name).setLevel(logging.INFO)
    for noisy in ("openai", "httpx", "httpcore"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


async def build() -> tuple[Companion, str]:
    """Wire up storage + LLM + memory + companion. Returns (companion, resolved_model_id)."""
    conn = connect(config.DB_PATH)
    conv_store = SqliteConversationStore(conn)
    mem_store = SqliteMemoryStore(conn)
    meta_store = SqliteMetaStore(conn)
    thought_store = SqliteThoughtStore(conn)
    intention_store = SqliteIntentionStore(conn)

    llm = LLMClient(config.BASE_URL, config.API_KEY, config.MODEL, config.TEMPERATURE,
                    no_think=config.NO_THINK,
                    frequency_penalty=config.FREQUENCY_PENALTY,
                    presence_penalty=config.PRESENCE_PENALTY,
                    max_retries=config.LLM_MAX_RETRIES)
    model = await llm.resolve_model()

    embedder = Embedder(llm.client, config.EMBED_MODEL)
    brain_model = config.BRAIN_MODEL or model  # reuse chat model unless overridden
    memory = MemoryManager(
        embedder, mem_store, llm, brain_model,
        config.RECALL_TOP_K, config.RECALL_MIN_SIMILARITY,
        config.MEMORY_RELATE_TOP_K, config.MEMORY_RELATE_SIMILARITY,
        core_max=config.CORE_MEMORY_MAX, dup_sim=config.MEMORY_DUP_SIMILARITY,
    )

    emotion = await _build_emotion(meta_store)

    # Internal drives (multi-drive proactivity). Cheap deterministic state, persisted like
    # mood; the tick's DriveDriftJob integrates them, and the companion relieves them on
    # contact. Reach-out gates on `connection` and reflection on `restlessness` (arc A1); energy
    # biases sleep (A2).
    drives = DriveManager(meta_store, config.DRIVE_AWAY_AFTER) if config.DRIVES_ENABLED else None

    # Sleep/standby needs the `lms` CLI; auto-disable if it isn't on this machine.
    model_manager = None
    if config.SLEEP_ENABLED:
        mm = LmsModelManager(config.LMS_PATH)
        if mm.available():
            model_manager = mm
        else:
            logging.getLogger("bootstrap").warning(
                "SLEEP_ENABLED but `%s` CLI not found; sleep/standby disabled", config.LMS_PATH)

    # Resume the most recent conversation (tabs); create one only if there are none.
    active = conv_store.latest_session() or conv_store.create_session()
    history = conv_store.session_messages(active, config.HISTORY_TURNS)
    # Recover the unconsolidated tail (GLOBAL, across all conversations): any messages
    # logged after the last consolidation checkpoint (e.g. dropped by a hard kill).
    watermark = meta_store.get_int(CONSOLIDATED_WATERMARK_KEY, 0)
    unconsolidated = conv_store.messages_after(watermark)

    # Tool table (pillar 4): the hot-swappable set of things Mari can call mid-turn.
    # Add a Tool here (or at runtime via companion.tools.register) and she can use it
    # next message — the chat loop consults this table every turn, nothing else changes.
    tools = None
    if config.TOOLS_ENABLED:
        tools = ToolRegistry([
            make_time_tool(),
            make_reminisce_tool(thought_store, conv_store),
        ])

    companion = Companion(llm, conv_store, memory, meta_store, active,
                          history, unconsolidated, emotion=emotion, thoughts=thought_store,
                          model_manager=model_manager, tools=tools,
                          tool_max_iters=config.TOOL_MAX_ITERS, drives=drives,
                          intentions=intention_store,
                          session_title=conv_store.session_title(active))

    # Proactivity heartbeat. Created, not started; the entry point starts it so eval/test
    # harnesses that call build() don't tick. Internal jobs live here; surface-specific
    # jobs (reach-out, which needs the WebSocket) are registered by the entry point.
    if config.TICK_ENABLED:
        jobs = [
            MoodDriftJob(companion, emotion, config.TICK_INTERVAL, config.TICK_IDLE_SECONDS),
            IdleConsolidationJob(companion, config.TICK_INTERVAL, config.IDLE_CONSOLIDATE_AFTER),
        ]
        if drives is not None:
            jobs.append(DriveDriftJob(companion, drives, config.TICK_INTERVAL))
        if config.REFLECT_ENABLED:
            jobs.append(ReflectionJob(companion, config.TICK_INTERVAL,
                                      config.REFLECT_MIN_IDLE, config.REFLECT_COOLDOWN,
                                      drives=drives, threshold=config.DRIVE_RESTLESSNESS_THRESHOLD))
        if config.INTENTION_ENABLED:
            jobs.append(IntentionJob(companion, config.TICK_INTERVAL,
                                     config.INTENTION_MIN_IDLE, config.INTENTION_COOLDOWN))
        if config.SELFNOTES_ENABLED:
            jobs.append(SelfNotesJob(companion, config.TICK_INTERVAL,
                                     config.SELFNOTES_MIN_IDLE, config.SELFNOTES_COOLDOWN))
        if config.PERSONA_EDIT_ENABLED:
            jobs.append(PersonaEditJob(companion, config.TICK_INTERVAL,
                                       config.PERSONA_EDIT_MIN_IDLE, config.PERSONA_EDIT_COOLDOWN,
                                       config.PERSONA_MIN_MESSAGES))
        if model_manager is not None:
            jobs.append(SleepJob(companion, config.TICK_INTERVAL, config.SLEEP_AFTER_IDLE,
                                 drives=drives, energy_threshold=config.ENERGY_SLEEP_THRESHOLD,
                                 energy_min_idle=config.ENERGY_SLEEP_MIN_IDLE))
        companion.tick = TickLoop(jobs, interval=config.TICK_INTERVAL)

    logging.getLogger("bootstrap").info(
        "ready: chat=%s, brain=%s, embed=%s, emotion=%s, tools=%s | %d logged msgs, %d memories, "
        "%d carried, %d unconsolidated recovered",
        model, brain_model, config.EMBED_MODEL,
        config.EMOTION_MODEL if emotion else "off",
        ",".join(tools.names()) if tools else "off",
        conv_store.message_count(), mem_store.count(), len(history), len(unconsolidated),
    )
    return companion, model


async def _build_emotion(meta_store) -> EmotionManager | None:
    """Load the CPU emotion classifier (off the event loop). Degrade gracefully:
    if it's disabled or fails to load, chat still runs — emotion just stays off."""
    if not config.EMOTION_ENABLED:
        return None
    try:
        # Import here so a disabled build never pays the torch/transformers import.
        from infrastructure.emotion_classifier import EmotionClassifier
        classifier = await asyncio.to_thread(EmotionClassifier, config.EMOTION_MODEL)
    except Exception:  # noqa: BLE001 - emotion is an enhancement, never a hard dependency
        logging.getLogger("bootstrap").warning(
            "emotion classifier failed to load; continuing without emotion", exc_info=True
        )
        return None
    return EmotionManager(classifier, meta_store,
                          pull_strength=config.EMOTION_PULL_STRENGTH,
                          noise_floor=config.EMOTION_NOISE_FLOOR)
