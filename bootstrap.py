"""Single composition root.

Both entry points (the REPL and the web app) build the Companion here so wiring
lives in exactly one place.
"""
import asyncio
import logging
import sys

import config
from core.companion import CONSOLIDATED_WATERMARK_KEY, Companion
from core.emotion_manager import EmotionManager
from core.memory_manager import MemoryManager
from infrastructure.conversation_store import SqliteConversationStore
from infrastructure.db import connect
from infrastructure.embedder import Embedder
from infrastructure.llm_client import LLMClient
from infrastructure.memory_store import SqliteMemoryStore
from infrastructure.meta_store import SqliteMetaStore


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

    llm = LLMClient(config.BASE_URL, config.API_KEY, config.MODEL, config.TEMPERATURE,
                    no_think=config.NO_THINK)
    model = await llm.resolve_model()

    embedder = Embedder(llm.client, config.EMBED_MODEL)
    brain_model = config.BRAIN_MODEL or model  # reuse chat model unless overridden
    memory = MemoryManager(
        embedder, mem_store, llm, brain_model,
        config.RECALL_TOP_K, config.RECALL_MIN_SIMILARITY,
        config.MEMORY_RELATE_TOP_K, config.MEMORY_RELATE_SIMILARITY,
    )

    emotion = await _build_emotion(meta_store)

    history = conv_store.recent_messages(config.HISTORY_TURNS)
    # Recover the unconsolidated tail: any messages logged after the last
    # consolidation checkpoint (e.g. dropped by a hard kill before flush).
    watermark = meta_store.get_int(CONSOLIDATED_WATERMARK_KEY, 0)
    unconsolidated = conv_store.messages_after(watermark)
    session_id = conv_store.create_session()
    companion = Companion(llm, conv_store, memory, meta_store, session_id,
                          history, unconsolidated, emotion=emotion)

    logging.getLogger("bootstrap").info(
        "ready: chat=%s, brain=%s, embed=%s, emotion=%s | %d logged msgs, %d memories, "
        "%d carried, %d unconsolidated recovered",
        model, brain_model, config.EMBED_MODEL,
        config.EMOTION_MODEL if emotion else "off",
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
