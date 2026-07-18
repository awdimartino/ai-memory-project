from core.tick_system import TickSystem
from infrastructure.database import DatabaseConnection
from infrastructure.llm_client import LLMClient
from infrastructure import config
from infrastructure.embedder import Embedder
from infrastructure.memory_store import MemoryStore
from infrastructure.conversation_store import ConversationStore
from core.companion import Companion
from core.memory_manager import MemoryManager
from core.conversation_manager import ConversationManager
from core.emotion_manager import EmotionManager
from core.chat_loop import ChatLoop
import logging
import threading
from openai import OpenAI as oai


client = oai(
            base_url=config.AI_BASE_URL,
            api_key=config.AI_API_KEY
        )


def configure_logging():
    """Keep the root/third-party loggers quiet; only the app's own loggers
    ('core', 'infrastructure') follow DEBUG_MODE. Otherwise httpcore/httpx/
    huggingface DEBUG spam floods the chat prompt."""
    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    app_level = logging.DEBUG if config.DEBUG_MODE else logging.INFO
    for name in ("core", "infrastructure"):
        logging.getLogger(name).setLevel(app_level)
    for noisy in ("httpcore", "httpx", "openai", "urllib3",
                  "huggingface_hub", "transformers", "filelock"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


def main():
    configure_logging()

    # Create databae objects
    db = DatabaseConnection()
    memory_store = MemoryStore(db)
    conversation_store = ConversationStore(db)

    # Create LLM clients
    embedder_instance = Embedder(client)
    llm_client = LLMClient(client)

    # Create managers
    memory_manager = MemoryManager(memory_store, embedder_instance, llm_client)
    conversation_manager = ConversationManager(conversation_store)
    emotion_manager = EmotionManager()
    
    # Create companion
    companion = Companion(
        llm_client=llm_client,
        memory_manager=memory_manager,
        conversation_manager=conversation_manager,
        emotion_manager=emotion_manager,
    )

    # Create the chat loop with the companion and tick system
    tick_system = TickSystem(companion, interval=30, lock=threading.Lock())
    chat_loop = ChatLoop(companion, tick_system)
    chat_loop.start()


if __name__ == "__main__":
    main()