"""Configuration for the companion.

Values come from the environment, optionally seeded by a local .env file.
Keep this the single place that reads configuration.
"""
import os


def _load_dotenv(path: str = ".env") -> None:
    """Minimal .env loader so we don't take a dependency just for config.

    Existing environment variables win over .env (setdefault), matching how
    real dotenv libraries behave.
    """
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv()

# LM Studio's OpenAI-compatible endpoint (v1 default).
BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
API_KEY = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")  # value is ignored on localhost

# Empty MODEL => auto-detect the first model loaded in LM Studio.
MODEL = os.environ.get("MODEL", "")
TEMPERATURE = float(os.environ.get("TEMPERATURE", "0.8"))
# Discourage the model from repeating itself (verbatim lines/metaphors across a
# reply and, with the anti-repeat persona rule, across turns). OpenAI-style
# penalties, applied to chat generation only (structured brain calls stay crisp).
FREQUENCY_PENALTY = float(os.environ.get("FREQUENCY_PENALTY", "0.4"))
PRESENCE_PENALTY = float(os.environ.get("PRESENCE_PENALTY", "0.3"))
# LM Studio intermittently 400s ("predict request failed: fetch failed"). Retry a
# few times so a transient hiccup doesn't surface as a chat error or drop a
# consolidation. Chat retries only before the first visible token (never mid-stream).
LLM_MAX_RETRIES = int(os.environ.get("LLM_MAX_RETRIES", "3"))

# For reasoning models (qwen3 etc.), append "/no_think" so they answer directly
# instead of spending latency (and token budget) on hidden reasoning. Set true
# when MODEL is a reasoning model; harmless-but-pointless on non-reasoning ones.
NO_THINK = os.environ.get("NO_THINK", "false").lower() in ("1", "true", "yes")

BOT_NAME = os.environ.get("BOT_NAME", "Mari")

# Max number of prior messages (user+assistant) kept in the prompt / carried
# across restarts as context.
HISTORY_TURNS = int(os.environ.get("HISTORY_TURNS", "20"))

# Persistence.
DB_PATH = os.environ.get("DB_PATH", "companion.db")

# Semantic memory.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
# Model used for backgrounded memory consolidation. Empty => reuse the chat model
# (VRAM-safe: no second large model loaded). Set e.g. qwen3-8b for sharper extraction.
BRAIN_MODEL = os.environ.get("BRAIN_MODEL", "")
RECALL_TOP_K = int(os.environ.get("RECALL_TOP_K", "5"))
# Similarity floor for injecting a memory. Calibrate on real data (v1 lesson) —
# a too-high cutoff silently drops real matches.
# Calibrated on nomic-embed-v1.5: real matches ~0.59-0.65, unrelated ~0.50.
RECALL_MIN_SIMILARITY = float(os.environ.get("RECALL_MIN_SIMILARITY", "0.55"))
# Consolidate once this many unconsolidated messages accumulate (a context window).
CONSOLIDATE_WINDOW = int(os.environ.get("CONSOLIDATE_WINDOW", str(HISTORY_TURNS)))

# Lifecycle: when consolidating, compare a new fact against existing memories this
# similar (higher than recall — only genuinely related facts get an LLM decision).
MEMORY_RELATE_SIMILARITY = float(os.environ.get("MEMORY_RELATE_SIMILARITY", "0.6"))
MEMORY_RELATE_TOP_K = int(os.environ.get("MEMORY_RELATE_TOP_K", "5"))

# Emotion (pillar 2). A local RoBERTa GoEmotions classifier maps each message to
# 28 emotions, folded into 6 mood channels that decay toward a baseline. Runs on
# CPU (the 9070XT can't use the v1 CUDA path) so it keeps all VRAM for the LLMs.
EMOTION_ENABLED = os.environ.get("EMOTION_ENABLED", "true").lower() in ("1", "true", "yes")
EMOTION_MODEL = os.environ.get("EMOTION_MODEL", "SamLowe/roberta-base-go_emotions")
# How reactive mood is to a single message (v1-tuned); noise floor drops weak labels.
EMOTION_PULL_STRENGTH = float(os.environ.get("EMOTION_PULL_STRENGTH", "0.4"))
EMOTION_NOISE_FLOOR = float(os.environ.get("EMOTION_NOISE_FLOOR", "0.05"))

# Web server.
WEB_HOST = os.environ.get("WEB_HOST", "127.0.0.1")
WEB_PORT = int(os.environ.get("WEB_PORT", "8000"))
