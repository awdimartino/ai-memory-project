"""Configuration for the companion.

Values come from the environment, optionally seeded by a local .env file.
Keep this the single place that reads configuration.
"""
import os


# The repo root, resolved from this file rather than the working directory.
_ROOT = os.path.dirname(os.path.abspath(__file__))


def _load_dotenv(path: str = os.path.join(_ROOT, ".env")) -> None:
    """Minimal .env loader so we don't take a dependency just for config.

    Existing environment variables win over .env (setdefault), matching how
    real dotenv libraries behave.

    The path is anchored to this file, NOT the working directory. It used to be a
    bare ".env", so launching from anywhere but the repo root silently skipped the
    file and fell back to defaults — which are not merely incomplete but *wrong*:
    MODEL="" auto-detects whatever model happens to be loaded, and NO_THINK=False
    turns reasoning back ON, the opposite of the production config.
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
# Consolidate once this many unconsolidated messages accumulate. Kept small (10) so a single
# durable fact isn't drowned in a large, low-signal window — extraction misses a lone fact
# buried in 20+ messages of banter (measured). Each consolidation is still backgrounded.
CONSOLIDATE_WINDOW = int(os.environ.get("CONSOLIDATE_WINDOW", "10"))

# Lifecycle: when consolidating, compare a new fact against existing memories this
# similar (higher than recall — only genuinely related facts get an LLM decision).
MEMORY_RELATE_SIMILARITY = float(os.environ.get("MEMORY_RELATE_SIMILARITY", "0.6"))
MEMORY_RELATE_TOP_K = int(os.environ.get("MEMORY_RELATE_TOP_K", "5"))
# Consolidation batches all lifecycle decisions into ONE model call (speed). Two facts in the
# SAME window this cosine-similar are treated as the same fact and collapsed without a model
# call (near-verbatim only, so genuine "two dogs" still coexist). Keep high.
MEMORY_DUP_SIMILARITY = float(os.environ.get("MEMORY_DUP_SIMILARITY", "0.97"))

# Core memory: a small set of identity-defining facts (name, key people, job, where they
# live, defining traits) that are ALWAYS injected into the prompt, not just when recall
# surfaces them. The extractor marks facts as core; when the set exceeds this cap, the
# brain re-ranks and demotes the least essential back to regular (searchable) memory.
CORE_MEMORY_MAX = int(os.environ.get("CORE_MEMORY_MAX", "12"))

# Emotion (pillar 2). A local RoBERTa GoEmotions classifier maps each message to
# 28 emotions, folded into 6 mood channels that decay toward a baseline. Runs on
# CPU (the 9070XT can't use the v1 CUDA path) so it keeps all VRAM for the LLMs.
EMOTION_ENABLED = os.environ.get("EMOTION_ENABLED", "true").lower() in ("1", "true", "yes")
EMOTION_MODEL = os.environ.get("EMOTION_MODEL", "SamLowe/roberta-base-go_emotions")
# How reactive mood is to a single message (v1-tuned); noise floor drops weak labels.
EMOTION_PULL_STRENGTH = float(os.environ.get("EMOTION_PULL_STRENGTH", "0.4"))
EMOTION_NOISE_FLOOR = float(os.environ.get("EMOTION_NOISE_FLOOR", "0.05"))

# Tick loop (pillar 3, proactivity). An internal heartbeat that runs background jobs
# on a cadence: for this slice, mood drift toward baseline while the user is away, and
# idle consolidation of the pending buffer. Outward reach-out (unprompted messages) is
# a later slice. Jobs only act once the user has been idle for TICK_IDLE_SECONDS, so
# nothing fires mid-conversation.
TICK_ENABLED = os.environ.get("TICK_ENABLED", "true").lower() in ("1", "true", "yes")
TICK_INTERVAL = float(os.environ.get("TICK_INTERVAL", "60"))          # base loop cadence (s)
TICK_IDLE_SECONDS = float(os.environ.get("TICK_IDLE_SECONDS", "90"))  # "user is away" threshold
# Consolidate a non-empty pending buffer once the user has been idle this long.
IDLE_CONSOLIDATE_AFTER = float(os.environ.get("IDLE_CONSOLIDATE_AFTER", "180"))

# Proactive reach-out: after the user has been away REACHOUT_MIN_IDLE seconds, the
# tick may generate an unprompted message (Mari decides whether it's worth it and can
# decline). REACHOUT_COOLDOWN throttles attempts (persisted across restarts) so she's
# a companion, not a nag. Only the web UI surfaces these; the REPL runs internal jobs only.
REACHOUT_ENABLED = os.environ.get("REACHOUT_ENABLED", "true").lower() in ("1", "true", "yes")
REACHOUT_MIN_IDLE = float(os.environ.get("REACHOUT_MIN_IDLE", "900"))    # 15 min away
REACHOUT_COOLDOWN = float(os.environ.get("REACHOUT_COOLDOWN", "7200"))   # >=2h between attempts

# Phone push (self-hosted Bark, §8-D): when Mari *reaches out* (a proactive message) and NOTIFY_URL
# is set, POST it to your Bark server so it pushes to your phone via APNs even with the tab closed.
# Empty => disabled (no-op). NOTIFY_URL is the full Bark device endpoint, e.g.
# http://alex-pi:8090/<device_key>. NOTIFY_UI_URL is an optional tap-to-open link (your Tailscale web
# UI address) so tapping the notification opens Mari to reply. Follow-ups stay in-tab (not pushed).
NOTIFY_URL = os.environ.get("NOTIFY_URL", "")
NOTIFY_UI_URL = os.environ.get("NOTIFY_UI_URL", "")
NOTIFY_TITLE = os.environ.get("NOTIFY_TITLE", BOT_NAME)
# Optional custom notification image (Bark `icon`): a URL the *phone* fetches when the push arrives,
# so it must be reachable from your iPhone. Easiest: drop an image in web/static/ and point here at
# the Tailscale web-UI, e.g. https://<pc>.<tailnet>.ts.net/static/mari.png. iOS caches it by URL.
NOTIFY_ICON = os.environ.get("NOTIFY_ICON", "")

# Follow-up messages: after Mari replies, she may fire off a spontaneous second message a tick
# or a few later (an afterthought / "double-text"), if she genuinely has something to add — she
# decides and can PASS. A per-tick CHANCE spreads the timing so it isn't clockwork; it only fires
# within a short WINDOW after her reply (long idle is reach-out's job) and is capped per turn.
# Web-only (pushed over the WebSocket, like reach-out).
FOLLOWUP_ENABLED = os.environ.get("FOLLOWUP_ENABLED", "true").lower() in ("1", "true", "yes")
FOLLOWUP_CHANCE = float(os.environ.get("FOLLOWUP_CHANCE", "0.2"))        # chance per eligible tick (low: rare)
FOLLOWUP_MIN_DELAY = float(os.environ.get("FOLLOWUP_MIN_DELAY", "0"))    # min secs after her reply
FOLLOWUP_WINDOW = float(os.environ.get("FOLLOWUP_WINDOW", "60"))         # must land soon or the moment's gone
FOLLOWUP_MAX_PER_TURN = int(os.environ.get("FOLLOWUP_MAX_PER_TURN", "1"))  # follow-ups per user turn

# Self-reflection: while the user is away, Mari writes a short private thought (a journal
# to herself, never shown in chat) about how she's doing and the conversations. Internal
# cognition — the substrate for reminisce and, later, the self-modifying persona.
REFLECT_ENABLED = os.environ.get("REFLECT_ENABLED", "true").lower() in ("1", "true", "yes")
REFLECT_MIN_IDLE = float(os.environ.get("REFLECT_MIN_IDLE", "120"))      # think after 2 min away
REFLECT_COOLDOWN = float(os.environ.get("REFLECT_COOLDOWN", "600"))      # at most every ~10 min

# Intentions (the "planning" pillar): during idle, Mari notes forward intentions — things she means to
# bring up or find out next time — which reach-out then draws on to be purposeful. Its own idle cadence.
INTENTION_ENABLED = os.environ.get("INTENTION_ENABLED", "true").lower() in ("1", "true", "yes")
INTENTION_MIN_IDLE = float(os.environ.get("INTENTION_MIN_IDLE", "180"))   # note them a few min after a chat
INTENTION_COOLDOWN = float(os.environ.get("INTENTION_COOLDOWN", "900"))   # at most ~every 15 min
INTENTION_MAX_ACTIVE = int(os.environ.get("INTENTION_MAX_ACTIVE", "8"))   # cap the open agenda (drop oldest)
# Stale intentions expire so the agenda doesn't linger on things that stopped mattering (0 = never).
INTENTION_MAX_AGE_DAYS = float(os.environ.get("INTENTION_MAX_AGE_DAYS", "7"))

# Learned operating-notes (the self-improvement loop): during idle, Mari distills short notes on HOW to be
# with this person from recent experience ("ease off the questions"), injected live into her prompt. Slower
# than reflection (a lesson, not a mood) but faster than the persona rewrite (behavior, not identity).
SELFNOTES_ENABLED = os.environ.get("SELFNOTES_ENABLED", "true").lower() in ("1", "true", "yes")
SELFNOTES_MIN_IDLE = float(os.environ.get("SELFNOTES_MIN_IDLE", "300"))    # 5 min away
SELFNOTES_COOLDOWN = float(os.environ.get("SELFNOTES_COOLDOWN", "1800"))   # at most ~every 30 min
SELFNOTES_MAX_CHARS = int(os.environ.get("SELFNOTES_MAX_CHARS", "400"))    # keep the injected block small

# Self-modifying persona: during idle ticks Mari rewrites a bot-owned "who you've become"
# slot in her system prompt, reading her thought journal + core memories. How far it may
# drift is gated by a familiarity meter (derived from message count) so a stranger doesn't
# rewrite herself into a close friend on day one. Slow and rare (a personality shifts gradually).
PERSONA_EDIT_ENABLED = os.environ.get("PERSONA_EDIT_ENABLED", "true").lower() in ("1", "true", "yes")
PERSONA_EDIT_MIN_IDLE = float(os.environ.get("PERSONA_EDIT_MIN_IDLE", "300"))     # 5 min away
PERSONA_EDIT_COOLDOWN = float(os.environ.get("PERSONA_EDIT_COOLDOWN", "3600"))    # at most hourly
PERSONA_MIN_MESSAGES = int(os.environ.get("PERSONA_MIN_MESSAGES", "20"))          # need some history first
PERSONA_MAX_CHARS = int(os.environ.get("PERSONA_MAX_CHARS", "600"))               # keep the slot small
# Messages of interaction to reach full familiarity (stranger -> close friend), roughly.
FAMILIARITY_MESSAGES = int(os.environ.get("FAMILIARITY_MESSAGES", "400"))

# Sleep / standby (§2.8): after a long idle, Mari unloads the LLM from VRAM to free the
# machine (the brain sleeps; the heartbeat keeps ticking). The next message wakes her,
# reloading the model ("waking up…"). Model-using tick jobs pause while she's asleep.
# Requires the LM Studio `lms` CLI; auto-disables if it isn't found.
SLEEP_ENABLED = os.environ.get("SLEEP_ENABLED", "true").lower() in ("1", "true", "yes")
SLEEP_AFTER_IDLE = float(os.environ.get("SLEEP_AFTER_IDLE", "1800"))   # 30 min away -> sleep
LMS_PATH = os.environ.get("LMS_PATH", "lms")                           # LM Studio CLI

# Internal drives (multi-drive proactivity, roadmap arc A1 — "observe first" slice).
# Slow-integrating scalars (connection, restlessness) that rise while you're away,
# modulated by mood, and relax on contact — a more lifelike generalization of the tick's
# fixed idle gates (V2_PLAN §2.9). This slice only *observes* them (surfaced in the status
# panel); the behaviors still fire on their existing idle gates until the drives prove out.
# Persisted like mood so they survive restarts. Rise rates / weights live in core/drives.py.
DRIVES_ENABLED = os.environ.get("DRIVES_ENABLED", "true").lower() in ("1", "true", "yes")
DRIVE_AWAY_AFTER = float(os.environ.get("DRIVE_AWAY_AFTER", "90"))  # idle (s) before drives rise
# Behavior thresholds: reach-out fires when `connection` crosses this, reflection when
# `restlessness` does (both still gated by their persisted cooldowns as a hard floor).
# 0.6 connection ≈ 15 min neutral / sooner after a warm or sad chat; 0.4 restlessness ≈ 5 min.
DRIVE_CONNECTION_THRESHOLD = float(os.environ.get("DRIVE_CONNECTION_THRESHOLD", "0.6"))
DRIVE_RESTLESSNESS_THRESHOLD = float(os.environ.get("DRIVE_RESTLESSNESS_THRESHOLD", "0.4"))
# Energy / body cycle (arc A2): sleep ALSO fires when the fatigue reserve drops to this level
# and she's been briefly idle (ENERGY_SLEEP_MIN_IDLE — so she doesn't nod off mid-chat), on top
# of the long-idle SLEEP_AFTER_IDLE trigger. Deplete/restore rates live in core/drives.py.
ENERGY_SLEEP_THRESHOLD = float(os.environ.get("ENERGY_SLEEP_THRESHOLD", "0.15"))
ENERGY_SLEEP_MIN_IDLE = float(os.environ.get("ENERGY_SLEEP_MIN_IDLE", "120"))

# Tools (pillar 4): native function-calling. When enabled, the chat turn streams
# through a tool loop so Mari can call registered tools (clock, reminisce, more
# later). Verified 100% reliable on the chat model (scripts/tool_probe.py). The
# loop is capped at TOOL_MAX_ITERS tool round-trips per turn so it can never hang.
TOOLS_ENABLED = os.environ.get("TOOLS_ENABLED", "true").lower() in ("1", "true", "yes")
TOOL_MAX_ITERS = int(os.environ.get("TOOL_MAX_ITERS", "5"))

# Web server.
WEB_HOST = os.environ.get("WEB_HOST", "127.0.0.1")
WEB_PORT = int(os.environ.get("WEB_PORT", "8000"))
