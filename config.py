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


class ConfigError(ValueError):
    """A knob was set to something unparseable — names the key, unlike a bare ValueError."""


def _flag(name: str, default: bool) -> bool:
    """Read a boolean knob. Accepts 1/true/yes/on (any casing); anything else is False."""
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _num(name: str, default, cast):
    """Read a numeric knob, failing loudly enough to name the offender.

    A malformed value used to surface as a bare `ValueError: could not convert
    string to float: 'abc'` at import, with no indication of which of ~40 knobs
    was wrong.
    """
    raw = os.environ.get(name)
    if raw is None:
        return default
    try:
        return cast(raw.strip())
    except (TypeError, ValueError) as e:
        raise ConfigError(
            f"{name}={raw!r} is not a valid {cast.__name__} (check .env)") from e


def _f(name: str, default: float) -> float:
    return _num(name, default, float)


def _i(name: str, default: int) -> int:
    return _num(name, default, int)

# LM Studio's OpenAI-compatible endpoint (v1 default).
BASE_URL = os.environ.get("LMSTUDIO_BASE_URL", "http://localhost:1234/v1")
API_KEY = os.environ.get("LMSTUDIO_API_KEY", "lm-studio")  # value is ignored on localhost

# Empty MODEL => auto-detect the first model loaded in LM Studio.
MODEL = os.environ.get("MODEL", "")
TEMPERATURE = _f("TEMPERATURE", 0.8)
# Discourage the model from repeating itself (verbatim lines/metaphors across a
# reply and, with the anti-repeat persona rule, across turns). OpenAI-style
# penalties, applied to chat generation only (structured brain calls stay crisp).
FREQUENCY_PENALTY = _f("FREQUENCY_PENALTY", 0.4)
PRESENCE_PENALTY = _f("PRESENCE_PENALTY", 0.3)
# LM Studio intermittently 400s ("predict request failed: fetch failed"). Retry a
# few times so a transient hiccup doesn't surface as a chat error or drop a
# consolidation. Chat retries only before the first visible token (never mid-stream).
LLM_MAX_RETRIES = _i("LLM_MAX_RETRIES", 3)

# For reasoning models (qwen3 etc.), append "/no_think" so they answer directly
# instead of spending latency (and token budget) on hidden reasoning. Set true
# when MODEL is a reasoning model; harmless-but-pointless on non-reasoning ones.
NO_THINK = _flag("NO_THINK", False)

BOT_NAME = os.environ.get("BOT_NAME", "Mari")

# Max number of prior messages (user+assistant) kept in the prompt / carried
# across restarts as context.
HISTORY_TURNS = _i("HISTORY_TURNS", 20)

# How many recent prompts the inspector keeps (chat turns, reach-outs, follow-ups).
# In-memory and bounded on purpose: it answers "why did she just say that", and a
# follow-up firing right after a reply would otherwise push the interesting one out
# of a single-slot buffer. Deliberately NOT persisted — the value expires in a few
# turns and a table would cost a migration to store debugging chaff.
PROMPT_LOG_MAX = _i("PROMPT_LOG_MAX", 12)

# Persistence.
DB_PATH = os.environ.get("DB_PATH", "companion.db")

# Semantic memory.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "text-embedding-nomic-embed-text-v1.5")
# Model used for backgrounded memory consolidation. Empty => reuse the chat model
# (VRAM-safe: no second large model loaded). Set e.g. qwen3-8b for sharper extraction.
BRAIN_MODEL = os.environ.get("BRAIN_MODEL", "")
RECALL_TOP_K = _i("RECALL_TOP_K", 5)
# Similarity floor for injecting a memory. Calibrate on real data (v1 lesson) —
# a too-high cutoff silently drops real matches.
# Calibrated on nomic-embed-v1.5: real matches ~0.59-0.65, unrelated ~0.50.
RECALL_MIN_SIMILARITY = _f("RECALL_MIN_SIMILARITY", 0.55)

# --- contrast gate (2026-07-20) -----------------------------------------------
# The floor above is kept, but it is no longer the ONLY way in. Measured on the
# gold set's six facts (scripts/recall_margin_probe.py), the absolute score turned
# out to be a poor discriminator: correct top-1 hits ranged 0.448-0.644 while
# unrelated queries reached 0.576, so positives and negatives OVERLAP almost
# completely and no floor separates them. At 0.55 the correct fact was top-1 every
# time and thrown away anyway — 4/12 recall.
#
# What does separate them is CONTRAST: how far the top hit stands above the rest
# of the corpus. nomic gives each query its own baseline offset (some queries score
# ~0.5 against everything), and subtracting the corpus median cancels that offset.
# Same measurement, contrast gate: 11/12 recall. Lowering the floor to 0.50 instead
# would have scored 10/12 but with more false positives — the gate is strictly the
# better instrument, which is why the floor did NOT move.
#
# The median (not the mean) is the background estimate: it barely shifts when one
# or two facts are genuinely relevant, so a compound question ("my dog and my job")
# still clears the gate on both.
RECALL_CONTRAST_GAP = _f("RECALL_CONTRAST_GAP", 0.06)
# Backstop so contrast alone can't admit a hit that is weak in absolute terms —
# in a corpus where everything scores badly, standing out means little.
RECALL_CONTRAST_FLOOR = _f("RECALL_CONTRAST_FLOOR", 0.42)
# Below this many memories the median IS the top hit (or its neighbour), so
# "contrast" is degenerate and only the absolute floor applies.
RECALL_CONTRAST_MIN_CORPUS = _i("RECALL_CONTRAST_MIN_CORPUS", 3)
# Consolidate once this many unconsolidated messages accumulate. Kept small (10) so a single
# durable fact isn't drowned in a large, low-signal window — extraction misses a lone fact
# buried in 20+ messages of banter (measured). Each consolidation is still backgrounded.
CONSOLIDATE_WINDOW = _i("CONSOLIDATE_WINDOW", 10)
# Salience gate on consolidation. A window of "morning" / "ok" should not spend the
# same machinery as one where he says his dad is sick, and a fixed cadence spends it
# identically on both -- which fills memory with trivia. Generative Agents fires
# reflection on an ACCUMULATED importance threshold rather than a timer; the emotion
# classifier already gives us the accumulator for free.
# The window stays a HARD CEILING (consolidate at CONSOLIDATE_WINDOW regardless) so a
# long flat stretch still gets saved; salience only lets an emotionally loaded window
# fire EARLY, never late. Set SALIENCE_MIN_MESSAGES > 0 to require some minimum.
CONSOLIDATE_SALIENCE = _f("CONSOLIDATE_SALIENCE", 2.0)   # summed |mood delta| to fire early
SALIENCE_MIN_MESSAGES = _i("SALIENCE_MIN_MESSAGES", 4)   # never fire below this many

# Lifecycle: when consolidating, compare a new fact against existing memories this
# similar (higher than recall — only genuinely related facts get an LLM decision).
MEMORY_RELATE_SIMILARITY = _f("MEMORY_RELATE_SIMILARITY", 0.6)

# Guard on the model's "duplicate" verdict: honour it only when the two texts really
# are near-identical. A "duplicate" decision DELETES information (the candidate is
# dropped and never stored), and the model over-applies it to NARROWING facts —
# measured 3 of 8 refinements called duplicates, e.g. "The user is a nurse" against
# "The user works in healthcare" (scripts/duplicate_guard_probe.py). Each one is a
# fact silently lost, in the subsystem the project exists for.
#
# The geometry separates the two classes cleanly where the model doesn't:
#   refinements (must store)  0.808 - 0.906
#   duplicates  (must skip)   0.924 - 0.986
# 0.92 sits in that gap. Set toward the TOP of the gap on purpose: too high only
# costs a redundant row, too low loses a fact, and those are not equal errors.
# Calibrated on 8 pairs per class — re-run the probe before moving it.
#
# ⚠️ This is cosine between the BARE fact texts, which is why the guard re-embeds
# them (MemoryManager._bare_duplicate_sims) instead of reusing the stored vectors:
# those carry KEY EXPANSION (fact + originating transcript), and the context dilutes
# the score unevenly — nurse/healthcare is 0.844 bare but 0.735 expanded, welder/
# welder 0.943 bare but 0.835 expanded. Applying this threshold to expanded vectors
# lets real duplicates through. Lexical overlap (core/textsim) does NOT work here
# either: measured, the two classes overlap completely ("The user is called Alex"
# vs "The user's name is Alex" is a true duplicate scoring only 0.200).
MEMORY_DUPLICATE_MIN_SIMILARITY = _f("MEMORY_DUPLICATE_MIN_SIMILARITY", 0.92)
MEMORY_RELATE_TOP_K = _i("MEMORY_RELATE_TOP_K", 5)
# Consolidation batches all lifecycle decisions into ONE model call (speed). Two facts in the
# SAME window this cosine-similar are treated as the same fact and collapsed without a model
# call (near-verbatim only, so genuine "two dogs" still coexist). Keep high.
MEMORY_DUP_SIMILARITY = _f("MEMORY_DUP_SIMILARITY", 0.97)

# Core memory: a small set of identity-defining facts (name, key people, job, where they
# live, defining traits) that are ALWAYS injected into the prompt, not just when recall
# surfaces them. The extractor marks facts as core; when the set exceeds this cap, the
# brain re-ranks and demotes the least essential back to regular (searchable) memory.
CORE_MEMORY_MAX = _i("CORE_MEMORY_MAX", 12)
# Sticky / cooldown on core-fact injection, counted in TURNS. Core facts were injected
# unconditionally on every single turn, which is the structural cause of a companion
# sounding like it's reading off a card -- a fact that is ALWAYS present reads as
# recitation no matter how the persona is worded.
# STICKY: once injected, keep it in for this many turns so it can't flicker out
# mid-topic. COOLDOWN: after that, hold it back until this many turns have passed.
CORE_STICKY_TURNS = _i("CORE_STICKY_TURNS", 3)
CORE_COOLDOWN_TURNS = _i("CORE_COOLDOWN_TURNS", 8)
# Facts that must never be held back -- her name is not recitation, it's knowing you.
CORE_ALWAYS_PATTERN = os.environ.get("CORE_ALWAYS_PATTERN", "name is")

# Emotion (pillar 2). A local RoBERTa GoEmotions classifier maps each message to
# 28 emotions, folded into 6 mood channels that decay toward a baseline. Runs on
# CPU (the 9070XT can't use the v1 CUDA path) so it keeps all VRAM for the LLMs.
EMOTION_ENABLED = _flag("EMOTION_ENABLED", True)
EMOTION_MODEL = os.environ.get("EMOTION_MODEL", "SamLowe/roberta-base-go_emotions")
# How reactive mood is to a single message (v1-tuned); noise floor drops weak labels.
EMOTION_PULL_STRENGTH = _f("EMOTION_PULL_STRENGTH", 0.4)
EMOTION_NOISE_FLOOR = _f("EMOTION_NOISE_FLOOR", 0.05)

# Tick loop (pillar 3, proactivity). An internal heartbeat that runs background jobs
# on a cadence: for this slice, mood drift toward baseline while the user is away, and
# idle consolidation of the pending buffer. Outward reach-out (unprompted messages) is
# a later slice. Jobs only act once the user has been idle for TICK_IDLE_SECONDS, so
# nothing fires mid-conversation.
TICK_ENABLED = _flag("TICK_ENABLED", True)
TICK_INTERVAL = _f("TICK_INTERVAL", 60)          # base loop cadence (s)
TICK_IDLE_SECONDS = _f("TICK_IDLE_SECONDS", 90)  # "user is away" threshold
# Consolidate a non-empty pending buffer once the user has been idle this long.
IDLE_CONSOLIDATE_AFTER = _f("IDLE_CONSOLIDATE_AFTER", 180)

# Proactive reach-out: after the user has been away REACHOUT_MIN_IDLE seconds, the
# tick may generate an unprompted message (Mari decides whether it's worth it and can
# decline). REACHOUT_COOLDOWN throttles attempts (persisted across restarts) so she's
# a companion, not a nag. Only the web UI surfaces these; the REPL runs internal jobs only.
REACHOUT_ENABLED = _flag("REACHOUT_ENABLED", True)
REACHOUT_MIN_IDLE = _f("REACHOUT_MIN_IDLE", 900)    # 15 min away
REACHOUT_COOLDOWN = _f("REACHOUT_COOLDOWN", 7200)   # >=2h between attempts

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
FOLLOWUP_ENABLED = _flag("FOLLOWUP_ENABLED", True)
FOLLOWUP_CHANCE = _f("FOLLOWUP_CHANCE", 0.2)        # chance per eligible tick (low: rare)
FOLLOWUP_MIN_DELAY = _f("FOLLOWUP_MIN_DELAY", 0)    # min secs after her reply
FOLLOWUP_WINDOW = _f("FOLLOWUP_WINDOW", 60)         # must land soon or the moment's gone
FOLLOWUP_MAX_PER_TURN = _i("FOLLOWUP_MAX_PER_TURN", 1)  # follow-ups per user turn

# Self-reflection: while the user is away, Mari writes a short private thought (a journal
# to herself, never shown in chat) about how she's doing and the conversations. Internal
# cognition — the substrate for reminisce and, later, the self-modifying persona.
REFLECT_ENABLED = _flag("REFLECT_ENABLED", True)
REFLECT_MIN_IDLE = _f("REFLECT_MIN_IDLE", 120)      # think after 2 min away
REFLECT_COOLDOWN = _f("REFLECT_COOLDOWN", 600)      # at most every ~10 min

# Intentions (the "planning" pillar): during idle, Mari notes forward intentions — things she means to
# bring up or find out next time — which reach-out then draws on to be purposeful. Its own idle cadence.
INTENTION_ENABLED = _flag("INTENTION_ENABLED", True)
INTENTION_MIN_IDLE = _f("INTENTION_MIN_IDLE", 180)   # note them a few min after a chat
INTENTION_COOLDOWN = _f("INTENTION_COOLDOWN", 900)   # at most ~every 15 min
INTENTION_MAX_ACTIVE = _i("INTENTION_MAX_ACTIVE", 8)   # cap the open agenda (drop oldest)
# Stale intentions expire so the agenda doesn't linger on things that stopped mattering (0 = never).
INTENTION_MAX_AGE_DAYS = _f("INTENTION_MAX_AGE_DAYS", 7)

# Learned operating-notes (the self-improvement loop): during idle, Mari distills short notes on HOW to be
# with this person from recent experience ("ease off the questions"), injected live into her prompt. Slower
# than reflection (a lesson, not a mood) but faster than the persona rewrite (behavior, not identity).
SELFNOTES_ENABLED = _flag("SELFNOTES_ENABLED", True)
SELFNOTES_MIN_IDLE = _f("SELFNOTES_MIN_IDLE", 300)    # 5 min away
SELFNOTES_COOLDOWN = _f("SELFNOTES_COOLDOWN", 1800)   # at most ~every 30 min
SELFNOTES_MAX_CHARS = _i("SELFNOTES_MAX_CHARS", 400)    # keep the injected block small

# Self-modifying persona: during idle ticks Mari rewrites a bot-owned "who you've become"
# slot in her system prompt, reading her thought journal + core memories. How far it may
# drift is gated by a familiarity meter (derived from message count) so a stranger doesn't
# rewrite herself into a close friend on day one. Slow and rare (a personality shifts gradually).
PERSONA_EDIT_ENABLED = _flag("PERSONA_EDIT_ENABLED", True)
PERSONA_EDIT_MIN_IDLE = _f("PERSONA_EDIT_MIN_IDLE", 300)     # 5 min away
PERSONA_EDIT_COOLDOWN = _f("PERSONA_EDIT_COOLDOWN", 3600)    # at most hourly
PERSONA_MIN_MESSAGES = _i("PERSONA_MIN_MESSAGES", 20)          # need some history first
PERSONA_MAX_CHARS = _i("PERSONA_MAX_CHARS", 600)               # keep the slot small
# Messages of interaction to reach full familiarity (stranger -> close friend), roughly.
FAMILIARITY_MESSAGES = _i("FAMILIARITY_MESSAGES", 400)

# Sleep / standby (§2.8): after a long idle, Mari unloads the LLM from VRAM to free the
# machine (the brain sleeps; the heartbeat keeps ticking). The next message wakes her,
# reloading the model ("waking up…"). Model-using tick jobs pause while she's asleep.
# Requires the LM Studio `lms` CLI; auto-disables if it isn't found.
SLEEP_ENABLED = _flag("SLEEP_ENABLED", True)
SLEEP_AFTER_IDLE = _f("SLEEP_AFTER_IDLE", 1800)   # 30 min away -> sleep
LMS_PATH = os.environ.get("LMS_PATH", "lms")                           # LM Studio CLI

# When the user says goodnight, ask her whether she wants to go quiet too, instead of
# only ever timing out into standby 30 minutes later. Costs ONE structured call, and
# only on turns where core/cues.py sees a farewell — not on every turn. Declining is
# free: SLEEP_AFTER_IDLE still fires, so this can only make standby earlier and
# CHOSEN, never prevent it. Needs SLEEP_ENABLED (it's the same model_manager).
SLEEP_ON_FAREWELL_ENABLED = _flag("SLEEP_ON_FAREWELL_ENABLED", True)

# Internal drives (multi-drive proactivity, roadmap arc A1 — "observe first" slice).
# Slow-integrating scalars (connection, restlessness) that rise while you're away,
# modulated by mood, and relax on contact — a more lifelike generalization of the tick's
# fixed idle gates (V2_PLAN §2.9). This slice only *observes* them (surfaced in the status
# panel); the behaviors still fire on their existing idle gates until the drives prove out.
# Persisted like mood so they survive restarts. Rise rates / weights live in core/drives.py.
DRIVES_ENABLED = _flag("DRIVES_ENABLED", True)
DRIVE_AWAY_AFTER = _f("DRIVE_AWAY_AFTER", 90)  # idle (s) before drives rise
# Behavior thresholds: reach-out fires when `connection` crosses this, reflection when
# `restlessness` does (both still gated by their persisted cooldowns as a hard floor).
# 0.6 connection ≈ 15 min neutral / sooner after a warm or sad chat; 0.4 restlessness ≈ 5 min.
DRIVE_CONNECTION_THRESHOLD = _f("DRIVE_CONNECTION_THRESHOLD", 0.6)
DRIVE_RESTLESSNESS_THRESHOLD = _f("DRIVE_RESTLESSNESS_THRESHOLD", 0.4)
# Energy / body cycle (arc A2): sleep ALSO fires when the fatigue reserve drops to this level
# and she's been briefly idle (ENERGY_SLEEP_MIN_IDLE — so she doesn't nod off mid-chat), on top
# of the long-idle SLEEP_AFTER_IDLE trigger. Deplete/restore rates live in core/drives.py.
ENERGY_SLEEP_THRESHOLD = _f("ENERGY_SLEEP_THRESHOLD", 0.15)
ENERGY_SLEEP_MIN_IDLE = _f("ENERGY_SLEEP_MIN_IDLE", 120)

# --- Pursuits (§A4: she can make herself unavailable) ---
#
# A3: during idle, mine grounded open questions about the user (cited to a real
# message id; uncited output is rejected). A slow, "nightly deep pass" cadence.
PURSUIT_MINING_ENABLED = _flag("PURSUIT_MINING_ENABLED", True)
PURSUIT_MINING_MIN_IDLE = _f("PURSUIT_MINING_MIN_IDLE", 1800)     # 30 min away
PURSUIT_MINING_COOLDOWN = _f("PURSUIT_MINING_COOLDOWN", 86400)    # ~once a day
PURSUIT_WINDOW_MESSAGES = _i("PURSUIT_WINDOW_MESSAGES", 40)       # how far back to mine
PURSUIT_MIN_MESSAGES = _i("PURSUIT_MIN_MESSAGES", 8)              # never mine a barren window
PURSUIT_MAX_ACTIVE = _i("PURSUIT_MAX_ACTIVE", 5)                  # cap the open backlog
# Salience floor to mine (reuses the same signal CONSOLIDATE_SALIENCE reads).
PURSUIT_SALIENCE = _f("PURSUIT_SALIENCE", CONSOLIDATE_SALIENCE)

# A4: while restless enough, she may step away to do one of a closed set of real
# things (journal, revise self-notes/persona, sit with an A3-mined question), for an
# "educated random" bounded duration, and won't reply until she's back or you write
# again (she'll just acknowledge — see PursuitReturnJob). OFF by default: enable only
# once live-tested.
PURSUIT_UNAVAILABLE_ENABLED = _flag("PURSUIT_UNAVAILABLE_ENABLED", False)
PURSUIT_UNAVAILABLE_MIN_IDLE = _f("PURSUIT_UNAVAILABLE_MIN_IDLE", 900)     # 15 min away
PURSUIT_UNAVAILABLE_COOLDOWN = _f("PURSUIT_UNAVAILABLE_COOLDOWN", 21600)   # >=6h between
# Higher than DRIVE_RESTLESSNESS_THRESHOLD (0.4, journaling's own gate) — stepping away
# entirely is a bigger deal than a private journal entry, and restlessness (not
# connection) is the drive because it's idle/boredom-driven, not warmth-driven, so
# this can't correlate with how invested the user is in the conversation.
PURSUIT_UNAVAILABLE_THRESHOLD = _f("PURSUIT_UNAVAILABLE_THRESHOLD", 0.8)
# Never step away right after a heavy disclosure — the same signal A3 uses to decide
# TO mine, applied as a ceiling instead of a floor.
PURSUIT_UNAVAILABLE_CALM_CEILING = _f("PURSUIT_UNAVAILABLE_CALM_CEILING", CONSOLIDATE_SALIENCE)

# Per-pursuit "educated random" duration ranges (seconds) — bounded, minutes not hours.
PURSUIT_JOURNAL_MIN = _f("PURSUIT_JOURNAL_MIN", 60)
PURSUIT_JOURNAL_MAX = _f("PURSUIT_JOURNAL_MAX", 240)
PURSUIT_SELFNOTES_MIN = _f("PURSUIT_SELFNOTES_MIN", 60)
PURSUIT_SELFNOTES_MAX = _f("PURSUIT_SELFNOTES_MAX", 180)
PURSUIT_PERSONA_MIN = _f("PURSUIT_PERSONA_MIN", 120)
PURSUIT_PERSONA_MAX = _f("PURSUIT_PERSONA_MAX", 360)
PURSUIT_SIT_MIN = _f("PURSUIT_SIT_MIN", 180)
PURSUIT_SIT_MAX = _f("PURSUIT_SIT_MAX", 600)

# Tools (pillar 4): native function-calling. When enabled, the chat turn streams
# through a tool loop so Mari can call registered tools (clock, reminisce, more
# later). Verified 100% reliable on the chat model (scripts/tool_probe.py). The
# loop is capped at TOOL_MAX_ITERS tool round-trips per turn so it can never hang.
TOOLS_ENABLED = _flag("TOOLS_ENABLED", True)
TOOL_MAX_ITERS = _i("TOOL_MAX_ITERS", 5)

# Web server.
WEB_HOST = os.environ.get("WEB_HOST", "127.0.0.1")
WEB_PORT = _i("WEB_PORT", 8000)
