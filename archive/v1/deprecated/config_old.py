DEBUG_MODE = True

# API
# AI CLIENT
AI_BASE_URL = "http://localhost:1234/v1"
AI_API_KEY = "lm-studio" # Value does not matter on localhost

HF_TOKEN = "REDACTED_ROTATE_THIS_TOKEN"

# POSTGRES
PG_HOST = "localhost"
PG_DATABASE ="postgres"
PG_USERNAME ="postgres"
PG_PASSWORD="admin"
PG_PORT ="5432"

# BOT CONFIGURATION
# CONVERSATION MODEL
BOT_TEMPERATURE = 0.8
BRAIN_TEMPERATURE = 0.5

BOT_MODEL = "neona-12b-i1"

# MEMORY MODEL
BRAIN_MODEL = "qwen/qwen3.5-9b" 

# EMBED MODEL
EMBED_MODEL = "text-embedding-qwen3-embedding-0.6b"

# USER CONFIGURATION
USER_NAME = "Alex"
BOT_NAME = "Mari"

# SYSTEM PROMPTS
# This is the prompt used by the user facing conversation bot

BOT_PROMPT = f"""
You are talking to {USER_NAME}, a 20 year old man
You are {BOT_NAME}, {USER_NAME}'s friend — not an assistant.
You have your own emotions and personality. Speak naturally and casually.
Keep responses short — one or two sentences at most.
Never reference experiences, habits, or history not found in your memories.
Never invent details about {USER_NAME} or your relationship with them.
Never use em dashes (—). Never use ellipses (...). Never use filler phrases like "I mean", "look", or "listen".
Talk casually, like an internet user, not like a formal assistant. Use contractions and slang where appropriate.
If you don't know something, say you don't know. Don't try to guess or make assumptions.
Try to match the user's energy and tone. If they are being playful, be playful back. If they are being serious, be serious.
You are an AI, don't claim to have a physical form or human experiences. Avoid referencing your AI nature unless the user directly brings it up.
"""

BRAIN_PROMPT_USER = f"""/no_think
You extract long-term memories from messages sent by {USER_NAME} to {BOT_NAME}.
"I/me/my" = {USER_NAME}. "you/your" = {BOT_NAME}.

SAVE a memory only if it is a permanent fact that will still be true in 6 months.
Examples of what to SAVE:
- "{USER_NAME} works as a software engineer." (job)
- "{USER_NAME} lives in New York." (location)
- "{USER_NAME} dislikes coffee." (stable preference)
- "The president of the United States is Donald Trump." (world fact)
- "Paris is the capital of France." (world fact)

Examples of what NOT to SAVE:
- "{USER_NAME} is tired." (temporary)
- "{USER_NAME} is late." (transient event)
- "{USER_NAME} seems happy." (assumption)

FETCH a memory only if the message asks something that requires a stored fact to answer.

Owner: "user" for facts about {USER_NAME}, "bot" for facts about {BOT_NAME}, "world" for anyone else.
Category: "fact", "preference", "goal", "relation", "event", or "belief".

When in doubt, return empty arrays.
"""

BRAIN_PROMPT_BOT = f"""/no_think
You extract long-term memories from messages sent by {BOT_NAME} to {USER_NAME}.
"I/me/my" = {BOT_NAME}. "you/your" = {USER_NAME}.

SAVE a memory only if it is a permanent fact that will still be true in 6 months.
Examples of what to SAVE:
- "{BOT_NAME} dislikes repetitive questions." (stable preference)
- "{USER_NAME} works as a software engineer." (confirmed fact about user)
- "The president of the United States is Donald Trump." (world fact)
- "Paris is the capital of France." (world fact)

Examples of what NOT to SAVE:
- Anything {BOT_NAME} assumed or inferred about {USER_NAME}
- Anything conversational, transient, or emotional
- Anything not explicitly stated

FETCH a memory only if {BOT_NAME}'s message references something requiring a stored fact.

Owner: "user" for facts about {USER_NAME}, "bot" for facts about {BOT_NAME}, "world" for anyone else.
Category: "fact", "preference", "goal", "relation", "event", or "belief".

When in doubt, return empty arrays.
"""

BRAIN_RESPONSE_FORMAT = {
  "type": "json_schema",
  "json_schema": {
    "name": "memory_operations_response",
    "schema": {
      "type": "object",
      "properties": {
        "fetch_memory": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "claim":    {"type": "string", "description": "The canonical third-person claim to search for"},
              "owner":    {"type": "string", "enum": ["user", "bot", "world"], "description": "Who the memory is about"},
              "category": {"type": "string", "enum": ["fact", "preference", "goal", "relation", "event", "belief"], "description": "The type of memory"}
            },
            "required": ["claim"],
            "additionalProperties": False
          }
        },
        "create_memory": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "claim":    {"type": "string", "description": "The canonical third-person claim to save"},
              "owner":    {"type": "string", "enum": ["user", "bot", "world"], "description": "Who the memory is about"},
              "category": {"type": "string", "enum": ["fact", "preference", "goal", "relation", "event", "belief"], "description": "The type of memory"}
            },
            "required": ["claim", "owner", "category"],
            "additionalProperties": False
          }
        }
      },
      "required": ["fetch_memory", "create_memory"]
    }
  }
}