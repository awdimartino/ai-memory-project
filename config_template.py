# API
# AI CLIENT
AI_BASE_URL = "", # AI API endpoint
AI_API_KEY = "" # Value does not matter on localhost

# POSTGRES
PG_HOST = "",
PG_DATABASE ="",
PG_USERNAME ="",
PG_PASSWORD="",
PG_PORT =""

# BOT CONFIGURATION
# CONVERSATION MODEL
BOT_MODEL = "" 

# MEMORY MODEL
BRAIN_MODEL = ""

# EMBED MODEL
EMBED_MODEL = ""

# USER CONFIGURATION
USER_NAME = ""
BOT_NAME = ""

# SYSTEM PROMPTS
# This is the prompt used by the user facing conversation bot
BOT_PROMPT = f"""

"""

# This is the prompt for memory classification used by the brain bot
BRAIN_PROMPT = f"""

"""

# This is the JSON SCHEMA used by the brain bot
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
              "claim": {"type": "string", "description": "The claim being made"},
              "category": {"type": "string", "description": "The category this claim belongs to"}
            },
            "required": ["claim", "category"],
            "additionalProperties": False,
            "description": "A memory entry to fetch."
          }
        },
        "create_memory": {
          "type": "array",
          "items": {
            "type": "object",
            "properties": {
              "claim": {"type": "string", "description": "The claim being made"},
              "category": {"type": "string", "description": "The category this claim belongs to"}
            },
            "required": ["claim", "category"],
            "additionalProperties": False,
            "description": "A memory entry to create."
          }
        }
      },
      "required": ["fetch_memory", "create_memory"],
      "description": "Schema defining how an AI agent should format its response as a list of memory fetch and addition operations with parameters."
    }
  }
}