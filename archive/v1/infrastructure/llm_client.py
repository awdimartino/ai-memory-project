import json
import logging
import re

import infrastructure.config as config

logger = logging.getLogger(__name__)

_THINK_OPEN = "<think>"
_THINK_BLOCK = re.compile(r"(?s)^\s*<think>.*?</think>\s*")


class LLMClient:
    """Responsible for interfacing with the OpenAI API to generate responses and embeddings."""
    def __init__(self, client):
        self.client = client

    def stream(self, messages):
            """Stream a reply, hiding any leading <think>...</think> reasoning block.

            qwen3.5 emits a tagged reasoning block before the answer. We suppress it
            from both the live output and the stored response, streaming the visible
            answer once the block closes.
            """
            stream = self.client.chat.completions.create(
                model=config.BOT_MODEL,
                temperature=config.BOT_TEMPERATURE,
                messages=messages,
                stream=True
            )
            raw = ""
            visible_started = False
            for chunk in stream:
                if not (chunk.choices and chunk.choices[0].delta.content is not None):
                    continue
                raw += chunk.choices[0].delta.content
                if visible_started:
                    print(chunk.choices[0].delta.content, end='', flush=True)
                    continue
                stripped = raw.lstrip()
                if stripped.startswith(_THINK_OPEN):
                    # Inside a reasoning block — wait for it to close, then emit the rest.
                    if "</think>" in raw:
                        visible_started = True
                        print(raw.split("</think>", 1)[1].lstrip(), end='', flush=True)
                elif len(stripped) >= len(_THINK_OPEN):
                    # Enough text to know there's no reasoning block; stream normally.
                    visible_started = True
                    print(raw, end='', flush=True)
                # else: ambiguous prefix (e.g. "<th") — keep buffering.
            print("\n")
            return _THINK_BLOCK.sub("", raw).strip()
    
    def query(self, messages):
            """Get a complete response from the LLM based on the provided messages."""
            response = self.client.chat.completions.create(
                model=config.BOT_MODEL,
                temperature=config.BOT_TEMPERATURE,
                messages=messages,
                stream=False
                )
            return response.choices[0].message.content
    
    def memory_classification(self, messages):
        """Return a list of memories to save."""
        response = self.client.chat.completions.create(
            model=config.BRAIN_MODEL,
            temperature=config.BRAIN_TEMPERATURE,
            messages=messages,
            response_format=config.BRAIN_RESPONSE_FORMAT
        )
        content = response.choices[0].message.content
        logger.debug("memory_classification raw content: %s", content)

        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            return []

        # Schema wraps the list under "memories"; tolerate a bare list too.
        if isinstance(data, dict):
            memories = data.get("memories", [])
        elif isinstance(data, list):
            memories = data
        else:
            return []

        return memories if isinstance(memories, list) else []