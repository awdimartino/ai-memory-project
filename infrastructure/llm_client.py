# File imports
import json

import infrastructure.config as config
# Library imports
from openai import OpenAI as oai

class LLMClient:
    """Responsible for interfacing with the OpenAI API to generate responses and embeddings."""
    def __init__(self, client):
        self.client = client

    def stream(self, messages):
            """Stream a response from the LLM based on the provided messages."""
            stream = self.client.chat.completions.create(
                model=config.BOT_MODEL,
                temperature=config.BOT_TEMPERATURE,
                messages=messages,
                stream=True
            )
            response = ""
            for chunk in stream:
                if chunk.choices and chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        print(content, end='', flush=True)
                        response += content
            print("\n")
            return response
    
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
            """Classify a query for memory management purposes."""
            response = self.client.chat.completions.create(
                model=config.BRAIN_MODEL,
                temperature=config.BRAIN_TEMPERATURE,
                messages=messages,
                response_format=config.BRAIN_RESPONSE_FORMAT
                )
            try:
                results = json.loads(response.choices[0].message.content)
            except json.JSONDecodeError as e:
                results = {"create_memory": [], "fetch_memory": []}
            return results