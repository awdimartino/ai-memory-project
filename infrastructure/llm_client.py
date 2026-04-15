# File imports
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
                        response += content
            print("\n")
            return response
    
    @staticmethod
    def query(self,messages):
            """Get a complete response from the LLM based on the provided messages."""
            response = self.client.chat.completions.create(
                model=config.BOT_MODEL,
                temperature=config.BOT_TEMPERATURE,
                messages=messages,
                stream=False
                )
            return response.choices[0].message.content