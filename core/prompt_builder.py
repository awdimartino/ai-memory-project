import datetime
from infrastructure import config
from core.models import MemoryRecord


class PromptBuilder:
    @staticmethod
    def build_response_prompt(
        query,
        conversation,
        emotions,
        memories="",
        last_thought=""
    ):
        messages = [
            {
                "role": "system",
                "content": (
                    f"System Prompt:\n"
                    f"{config.BOT_PROMPT}\n"
                    f"Your self described personality:\n"
                    f"PLACEHOLDER\n"
                    f"Current date and time: "
                    f"{datetime.datetime.now():%A, %b %d at %I:%M %p}\n"
                    f"{emotions}"
                )
            }
        ]

        if last_thought:
            messages.append({
                "role": "assistant",
                "content": f"Internal reasoning (not spoken):\n{last_thought}"
            })

        if memories:
            memory_list = []
            for memory in memories:
                memory_list.append(memory.content)
            messages.append({
                "role": "assistant",
                "content": (
                    "Your relevant memories:\n"
                    f"{memory_list}"
                )
            })

        messages.extend(conversation[-10:])

        messages.append({
            "role": "user",
            "content": query
        })

        return messages
        
    def build_thought_prompt(self, last_thought):
        pass

    def build_classify_prompt(self, query, response):
        messages = [
            {
                "role": "system",
                "content": f"{config.BRAIN_PROMPT}"
            },
            {
                "role": "user",
                "content": query
            },
            {
                "role": "assistant",
                "content": response
            }
        ]
        return messages