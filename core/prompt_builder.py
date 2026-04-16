import datetime
from infrastructure import config

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
            messages.append({
                "role": "assistant",
                "content": (
                    "Your relevant memories:\n"
                    f"{memories}"
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
