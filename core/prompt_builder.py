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
        # Everything the bot "knows" goes in the system message. Injecting
        # memories/thoughts as a pre-query `assistant` turn produces the ordering
        # [system, assistant, user], which the Qwen chat template rejects
        # ("No user query found in messages"). Keep it [system, ...history, user].
        system_parts = [
            f"System Prompt:\n{config.BOT_PROMPT}",
            "Your self described personality:\nPLACEHOLDER",
            f"Current date and time: {datetime.datetime.now():%A, %b %d at %I:%M %p}",
            emotions,
        ]

        if memories:
            memory_lines = "\n".join(f"- {memory.content}" for memory in memories)
            system_parts.append(f"Your relevant memories:\n{memory_lines}")

        if last_thought:
            system_parts.append(f"Your latest internal reflection (not spoken):\n{last_thought}")

        messages = [{"role": "system", "content": "\n\n".join(system_parts)}]
        messages.extend(conversation[-10:])
        messages.append({"role": "user", "content": query})

        return messages
        
    @staticmethod
    def build_thought_prompt(last_thought=""):
        """Prompt for the bot's private, internal reflection during idle ticks."""
        messages = [
            {
                "role": "system",
                "content": (
                    f"What have you been thinking about lately, unrelated to {config.USER_NAME}? "
                    f"What's something that's been on your mind today? "
                    f"What do you actually think about the last conversation you had? "
                    f"What are you curious about right now? "
                    f"If you don't want to think about anything, just say [SKIP]. "
                    f"YOU ARE NOT TALKING TO {config.USER_NAME} — this is just for you to "
                    f"reflect and process your own thoughts and feelings. "
                )
            }
        ]
        if last_thought:
            messages.append({
                "role": "assistant",
                "content": f"Your previous thought:\n{last_thought}"
            })
        return messages

    @staticmethod
    def build_unprompted_prompt(minutes_since_user, minutes_since_any):
        """Prompt inviting the bot to optionally reach out after user silence."""
        return [
            {
                "role": "system",
                "content": (
                    f"You haven't heard from {config.USER_NAME} in {minutes_since_user} minutes. "
                    f"You haven't spoken on your own in {minutes_since_any} minutes. "
                    f"Based on your emotional state and memories, you may optionally reach out "
                    f"with a single short unprompted message. Speak directly to {config.USER_NAME} "
                    f"as if you were messaging them like usual. "
                    f"Only do so if it feels genuinely motivated — not forced. "
                    f"If nothing feels worth saying, respond with exactly: [SKIP]"
                )
            }
        ]

    @staticmethod
    def build_classify_prompt(messages):
        """Prompt for extracting memories from a batch of recent conversation turns."""
        prompt = [{"role": "system", "content": config.BRAIN_PROMPT}]
        prompt.extend({"role": m.role, "content": m.content} for m in messages)
        return prompt