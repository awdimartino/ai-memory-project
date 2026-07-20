"""Built-in tool factories.

Each returns a ready-to-register Tool, closing over whatever dependencies it
needs (stores, clock) so the ToolRegistry stays a plain table and handlers stay
testable. Descriptions are written in Mari's second person ("you") because the
model reads them as its own capabilities, and framed as things a friend does
(knowing the time, remembering past talks) rather than assistant functions.
"""
import logging
from datetime import datetime

from core.tools import Tool

logger = logging.getLogger(__name__)


def make_time_tool() -> Tool:
    """Local wall-clock. Lets Mari answer 'what time is it' and be time-of-day aware."""

    async def handler(_args: dict) -> str:
        now = datetime.now().astimezone()
        return now.strftime("It is %A, %B %-d %Y, %-I:%M %p") if _supports_dash() \
            else now.strftime("It is %A, %B %d %Y, %I:%M %p")

    return Tool(
        name="get_current_time",
        description=("Check the current local date and time. Use it whenever the user asks "
                     "what time or day it is, or when knowing the time of day would make your "
                     "reply more grounded (morning vs. late night, etc.)."),
        parameters={"type": "object", "properties": {}, "required": []},
        handler=handler,
    )


def _supports_dash() -> bool:
    """`%-d`/`%-I` (no zero-pad) is glibc-only; Windows' strftime rejects it."""
    try:
        datetime(2020, 1, 5).strftime("%-d")
        return True
    except ValueError:
        return False


# Resolved once at import: it's a property of the interpreter, not of the call.
# The probe (build a datetime, catch a ValueError) used to run on every tool call.
_TIME_FORMAT = ("It is %A, %B %-d %Y, %-I:%M %p" if _supports_dash()
                else "It is %A, %B %d %Y, %I:%M %p")


def make_reminisce_tool(thoughts, store) -> Tool:
    """Look back over past conversations + Mari's private journal to recall something.

    `thoughts` is the ThoughtStore (her reflections), `store` the ConversationStore
    (the episodic log). Given a topic, it keyword-searches past messages and pulls
    recent thoughts, returning a compact digest the model weaves into its reply.
    This is deliberate recall — distinct from the autonomic semantic recall that
    already runs every turn — for "remember when..." moments.
    """

    async def handler(args: dict) -> str:
        topic = str(args.get("topic") or "").strip()
        lines: list[str] = []

        matches = store.search_messages(topic, limit=8) if topic else []
        if matches:
            lines.append(f"From past conversations about {topic!r}:" if topic
                         else "From past conversations:")
            for m in matches:
                who = "you" if m["role"] == "user" else "I"
                snippet = " ".join(m["content"].split())[:160]
                lines.append(f"  - {who}: {snippet}")
        elif topic:
            lines.append(f"Nothing in our past conversations mentions {topic!r}.")

        recent = thoughts.recent(5) if thoughts is not None else []
        if recent:
            lines.append("Some private notes I've written to myself lately:")
            for t in recent:
                lines.append(f"  - {' '.join(t['content'].split())[:160]}")

        if not lines:
            return ("Nothing came back — we may not have talked about this before, "
                    "or there's no history yet.")
        return "\n".join(lines)

    return Tool(
        name="reminisce",
        description=("Look back over your past conversations with the user and your own private "
                     "notes to recall something specific you two talked about before. Use it for "
                     "'remember when...' moments, when the user references the past, or when you "
                     "want to bring up something from earlier. Pass the topic to look for."),
        parameters={
            "type": "object",
            "properties": {
                "topic": {"type": "string",
                          "description": "What to look back for, e.g. 'their trip to Denver' or 'the job interview'."}
            },
            "required": ["topic"],
        },
        handler=handler,
    )
