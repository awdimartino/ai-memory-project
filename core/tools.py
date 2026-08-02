"""Tool framework (pillar 4): a registry the companion can call during a turn.

A Tool pairs an OpenAI function schema (what the model sees) with an async
handler (what actually runs). The ToolRegistry hands the LLM the specs and
executes calls the model requests. It's the hot-swappable table the chat loop
consults every turn — register a Tool and the companion can use it next message; no other
code changes.

Robustness is the whole point (the user's ask: "make sure tool calling is robust
so we can add more tools later"). Every failure a real tool loop hits is turned
into a *string result fed back to the model*, never an exception that aborts the
turn:
  - unknown tool name         -> "[error: ...]" result, model recovers
  - handler raises            -> "[error: ...]" result (logged), model recovers
  - handler returns non-str   -> coerced to str
  - malformed argument JSON    -> handled one layer up (LLMClient), same shape

The model sees the error text and can apologize, retry with different args, or
answer without the tool. The turn always completes.
"""
import inspect
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable

logger = logging.getLogger(__name__)

# A handler takes the parsed argument dict and returns a result string (sync or async).
Handler = Callable[[dict], Awaitable[str] | str]


@dataclass
class Tool:
    """One callable capability.

    `parameters` is a JSON Schema object ({"type": "object", "properties": {...},
    "required": [...]}); use an empty-properties object for a no-arg tool. `handler`
    receives the parsed args dict and returns a short result string the model reads
    back. It may be sync or async — the registry awaits accordingly.
    """
    name: str
    description: str
    parameters: dict
    handler: Handler

    def spec(self) -> dict:
        """The OpenAI tool spec the model is shown."""
        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": self.parameters,
            },
        }


# How long a single tool result may be before we truncate it (keeps a chatty tool
# from blowing up the context window on the follow-up call).
MAX_RESULT_CHARS = 4000


class ToolRegistry:
    """The live table of tools available to the chat loop.

    Mutable at runtime: register()/unregister() so tools can be added later
    without touching the loop. `bool(registry)` is False when empty, which is how
    the Companion decides whether to send `tools` at all (no tools => a plain
    streaming turn, zero overhead).
    """

    def __init__(self, tools: list[Tool] | None = None):
        self._tools: dict[str, Tool] = {}
        for t in tools or []:
            self.register(t)

    def register(self, tool: Tool) -> None:
        if tool.name in self._tools:
            logger.info("replacing already-registered tool %r", tool.name)
        self._tools[tool.name] = tool

    def unregister(self, name: str) -> None:
        self._tools.pop(name, None)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def names(self) -> list[str]:
        return list(self._tools)

    def specs(self) -> list[dict]:
        """Tool specs for the API call, or [] when empty (callers pass None then)."""
        return [t.spec() for t in self._tools.values()]

    def __bool__(self) -> bool:
        return bool(self._tools)

    def __len__(self) -> int:
        return len(self._tools)

    async def execute(self, name: str, args: dict) -> str:
        """Run one requested tool and return its result string.

        Never raises: an unknown tool or a handler exception becomes an "[error: ...]"
        string so the model can see what went wrong and carry on. `args` is the
        already-parsed dict (arg-JSON parsing failures are handled by the caller).
        """
        tool = self._tools.get(name)
        if tool is None:
            logger.warning("model called unknown tool %r (have: %s)", name, self.names())
            return (f"[error: no tool named {name!r}. "
                    f"Available tools: {', '.join(self.names()) or 'none'}.]")
        if not isinstance(args, dict):
            args = {}
        try:
            result = tool.handler(args)
            if inspect.isawaitable(result):
                result = await result
        except Exception as e:  # noqa: BLE001 - a tool must never crash the turn
            logger.exception("tool %r raised", name)
            return f"[error running {name}: {type(e).__name__}: {e}]"
        text = result if isinstance(result, str) else str(result)
        if len(text) > MAX_RESULT_CHARS:
            text = text[:MAX_RESULT_CHARS] + "\n[...truncated]"
        return text
