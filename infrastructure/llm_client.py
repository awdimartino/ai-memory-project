"""Async wrapper over the OpenAI SDK pointed at LM Studio.

Async so token streaming never blocks the event loop (the runtime decision:
single async process). Hides any leading <think> reasoning block and measures
time-to-first-token and tokens/sec.

All model calls go through a single lock: LM Studio (llama.cpp) can't safely
serve two concurrent requests to a model, so background consolidation and live
chat must never overlap. This is the seed of the §1.2 priority-queue arbiter.
"""
import asyncio
import json
import logging
import re
import time
from typing import Awaitable, Callable

from openai import AsyncOpenAI

logger = logging.getLogger(__name__)

_THINK_OPEN = "<think>"
_THINK_BLOCK = re.compile(r"(?s)^\s*<think>.*?</think>\s*")


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float,
                 no_think: bool = False):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.no_think = no_think
        self._model_lock = asyncio.Lock()

    def _prep(self, messages):
        """Optionally append the /no_think directive to the system message."""
        if not self.no_think:
            return messages
        out = [dict(m) for m in messages]
        for m in out:
            if m.get("role") == "system":
                m["content"] = m["content"].rstrip() + "\n\n/no_think"
                return out
        return [{"role": "system", "content": "/no_think"}, *out]

    async def resolve_model(self) -> str:
        """Return the configured model, or auto-detect the first one LM Studio has loaded."""
        if self.model:
            return self.model
        models = await self.client.models.list()
        ids = [m.id for m in models.data]
        if not ids:
            raise RuntimeError("No model is loaded in LM Studio and MODEL is not set.")
        self.model = ids[0]
        logger.info("auto-detected model: %s", self.model)
        return self.model

    async def stream(self, messages, on_token: Callable[[str], Awaitable]):
        """Stream a reply, awaiting on_token(text) for each visible chunk.

        Returns (full_text, stats) with ttft, tok_per_s, tokens, estimated.
        """
        raw = ""
        visible_started = False
        first_token_at = None
        completion_tokens = None

        async with self._model_lock:
            start = time.perf_counter()
            stream = await self.client.chat.completions.create(
                model=self.model,
                temperature=self.temperature,
                messages=self._prep(messages),
                stream=True,
                stream_options={"include_usage": True},
            )

            async for chunk in stream:
                if getattr(chunk, "usage", None):
                    completion_tokens = chunk.usage.completion_tokens
                if not (chunk.choices and chunk.choices[0].delta.content is not None):
                    continue

                delta = chunk.choices[0].delta.content
                raw += delta

                if visible_started:
                    await on_token(delta)
                    continue

                stripped = raw.lstrip()
                if not stripped:
                    continue
                if stripped.startswith(_THINK_OPEN):
                    # Inside a reasoning block; wait for it to close, then emit the rest.
                    if "</think>" in raw:
                        visible_started = True
                        first_token_at = time.perf_counter()
                        rest = raw.split("</think>", 1)[1].lstrip()
                        if rest:
                            await on_token(rest)
                elif _THINK_OPEN.startswith(stripped):
                    # Still could become "<think>" (e.g. "<", "<th"); keep buffering.
                    continue
                else:
                    # Definitely not a reasoning block; flush what we have.
                    visible_started = True
                    first_token_at = time.perf_counter()
                    await on_token(raw)

            end = time.perf_counter()

        text = _THINK_BLOCK.sub("", raw).strip()

        # Flush anything still buffered (short replies that never tripped the checks
        # above, or a reply that ended while still ambiguous).
        if not visible_started and text:
            first_token_at = first_token_at or end
            await on_token(text)

        ft = first_token_at or end
        gen_time = max(end - ft, 1e-6)
        estimated = completion_tokens is None
        tokens = completion_tokens if completion_tokens else max(1, round(len(text) / 4))
        stats = {
            "ttft": ft - start,
            "tok_per_s": tokens / gen_time,
            "tokens": tokens,
            "estimated": estimated,
        }
        return text, stats

    async def structured(self, messages, schema: dict, model: str | None = None) -> list:
        """Schema-constrained (Tier-2) call. Returns the `memories` list.

        Tolerates a bare list too. Low temperature for stable extraction.
        """
        async with self._model_lock:
            resp = await self.client.chat.completions.create(
                model=model or self.model,
                temperature=0.2,
                messages=self._prep(messages),
                response_format=schema,
            )
        content = resp.choices[0].message.content
        logger.debug("structured raw: %s", content)
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            logger.warning("structured output was not valid JSON")
            return []
        if isinstance(data, dict):
            items = data.get("memories", [])
        elif isinstance(data, list):
            items = data
        else:
            items = []
        return items if isinstance(items, list) else []

    async def structured_json(self, messages, schema: dict, model: str | None = None) -> dict:
        """Schema-constrained call returning a single parsed object ({} on failure)."""
        async with self._model_lock:
            resp = await self.client.chat.completions.create(
                model=model or self.model,
                temperature=0.2,
                messages=self._prep(messages),
                response_format=schema,
            )
        content = resp.choices[0].message.content
        logger.debug("structured_json raw: %s", content)
        try:
            data = json.loads(content)
        except (json.JSONDecodeError, TypeError):
            logger.warning("structured_json output was not valid JSON")
            return {}
        return data if isinstance(data, dict) else {}
