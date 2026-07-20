"""Async wrapper over the OpenAI SDK pointed at LM Studio.

Async so token streaming never blocks the event loop (the runtime decision:
single async process). Hides any leading <think> reasoning block and measures
time-to-first-token and tokens/sec.

All model calls go through a single lock: LM Studio (llama.cpp) can't safely
serve two concurrent requests to a model, so background consolidation and live
chat must never overlap. This is the seed of the §1.2 priority-queue arbiter.

Transient LM Studio failures ("predict request failed: fetch failed", connection
resets) are retried a few times. Chat retries only *before the first visible
token* so we never re-emit a partial stream; structured calls are non-streaming
and always safe to retry.
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

_RETRYABLE_MARKERS = ("fetch failed", "predict request", "connection error",
                      "connection reset", "timeout", "temporarily")


def _is_retryable(exc: Exception) -> bool:
    return any(m in str(exc).lower() for m in _RETRYABLE_MARKERS)


class LLMClient:
    def __init__(self, base_url: str, api_key: str, model: str, temperature: float,
                 no_think: bool = False, frequency_penalty: float = 0.0,
                 presence_penalty: float = 0.0, max_retries: int = 0):
        self.client = AsyncOpenAI(base_url=base_url, api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.no_think = no_think
        self.frequency_penalty = frequency_penalty
        self.presence_penalty = presence_penalty
        self.max_retries = max_retries
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

        Returns (full_text, stats) with ttft, tok_per_s, tokens, estimated (plus an
        always-empty `tools`, for one stats shape across both entry points). Retries
        transient failures, but only while nothing has been emitted yet.

        This is `stream_with_tools`' single round with no tools attached — the same
        call `stream_with_tools` already makes to force a plain answer when the tool
        loop hits `max_iters`. Sharing it keeps the <think>-stripping state machine
        and the retry-before-first-emit guard as ONE implementation; they were
        previously duplicated here and in `_tool_step_once`, free to drift apart.
        """
        start = time.perf_counter()
        text, _calls, tokens, first_at, reasoning = await self._tool_step(
            self._prep(messages), on_token, None)
        return text, self._tool_stats(start, first_at, tokens or 0, text, [], reasoning)

    # ---- native tool-calling (pillar 4) --------------------------------------
    # LM Studio streams tool calls as deltas (finish_reason="tool_calls", empty
    # content), so we keep token-streaming for the final answer and only loop when
    # the model actually asks for a tool. The loop is defensive at every seam: a
    # malformed-args call, an unknown tool, or a handler error all come back as a
    # result *string* the model reads and recovers from — the turn always ends.

    async def stream_with_tools(self, messages, on_token: Callable[[str], Awaitable],
                                registry, max_iters: int = 5):
        """Stream a reply, letting the model call tools from `registry` in a loop.

        Same contract as stream(): awaits on_token(text) for each visible chunk and
        returns (final_text, stats). Each round streams a completion with the tools
        attached; if the model asks for tools (content empty), we run them, append
        the results, and loop; the round that streams real content is the answer.
        `registry` needs `.specs()` and `async .execute(name, args)`.
        """
        convo = self._prep([dict(m) for m in messages])
        start = time.perf_counter()
        first_at = None
        total_tokens = 0
        invocations: list[dict] = []  # what ran this turn, surfaced for the UI inspector
        # Thinking from EVERY round, not just the last: on a tool turn the interesting
        # reasoning is usually the round that decided to call the tool.
        thinking: list[str] = []

        for _ in range(max_iters):
            text, calls, ct, ft, reasoning = await self._tool_step(
                convo, on_token, registry.specs())
            if ft is not None and first_at is None:
                first_at = ft
            if ct:
                total_tokens += ct
            if reasoning:
                thinking.append(reasoning)
            if not calls:  # no tool requested -> this was the final answer
                return text, self._tool_stats(start, first_at, total_tokens, text,
                                              invocations, "\n\n---\n\n".join(thinking))

            # Record the assistant's tool-call turn, then feed back each result.
            convo.append({
                "role": "assistant",
                "content": text or "",
                "tool_calls": [
                    {"id": c["id"], "type": "function",
                     "function": {"name": c["name"], "arguments": c["raw_args"]}}
                    for c in calls
                ],
            })
            for c in calls:
                result = await self._run_call(registry, c)
                invocations.append({"name": c["name"], "args": c["raw_args"],
                                    "result": result[:200]})
                convo.append({"role": "tool", "tool_call_id": c["id"], "content": result})

        # Safety net: the model kept calling tools past the cap. Force one final
        # answer with no tools so a turn can never hang in a tool loop.
        logger.warning("tool loop hit max_iters=%d; forcing a plain answer", max_iters)
        text, _, ct, ft, reasoning = await self._tool_step(convo, on_token, None)
        if ft is not None and first_at is None:
            first_at = ft
        if reasoning:
            thinking.append(reasoning)
        return text, self._tool_stats(start, first_at, total_tokens + (ct or 0), text,
                                      invocations, "\n\n---\n\n".join(thinking))

    async def _run_call(self, registry, call: dict) -> str:
        """Parse one call's args and execute it, turning any failure into a result string."""
        raw = call["raw_args"]
        try:
            args = json.loads(raw) if raw and raw.strip() else {}
            if not isinstance(args, dict):
                raise ValueError("arguments were not a JSON object")
        except (json.JSONDecodeError, TypeError, ValueError) as e:
            logger.warning("tool %r got malformed args %r: %s", call["name"], raw, e)
            return (f"[error: could not parse arguments for {call['name']} ({e}). "
                    f"Call it again with a valid JSON object.]")
        result = await registry.execute(call["name"], args)
        logger.info("tool %s(%s) -> %s", call["name"], args, result[:120].replace("\n", " "))
        return result

    async def _tool_step(self, messages, on_token: Callable[[str], Awaitable], tools):
        """One streaming completion, with retry-before-first-emit (like stream())."""
        emitted = {"any": False}

        async def guarded(t: str) -> None:
            emitted["any"] = True
            await on_token(t)

        for attempt in range(self.max_retries + 1):
            emitted["any"] = False
            try:
                return await self._tool_step_once(messages, guarded, tools)
            except Exception as e:  # noqa: BLE001
                if emitted["any"] or attempt >= self.max_retries or not _is_retryable(e):
                    raise
                logger.warning("tool-step failed (%s); retry %d/%d", e, attempt + 1, self.max_retries)
                await asyncio.sleep(0.5 * (attempt + 1))

    async def _tool_step_once(self, messages, on_token: Callable[[str], Awaitable], tools):
        """Stream one completion: emit visible content (think-stripped) AND accumulate
        any tool_call deltas. Returns
        (text, calls, completion_tokens, first_token_at, reasoning).
        `calls` is [{id, name, raw_args}] — empty when the model just answered.

        `reasoning` is the model's hidden thinking, captured for the inspector. Note
        reasoning arrives on TWO different channels and only one was ever handled:
        an inline `<think>…</think>` block inside `content` (stripped below), and a
        separate `reasoning_content` delta field, which LM Studio uses for qwen and
        which this dropped on the floor. Bounded thinking (§0) made that content
        worth seeing, so it is now kept rather than discarded.
        """
        raw = ""
        reasoning = ""
        visible_started = False
        first_token_at = None
        completion_tokens = None
        tool_frags: dict[int, dict] = {}  # call index -> {id, name, args} (args arrive in pieces)

        async with self._model_lock:
            kwargs = dict(
                model=self.model,
                temperature=self.temperature,
                frequency_penalty=self.frequency_penalty,
                presence_penalty=self.presence_penalty,
                messages=messages,
                stream=True,
                stream_options={"include_usage": True},
            )
            if tools:
                kwargs["tools"] = tools
                kwargs["tool_choice"] = "auto"
            stream = await self.client.chat.completions.create(**kwargs)

            async for chunk in stream:
                if getattr(chunk, "usage", None):
                    completion_tokens = chunk.usage.completion_tokens
                if not chunk.choices:
                    continue
                delta = chunk.choices[0].delta

                # getattr: LM Studio is the server and its delta shape has varied;
                # a missing key must not crash a plain no-tool reply.
                for tc in (getattr(delta, "tool_calls", None) or []):  # names whole, args streamed
                    slot = tool_frags.setdefault(tc.index, {"id": None, "name": "", "args": ""})
                    if tc.id:
                        slot["id"] = tc.id
                    if tc.function and tc.function.name:
                        slot["name"] = tc.function.name
                    if tc.function and tc.function.arguments:
                        slot["args"] += tc.function.arguments

                # The separate reasoning channel (getattr: not every server sends it).
                reasoning += getattr(delta, "reasoning_content", None) or ""

                if delta.content is None:
                    continue
                raw += delta.content

                if visible_started:
                    await on_token(delta.content)
                    continue
                stripped = raw.lstrip()
                if not stripped:
                    continue
                if stripped.startswith(_THINK_OPEN):
                    if "</think>" in raw:
                        visible_started = True
                        first_token_at = time.perf_counter()
                        rest = raw.split("</think>", 1)[1].lstrip()
                        if rest:
                            await on_token(rest)
                elif _THINK_OPEN.startswith(stripped):
                    continue
                else:
                    visible_started = True
                    first_token_at = time.perf_counter()
                    await on_token(raw)

        # An inline <think> block is the OTHER reasoning channel; keep it too, so the
        # inspector shows thinking regardless of which way the server delivered it.
        inline = _THINK_BLOCK.search(raw)
        if inline and not reasoning:
            reasoning = inline.group(0).strip()

        text = _THINK_BLOCK.sub("", raw).strip()
        # Flush a short answer that never tripped the emit checks — but only on a
        # pure answer turn (when there's a tool call, content is scaffolding, not reply).
        if not visible_started and text and not tool_frags:
            first_token_at = first_token_at or time.perf_counter()
            await on_token(text)

        calls = [
            {"id": v["id"] or f"call_{i}", "name": v["name"], "raw_args": v["args"]}
            for i, v in sorted(tool_frags.items())
            if v["name"]  # ignore an empty fragment (never a real call)
        ]
        return text, calls, completion_tokens, first_token_at, reasoning.strip()

    def _tool_stats(self, start: float, first_at: float | None, tokens: int, text: str,
                    invocations: list[dict], reasoning: str = "") -> dict:
        """Same stats shape as stream() plus a `tools` list; ttft spans any tool
        round-trips (honest latency: a tool turn genuinely takes longer)."""
        end = time.perf_counter()
        ft = first_at or end
        gen_time = max(end - ft, 1e-6)
        estimated = not tokens
        tokens = tokens or max(1, round(len(text) / 4))
        return {"ttft": ft - start, "tok_per_s": tokens / gen_time,
                "tokens": tokens, "estimated": estimated, "tools": invocations,
                "reasoning": reasoning}

    async def _create_retry(self, **kwargs):
        """Non-streaming completion with the model lock + transient-error retries."""
        async with self._model_lock:
            for attempt in range(self.max_retries + 1):
                try:
                    return await self.client.chat.completions.create(**kwargs)
                except Exception as e:  # noqa: BLE001
                    if attempt >= self.max_retries or not _is_retryable(e):
                        raise
                    logger.warning("structured call failed (%s); retry %d/%d",
                                   e, attempt + 1, self.max_retries)
                    await asyncio.sleep(0.5 * (attempt + 1))

    async def structured(self, messages, schema: dict, model: str | None = None) -> list:
        """Schema-constrained (Tier-2) call. Returns the `memories` list.

        Tolerates a bare list too. Low temperature for stable extraction.
        """
        resp = await self._create_retry(
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
        resp = await self._create_retry(
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
