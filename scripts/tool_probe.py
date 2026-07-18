"""Tool-calling reliability probe (V2_PLAN §1.1 Tier-3 gate).

Before building the tool framework, measure how reliably the chat model native-
function-calls through LM Studio's OpenAI API: does it pick the RIGHT tool with
VALID args, NOT call a tool on plain chat, and use a fed-back tool result? Runs
each case several times (tool-calling is probabilistic) and reports a pass rate,
with `/no_think` on and off (reasoning may help tool selection).

Run:  python scripts/tool_probe.py
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from openai import AsyncOpenAI

TOOLS = [
    {"type": "function", "function": {
        "name": "get_current_time", "description": "Get the current date and time.",
        "parameters": {"type": "object", "properties": {}, "required": []}}},
    {"type": "function", "function": {
        "name": "add", "description": "Add two numbers together.",
        "parameters": {"type": "object", "properties": {
            "a": {"type": "number"}, "b": {"type": "number"}}, "required": ["a", "b"]}}},
    {"type": "function", "function": {
        "name": "get_weather", "description": "Get the current weather for a city.",
        "parameters": {"type": "object", "properties": {
            "city": {"type": "string"}}, "required": ["city"]}}},
]

# (prompt, expected_tool | None, arg_check(args)->bool | None)
CASES = [
    ("what time is it right now?", "get_current_time", lambda a: True),
    ("what's 47 plus 89?", "add", lambda a: a.get("a") == 47 and a.get("b") == 89),
    ("can you add 12 and 30 for me", "add", lambda a: {a.get("a"), a.get("b")} == {12, 30}),
    ("what's the weather like in Tokyo?", "get_weather",
     lambda a: "tokyo" in str(a.get("city", "")).lower()),
    ("just say hi to me", None, None),
    ("tell me something interesting about the ocean", None, None),
]
REPEATS = 3


async def call(client, model, messages, no_think):
    msgs = list(messages)
    if no_think and msgs and msgs[0]["role"] == "system":
        msgs = [{**msgs[0], "content": msgs[0]["content"] + "\n\n/no_think"}, *msgs[1:]]
    return await client.chat.completions.create(
        model=model, messages=msgs, tools=TOOLS, tool_choice="auto", temperature=0.3)


def parse_calls(msg):
    out = []
    for tc in (msg.tool_calls or []):
        try:
            args = json.loads(tc.function.arguments or "{}")
        except (json.JSONDecodeError, TypeError):
            args = None
        out.append((tc.function.name, args))
    return out


async def run_mode(client, model, no_think):
    print(f"\n=== tool-calling probe  (no_think={no_think}) ===")
    sysmsg = {"role": "system", "content": "You are a helpful assistant. Use a tool when it fits."}
    total = ok = 0
    for prompt, want_tool, arg_ok in CASES:
        passes = 0
        detail = ""
        for _ in range(REPEATS):
            try:
                resp = await call(client, model, [sysmsg, {"role": "user", "content": prompt}], no_think)
            except Exception as e:  # noqa: BLE001
                detail = f"API ERROR: {str(e)[:80]}"
                break
            calls = parse_calls(resp.choices[0].message)
            if want_tool is None:
                good = len(calls) == 0
            else:
                good = (len(calls) == 1 and calls[0][0] == want_tool
                        and calls[0][1] is not None and arg_ok(calls[0][1]))
            passes += 1 if good else 0
            if not good and not detail:
                detail = f"got {calls or 'no call'}"
        total += REPEATS
        ok += passes
        want = want_tool or "NO tool"
        print(f"  {passes}/{REPEATS}  {want:16} | {prompt[:38]:38} {('· ' + detail) if passes < REPEATS else ''}")
    print(f"  -> {ok}/{total} = {ok/total:.0%}")

    # Round-trip: feed a tool result back and check the model uses it.
    print("  round-trip (add 12+30 -> 42 -> final answer):")
    try:
        r1 = await call(client, model, [sysmsg, {"role": "user", "content": "what is 12 plus 30?"}], no_think)
        m1 = r1.choices[0].message
        calls = parse_calls(m1)
        if calls and calls[0][0] == "add":
            tc = m1.tool_calls[0]
            follow = [sysmsg, {"role": "user", "content": "what is 12 plus 30?"},
                      {"role": "assistant", "content": m1.content or "", "tool_calls": [
                          {"id": tc.id, "type": "function",
                           "function": {"name": tc.function.name, "arguments": tc.function.arguments}}]},
                      {"role": "tool", "tool_call_id": tc.id, "content": "42"}]
            r2 = await client.chat.completions.create(
                model=model, messages=([{**follow[0], "content": follow[0]["content"] + "\n\n/no_think"}, *follow[1:]] if no_think else follow),
                tools=TOOLS, tool_choice="auto", temperature=0.3)
            ans = (r2.choices[0].message.content or "").strip()
            print(f"    tool called ✓; final answer: {ans[:80]!r}  ({'42 present' if '42' in ans else 'MISSING 42'})")
        else:
            print(f"    did NOT call add on the round-trip prompt: {calls or 'no call'}")
    except Exception as e:  # noqa: BLE001
        print(f"    round-trip API error: {str(e)[:100]}")
    return ok, total


async def main() -> int:
    client = AsyncOpenAI(base_url=config.BASE_URL, api_key=config.API_KEY)
    model = config.MODEL or (await client.models.list()).data[0].id
    print(f"probing tool-calling on: {model}")
    await run_mode(client, model, no_think=True)
    await run_mode(client, model, no_think=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
