"""Isolate what makes the model reach for reminisce under the real persona.

The smoke showed reminisce rarely fires on 'do you remember X?' phrasings. This
strips away the rest of the stack and asks directly: given the real persona system
prompt + the reminisce tool, which user phrasings trigger a tool call, and does
/no_think vs. thinking change it? Pure diagnostic — informs prompt/tool wording.

Run:  python scripts/reminisce_debug.py
"""
import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config
from core.prompts import build_system
from openai import AsyncOpenAI

REMINISCE_SPEC = {"type": "function", "function": {
    "name": "reminisce",
    "description": ("Look back over your past conversations with the user and your own private "
                    "notes to recall something specific you two talked about before. Use it for "
                    "'remember when...' moments, when the user references the past, or when you "
                    "want to bring up something from earlier. Pass the topic to look for."),
    "parameters": {"type": "object", "properties": {
        "topic": {"type": "string", "description": "What to look back for."}}, "required": ["topic"]}}}

PHRASINGS = [
    "do you remember what my dog's name is?",
    "remember when I told you about my dog?",
    "what did I say my dog was named earlier?",
    "hey what was that thing I mentioned about my trip last week?",
    "you should know this about me already, what's my job?",
]


async def probe(client, model, system, prompt, no_think):
    sysmsg = system + ("\n\n/no_think" if no_think else "")
    resp = await client.chat.completions.create(
        model=model, temperature=0.3, tools=[REMINISCE_SPEC], tool_choice="auto",
        messages=[{"role": "system", "content": sysmsg},
                  {"role": "user", "content": prompt}])
    msg = resp.choices[0].message
    calls = msg.tool_calls or []
    if calls:
        args = calls[0].function.arguments
        return f"CALL reminisce({args})"
    return f"no call · {(msg.content or '').strip()[:70]!r}"


async def main() -> int:
    client = AsyncOpenAI(base_url=config.BASE_URL, api_key=config.API_KEY)
    model = config.MODEL or (await client.models.list()).data[0].id
    system = build_system([], None, tools=["reminisce", "get_current_time"])
    print(f"model: {model}\n")
    print("=== system prompt sent ===")
    print(system[-600:])  # tail: the tools note + end of persona
    print("=" * 40)
    for no_think in (True, False):
        print(f"\n--- no_think={no_think} ---")
        for p in PHRASINGS:
            try:
                out = await probe(client, model, system, p, no_think)
            except Exception as e:  # noqa: BLE001
                out = f"ERROR: {str(e)[:70]}"
            print(f"  {out:45} | {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
