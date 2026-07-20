"""Does a bounded thinking budget buy reasoning back without losing the speed?

THE §0 QUESTION, MADE MEASURABLE. Production runs thinking OFF because unbounded
reasoning cost ~2000 hidden tokens per consolidation call (~50s vs ~3.5s). The cost of
that trade is reasoning: both qwen3.5-9b AND gemma-4-12b-qat answer `bat_ball` WRONG
with thinking off, while §0 recorded gpt-oss going WRONG -> OK with more reasoning.

A budget is only a win if it clears BOTH bars:
    (1) it fixes reasoning        -> bat_ball goes WRONG -> OK
    (2) consolidation stays fast  -> extraction near ~3.5s, not ~50s
Bar (2) is the make-or-break one. It is why thinking was turned off in the first place,
and it is measured here on a REAL extraction call rather than a chat prompt, because
that is the call that actually pays.

SETUP (all three are required; skip one and this measures nothing):
  1. LM Studio: enable "[BETA] Enable LM Studio Engine Protocol"
  2. Env: LLAMA_ARG_THINK_BUDGET set BEFORE LM Studio starts (see HANDOFF §0)
  3. Template: thinking back ON -- remove `{% set enable_thinking = false %}`.
     A budget bounds thinking; it cannot bound zero. If `baseline` below reads ~0
     reasoning chars, the template is still off and every other row is meaningless.

    python scripts/probe_thinking_budget.py

Run it once per budget value to find the knee. Env changes need an LM Studio restart --
unless the per-request sweep at the end turns out to work, which would make budgets
settable per call (a tight one for the brain, a looser one for chat).
"""
import os
import time

from _harness import raw_client  # repo-root path setup + UTF-8 stdout

import config  # noqa: E402

client = raw_client()

BAT_BALL = ("A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
            "How much does the ball cost?")

# The real extraction shape: structured output over a conversation window. This is the
# call that made thinking-off necessary, so it is the one that decides.
EXTRACT_MSGS = [
    {"role": "system", "content": "Extract durable facts about the user as JSON."},
    {"role": "user", "content":
     "user: my name is Alex and I work as a welder in Portland\n"
     "assistant: nice\n"
     "user: my border collie Pip is getting old, he's nearly twelve\n"
     "assistant: that's a good age\n"
     "user: my sister Kate keeps telling me to get another one"},
]
EXTRACT_SCHEMA = {"type": "json_schema", "json_schema": {
    "name": "facts", "strict": True,
    "schema": {"type": "object", "additionalProperties": False, "required": ["facts"],
               "properties": {"facts": {"type": "array", "items": {"type": "string"}}}}}}


def call(label, messages, temperature=0.2, **kw):
    body = kw.pop("extra_body", None)
    t = time.perf_counter()
    try:
        r = client.chat.completions.create(model=MODEL, messages=messages,
                                           temperature=temperature,
                                           extra_body=body or {}, **kw)
    except Exception as e:  # noqa: BLE001
        print(f"  {label:32} ERROR: {str(e)[:70]}")
        return None, 0.0, 0
    dt = time.perf_counter() - t
    m = r.choices[0].message
    rc = len(getattr(m, "reasoning_content", None) or "")
    print(f"  {label:32} {dt:6.1f}s   reasoning={rc:6} chars")
    return (m.content or ""), dt, rc


MODEL = config.MODEL or client.models.list().data[0].id
budget = os.environ.get("LLAMA_ARG_THINK_BUDGET", "(unset in THIS shell)")

print(f"=== thinking-budget probe on {MODEL} ===")
print(f"LLAMA_ARG_THINK_BUDGET (this shell): {budget}")
print("NOTE: what matters is the value LM STUDIO inherited at launch, not this shell.\n")

print("--- 1. is thinking on at all? ---")
_, _, base_rc = call("baseline chat", [{"role": "user", "content": "how's it going?"}], 0.8)
if base_rc == 0:
    print("\n  ^ 0 reasoning chars. Either the template still has enable_thinking=false,")
    print("    or Engine Protocol is off. Fix that first - nothing below is meaningful.\n")

print("\n--- 2. does it FIX REASONING? (bar 1) ---")
ans, _, rc = call("bat_ball", [{"role": "user", "content": BAT_BALL}], 0.2)
low = (ans or "").lower().replace(" ", "")
ok = any(k in low for k in ("$0.05", "0.05", "5cents", "fivecents"))
print(f"     answer: {(ans or '')[:100]!r}")
print(f"     -> bat_ball {'CORRECT ✅' if ok else 'WRONG ❌'}  (thinking-off baseline: WRONG)")

print("\n--- 3. does CONSOLIDATION stay fast? (bar 2 - the make-or-break) ---")
times = []
for i in range(3):
    _, dt, rc = call(f"extraction #{i + 1}", EXTRACT_MSGS, 0.2, response_format=EXTRACT_SCHEMA)
    times.append(dt)
avg = sum(times) / len(times)
print(f"     avg {avg:.1f}s per extraction call")
print("     reference: ~3.5s thinking-OFF (acceptable) vs ~50s unbounded (why it was disabled)")
if avg <= 8:
    print("     -> ACCEPTABLE ✅")
elif avg <= 20:
    print("     -> MARGINAL ⚠️  consolidation gets noticeably slower")
else:
    print("     -> TOO SLOW ❌  this is the failure mode thinking-off was created to avoid")

print("\n--- 4. bonus: does PER-REQUEST budget work under Engine Protocol? ---")
print("    (previously stripped by the in-process path. If these differ, the brain can")
print("     run a tight budget and chat a looser one, instead of one global setting.)")
for n in (64, 512):
    call(f"thinking_budget_tokens={n}", [{"role": "user", "content": BAT_BALL}], 0.2,
         extra_body={"thinking_budget_tokens": n})

print("\nVERDICT: a budget is worth adopting only if section 2 is CORRECT *and* section 3 is")
print("ACCEPTABLE. Either one alone is not the win section 0 was waiting for.")
