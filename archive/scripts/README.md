# Archived scripts

These were live tooling once. Each answered a specific question that is now closed and
recorded in `HANDOFF.md` / `V2_PLAN.md`, so they are kept for provenance rather than use.
Nothing in the app imports them. Archived 2026-07-19.

| script | the question it answered | where the answer lives |
|---|---|---|
| `tool_probe.py` | Is native function-calling reliable enough on qwen3.5-9b to build the Tier-3 framework on? | **Yes, 100% (18/18).** HANDOFF §6; the framework shipped (§2, pillar 4). Probed a synthetic `add`/`get_weather` toolset that doesn't exist in the app — `scripts/tool_eval.py` measures the real one. |
| `prompt_test.py` | Fast iteration on the seed persona, against `gemma-3-4b-it` / `llama-3.2-3b-instruct`. | Persona work resolved (HANDOFF §7); both target models are long out of play (chat model is qwen3.5-9b). `scripts/bakeoff_personality.py` is the better-instrumented successor. |
| `bench_specdec.py` | Does speculative decoding make reasoning fast enough to afford bounded thinking? | **No — net loss.** 45 tok/s base, +27% predictable / −50% creative; chat is the creative path. HANDOFF §0 lists it under "settled dead ends (don't relitigate)". |
| `reminisce_debug.py` | Why does `reminisce` under-trigger? | Phrasing-sensitivity recorded in HANDOFF §7 (~63%, "inherently harder to trigger"); the remaining lever is parked in §0. The only script referenced in neither planning doc. |

**Deliberately NOT archived:** `probe_reasoning_control.py` looks settled (all reasoning knobs
are no-ops for qwen3.5) but HANDOFF §0 assigns it two live jobs — verifying the LM Studio
template state after the 2026-07-19 flip, and re-probing if LM Studio ships a reasoning budget
(#1838 / #1974). "Wait and re-run" is not the same as "settled".
