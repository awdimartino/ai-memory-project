"""Offline coverage for the prompt builders that had none.

These are pure string functions with no I/O, so they're the cheapest tests in the
suite — and prompt regressions here are the expensive kind, because they surface
as *behavior* (she stops calling a tool, or manufactures a lesson) rather than as
an error. The properties pinned below are the ones the build log records as having
actually broken:

  - numbering contracts. `build_batch_decision_user` and `build_core_rerank_user`
    both ask the model to answer with NUMBERS that index back into a list. An
    off-by-one here silently retires the wrong memory.
  - the tools note must OVERRIDE the persona's "you just met / you can't sense
    anything" rules, or the model disclaims memory instead of reaching for
    reminisce (measured: 19/30 before the override wording, 25/30 after).
  - the reach-out cue must carry elapsed time, because the model can't see a clock
    and the history makes the last exchange look immediate.

Run:  python tests/test_prompts.py
"""
from _harness import case, run  # also puts the repo root on sys.path

import config
from core.prompts import (
    build_batch_decision_user,
    build_core_rerank_user,
    build_intentions_user,
    build_persona_edit_system,
    build_persona_edit_user,
    build_reachout_cue,
    build_reflect_system,
    build_tools_note,
)


# --- tools note ------------------------------------------------------------------

@case
async def tools_note_is_none_without_tools():
    assert build_tools_note(None) is None
    assert build_tools_note([]) is None, "no tools -> nothing injected, zero overhead"


@case
async def tools_note_overrides_the_persona_rules_it_conflicts_with():
    note = build_tools_note(["get_current_time", "reminisce"]).lower()
    assert "override" in note, (
        "must explicitly override the 'you just met / can't sense anything' rules — "
        "without that the model disclaims memory instead of calling reminisce")
    assert "call" in note


@case
async def tools_note_only_describes_registered_tools():
    only_time = build_tools_note(["get_current_time"])
    assert "get_current_time" in only_time
    assert "reminisce" not in only_time, "must not advertise a tool that isn't registered"

    only_rem = build_tools_note(["reminisce"])
    assert "reminisce" in only_rem
    assert "get_current_time" not in only_rem


@case
async def tools_note_time_rule_is_ask_only():
    # She volunteered the time as filler; the rule was tightened to fire only when
    # DIRECTLY asked. Guard the wording so that fix can't quietly regress.
    note = build_tools_note(["get_current_time"]).lower()
    assert "directly" in note, "time tool must be gated on a direct request"
    assert "small talk" in note, "must forbid using the time to fill a lull"


@case
async def unknown_tools_still_get_a_line():
    note = build_tools_note(["get_current_time", "web_search"])
    assert "web_search" in note, "a newly registered tool must appear without editing prompts.py"


# --- reach-out cue ---------------------------------------------------------------

@case
async def reachout_cue_carries_elapsed_time_and_offers_pass():
    cue = build_reachout_cue("20 minutes")
    assert "20 minutes" in cue, "the model can't see a clock; the gap must be in the cue"
    assert "PASS" in cue, "she must be able to decline"


# --- numbering contracts (an off-by-one here retires the wrong memory) -----------

@case
async def batch_decision_numbers_candidates_from_one():
    out = build_batch_decision_user([
        ("The user lives in Boston", ["The user lives in New York"]),
        ("The user owns a cat", ["The user owns a dog", "The user owns a bird"]),
    ])
    assert "Candidate 1:" in out and "Candidate 2:" in out, "candidates are 1-indexed"
    assert "Candidate 0:" not in out
    # related memories are numbered WITHIN their candidate, also from 1
    assert "  1. The user lives in New York" in out
    assert "  1. The user owns a dog" in out
    assert "  2. The user owns a bird" in out
    assert "by its number" in out


@case
async def core_rerank_numbers_facts_and_states_the_cap():
    out = build_core_rerank_user(["name is Alex", "is a nurse", "lives in Seattle"], max_keep=2)
    assert "1. name is Alex" in out
    assert "3. lives in Seattle" in out
    assert "0." not in out, "1-indexed, matching the schema's 'numbers to keep'"
    assert "at most 2" in out


# --- intentions ------------------------------------------------------------------

@case
async def intentions_user_shows_the_open_agenda_for_dedupe():
    out = build_intentions_user(
        [{"role": "user", "content": "work was rough"},
         {"role": "assistant", "content": "that sounds heavy"}],
        ["ask how Deadlock is going"])
    assert "work was rough" in out, "the window it learns from"
    assert "ask how Deadlock is going" in out, (
        "the open agenda must be visible or form_intentions re-adds duplicates")


@case
async def intentions_user_handles_an_empty_agenda():
    out = build_intentions_user([{"role": "user", "content": "hey"}], [])
    assert "hey" in out  # must not crash or emit a stray empty block


# --- persona edit ----------------------------------------------------------------

@case
async def persona_edit_system_states_familiarity_and_budget():
    out = build_persona_edit_system("a new acquaintance", 600)
    assert "a new acquaintance" in out, (
        "familiarity is the ceiling on drift — a stranger must not rewrite herself "
        "into a best friend")
    assert "600" in out, "character budget must be stated to the model"


@case
async def persona_edit_user_carries_current_thoughts_and_core():
    out = build_persona_edit_user("You are still getting to know him.",
                                  ["I wonder if he's sleeping enough"],
                                  ["The user's name is Alex"])
    assert "You are still getting to know him." in out, "current slot, so it revises not restarts"
    assert "I wonder if he's sleeping enough" in out, "her journal is the raw material"
    assert "The user's name is Alex" in out


@case
async def persona_edit_user_survives_empty_inputs():
    out = build_persona_edit_user("", [], [])   # first ever edit: nothing to build on
    assert isinstance(out, str) and out.strip()


# --- reflection ------------------------------------------------------------------

@case
async def reflect_system_carries_recent_thoughts_to_avoid_repeats():
    out = build_reflect_system([], None, ["I keep circling the same worry"])
    assert "I keep circling the same worry" in out, (
        "recent thoughts are injected precisely so she doesn't journal the same line twice")


@case
async def reflect_system_is_addressed_to_her_not_about_the_user():
    out = build_reflect_system(["The user is a nurse"], "warm", [], core=["name is Alex"])
    assert config.BOT_NAME in out or "you" in out.lower()
    assert "The user is a nurse" in out, "recalled context still reaches the reflection"


if __name__ == "__main__":
    raise SystemExit(run())
