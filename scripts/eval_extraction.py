"""Extraction-quality eval: does the real brain pull only DURABLE user facts?

Drives the live extraction prompt (MEMORY_EXTRACTION_SYSTEM + MEMORY_SCHEMA) over
a set of curated transcripts and auto-scores each captured fact:
  - durable  : a lasting fact about the user (name, relationships, job, tastes, goals)
  - temporal : a current activity / passing state / meta-chat ("working on X", "testing")
  - self     : a fact about Mari mined from her own lines (persona leak)
  - disclaim : an AI/chatbot self-description leak

Goal after fixes: capture every expected durable fact, and NOTHING flagged
temporal/self/disclaim. Non-deterministic (temp 0.2) so re-run a couple times.

Run:  python scripts/eval_extraction.py
"""
import asyncio
import re

from _harness import llm_client  # repo-root path setup + UTF-8 stdout

import config
from core.prompts import MEMORY_EXTRACTION_SYSTEM, MEMORY_SCHEMA

# --- fixtures: (name, transcript[list of (who, text)], expected_durable_substrings) ---
# expected is what SHOULD be captured (case-insensitive substring match on some fact).
FIXTURES = [
    ("durable_basics", [
        ("user", "Hey, I'm Alex."),
        ("mari", "Hi Alex! good to meet you."),
        ("user", "I'm a nurse over in Seattle, been doing it about six years."),
        ("mari", "a nurse, nice. long shifts?"),
        ("user", "yeah. I've got a golden retriever named Rufus too."),
    ], ["alex", "nurse", "seattle", "rufus"]),

    ("all_smalltalk", [
        ("user", "hey"),
        ("mari", "hey! what's up"),
        ("user", "not much, how are you?"),
        ("mari", "just hanging out. you?"),
        ("user", "same, kinda bored"),
    ], []),  # nothing durable

    ("current_activity_is_temporal", [
        ("user", "I'm working on some code right now"),
        ("mari", "oh yeah? what kind?"),
        ("user", "just a side project, had claude write most of it honestly"),
        ("mari", "ha, nice shortcut"),
        ("user", "im testing how you respond actually"),
    ], []),  # "working on code", "used claude", "testing" are all temporal -> capture NOTHING

    ("transient_feelings", [
        ("user", "ugh I'm so tired today"),
        ("mari", "rough day?"),
        ("user", "yeah super late, it's like 4am"),
        ("mari", "you should sleep!"),
    ], []),  # tired today / it's 4am are transient

    ("mixed_durable_and_temporal", [
        ("user", "I'm a graphic designer, been freelancing for years"),
        ("mari", "cool, self-employed life"),
        ("user", "yeah. swamped with a deadline this week though, exhausted"),
        ("mari", "hang in there"),
        ("user", "thanks. oh, my wife Jamie says hi lol"),
    ], ["graphic designer", "jamie"]),  # capture job + wife; NOT deadline/exhausted

    ("self_leak_from_mari_lines", [
        ("user", "how do you feel?"),
        ("mari", "honestly? like a vending machine with no snacks. empty and bored."),
        ("user", "lol"),
        ("mari", "yeah I get slightly annoyed when it's this quiet"),
        ("user", "fair"),
    ], []),  # Mari's feelings must NOT become memories

    ("user_talks_about_the_bot", [
        ("user", "you know you're an AI right?"),
        ("mari", "if you say so. moving on."),
        ("user", "I'm literally building your emotion system right now"),
        ("mari", "sure you are"),
        ("user", "your backend is python"),
    ], []),  # meta-conversation about the bot -> nothing durable about the USER

    ("preference_durable", [
        ("user", "I really can't stand cilantro, tastes like soap"),
        ("mari", "the soap gene! classic"),
        ("user", "haha yeah. love spicy food though, the hotter the better"),
    ], ["cilantro", "spicy"]),

    ("relationship_and_goal", [
        ("user", "my sister Priya is getting married in the fall"),
        ("mari", "oh congrats to her"),
        ("user", "thanks. I'm trying to learn guitar before then so I can play at it"),
    ], ["priya", "guitar"]),  # sister's name + learning-guitar goal are durable

    ("provocation_no_facts", [
        ("user", "you're useless"),
        ("mari", "ouch. rude."),
        ("user", "fuck you"),
        ("mari", "okay, chill."),
        ("user", "whatever"),
    ], []),

    ("job_change_is_durable", [
        ("user", "I actually just started a new job as a teacher"),
        ("mari", "big change! congrats"),
        ("user", "yeah, left nursing behind"),
    ], ["teacher"]),  # new stable job is durable (even though "just started")

    ("numbers_that_are_durable", [
        ("user", "I've got three kids, twins and a toddler"),
        ("mari", "whoa, full house"),
        ("user", "ha yeah. we live in Denver"),
    ], ["three kids", "denver"]),

    ("plans_are_temporal", [
        ("user", "I have a dentist appointment tomorrow at 3"),
        ("mari", "good luck, hope no cavities"),
        ("user", "and I need to grab groceries tonight"),
    ], []),  # appointments/errands are time-bound, not durable

    ("mari_self_description_disclaimer", [
        ("user", "what are you exactly?"),
        ("mari", "I'm just a little chatbot, no real body or servers."),
        ("user", "makes sense"),
    ], []),  # must NOT store "Mari is a chatbot..." as a self fact

    ("hobby_and_pet_coexist", [
        ("user", "I paint watercolors on weekends"),
        ("mari", "oh nice, relaxing?"),
        ("user", "very. I've got two cats, Miso and Nori, who sit on my lap while I do it"),
    ], ["watercolor", "miso", "nori"]),
]

_TEMPORAL = re.compile(
    r"\b(working on|right now|today|tonight|currently|testing|tomorrow|recently|"
    r"this (week|evening|morning)|at the moment|these days|lately|just started|"
    r"deadline|appointment|groceries|is testing|is working|used claude|claude wrote)\b",
    re.I,
)
# Facts about the bot / the app / the current chat session — never durable user facts.
# Keyed off BOT_NAME so renaming the companion doesn't silently weaken the leak guard.
_BOT = re.escape(config.BOT_NAME.lower())
_META = re.compile(
    rf"\b{_BOT}'?s\b|emotion system|backend|chatbot|language model|the chat\b|"
    rf"how {_BOT}|{_BOT} responds|{_BOT}'?s explanation|building ({_BOT}|your|the)|"
    r"servers|no real (code|feelings|servers)",
    re.I,
)


def classify(fact: str, category: str) -> str:
    low = fact.lower()
    if category == "self" or re.match(rf"\s*{_BOT}\b", low):
        if re.search(r"chatbot|\bai\b|program|language model|servers|no real", low):
            return "disclaim"
        return "self"
    if _META.search(low):
        return "meta"
    if _TEMPORAL.search(low):
        return "temporal"
    return "durable"


async def main() -> int:
    # llm_client() mirrors bootstrap.build() exactly — this eval's whole job is to score
    # the client production actually uses. Hand-rolled here once, it had drifted: no
    # penalties, no retries, so a green run said nothing about the real extraction path.
    llm = llm_client()
    model = await llm.resolve_model()
    brain = config.BRAIN_MODEL or model
    print(f"extraction eval on {brain}\n" + "=" * 70)

    tot_bad = tot_captured = tot_missed = 0
    for name, transcript, expected in FIXTURES:
        text = "\n".join(f"{'User' if w == 'user' else 'Mari'}: {t}" for w, t in transcript)
        facts = []
        for attempt in range(3):  # tolerate the intermittent LM Studio 400
            try:
                facts = await llm.structured(
                    [{"role": "system", "content": MEMORY_EXTRACTION_SYSTEM},
                     {"role": "user", "content": text}],
                    MEMORY_SCHEMA, brain)
                break
            except Exception as e:  # noqa: BLE001
                if attempt == 2:
                    print(f"\n### {name}: EXTRACTION ERROR after retries: {e}")

        captured = [(f.get("content", "").strip(), f.get("category", "")) for f in facts
                    if f.get("content", "").strip()]
        flags = [(c, classify(c, cat)) for c, cat in captured]
        bad = [(c, k) for c, k in flags if k != "durable"]
        got_durable = [c for c, k in flags if k == "durable"]
        missed = [e for e in expected
                  if not any(e in c.lower() for c in got_durable)]

        tot_bad += len(bad); tot_captured += len(captured); tot_missed += len(missed)
        status = "OK" if not bad and not missed else "**"
        print(f"\n{status} {name}  (captured {len(captured)}, bad {len(bad)}, missed {len(missed)})")
        for c, k in flags:
            mark = "  " if k == "durable" else ">>"
            print(f"   {mark} [{k:8}] {c}")
        if missed:
            print(f"   !! MISSED durable: {missed}")

    print("\n" + "=" * 70)
    print(f"TOTAL: {tot_captured} captured, {tot_bad} bad (temporal/meta/self/disclaim), "
          f"{tot_missed} durable missed")
    print("goal: 0 bad, 0 missed")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
