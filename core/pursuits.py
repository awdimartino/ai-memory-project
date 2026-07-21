"""The pursuit-tool registry (§A4: she can make herself unavailable).

A "pursuit" is a tool call Mari chooses for HERSELF, not one the user (or the chat
loop) invokes — deliberately NOT `core/tools.py`'s native function-calling registry,
which is user-facing, exposed to the model mid-turn, and measured too unreliable
(~23/30) to gate something as consequential as whether she goes silent on the user.
The choice here is a structured-output decision instead (`core/prompts.py`'s
`build_pursuit_choice_*`), and every entry maps to a REAL `Companion` method — this is
what keeps an absence honest: she never invents an errand, she does one of a small,
closed, real set of things (journal, revise her notes/persona, sit with an open
question) and the artifact she comes back with is genuine.
"""
import random
from dataclasses import dataclass
from typing import Awaitable, Callable

import config

Handler = Callable[[], Awaitable[str | None]]


@dataclass
class PursuitTool:
    """One thing she can choose to step away and do.

    `description` is shown to the choice call AND (for the non-seeded pursuits)
    templates the UI's "why she's away" string directly — it is fixed prose, never
    freely generated, so there's nothing for the embodiment filter to need to catch.
    `(min_seconds, max_seconds)` is the "educated random" duration range: how long
    this particular kind of pursuit plausibly takes.
    """
    name: str
    description: str
    min_seconds: float
    max_seconds: float
    handler: Handler

    def sample_duration(self, rng: Callable[[], float] = random.random) -> float:
        return self.min_seconds + rng() * (self.max_seconds - self.min_seconds)


class PursuitRegistry:
    """The closed menu offered each time she considers stepping away."""

    def __init__(self, tools: list[PursuitTool] | None = None):
        self._tools: dict[str, PursuitTool] = {t.name: t for t in (tools or [])}

    def get(self, name: str) -> PursuitTool | None:
        return self._tools.get(name)

    def available(self, has_open_pursuit: bool) -> list[PursuitTool]:
        """`sit_with_question` only appears on the menu when a real A3-mined pursuit
        exists — she is never offered a choice with nothing real behind it."""
        return [t for t in self._tools.values()
                if t.name != "sit_with_question" or has_open_pursuit]

    def __bool__(self) -> bool:
        return bool(self._tools)


def make_journal_pursuit(companion) -> PursuitTool:
    async def handler() -> str | None:
        return await companion.reflect()
    return PursuitTool("journal", "sit and write a private journal entry for a bit",
                       config.PURSUIT_JOURNAL_MIN, config.PURSUIT_JOURNAL_MAX, handler)


def make_self_notes_pursuit(companion) -> PursuitTool:
    async def handler() -> str | None:
        return await companion.update_self_notes()
    return PursuitTool("self_notes", "think over how she's been with him lately",
                       config.PURSUIT_SELFNOTES_MIN, config.PURSUIT_SELFNOTES_MAX, handler)


def make_persona_pursuit(companion) -> PursuitTool:
    async def handler() -> str | None:
        return await companion.edit_persona()
    return PursuitTool("persona", "sit with who she's been becoming",
                       config.PURSUIT_PERSONA_MIN, config.PURSUIT_PERSONA_MAX, handler)


def make_sit_with_question_pursuit(companion) -> PursuitTool:
    async def handler() -> str | None:
        return await companion.sit_with_open_question()
    return PursuitTool("sit_with_question", "sit with something she's been wondering about him",
                       config.PURSUIT_SIT_MIN, config.PURSUIT_SIT_MAX, handler)


def build_default_registry(companion) -> PursuitRegistry:
    """The standard closed menu, bound to one Companion instance."""
    return PursuitRegistry([
        make_journal_pursuit(companion),
        make_self_notes_pursuit(companion),
        make_persona_pursuit(companion),
        make_sit_with_question_pursuit(companion),
    ])
