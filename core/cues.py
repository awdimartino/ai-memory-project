"""Lexical cues in the user's message that change what happens AFTER a turn.

Deliberately lexical, for the same reason `core/textsim.py` is: this runs on the hot
chat path and must not cost a model call. The point of a cue is to decide whether an
expensive decision is worth making at all — a gate in front of a generation, never a
replacement for one.

Currently one cue: the user saying goodnight. It gates the sleep decision (§A2), so
she only considers resting when there's an actual reason to, instead of paying a
structured-output call at the end of every turn.
"""
import re

# The user announcing THEY are going, now. Ordered roughly by how unambiguous it is.
#
# Conservative on purpose. A false positive costs one structured-output call and a
# likely "stay up" answer, which is cheap; but it can also put her to sleep in the
# middle of a live conversation, which is not. So every pattern below requires the
# user to be the one leaving, in the present or immediate future.
# Optional leading subject: "i" / "im" / "i'm" / "i am". Optional because people drop
# it constantly when signing off ("heading to bed", "gonna hit the hay") — requiring it
# missed a third of the real phrasings.
_SUBJ = r"(?:i(?:'m|m| am)?\s+)?"

_FAREWELL = [
    r"\bg(?:'|o+)?d?\s*night\b",              # goodnight, g'night, gnight, good night
    r"\bnight[- ]?night\b",
    r"\bnighty[- ]?night\b",
    _SUBJ + r"(?:gonna|going to|gunna|off to|headed|heading)\s+(?:go\s+)?(?:to\s+)?(?:bed|sleep|crash)\b",
    _SUBJ + r"(?:gonna|going to|gunna)\s+(?:hit the (?:hay|sack)|turn in|pass out)\b",
    r"\b(?:turning|gonna turn) in (?:for the night|now)?\b",
    r"\bcalling it (?:a night|here for the night)\b",
    r"\b(?:off to|time for) bed\b",
    r"\b(?:talk|see|catch)\s+(?:to\s+)?(?:ya\s+|you\s+)?(?:tomorrow|in the morning)\b",
    r"\bsleep well\b",
    r"\b(?:get|getting|grab|grabbing)\s+some\s+(?:sleep|rest)\b",
]

# Phrases that contain a farewell word but announce nothing. Checked first — an
# overlapping match exonerates the hit, same shape as core/embodiment.py's allow-list.
#
# The dominant false positive is the PAST tense: "last night", "how was your night",
# "i slept badly" are all about a night that already happened, and treating them as a
# goodbye would put her to sleep mid-conversation.
_NOT_FAREWELL = [
    r"\b(?:last|the other|that|yester)\s*night\b",
    r"\bhow (?:was|did) (?:your|the) night\b",
    r"\b(?:all|every|one|some|late at|during the|at) night\b",
    r"\bnight(?:s)? (?:shift|out|owl|before|after|terror)\b",
    r"\bdid(?:n't| not)? sleep well\b",
    r"\btomorrow\b.{0,20}\b(?:i have|is|we|there'?s|what)\b",   # "tomorrow i have work"
]

_FAREWELL_RE = [re.compile(p, re.I) for p in _FAREWELL]
_NOT_FAREWELL_RE = [re.compile(p, re.I) for p in _NOT_FAREWELL]


def farewell_cue(text: str) -> str | None:
    """Return the matched phrase if the user is signing off for the night, else None."""
    if not text:
        return None
    for pattern in _FAREWELL_RE:
        m = pattern.search(text)
        if not m:
            continue
        window = text[max(0, m.start() - 30):m.end() + 30]
        if any(a.search(window) for a in _NOT_FAREWELL_RE):
            continue
        return m.group(0).strip()
    return None
