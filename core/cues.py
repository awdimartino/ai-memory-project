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

# The user announcing THEY are going, now — or telling her to rest.
#
# **Deliberately broad, and that is a measured decision.** The first version required
# an explicit subject and an explicit "gonna", which missed how people actually sign
# off: "hit the hay", "i'm sleeping so you should too", "bedtime". It was written tight
# because the downstream decision was unmeasured.
#
# It since was (2026-07-21, 10 samples per context):
#
#     clear goodnight, low energy        SLEEP 10/10
#     clear goodnight, high energy       SLEEP 10/10
#     says goodnight but keeps talking   SLEEP  0/10   <- refuses outright
#     goodnight mid-heavy-conversation   SLEEP  3/10
#
# The decision discriminates, so it — not this list — is the real gate. A false
# positive here now costs one structured call that answers STAY, which is cheap. That
# frees this list to favour recall.
#
# What it must STILL never match is the past tense: "last night was rough" is not a
# goodbye, and treating it as one asks the question mid-conversation where the heavy
# context above still says SLEEP 3/10.
# Optional leading subject: "i" / "im" / "i'm" / "i am". Optional because people drop
# it constantly when signing off ("heading to bed", "gonna hit the hay") — requiring it
# missed a third of the real phrasings.
_SUBJ = r"(?:i(?:'m|m| am)?\s+)?"

# Optional intent verb: "gonna", "going to", "need to", "should", "have to", "gotta".
_INTENT = r"(?:(?:gonna|gunna|going to|need to|needa|gotta|got to|have to|hafta|should|must|will|wanna|want to)\s+)?"

_FAREWELL = [
    # --- explicit goodnights -----------------------------------------------------
    r"\bg(?:'|o+)?d?\s*night\b",              # goodnight, g'night, gnight, good night
    r"\bnight[- ]?night\b",
    r"\bnighty[- ]?night\b",
    r"\bsweet dreams\b",
    r"\bnight\s*[!.]*\s*$",                   # a bare "night" ENDING the message
    # --- the user announcing they're going ---------------------------------------
    _SUBJ + _INTENT + r"(?:go\s+)?(?:off\s+)?(?:to\s+)?(?:bed|sleep|crash)\b",
    _SUBJ + _INTENT + r"(?:hit the (?:hay|sack)|turn(?:ing)? in|pass out|conk out|doze off)\b",
    _SUBJ + r"(?:'m|m| am)?\s*(?:sleeping|going to sleep|off to sleep|in bed)\b",
    r"\b(?:off to|time for|ready for) bed\b",
    r"\bbed\s*time\b",
    r"\bcalling it (?:a night|here|quits)\b",
    r"\b(?:signing|logging) off (?:for the night|now)?\b",
    r"\b(?:talk|see|catch)\s+(?:to\s+)?(?:ya\s+|you\s+)?(?:tomorrow|in the morning)\b",
    # --- the user telling HER to rest --------------------------------------------
    # "i'm sleeping so you should too" - the cue is about her, not about them leaving.
    r"\byou\s+(?:should|can|could|ought to)\s+(?:sleep|rest|go to bed|power down|sleep too|rest too)\b",
    r"\b(?:get|getting|grab|grabbing)\s+some\s+(?:sleep|rest)\b",
    r"\b(?:rest|sleep)\s+(?:up|well)\b",
    r"\byou\s+(?:should\s+)?(?:get\s+)?some\s+(?:sleep|rest)\b",
]

# Phrases that contain a farewell word but announce nothing. Checked first — an
# overlapping match exonerates the hit, same shape as core/embodiment.py's allow-list.
#
# The dominant false positive is the PAST tense: "last night", "how was your night",
# "i slept badly" are all about a night that already happened, and treating them as a
# goodbye would put her to sleep mid-conversation.
_NOT_FAREWELL = [
    # Past tense — the dominant and most expensive false positive.
    r"\b(?:last|the other|that|yester)\s*night\b",
    r"\bhow (?:was|did|were) (?:your|the|you)\b",
    r"\b(?:all|every|one|some|late at|during the|at|by|this) night\b",
    r"\bnight(?:s)? (?:shift|out|owl|before|after|terror|sky|air)\b",
    r"\b(?:did|could)(?:n't| not)?\s+sleep\b",
    r"\bcan'?t sleep\b",
    r"\bslept\b", r"\bwoke up\b", r"\bwas sleeping\b", r"\bhave been sleeping\b",
    # Hypothetical / negated / about someone else, not an announcement.
    r"\b(?:if|when|whenever|before|after|unless)\s+i\s+(?:go to bed|sleep|hit the)\b",
    # Someone ELSE going to bed. `_SUBJ` is optional (so "heading to bed" is caught),
    # which is exactly what lets "the kids went to bed an hour ago" through — so the
    # third person has to be excluded explicitly. Note "you" is deliberately NOT in
    # here: "you should go to bed" is a real cue, the user telling HER to rest.
    r"\b(?:he|she|they|his|her|their|the|my)\s+[\w']{0,15}\s*(?:went|goes|going|has gone)\s+to\s+bed\b",
    r"\b(?:he|she|they|it)(?:'s|'re|s| is| are| was| were)\s+sleep(?:ing)?\b",
    r"\bsleep(?:ing|s)?\s+over\b",
    r"\bnot\s+(?:going to|gonna)\s+(?:bed|sleep)\b",
    r"\bdon'?t\s+(?:want to|wanna)\s+(?:go to bed|sleep)\b",
    # Talking ABOUT sleep as a topic rather than doing it.
    r"\b(?:trouble|problems?|issues?) (?:with )?sleep(?:ing)?\b",
    r"\bsleep (?:schedule|apnea|tracker|debt|study|deprived|deprivation)\b",
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
