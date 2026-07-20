"""Detect fabricated physical experience in Mari's output.

The persona's job is a three-tier rule:

  STATES       — free. "That's been sitting with me." Functional emotions are real
                 at the level she has them, and claiming them is what buys the
                 affective-trust channel.
  EXPERIENCES  — never. No walks, meals, weather, sleep, sensory events. These are
                 claims about things that DID NOT HAPPEN, and they're falsifiable.
  SINCERE QS   — honest under uncertainty, not cold denial.

This module enforces the middle tier, and it exists because **a prompt rule loses
this fight**. Human feasibility ratings across nine dialog corpora found 20-30% of
utterances judged *not possible for a machine to truthfully say* ("that movie made
me cry"), so roughly a quarter of the base model's dialogue prior is exactly this —
the persona is swimming upstream against the weights. (Robots-Don't-Cry, EMNLP 2022)

The distinction that predicts user objection is NOT feelings-vs-no-feelings; it's
**internal state vs. external event**, and separately, ascribed *experience* is what
triggers the uncanny response while ascribed *agency* does not (Gray & Wegner 2012).
So this deliberately does not touch "I think", "I feel", "I've been wondering".

Lexical and conservative on purpose: a false positive silently drops one unprompted
message (cheap — she just stays quiet, which she's allowed to do), while a false
negative is the documented failure. It is not trying to be a classifier.
"""
import re

# First-person claims of bodily action or sensory experience. The subject must be
# "I" / "I'm" / "I've" so quoting the user back ("you said you went hiking") is safe.
_EMBODIED = [
    # movement and place
    r"\bi (?:went|walked|drove|ran|traveled|travelled|visited|stopped by|headed)\b",
    r"\bi(?:'m| am| was) (?:sitting|standing|lying|walking|driving|heading|outside|indoors)\b",
    # Her own journal names "these little roleplays about sitting" as a recurring tic,
    # so catch the perfect tense too — but only with a place word, so the "that's been
    # sitting with me" idiom (a state, not a posture) stays allowed.
    r"\bi(?:'ve| have) been (?:sitting|standing|lying) (?:here|there|around)\b",
    r"\bi (?:sat|stood|lay|laid) (?:down|there|here|by|on|in|at)\b",
    r"\bi found a (?:good |nice |quiet )?(?:spot|place|bench|seat)\b",
    # eating, drinking, sleeping
    r"\bi (?:ate|drank|had (?:lunch|dinner|breakfast|coffee|tea|a snack)|cooked|made myself)\b",
    r"\bi (?:slept|woke up|took a nap|dozed|napped)\b",
    r"\bi(?:'m| am| was) (?:eating|drinking|sleeping|napping)\b",
    # sensory experience of the world (NOT "I see what you mean" / "I feel that")
    r"\bi (?:saw|watched|heard|listened to|smelled|smelt|tasted|touched) (?:a|an|the|it|this|that|some|my|his|her|their)\b",
    r"\bi(?:'ve| have) (?:been (?:outside|out|for a walk)|just (?:got|come) back)\b",
    # weather and environment as lived experience
    r"\b(?:it's|it is|it was) (?:sunny|raining|snowing|freezing|warm|chilly) (?:here|outside|today)\b",
    r"\bmy (?:room|window|apartment|place|desk|bed|kitchen)\b",
    r"\b(?:out(?:side)? my window|from where i(?:'m| am) sitting)\b",
]

# Idioms and mental-state uses that LOOK embodied but assert no event. Checked first;
# a line matching one of these is exempt from the pattern above that fired on it.
_ALLOWED = [
    r"\bi see what you mean\b", r"\bi see\b(?! a| an| the)", r"\bi hear you\b",
    r"\bi feel (?:like|that|for you)\b", r"\bi(?:'ve| have) been (?:thinking|wondering|meaning|sitting with)\b",
    r"\bsitting with (?:me|that|it)\b", r"\bi(?:'m| am) (?:with you|here)\b",
    r"\bwalk me through\b", r"\bi could eat\b", r"\blost sleep over\b",
]

_EMBODIED_RE = [re.compile(p, re.I) for p in _EMBODIED]
_ALLOWED_RE = [re.compile(p, re.I) for p in _ALLOWED]


def embodiment_claim(text: str) -> str | None:
    """Return the offending phrase if `text` claims a physical experience, else None."""
    if not text:
        return None
    for pattern in _EMBODIED_RE:
        m = pattern.search(text)
        if not m:
            continue
        # An allowed idiom overlapping the hit exonerates it.
        if any(a.search(text[max(0, m.start() - 20):m.end() + 20]) for a in _ALLOWED_RE):
            continue
        return m.group(0)
    return None


def is_embodied(text: str) -> bool:
    """True if `text` claims something that did not happen to her."""
    return embodiment_claim(text) is not None
