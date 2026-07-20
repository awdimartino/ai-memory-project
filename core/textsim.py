"""Cheap lexical similarity, shared by the repeat guards and the RRR diagnostic.

Deliberately NOT embeddings. Two reasons: these checks run on paths that must not
depend on LM Studio being up (a reflection shouldn't fail because the embedder is
busy), and the failure being caught is *surface* repetition — the model reaching for
the same phrasing again — which lexical overlap measures directly.

One implementation so the guard that rejects a repeat and the diagnostic that reports
the rate can never disagree about what "the same thing again" means.
"""
import re

# Function words carry no signal about whether two thoughts are the same thought.
_STOP = {"the", "a", "an", "and", "or", "but", "is", "it", "its", "to", "of", "in",
         "that", "this", "i", "im", "my", "me", "you", "your", "just", "so", "like",
         "s", "t", "for", "with", "as", "at", "on", "be", "was", "if", "how", "what"}

# Above this Jaccard overlap, treat two texts as the same thought restated.
#
# CALIBRATED against the real journal (24 entries, scripts/rrr_diagnostic.py), not
# guessed. That data has four byte-identical pairs at 1.00, one at 0.93, then a band
# of genuine restatements clustered at 0.42-0.56 — e.g. "my irritation settles into
# this heavy quiet ... waiting for someone to fix the mood" vs "my irritation just
# sits there heavy, waiting for someone to fix the mood", which scores 0.42. There is
# no clean gap, so the threshold is set from cost asymmetry instead:
#
#   false positive (reject a fresh thought) -> she skips one journal entry. Cheap.
#   false negative (store a restatement)    -> the journal that feeds the persona
#                                              edit fills with duplicates. This is
#                                              the documented harm.
#
# So err toward rejecting. 0.40 catches the restatement band while leaving genuinely
# unrelated thoughts (which score near 0) untouched.
REPEAT_THRESHOLD = 0.40


def tokens(text: str) -> set[str]:
    """Content words, lowercased, stop-words and very short tokens dropped."""
    return {w for w in re.findall(r"[a-z']+", text.lower())
            if w not in _STOP and len(w) > 2}


def similarity(a: str, b: str) -> float:
    """Jaccard overlap of content words, 0.0-1.0."""
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def is_repeat(candidate: str, previous: list[str],
              threshold: float = REPEAT_THRESHOLD) -> bool:
    """True if `candidate` restates anything in `previous`.

    Used to reject a reflection before it's stored. The reflection prompt already
    shows her recent thoughts and asks her not to repeat — measured RRR 0.26 with
    five near-verbatim pairs and three byte-identical entries, so prompting alone
    demonstrably does not hold. The study's own mitigation was programmatic
    extraction over free-form self-diagnosis (RRR 0.64 -> 0.10).
    """
    return any(similarity(candidate, p) >= threshold for p in previous)
