"""Shared fakes for the offline suite.

These were redeclared per test file, and the copies had drifted — three classes named
`FakeMeta` with three different surfaces (one of them a stub whose `get()` always
returned `None`), two `FakeConv`s where one lacked `message_count`. A behavior asserted
against a stub isn't the behavior the app sees, so the copies are consolidated here and
the *superset* wins in each case.

Import alongside the harness:

    from _harness import case, run, temp_db
    from helpers import OneHotEmbedder, ScriptedLLM, fact
"""
import numpy as np


# --- memory-tier fakes ------------------------------------------------------------

class OneHotEmbedder:
    """Topic keyword -> one-hot vector, so same-topic facts are cosine-similar.

    `topics` is the only thing that ever differed between the three copies of this
    class; anything not matching a topic lands in a shared "misc" bucket at the end.
    """

    def __init__(self, topics: list[str]):
        self.topics = list(topics)

    def _vec(self, text: str) -> list[float]:
        v = np.zeros(len(self.topics) + 1, dtype=np.float32)
        low = text.lower()
        idx = len(self.topics)  # "misc" bucket
        for i, k in enumerate(self.topics):
            if k in low:
                idx = i
                break
        v[idx] = 1.0
        return v.tolist()

    async def embed_query(self, text: str):
        return self._vec(text)

    async def embed_document(self, text: str):
        return self._vec(text)

    async def embed_documents(self, texts):
        return [self._vec(t) for t in texts]


class ScriptedLLM:
    """Scripted extraction (fact lists) + BATCHED lifecycle decisions.

    `decisions` is one batched dict per consolidate call that has at least one fact
    needing a decision: `{"decisions": [{"candidate", "action", "target"}, ...]}`.
    Also serves the core-rerank path, which reads `{"keep": [...]}` off the same queue.
    """

    def __init__(self, fact_lists=None, decisions=None):
        self.fact_lists = list(fact_lists or [])
        self.decisions = list(decisions or [])

    async def structured(self, messages, schema, model=None):
        return self.fact_lists.pop(0) if self.fact_lists else []

    async def structured_json(self, messages, schema, model=None):
        return self.decisions.pop(0) if self.decisions else {"decisions": []}


def fact(content: str, core: bool = False, category: str = "user") -> dict:
    """One extracted-fact dict, in the shape the extractor returns.

    `category` is vestigial in production (it was a single-value enum and was dropped
    from MEMORY_SCHEMA) but the store signature still carries it, so it stays here.
    """
    return {"content": content, "category": category, "core": core}


# --- companion-side fakes ---------------------------------------------------------

class FakeMemory:
    """A memory manager that recalls nothing — for tests about everything else."""

    def __init__(self):
        self.injected: list[tuple[list[int], int]] = []

    async def recall(self, text):
        return []

    def core_memories(self):
        return []

    def core_for_turn(self, turn):
        """(contents, ids) eligible this turn — nothing, for tests about other things."""
        return [], []

    def mark_injected(self, memory_ids, turn):
        self.injected.append((list(memory_ids), turn))

    def clear(self) -> int:
        """Admin wipe. Returns how many were removed (always 0 — it holds nothing)."""
        return 0


class FakeConv:
    """Conversation store that hands out incrementing ids and forgets everything.

    `add_message` DOES retain each message (id/role/content only) so
    `recent_messages_with_ids` — the citation surface for A3's open-question mining —
    has something real to return; nothing else here is persisted.
    """

    def __init__(self, message_count: int = 100):
        self._id = 0
        self._count = message_count
        self.self_notes_log: list[str] = []
        self._messages: list[dict] = []

    def add_message(self, sid, role, content):
        self._id += 1
        self._messages.append({"id": self._id, "role": role, "content": content})
        return self._id

    def message_count(self):
        return self._count

    def recent_messages_with_ids(self, limit):
        return self._messages[-limit:]

    def set_title(self, sid, title):
        pass

    def log_self_notes(self, content):
        """Revision log for the operating-notes slot (RRR scoring)."""
        self.self_notes_log.append(content)
        return len(self.self_notes_log)

    def recent_self_notes(self, limit):
        return list(reversed(self.self_notes_log[-limit:]))


class InMemoryMeta:
    """A real dict-backed MetaStore.

    Deliberately the full surface (`get`/`set`/`get_json`/`set_json`) rather than the
    always-None stub one file used: a test that reads back what it wrote is asserting
    something, and a stub silently makes those assertions vacuous.
    """

    def __init__(self, data: dict | None = None):
        self.data = dict(data or {})

    def get(self, k):
        return self.data.get(k)

    def set(self, k, v):
        self.data[k] = v

    def get_json(self, k, default=None):
        return self.data.get(k, default)

    def set_json(self, k, v):
        self.data[k] = v

    def clear(self):
        """Wipe every key — mood, drives, persona, watermark, cooldowns."""
        self.data.clear()


class FakeClassifier:
    """Emotion classifier returning scripted label scores (empty = neutral)."""

    def __init__(self, scripted=None):
        self.scripted = list(scripted or [])

    def classify(self, text):
        return self.scripted.pop(0) if self.scripted else []
