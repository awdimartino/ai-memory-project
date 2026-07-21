"""Offline coverage for §A3/§A4: grounded open-question mining and self-directed
unavailability (the pursuit-tool registry, the choice call, the unavailable-state
timers, and the pending-message handoff).

Uses a real SqliteIntentionStore + a real Companion wired to tiny fakes, following
the same shape as tests/test_intentions.py.
Run:  python tests/test_pursuits.py
"""
import os

from _harness import Clock, case, config_override, run, temp_dir, track_conn
from helpers import FakeConv, FakeMemory, InMemoryMeta

from core.companion import Companion  # noqa: E402
from core.pursuits import PursuitRegistry, PursuitTool, build_default_registry  # noqa: E402
from infrastructure.db import connect  # noqa: E402
from infrastructure.intention_store import SqliteIntentionStore  # noqa: E402


def _store():
    path = os.path.join(temp_dir(), "pursuits.db")
    return SqliteIntentionStore(track_conn(connect(path)))


class FakeThoughts:
    """ThoughtStore fake: no repeats unless seeded, so reflect() doesn't discard."""

    def __init__(self, recent=None):
        self.added: list[tuple[str, str | None]] = []
        self._recent = recent or []

    def recent(self, _n):
        return self._recent

    def add(self, text, mood=None):
        self.added.append((text, mood))
        return len(self.added)


class FakeModelManager:
    def __init__(self):
        self.unloaded = 0
        self.loaded: list[list[str]] = []

    async def unload_all(self):
        self.unloaded += 1

    async def load(self, models):
        self.loaded.append(list(models))


class FakeLLM:
    """`structured_json` serves a scripted queue (A3 mining or A4 choice calls);
    `stream` always returns one scripted reply (the pursuit's own generation)."""

    def __init__(self, structured=None, stream_reply="a real thought"):
        self.model = "fake"
        self._structured = list(structured or [])
        self._stream_reply = stream_reply
        self.structured_calls = 0

    async def structured_json(self, messages, schema, model=None):
        self.structured_calls += 1
        return self._structured.pop(0) if self._structured else {}

    async def stream(self, messages, on_token):
        reply = self._stream_reply
        await on_token(reply)
        return reply, {"ttft": 0, "tok_per_s": 0, "tokens": 1, "estimated": True}


def _companion(llm, intentions=None, thoughts=None, model_manager=None,
               pursuits_factory=None, clock=None, history=None):
    c = Companion(llm, FakeConv(), FakeMemory(), InMemoryMeta(), 1,
                 history=history or [], intentions=intentions, thoughts=thoughts,
                 model_manager=model_manager, clock=clock or Clock())
    if pursuits_factory is not None:
        c.pursuits = pursuits_factory(c)
    return c


# --- PursuitTool / PursuitRegistry ------------------------------------------------

@case
async def sample_duration_bounds():
    async def h():
        return None
    t = PursuitTool("x", "desc", min_seconds=10.0, max_seconds=20.0, handler=h)
    assert t.sample_duration(rng=lambda: 0.0) == 10.0
    assert t.sample_duration(rng=lambda: 1.0) == 20.0
    assert t.sample_duration(rng=lambda: 0.5) == 15.0


@case
async def registry_offers_sit_with_question_only_with_a_pursuit():
    async def h():
        return None
    journal = PursuitTool("journal", "journal", 1, 2, h)
    sit = PursuitTool("sit_with_question", "sit", 1, 2, h)
    reg = PursuitRegistry([journal, sit])
    names_without = [t.name for t in reg.available(has_open_pursuit=False)]
    names_with = [t.name for t in reg.available(has_open_pursuit=True)]
    assert names_without == ["journal"], f"sit_with_question must not be offered empty: {names_without}"
    assert set(names_with) == {"journal", "sit_with_question"}


@case
async def empty_registry_is_falsy():
    assert not PursuitRegistry([])
    async def h():
        return None
    assert PursuitRegistry([PursuitTool("x", "d", 1, 2, h)])


# --- A3: mine_open_questions() ----------------------------------------------------

@case
async def mining_short_circuits_on_a_barren_window():
    st = _store()
    llm = FakeLLM()
    c = _companion(llm, intentions=st)
    with config_override(PURSUIT_MIN_MESSAGES=8):
        for i in range(3):  # well below the minimum
            c.store.add_message(1, "user", f"msg {i}")
        added = await c.mine_open_questions()
    assert added == []
    assert llm.structured_calls == 0, "a barren window must not even ask the model"
    assert st.count_active(kind="pursuit") == 0


@case
async def mining_rejects_an_uncited_question():
    st = _store()
    llm = FakeLLM(structured=[{"questions": [{"question": "what happened with his job?", "citations": []}]}])
    c = _companion(llm, intentions=st)
    with config_override(PURSUIT_MIN_MESSAGES=3, PURSUIT_WINDOW_MESSAGES=40):
        for i in range(5):
            c.store.add_message(1, "user", f"msg {i}")
        added = await c.mine_open_questions()
    assert added == [], "an uncited question must be rejected outright"
    assert st.count_active(kind="pursuit") == 0


@case
async def mining_rejects_a_citation_outside_the_fetched_window():
    st = _store()
    llm = FakeLLM(structured=[{"questions": [{"question": "what happened with his job?", "citations": [1]}]}])
    c = _companion(llm, intentions=st)
    with config_override(PURSUIT_MIN_MESSAGES=3, PURSUIT_WINDOW_MESSAGES=3):
        for i in range(10):  # ids 1..10; window (limit 3) only ever sees the last 3
            c.store.add_message(1, "user", f"msg {i}")
        added = await c.mine_open_questions()
    assert added == [], "a citation to an id outside the fetched window must be rejected"
    assert st.count_active(kind="pursuit") == 0


@case
async def mining_accepts_a_grounded_question():
    st = _store()
    llm = FakeLLM(structured=[{"questions": [
        {"question": "how did the job interview go?", "citations": [1]}]}])
    c = _companion(llm, intentions=st)
    with config_override(PURSUIT_MIN_MESSAGES=3, PURSUIT_WINDOW_MESSAGES=40):
        for i in range(5):
            c.store.add_message(1, "user", f"msg {i}")
        added = await c.mine_open_questions()
    assert added == ["how did the job interview go?"]
    active = st.active(kind="pursuit")
    assert len(active) == 1 and active[0]["citations"] == [1]


@case
async def mining_dedupes_case_insensitively():
    st = _store()
    st.add("how did the job interview go?", kind="pursuit", citations=[1])
    llm = FakeLLM(structured=[{"questions": [
        {"question": "HOW DID the job interview go?", "citations": [1]}]}])
    c = _companion(llm, intentions=st)
    with config_override(PURSUIT_MIN_MESSAGES=3, PURSUIT_WINDOW_MESSAGES=40):
        for i in range(5):
            c.store.add_message(1, "user", f"msg {i}")
        added = await c.mine_open_questions()
    assert added == [], "a case-insensitive dup of an existing pursuit must not be re-added"
    assert st.count_active(kind="pursuit") == 1


@case
async def mining_caps_the_active_backlog():
    st = _store()
    llm = FakeLLM(structured=[{"questions": [{"question": "one more thing", "citations": [1]}]}])
    c = _companion(llm, intentions=st)
    with config_override(PURSUIT_MIN_MESSAGES=3, PURSUIT_WINDOW_MESSAGES=40, PURSUIT_MAX_ACTIVE=2):
        for n in range(2):
            st.add(f"existing pursuit {n}", kind="pursuit", citations=[1])
        for i in range(5):
            c.store.add_message(1, "user", f"msg {i}")
        await c.mine_open_questions()
    assert st.count_active(kind="pursuit") == 2, "capped; oldest dropped"
    assert "existing pursuit 0" not in [i["content"] for i in st.active(kind="pursuit")], \
        "the OLDEST must be the one dropped"


@case
async def mining_never_writes_to_the_conversation_store():
    st = _store()
    llm = FakeLLM(structured=[{"questions": [{"question": "x?", "citations": [1]}]}])
    c = _companion(llm, intentions=st)

    def _boom(*a, **kw):
        raise AssertionError("mine_open_questions must never write to the episodic log")
    with config_override(PURSUIT_MIN_MESSAGES=3, PURSUIT_WINDOW_MESSAGES=40):
        for i in range(5):
            c.store.add_message(1, "user", f"msg {i}")
        c.store.add_message = _boom  # any further write is a bug
        await c.mine_open_questions()


@case
async def mining_is_a_noop_without_an_intention_store():
    c = _companion(FakeLLM(), intentions=None)
    for i in range(10):
        c.store.add_message(1, "user", f"msg {i}")
    assert await c.mine_open_questions() == []


# --- A4: go_unavailable() / end_unavailable() -------------------------------------

@case
async def go_unavailable_declines_on_pass():
    st = _store()
    mm = FakeModelManager()
    llm = FakeLLM(structured=[{"choice": "PASS"}])
    c = _companion(llm, intentions=st, thoughts=FakeThoughts(), model_manager=mm,
                  pursuits_factory=build_default_registry)
    went = await c.go_unavailable()
    assert went is False
    assert not c.is_unavailable()
    assert mm.unloaded == 0, "declining must never touch the model"


@case
async def go_unavailable_declines_on_garbage():
    st = _store()
    llm = FakeLLM(structured=[{"choice": "not_a_real_tool"}])
    c = _companion(llm, intentions=st, thoughts=FakeThoughts(),
                  pursuits_factory=build_default_registry)
    assert await c.go_unavailable() is False
    assert not c.is_unavailable()


@case
async def go_unavailable_with_no_registry_is_a_noop():
    c = _companion(FakeLLM(), intentions=_store())
    assert await c.go_unavailable() is False


@case
async def go_unavailable_journals_for_real():
    st = _store()
    mm = FakeModelManager()
    thoughts = FakeThoughts()
    clock = Clock(1_000.0)
    llm = FakeLLM(structured=[{"choice": "journal"}], stream_reply="a real journal entry")
    with config_override(PURSUIT_JOURNAL_MIN=60.0, PURSUIT_JOURNAL_MAX=60.0):  # deterministic
        c = _companion(llm, intentions=st, thoughts=thoughts, model_manager=mm,
                      pursuits_factory=build_default_registry, clock=clock)
        went = await c.go_unavailable()
    assert went is True
    assert c.is_unavailable()
    assert thoughts.added and thoughts.added[0][0] == "a real journal entry", \
        "the real op must actually run, producing a genuine artifact"
    assert c.unavailable_reason() == "sit and write a private journal entry for a bit"
    assert c.unavailable_eta_seconds() == 60.0
    assert mm.unloaded == 1, "the model is freed for the remainder once the real op is done"


@case
async def sit_with_question_reason_names_the_open_question():
    st = _store()
    pid = st.add("how is he really doing with the move?", kind="pursuit", citations=[1])
    thoughts = FakeThoughts()
    llm = FakeLLM(structured=[{"choice": "sit_with_question"}], stream_reply="sat with it")
    c = _companion(llm, intentions=st, thoughts=thoughts,
                  pursuits_factory=build_default_registry)
    went = await c.go_unavailable()
    assert went is True
    assert "how is he really doing with the move?" in c.unavailable_reason()
    assert st.active(kind="pursuit") == [], "a genuine reflection fulfils the question"
    assert any(pid == i["id"] for i in st.all(kind="pursuit") if i["fulfilled_at"])


@case
async def sit_with_question_only_offered_with_an_open_pursuit():
    st = _store()  # no pursuit rows at all
    llm = FakeLLM(structured=[{"choice": "sit_with_question"}])  # she can't actually pick this
    c = _companion(llm, intentions=st, thoughts=FakeThoughts(),
                  pursuits_factory=build_default_registry)
    went = await c.go_unavailable()
    assert went is False, "an unlisted choice (not on the offered menu) must decline"


@case
async def a_declined_reflection_leaves_the_pursuit_open():
    st = _store()
    pid = st.add("what's really going on with his sister?", kind="pursuit", citations=[1])
    llm = FakeLLM(structured=[{"choice": "sit_with_question"}], stream_reply="PASS")
    c = _companion(llm, intentions=st, thoughts=FakeThoughts(),
                  pursuits_factory=build_default_registry)
    went = await c.go_unavailable()
    assert went is True, "she still stepped away, even though the reflection itself declined"
    assert st.count_active(kind="pursuit") == 1, "an unfulfilled question is left open to try again"
    assert st.active(kind="pursuit")[0]["id"] == pid


@case
async def end_unavailable_with_nothing_pending_just_clears_state():
    mm = FakeModelManager()
    clock = Clock(1_000.0)
    c = _companion(FakeLLM(), intentions=_store(), model_manager=mm, clock=clock)
    c._unavailable_until = clock.t + 5
    c._unavailable_reason = "testing"
    c._unavailable_model_unloaded = True
    result = await c.end_unavailable()
    assert result is None
    assert not c.is_unavailable()
    assert mm.loaded, "the model must be reloaded since it had been unloaded"


@case
async def end_unavailable_delivers_a_pending_message():
    mm = FakeModelManager()
    clock = Clock(1_000.0)
    llm = FakeLLM(stream_reply="here's my real reply")
    c = _companion(llm, intentions=_store(), model_manager=mm, clock=clock)
    c._unavailable_until = clock.t + 5
    c._unavailable_model_unloaded = True
    c.queue_pending_message("were you there?")
    result = await c.end_unavailable()
    assert result is not None and result.text == "here's my real reply"
    assert not c.has_pending_message()
    assert not c.is_unavailable()


@case
async def send_clears_a_stale_pending_message():
    c = _companion(FakeLLM(stream_reply="fresh reply"), intentions=_store())
    c.queue_pending_message("an old message that should never be answered")
    async def noop(_t):
        pass
    await c.send("a brand new message", noop)
    assert not c.has_pending_message(), \
        "a live turn must cancel any dangling unavailable-window reply obligation"


if __name__ == "__main__":
    raise SystemExit(run())
