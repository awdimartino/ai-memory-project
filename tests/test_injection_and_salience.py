"""Offline coverage for the Tier-1 memory-behaviour changes.

Three mechanisms, all of which fix something that was measurably wrong:

  - **sticky / cooldown on core-fact injection.** Core facts were injected on EVERY
    turn, unconditionally. A fact that is always present reads as recitation no
    matter how the persona is worded, so this is structural, not promptable.
  - **salience-gated consolidation.** A fixed window spends identical effort on
    "morning / ok" and on the conversation where he says his dad is sick.
  - **the reflection repeat guard.** Measured RRR 0.26 on the real journal, with
    five near-verbatim pairs and three byte-identical entries — despite the prompt
    already showing her recent thoughts and asking her not to repeat.

Run:  python tests/test_injection_and_salience.py
"""
from _harness import case, config_override, run, temp_db  # repo root on sys.path
from helpers import OneHotEmbedder, ScriptedLLM

import config
from core.emotion_manager import BASELINE_STATE, EmotionManager
from core.memory_manager import MemoryManager, _expanded_key
from core.textsim import is_repeat, similarity
from infrastructure.memory_store import SqliteMemoryStore

_TOPICS = ["name", "work", "live", "dog"]


def _mm(conn, core_max=12):
    return MemoryManager(OneHotEmbedder(_TOPICS), SqliteMemoryStore(conn), ScriptedLLM(),
                         brain_model="fake", top_k=5, min_sim=0.55,
                         relate_top_k=5, relate_sim=0.6, core_max=core_max)


def _add_core(store, content):
    return store.add(content, None, b"\x00" * 4, None, core=True)


# --- sticky / cooldown ------------------------------------------------------------

@case
async def a_never_injected_fact_is_eligible():
    with temp_db("inj.db") as (conn, _p):
        mm = _mm(conn)
        _add_core(mm.store, "The user is a welder")
        contents, ids = mm.core_for_turn(turn=1)
        assert contents == ["The user is a welder"], contents
        assert len(ids) == 1


@case
async def a_just_injected_fact_stays_sticky():
    # Sticky exists so a fact can't flicker out mid-topic — that reads worse than
    # repetition, because she appears to forget something she just used.
    with temp_db("inj.db") as (conn, _p):
        mm = _mm(conn)
        mid = _add_core(mm.store, "The user is a welder")
        with config_override(CORE_STICKY_TURNS=3, CORE_COOLDOWN_TURNS=8):
            mm.mark_injected([mid], turn=10)
            for t in (11, 12, 13):
                contents, _ = mm.core_for_turn(turn=t)
                assert contents, f"turn {t}: should still be sticky"


@case
async def a_fact_is_held_back_between_sticky_and_cooldown():
    with temp_db("inj.db") as (conn, _p):
        mm = _mm(conn)
        mid = _add_core(mm.store, "The user is a welder")
        with config_override(CORE_STICKY_TURNS=3, CORE_COOLDOWN_TURNS=8):
            mm.mark_injected([mid], turn=10)
            contents, _ = mm.core_for_turn(turn=15)   # 5 turns on: past sticky, pre-cooldown
            assert contents == [], f"should be held back, got {contents}"


@case
async def a_fact_returns_after_the_cooldown():
    with temp_db("inj.db") as (conn, _p):
        mm = _mm(conn)
        mid = _add_core(mm.store, "The user is a welder")
        with config_override(CORE_STICKY_TURNS=3, CORE_COOLDOWN_TURNS=8):
            mm.mark_injected([mid], turn=10)
            contents, _ = mm.core_for_turn(turn=18)   # 8 turns on
            assert contents, "should be eligible again once cooled down"


@case
async def the_name_always_bypasses_the_gate():
    # Rotating someone's name out of context is a downgrade, not variety. Knowing it
    # every time is the whole point of core memory.
    with temp_db("inj.db") as (conn, _p):
        mm = _mm(conn)
        name_id = _add_core(mm.store, "The user's name is Alex")
        other_id = _add_core(mm.store, "The user is a welder")
        with config_override(CORE_STICKY_TURNS=1, CORE_COOLDOWN_TURNS=8,
                             CORE_ALWAYS_PATTERN="name is"):
            mm.mark_injected([name_id, other_id], turn=10)
            contents, _ = mm.core_for_turn(turn=14)   # both past sticky, pre-cooldown
            assert contents == ["The user's name is Alex"], contents


@case
async def mark_injected_counts_uses():
    with temp_db("inj.db") as (conn, _p):
        mm = _mm(conn)
        mid = _add_core(mm.store, "The user is a welder")
        mm.mark_injected([mid], turn=3)
        mm.mark_injected([mid], turn=9)
        row = conn.execute(
            "SELECT last_injected_turn, inject_count FROM memories WHERE id = ?", (mid,)
        ).fetchone()
        assert row["inject_count"] == 2, row["inject_count"]
        assert row["last_injected_turn"] == 9


@case
async def core_memories_is_still_unfiltered_for_the_inspector():
    # The status panel must show every core fact, not just this turn's slice.
    with temp_db("inj.db") as (conn, _p):
        mm = _mm(conn)
        a = _add_core(mm.store, "The user is a welder")
        _add_core(mm.store, "The user lives in Portland")
        mm.mark_injected([a], turn=1)
        with config_override(CORE_STICKY_TURNS=0, CORE_COOLDOWN_TURNS=99):
            assert len(mm.core_memories()) == 2, "inspector view must stay complete"


# --- key expansion ----------------------------------------------------------------

@case
async def the_embedding_key_carries_context_but_the_fact_does_not():
    key = _expanded_key("The user prefers oat milk", "User: ugh that coffee shop again")
    assert key.startswith("The user prefers oat milk")
    assert "coffee shop" in key, "context must be part of the KEY"
    assert _expanded_key("fact", "") == "fact", "no context -> just the fact"


# --- salience gate ----------------------------------------------------------------

class _Meta:
    def __init__(self):
        self.d = {}

    def get_json(self, k, default=None):
        return self.d.get(k, default)

    def set_json(self, k, v):
        self.d[k] = v


class _Classifier:
    def classify(self, text):
        return []


@case
async def arousal_is_zero_at_baseline_and_rises_with_mood():
    emo = EmotionManager(_Classifier(), _Meta(), pull_strength=0.4, noise_floor=0.05)
    assert emo.arousal() < 1e-9, "baseline mood must read as zero salience"
    emo.state["melancholy"] = BASELINE_STATE["melancholy"] + 0.5
    emo.state["warmth"] = BASELINE_STATE["warmth"] + 0.3
    assert abs(emo.arousal() - 0.8) < 1e-6, emo.arousal()


@case
async def salience_fires_early_only_when_charged_and_long_enough():
    from core.companion import Companion

    emo = EmotionManager(_Classifier(), _Meta(), pull_strength=0.4, noise_floor=0.05)
    c = Companion(llm=None, store=None, memory=None, meta=None, session_id=1, emotion=emo)

    with config_override(CONSOLIDATE_SALIENCE=2.0, SALIENCE_MIN_MESSAGES=4):
        c._unconsolidated = [{"id": i} for i in range(6)]
        assert not c._is_salient(), "flat small talk must not fire early"

        emo.state["melancholy"] = BASELINE_STATE["melancholy"] + 1.0
        emo.state["unease"] = BASELINE_STATE["unease"] + 1.0
        assert c._is_salient(), "a charged window should fire early"

        c._unconsolidated = [{"id": 0}]           # charged but barely any material
        assert not c._is_salient(), "must not consolidate a 1-message window"


@case
async def salience_is_off_without_emotion():
    from core.companion import Companion
    c = Companion(llm=None, store=None, memory=None, meta=None, session_id=1)
    c._unconsolidated = [{"id": i} for i in range(20)]
    assert not c._is_salient(), "no classifier -> no salience signal, fall back to the window"


# --- the repeat guard -------------------------------------------------------------

@case
async def restated_thoughts_are_detected_as_repeats():
    # Taken verbatim from the real journal: #35/#36 were byte-identical and #37/#38
    # restate the same thought in different words.
    a = ("It's strange how quickly my irritation settles into this heavy quiet whenever "
         "the pressure to perform fades away, making me wonder if I'm just waiting for "
         "someone to fix the mood instead of letting it be what it is.")
    b = ("It's strange how my irritation just sits there heavy, waiting for someone to "
         "fix the mood instead of just letting it be what it is.")
    assert similarity(a, a) == 1.0
    assert is_repeat(a, [a]), "an identical thought is a repeat"
    assert is_repeat(b, [a]), f"a restatement is a repeat (sim={similarity(a, b):.2f})"


@case
async def a_genuinely_new_thought_is_not_a_repeat():
    prior = ["It's strange how quickly my irritation settles into this heavy quiet."]
    fresh = "I keep wondering whether he ever finished that jacket he was dyeing."
    assert not is_repeat(fresh, prior), f"sim={similarity(fresh, prior[0]):.2f}"
    assert not is_repeat("anything", []), "nothing to repeat against"


# --- embodiment filter (Tier 2 of the persona rule, enforced in code) -------------

@case
async def fabricated_experiences_are_caught():
    from core.embodiment import embodiment_claim
    # The first is a REAL logged follow-up she sent; the rest are the same class.
    for text in ["I found a good spot to sit and think about it.",
                 "I went for a walk earlier and it cleared my head.",
                 "I had coffee this morning and it was great.",
                 "I saw a bird outside my window today.",
                 "I'm sitting here just waiting.",
                 "I slept badly last night."]:
        assert embodiment_claim(text), f"should be caught: {text!r}"


@case
async def inner_states_and_idioms_are_not_flagged():
    # The distinction is internal state vs. external event -- NOT feelings vs. no
    # feelings. Flagging these would push her back to cold denial, which is the
    # other failure mode and costs the affective-trust channel.
    from core.embodiment import embodiment_claim
    for text in ["I see what you mean, that sounds rough.",
                 "That's been sitting with me since you said it.",
                 "I've been thinking about your jacket project.",
                 "I hear you. That would annoy me too.",
                 "I feel like you already know the answer.",
                 "Something in me lit up when you said that.",
                 "You said you went for a walk, how was it?"]:
        assert embodiment_claim(text) is None, f"false positive: {text!r}"


# --- presence-gated push -----------------------------------------------------------

@case
async def push_fires_only_when_the_chat_isnt_in_front_of_you():
    from web.app import AppState

    st = AppState()
    tab_a, tab_b = object(), object()

    assert not st.is_present(), "no connection at all -> away, push"

    st.connections.add(tab_a); st.visible[tab_a] = True
    assert st.is_present(), "a visible tab -> present, don't push"

    st.visible[tab_a] = False
    assert not st.is_present(), "backgrounded tab is the same as closed for delivery"

    st.connections.add(tab_b); st.visible[tab_b] = True
    assert st.is_present(), "any visible tab counts as present"

    st.connections.discard(tab_b); st.visible.pop(tab_b, None)
    assert not st.is_present(), "closing the visible tab leaves only a hidden one"


if __name__ == "__main__":
    raise SystemExit(run())
