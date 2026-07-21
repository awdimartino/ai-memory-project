"""Offline coverage for LLMClient's transient-error retry.

Fakes the OpenAI client so we can drive the retry logic deterministically (no LM
Studio): a retryable error ("fetch failed") is retried; a non-retryable error is
not; and a chat stream only retries BEFORE the first visible token (never after
tokens have been emitted, which would double-stream).

Run:  python tests/test_llm_retry.py
"""
import types

from _harness import case, run  # also puts the repo root on sys.path

from infrastructure.llm_client import LLMClient, _is_retryable

RETRYABLE = Exception("Error code: 400 - Engine protocol predict request failed: fetch failed")


def _chunk(content=None, tokens=None):
    if tokens is not None:
        return types.SimpleNamespace(usage=types.SimpleNamespace(completion_tokens=tokens),
                                     choices=[])
    # `tool_calls` is always present on a real SDK delta (None when the model isn't
    # calling a tool), so the fake carries it too — stream() shares one code path with
    # the tool loop, and a fake without it models a contract the SDK doesn't have.
    return types.SimpleNamespace(usage=None,
                                 choices=[types.SimpleNamespace(
                                     delta=types.SimpleNamespace(content=content,
                                                                 tool_calls=None))])


class FakeCompletions:
    """create() fails `fail_times`, then succeeds (JSON for non-stream, chunks for stream)."""

    def __init__(self, fail_times=0, err=RETRYABLE, content='{"memories": []}',
                 tokens=("Hel", "lo"), fail_mid_stream=False):
        self.fail_times = fail_times
        self.err = err
        self.content = content
        self.tokens = tokens
        self.fail_mid_stream = fail_mid_stream
        self.calls = 0

    async def create(self, **kwargs):
        self.calls += 1
        if self.calls <= self.fail_times:
            raise self.err
        if kwargs.get("stream"):
            fail_mid = self.fail_mid_stream

            async def gen():
                yield _chunk(self.tokens[0])
                if fail_mid:
                    raise RETRYABLE
                for t in self.tokens[1:]:
                    yield _chunk(t)
                yield _chunk(tokens=len(self.tokens))
            return gen()
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=self.content))])


def _client(fc):
    return types.SimpleNamespace(chat=types.SimpleNamespace(completions=fc))


def _llm(fc, retries=3):
    llm = LLMClient("http://localhost:1234/v1", "k", "m", 0.8, max_retries=retries)
    llm.client = _client(fc)
    return llm


@case
async def leaked_reasoning_is_stripped_from_the_stored_reply():
    """LM Studio bug #2147: a whole chain of thought arrives as `content`.

    Observed live 2026-07-20 — the CoT ended in `</think>` with NO opening tag, so the
    matched-pair regex didn't fire and the entire reasoning was streamed to the user
    AND written to her history, where it poisoned every later turn's context.

    Streaming can't be un-sent, but STORING it is the damage that compounds.
    """
    leak = ('" (Wait, I am not supposed to use the tool unless necessary).\n'
            '    *   "okay fine, i was wrong." (Direct)\n\n'
            "5.  **Refining for Persona:** I am irritated but trying to be direct.\n"
            "</think>\n\n"
            "fine, maybe i was wrong to say that.")
    fc = FakeCompletions(tokens=(leak,))
    text, stats = await _llm(fc).stream([{"role": "user", "content": "hi"}], _noop)

    assert text == "fine, maybe i was wrong to say that.", f"leak not stripped: {text!r}"
    assert "</think>" not in text
    assert "Refining for Persona" not in text, "reasoning survived into the stored reply"
    # It is thinking, so it belongs in the inspector rather than being thrown away.
    assert "Refining for Persona" in (stats.get("reasoning") or ""), "leak lost, not rerouted"


@case
async def leaked_reasoning_still_reaches_the_UI_so_done_must_carry_the_final_text():
    """The other half of the bug above, and the reason /ws sends `text` on "done".

    Stripping protects the STORE, but the tokens were already streamed — so the
    browser bubble keeps showing the chain of thought while the stored message is
    clean. Reported live 2026-07-21. This pins the divergence the UI fix relies on:
    what the user saw is NOT what she said. The web layer reconciles by treating the
    final text as authoritative on "done"; if this assertion ever flips to equal,
    that reconciliation has become dead code and should be removed.
    """
    leak = ("The user seems irritated. I should be direct and brief.\n"
            "</think>\n\n"
            "yeah, fair.")
    streamed = []

    async def collect(tok):
        streamed.append(tok)

    fc = FakeCompletions(tokens=(leak,))
    text, _ = await _llm(fc).stream([{"role": "user", "content": "hi"}], collect)
    shown = "".join(streamed)

    assert text == "yeah, fair.", text
    assert "</think>" in shown, "the leak really was streamed to the UI"
    assert shown != text, "streamed output and final text must diverge, or there'd be nothing to fix"


@case
async def a_matched_think_block_still_works():
    """The normal path must be untouched by the orphan fix."""
    fc = FakeCompletions(tokens=("<think>weighing it up</think>\n\nyeah, sounds right.",))
    text, stats = await _llm(fc).stream([{"role": "user", "content": "hi"}], _noop)
    assert text == "yeah, sounds right.", text
    assert "weighing it up" in (stats.get("reasoning") or "")


@case
async def a_reply_with_no_thinking_is_unchanged():
    """No tags at all: nothing may be stripped."""
    fc = FakeCompletions(tokens=("just a normal reply.",))
    text, stats = await _llm(fc).stream([{"role": "user", "content": "hi"}], _noop)
    assert text == "just a normal reply.", text
    assert not (stats.get("reasoning") or "")


async def _noop(_t):
    pass


@case
async def retryable_classification():
    assert _is_retryable(RETRYABLE)
    assert _is_retryable(Exception("Connection error."))
    assert not _is_retryable(ValueError("bad json schema"))


@case
async def structured_retries_then_succeeds():
    fc = FakeCompletions(fail_times=2, content='{"memories":[{"content":"x","category":"user"}]}')
    out = await _llm(fc).structured([{"role": "system", "content": "s"}], {}, "m")
    assert fc.calls == 3, f"expected 3 calls (2 fail + 1 ok), got {fc.calls}"
    assert out == [{"content": "x", "category": "user"}], out


@case
async def structured_gives_up_on_non_retryable():
    fc = FakeCompletions(fail_times=1, err=ValueError("nope"))
    try:
        await _llm(fc).structured([{"role": "system", "content": "s"}], {}, "m")
        assert False, "should have raised"
    except ValueError:
        pass
    assert fc.calls == 1, f"non-retryable must not retry; got {fc.calls} calls"


@case
async def structured_exhausts_retries_then_raises():
    fc = FakeCompletions(fail_times=99)  # always fails
    try:
        await _llm(fc, retries=2).structured([{"role": "system", "content": "s"}], {}, "m")
        assert False, "should have raised after exhausting retries"
    except Exception as e:  # noqa: BLE001
        assert "fetch failed" in str(e)
    assert fc.calls == 3, f"expected 1 + 2 retries = 3 calls, got {fc.calls}"


@case
async def stream_retries_before_first_token():
    fc = FakeCompletions(fail_times=2, tokens=("Hi", " there"))
    got = []
    text, stats = await _llm(fc).stream([{"role": "system", "content": "s"}], lambda t: _collect(got, t))
    assert fc.calls == 3, f"expected 3 calls, got {fc.calls}"
    assert text == "Hi there", repr(text)
    assert "".join(got) == "Hi there", got


@case
async def stream_does_not_retry_after_emitting():
    # Fails mid-stream AFTER a token was emitted -> must NOT retry (would double-stream).
    fc = FakeCompletions(fail_times=0, tokens=("Hi", " x"), fail_mid_stream=True)
    got = []
    try:
        await _llm(fc).stream([{"role": "system", "content": "s"}], lambda t: _collect(got, t))
        assert False, "should have raised (can't retry mid-stream)"
    except Exception as e:  # noqa: BLE001
        assert "fetch failed" in str(e)
    assert fc.calls == 1, f"must not retry after emitting; got {fc.calls} calls"
    assert "".join(got) == "Hi", got  # the one token that made it out


async def _collect(bucket, t):
    bucket.append(t)


if __name__ == "__main__":
    raise SystemExit(run())
