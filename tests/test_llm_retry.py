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
    return types.SimpleNamespace(usage=None,
                                 choices=[types.SimpleNamespace(
                                     delta=types.SimpleNamespace(content=content))])


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
