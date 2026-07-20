"""Offline coverage for the tool framework (pillar 4).

Two layers, no LM Studio needed:
  - ToolRegistry: specs/bool/len, and the robust execute() contract — success
    (sync + async handlers), unknown tool, handler exception, non-str result,
    non-dict args, oversize-result truncation.
  - LLMClient.stream_with_tools: the streaming tool loop, driven by a FAKE OpenAI
    client that yields scripted chunks (content and/or tool_call deltas). Verifies
    the call/execute/feed-back/answer cycle, argument fragments reassembling across
    deltas, and every failure seam turning into a fed-back result string rather than
    an aborted turn: malformed-args JSON, unknown tool, handler error, the
    max-iters safety net, a transient-error retry, and <think> stripping.

Run:  python tests/test_tools.py
"""
import sys
import types

from _harness import case, run  # also puts the repo root on sys.path

from core.builtin_tools import make_reminisce_tool, make_time_tool
from core.tools import MAX_RESULT_CHARS, Tool, ToolRegistry
from infrastructure.conversation_store import SqliteConversationStore
from infrastructure.db import connect
from infrastructure.llm_client import LLMClient
from infrastructure.thought_store import SqliteThoughtStore


# --- a registry with a couple of test tools -----------------------------------

def _add(args):  # sync handler
    return str(args["a"] + args["b"])


async def _aecho(args):  # async handler
    return f"echo:{args.get('msg', '')}"


def _boom(_args):
    raise RuntimeError("kaboom")


def build_registry():
    return ToolRegistry([
        Tool("add", "Add two numbers.",
             {"type": "object", "properties": {"a": {"type": "number"}, "b": {"type": "number"}},
              "required": ["a", "b"]}, _add),
        Tool("echo", "Echo a message.",
             {"type": "object", "properties": {"msg": {"type": "string"}}, "required": ["msg"]}, _aecho),
        Tool("boom", "Always fails.", {"type": "object", "properties": {}}, _boom),
    ])


# --- a fake streaming OpenAI client -------------------------------------------

class _Fn:
    def __init__(self, name, arguments):
        self.name = name
        self.arguments = arguments


class _TCDelta:
    """One tool_call delta: name arrives once, arguments stream in pieces."""
    def __init__(self, index, id=None, name=None, arguments=None):
        self.index = index
        self.id = id
        self.function = _Fn(name, arguments)


class _Chunk:
    def __init__(self, content=None, tool_calls=None, usage=None):
        delta = types.SimpleNamespace(content=content, tool_calls=tool_calls)
        self.choices = [types.SimpleNamespace(delta=delta)]
        self.usage = usage


class _Stream:
    def __init__(self, chunks):
        self._chunks = chunks

    def __aiter__(self):
        return self._gen()

    async def _gen(self):
        for c in self._chunks:
            yield c


class _Completions:
    def __init__(self, responder):
        self.responder = responder
        self.calls = []  # kwargs of each create() call, for assertions

    async def create(self, **kwargs):
        self.calls.append(kwargs)
        out = self.responder(kwargs, len(self.calls) - 1)
        if isinstance(out, Exception):
            raise out
        return _Stream(out)


class FakeClient:
    def __init__(self, responder):
        self.chat = types.SimpleNamespace(completions=_Completions(responder))


def make_llm(responder, max_retries=0):
    llm = LLMClient("http://x", "k", "test-model", 0.8, no_think=False, max_retries=max_retries)
    llm.client = FakeClient(responder)
    return llm


_USAGE = types.SimpleNamespace(completion_tokens=5)


def tool_call_chunks(name, args_json, call_id="call_1", index=0):
    """A tool call streamed like LM Studio: id+name first, arguments in two pieces."""
    mid = len(args_json) // 2
    return [
        _Chunk(tool_calls=[_TCDelta(index, id=call_id, name=name, arguments="")]),
        _Chunk(tool_calls=[_TCDelta(index, arguments=args_json[:mid])]),
        _Chunk(tool_calls=[_TCDelta(index, arguments=args_json[mid:])]),
        _Chunk(usage=_USAGE),
    ]


def content_chunks(text):
    return [_Chunk(content=text[i:i + 3]) for i in range(0, len(text), 3)] + [_Chunk(usage=_USAGE)]


async def _collect(llm, registry, prompt="hi", max_iters=5):
    """Run one tool-loop turn, capturing streamed tokens. Returns (text, stats, streamed)."""
    streamed = []

    async def on_token(t):
        streamed.append(t)

    messages = [{"role": "system", "content": "sys"}, {"role": "user", "content": prompt}]
    text, stats = await llm.stream_with_tools(messages, on_token, registry, max_iters)
    return text, stats, "".join(streamed)


# --- ToolRegistry unit tests --------------------------------------------------

@case
async def registry_specs_bool_len():
    reg = build_registry()
    assert bool(reg) and len(reg) == 3
    assert not ToolRegistry()
    specs = reg.specs()
    assert {s["function"]["name"] for s in specs} == {"add", "echo", "boom"}
    assert specs[0]["type"] == "function" and "parameters" in specs[0]["function"]


@case
async def registry_register_replaces_and_unregisters():
    reg = ToolRegistry()
    reg.register(Tool("t", "d", {"type": "object", "properties": {}}, lambda a: "1"))
    reg.register(Tool("t", "d2", {"type": "object", "properties": {}}, lambda a: "2"))  # replace
    assert len(reg) == 1 and reg.get("t").description == "d2"
    assert await reg.execute("t", {}) == "2"
    reg.unregister("t")
    assert not reg and reg.get("t") is None


@case
async def execute_success_sync_and_async():
    reg = build_registry()
    assert await reg.execute("add", {"a": 2, "b": 3}) == "5"
    assert await reg.execute("echo", {"msg": "hey"}) == "echo:hey"


@case
async def execute_unknown_tool_returns_error_string():
    reg = build_registry()
    out = await reg.execute("nope", {})
    assert out.startswith("[error: no tool named") and "nope" in out


@case
async def execute_handler_exception_is_caught():
    reg = build_registry()
    out = await reg.execute("boom", {})
    assert out.startswith("[error running boom:") and "kaboom" in out


@case
async def execute_coerces_nonstr_result_and_nondict_args():
    reg = ToolRegistry([Tool("n", "d", {"type": "object", "properties": {}}, lambda a: 42)])
    assert await reg.execute("n", {}) == "42"
    # a non-dict args payload must not crash the handler contract
    reg2 = ToolRegistry([Tool("k", "d", {"type": "object", "properties": {}},
                              lambda a: f"got:{a}")])
    assert await reg2.execute("k", ["not", "a", "dict"]) == "got:{}"


@case
async def execute_truncates_oversize_result():
    big = "x" * (MAX_RESULT_CHARS + 500)
    reg = ToolRegistry([Tool("big", "d", {"type": "object", "properties": {}}, lambda a: big)])
    out = await reg.execute("big", {})
    assert len(out) < len(big) and out.endswith("[...truncated]")


# --- stream_with_tools loop tests ---------------------------------------------

@case
async def loop_calls_tool_then_answers():
    # round 0: call add(2,3); round 1 (tools still attached): plain answer.
    def responder(kwargs, n):
        if n == 0:
            return tool_call_chunks("add", '{"a": 2, "b": 3}')
        return content_chunks("The sum is 5.")

    llm = make_llm(responder)
    reg = build_registry()
    text, stats, streamed = await _collect(llm, reg)
    assert text == "The sum is 5." and streamed == "The sum is 5."
    assert len(stats["tools"]) == 1
    inv = stats["tools"][0]
    assert inv["name"] == "add" and inv["result"] == "5"
    # the answer round was asked WITH tools available (auto), not forced
    assert llm.client.chat.completions.calls[1].get("tools")


@case
async def loop_reassembles_streamed_arguments():
    # arguments arrive in fragments across deltas; the loop must reassemble valid JSON.
    def responder(kwargs, n):
        return tool_call_chunks("add", '{"a": 40, "b": 2}') if n == 0 else content_chunks("42")

    llm = make_llm(responder)
    text, stats, _ = await _collect(llm, build_registry())
    assert stats["tools"][0]["result"] == "42" and text == "42"


@case
async def loop_malformed_args_feeds_error_and_recovers():
    def responder(kwargs, n):
        if n == 0:
            return tool_call_chunks("add", '{"a": 2, "b":')  # truncated JSON
        return content_chunks("hmm, let me try again")

    llm = make_llm(responder)
    text, stats, _ = await _collect(llm, build_registry())
    assert stats["tools"][0]["result"].startswith("[error: could not parse arguments for add")
    assert text == "hmm, let me try again"  # turn still completed


@case
async def loop_unknown_tool_feeds_error():
    def responder(kwargs, n):
        return tool_call_chunks("teleport", "{}") if n == 0 else content_chunks("can't do that")

    llm = make_llm(responder)
    _, stats, _ = await _collect(llm, build_registry())
    assert stats["tools"][0]["result"].startswith("[error: no tool named")


@case
async def loop_handler_error_feeds_error():
    def responder(kwargs, n):
        return tool_call_chunks("boom", "{}") if n == 0 else content_chunks("something broke")

    llm = make_llm(responder)
    _, stats, _ = await _collect(llm, build_registry())
    assert stats["tools"][0]["result"].startswith("[error running boom:")


@case
async def loop_max_iters_forces_plain_answer():
    # model NEVER stops calling tools when tools are attached; the loop must cap it
    # and force a final answer with NO tools so the turn can't hang.
    def responder(kwargs, n):
        if "tools" in kwargs:
            return tool_call_chunks("add", '{"a": 1, "b": 1}', call_id=f"c{n}")
        return content_chunks("forced answer")  # the tools=None escape hatch

    llm = make_llm(responder)
    text, stats, _ = await _collect(llm, build_registry(), max_iters=3)
    assert text == "forced answer"
    # 3 tool rounds (all with tools) + 1 forced round (no tools) = 4 calls
    calls = llm.client.chat.completions.calls
    assert len(calls) == 4 and "tools" not in calls[-1]
    assert len(stats["tools"]) == 3  # three tool round-trips recorded


@case
async def loop_plain_answer_no_tool():
    def responder(kwargs, n):
        return content_chunks("just chatting")

    llm = make_llm(responder)
    text, stats, streamed = await _collect(llm, build_registry())
    assert text == "just chatting" and streamed == "just chatting"
    assert stats["tools"] == []
    assert len(llm.client.chat.completions.calls) == 1  # answered in one round


@case
async def loop_strips_think_block_in_answer():
    def responder(kwargs, n):
        if n == 0:
            return tool_call_chunks("add", '{"a": 1, "b": 1}')
        return content_chunks("<think>the tool said 2</think>It's 2.")

    llm = make_llm(responder)
    text, _, streamed = await _collect(llm, build_registry())
    assert text == "It's 2." and "<think>" not in streamed and streamed == "It's 2."


@case
async def loop_retries_transient_error_before_emit():
    # first create() throws a retryable error; the loop retries and then answers.
    def responder(kwargs, n):
        if n == 0:
            return RuntimeError("predict request failed: fetch failed")
        return content_chunks("recovered")

    llm = make_llm(responder, max_retries=2)
    text, _, _ = await _collect(llm, build_registry())
    assert text == "recovered"
    assert len(llm.client.chat.completions.calls) == 2  # one failure + one success


# --- built-in tools: reminisce retrieval + get_time (store-backed, no LLM) -----

def _seed_store():
    """Fresh in-memory conversation + thought stores with a little history."""
    conn = connect(":memory:")
    conv = SqliteConversationStore(conn)
    thoughts = SqliteThoughtStore(conn)
    sid = conv.create_session("t")
    conv.add_message(sid, "user", "I went to Denver last week for a paramedic conference.")
    conv.add_message(sid, "assistant", "Nice! How was the conference in Denver?")
    conv.add_message(sid, "user", "Also I adopted a cat named Pixel.")
    thoughts.add("They seem happier lately; the Denver trip did them good.", "joy")
    return conv, thoughts


@case
async def search_messages_ranks_and_filters():
    conv, _ = _seed_store()
    hits = conv.search_messages("Denver conference", limit=5)
    assert hits, "expected matches for Denver"
    # both Denver messages hit; the one that also contains 'conference' ranks first
    assert "conference" in hits[0]["content"].lower() or "Denver" in hits[0]["content"]
    assert [h["id"] for h in hits] == sorted(h["id"] for h in hits)  # oldest-first
    assert conv.search_messages("submarines", 5) == []               # no match -> empty
    assert conv.search_messages("", 5) == []                         # empty query -> empty
    assert conv.search_messages("a", 5) == []                        # too-short token -> empty


@case
async def reminisce_returns_matches_and_thoughts():
    conv, thoughts = _seed_store()
    tool = make_reminisce_tool(thoughts, conv)
    out = await tool.handler({"topic": "Denver trip"})
    assert "Denver" in out
    assert "notes I've written" in out and "happier" in out  # journal folded in


@case
async def reminisce_reports_nothing_found():
    conv, thoughts = _seed_store()
    tool = make_reminisce_tool(thoughts, conv)
    out = await tool.handler({"topic": "submarines"})
    assert "submarines" in out and "Nothing" in out


@case
async def time_tool_reports_a_time():
    tool = make_time_tool()
    out = await tool.handler({})
    assert out.startswith("It is") and any(c.isdigit() for c in out)


if __name__ == "__main__":
    raise SystemExit(run())
