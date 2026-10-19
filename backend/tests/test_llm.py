import json

import httpx
import pytest

from agentplat.llm import DeepSeekLLM, LLMError, ScriptedLLM, Turn, tool_call


def sse(*chunks):
    lines = [f"data: {json.dumps(c)}\n\n" for c in chunks] + ["data: [DONE]\n\n"]
    return "".join(lines).encode()


def make(handler, **kw):
    return DeepSeekLLM("k", base_url="https://x", transport=httpx.MockTransport(handler), max_retries=0, **kw)


async def test_streams_content_and_reassembles_tool_call_fragments():
    seen = {}

    def handler(request):
        seen["body"] = json.loads(request.content)
        body = sse(
            {"choices": [{"delta": {"content": "Hel"}}]},
            {"choices": [{"delta": {"content": "lo"}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "id": "c1", "function": {"name": "get_error_rate", "arguments": "{\"window"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "_minutes\": 20}"}}]}}]},
            {"choices": [{"delta": {"tool_calls": [{"index": 1, "id": "c2", "function": {"name": "top_users", "arguments": "{}"}}]}}]},
            {"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 5}},
        )
        return httpx.Response(200, content=body, headers={"content-type": "text/event-stream"})

    tokens = []
    turn = await make(handler).complete([{"role": "user", "content": "hi"}], [{"type": "function"}], tokens.append)
    assert tokens == ["Hel", "lo"] and turn.content == "Hello"
    assert [c["function"]["name"] for c in turn.tool_calls] == ["get_error_rate", "top_users"]
    assert json.loads(turn.tool_calls[0]["function"]["arguments"]) == {"window_minutes": 20}
    assert turn.usage["completion_tokens"] == 5
    assert seen["body"]["thinking"] == {"type": "disabled"} and seen["body"]["stream"] is True
    assert "reasoning_effort" not in seen["body"]


async def test_reasoning_content_is_kept_for_the_next_request():
    def handler(request):
        return httpx.Response(200, content=sse({"choices": [{"delta": {"reasoning_content": "think"}}]},
                                                {"choices": [{"delta": {"content": "ok"}}]}))

    turn = await make(handler, thinking="enabled").complete([], None)
    assert turn.as_message() == {"role": "assistant", "content": "ok", "reasoning_content": "think"}


async def test_http_error_is_raised():
    llm = make(lambda r: httpx.Response(401, content=b'{"error":"bad key"}'))
    with pytest.raises(LLMError, match="401"):
        await llm.complete([], None)


async def test_scripted_llm_streams_words():
    got = []
    llm = ScriptedLLM([Turn(content="a b c"), Turn(tool_calls=[tool_call("x", {})])])
    t1 = await llm.complete([], [], got.append)
    t2 = await llm.complete([], [], got.append)
    assert "".join(got) == "a b c" and t1.content == "a b c" and t2.tool_calls[0]["function"]["name"] == "x"
