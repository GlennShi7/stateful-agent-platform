import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agentplat.graph import SYSTEM_PROMPT, RunContext, build_graph, config_for, run_turn
from agentplat.llm import ScriptedLLM, Turn, tool_call

from .helpers import FakeTools


def setup(script, tools=None, thread="t1", max_iterations=10, **kw):
    graph = build_graph(InMemorySaver())
    llm = ScriptedLLM(script)
    tools = tools or FakeTools()
    ctx = RunContext(user="bob", caller_token="signed-bob", thread_id=thread, llm=llm, tools=tools,
                     max_iterations=max_iterations, **kw)
    return graph, llm, tools, ctx


def user(text):
    return {"messages": [{"role": "user", "content": text}]}


async def collect(graph, inp, ctx):
    events = []

    async def on_event(e):
        events.append(e)
    outcome = await run_turn(graph, inp, ctx, on_event)
    return outcome, events


async def test_plain_answer_streams_tokens_and_finishes():
    graph, llm, tools, ctx = setup([Turn(content="all systems normal")])
    (status, _), events = await collect(graph, user("status?"), ctx)
    assert status == "done"
    assert "".join(e["text"] for e in events if e["type"] == "token") == "all systems normal"
    assert [e["stage"] for e in events if e["type"] == "stage"] == ["intake", "agent"]
    assert events[-1]["type"] == "done"
    sent = llm.seen[0]["messages"]
    assert sent[0] == {"role": "system", "content": SYSTEM_PROMPT} and sent[1]["content"] == "status?"


async def test_react_loop_calls_tool_then_answers():
    tools = FakeTools(results={"get_error_rate": [{"event_type": "contract_upload", "error_rate": 0.22}]})
    graph, llm, tools, ctx = setup([Turn(tool_calls=[tool_call("get_error_rate", {"window_minutes": 20}, "c1")]),
                                    Turn(content="contract_upload is at 22%")], tools)
    (status, _), events = await collect(graph, user("errors?"), ctx)
    assert status == "done" and tools.executed == [("get_error_rate", {"window_minutes": 20})]
    st = await graph.aget_state(config_for("t1"))
    roles = [m["role"] for m in st.values["messages"]]
    assert roles == ["user", "assistant", "tool", "assistant"]
    assert st.values["messages"][2]["tool_call_id"] == "c1"
    assert [e["type"] for e in events if e["type"].startswith("tool")] == ["tool_start", "tool_end"]


async def test_iteration_cap_stops_a_runaway_loop():
    def always_tool(messages, tools):
        return Turn(tool_calls=[tool_call("get_error_rate", {}, f"c{len(messages)}")])

    graph, llm, tools, ctx = setup(always_tool)
    (status, _), events = await collect(graph, user("loop forever"), ctx)
    assert status == "capped"
    assert llm.calls == 10 and len(tools.executed) == 10
    assert events[-1] == {**events[-1], "type": "done", "status": "capped", "iterations": 10}


async def test_destructive_tool_waits_for_approval_and_runs_when_approved():
    script = [Turn(tool_calls=[tool_call("delete_alert_rule", {"rule_id": 3}, "d1")]), Turn(content="deleted")]
    graph, llm, tools, ctx = setup(script)
    (status, payload), _ = await collect(graph, user("delete rule 3"), ctx)
    assert status == "approval" and payload["calls"][0]["name"] == "delete_alert_rule"
    assert tools.executed == []  # nothing ran before the decision
    st = await graph.aget_state(config_for("t1"))
    assert st.next == ("approval",) and st.interrupts

    (status, _), _ = await collect(graph, Command(resume={"d1": True}), ctx)
    assert status == "done" and tools.executed == [("delete_alert_rule", {"rule_id": 3})]


async def test_declined_action_is_reported_to_the_model_not_executed():
    script = [Turn(tool_calls=[tool_call("delete_alert_rule", {"rule_id": 3}, "d1")]), Turn(content="ok, kept it")]
    graph, llm, tools, ctx = setup(script)
    await collect(graph, user("delete rule 3"), ctx)
    (status, _), _ = await collect(graph, Command(resume={"d1": False}), ctx)
    assert status == "done" and tools.executed == []
    tool_msg = llm.seen[-1]["messages"][-1]
    assert tool_msg["role"] == "tool" and "declined" in tool_msg["content"]


async def test_tool_outside_the_callers_list_is_refused_locally():
    script = [Turn(tool_calls=[tool_call("purge_events", {"older_than_days": 1}, "p1")]), Turn(content="cannot")]
    graph, llm, tools, ctx = setup(script)
    (status, _), _ = await collect(graph, user("purge"), ctx)
    assert status == "done" and tools.executed == []
    assert "not available" in llm.seen[-1]["messages"][-1]["content"]


async def test_bad_json_arguments_go_back_to_the_model():
    bad = {"id": "b1", "type": "function", "function": {"name": "get_error_rate", "arguments": "{oops"}}
    graph, llm, tools, ctx = setup([Turn(tool_calls=[bad]), Turn(content="retrying")])
    await collect(graph, user("x"), ctx)
    assert tools.executed == [] and "invalid JSON" in llm.seen[-1]["messages"][-1]["content"]


async def test_crash_in_tools_stage_resumes_without_repeating_the_write():
    script = [Turn(tool_calls=[tool_call("create_alert_rule", {"event_type": "api_call"}, "w1")]),
              Turn(content="created")]
    graph, llm, tools, ctx = setup(script)

    def crash(_name):
        raise RuntimeError("process died")  # after the write reached the tool server, before the checkpoint
    ctx.on_tool_done = crash
    with pytest.raises(RuntimeError):
        await collect(graph, user("make an alert"), ctx)
    st = await graph.aget_state(config_for("t1"))
    assert st.next == ("tools",)  # the last checkpoint is the agent step that requested the call

    ctx.on_tool_done = None
    (status, _), _ = await collect(graph, None, ctx)  # resume from the checkpoint
    assert status == "done"
    assert tools.executed == [("create_alert_rule", {"event_type": "api_call"})]  # executed once
    replayed = json.loads((await graph.aget_state(config_for("t1"))).values["messages"][2]["content"])
    assert replayed["replayed"] is True


async def test_identity_comes_from_the_runtime_context_each_time():
    script = [Turn(tool_calls=[tool_call("delete_alert_rule", {"rule_id": 1}, "d1")]), Turn(content="done")]
    graph, llm, tools, ctx = setup(script)
    await collect(graph, user("delete rule 1"), ctx)
    ctx.caller_token = "signed-bob-new"  # e.g. resumed later, from a fresh session
    await collect(graph, Command(resume={"d1": True}), ctx)
    assert tools.listed_with == ["signed-bob"]
    assert "signed-bob-new" not in json.dumps((await graph.aget_state(config_for("t1"))).values)


async def test_second_turn_keeps_conversation_and_resets_iterations():
    graph, llm, tools, ctx = setup([Turn(content="first"), Turn(content="second")])
    await collect(graph, user("one"), ctx)
    await collect(graph, user("two"), ctx)
    st = await graph.aget_state(config_for("t1"))
    assert [m["content"] for m in st.values["messages"]] == ["one", "first", "two", "second"]
    assert st.values["iterations"] == 1
