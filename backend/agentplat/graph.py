"""The agent: a LangGraph state machine with five stages, checkpointed after every step.

    intake ──► agent ──┬──► approval ──► tools ──► agent  (ReAct loop)
                       ├──► tools ─────────────► agent
                       └──► finalize ──► END

intake    load the tools this caller may use (filtered by SpiceDB through the MCP server)
agent     one LLM call; streams tokens; counts iterations and stops at the cap
approval  pauses the run (LangGraph interrupt) when a tool marked destructive is requested;
          the decision arrives later, possibly after a restart, as Command(resume=...)
tools     executes the requested calls through MCP with the signed caller context and an
          idempotency key "<thread>:<tool_call_id>"
finalize  records the outcome and emits `done`

Everything that must survive a crash lives in the graph state, which the Postgres checkpointer
writes after every node (durability="sync"). Things that must *not* be persisted, such as the
caller's identity and the service clients, come in through the runtime context on every call, so
a resumed run always acts with the identity of whoever resumed it, checked fresh.
"""
import asyncio
import json
import operator
import time
from dataclasses import dataclass
from typing import Annotated, Any, Callable, TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.runtime import Runtime
from langgraph.types import interrupt

SYSTEM_PROMPT = """You are the operations assistant for a real-time analytics platform.
- Get every number from a tool. Never estimate or invent metrics.
- Error rates from tools are fractions; present them as percentages (0.052 -> 5.2%).
- If a tool returns "permission denied", or the tool you would need is not available to you, tell the
  user they lack that permission. Do not look for a workaround with other tools.
- For changes (alert rules, keys, roles, deletions), state exactly what you changed when done.
- Deleting alert rules, revoking keys or roles, and purging data are approval-gated: when you call
  one of those tools, the platform asks the user to approve that exact call and runs it only if they
  approve. So call the tool as soon as the request is clear; do not ask for confirmation in chat
  first. Ask only when the request is ambiguous (for example, which rule). If the user declines,
  say that nothing was changed.
- All times are UTC. Be concise."""


def now_ms():
    return int(time.time() * 1000)


class AgentState(TypedDict, total=False):
    messages: Annotated[list[dict], operator.add]
    tool_specs: list[dict]
    destructive: list[str]
    iterations: int
    decisions: dict[str, bool]
    status: str
    turn_started_ms: int


@dataclass
class RunContext:
    user: str
    caller_token: str
    thread_id: str
    llm: Any
    tools: Any                     # ToolGateway
    max_iterations: int = 10
    telemetry: Any = None          # Telemetry or None
    on_tool_done: Callable | None = None  # fault-injection hook used by the crash/resume test


def _writer(runtime):
    return runtime.stream_writer or (lambda _chunk: None)


def _last_tool_calls(state):
    msgs = state.get("messages") or []
    return (msgs[-1].get("tool_calls") or []) if msgs and msgs[-1]["role"] == "assistant" else []


async def intake(state: AgentState, runtime: Runtime[RunContext]):
    ctx = runtime.context
    specs, destructive = await ctx.tools.list_for(ctx.caller_token)
    _writer(runtime)({"type": "stage", "stage": "intake", "tools": [s["function"]["name"] for s in specs],
                      "t": now_ms()})
    return {"tool_specs": specs, "destructive": destructive, "iterations": 0, "decisions": {},
            "status": "running", "turn_started_ms": now_ms()}


async def agent(state: AgentState, runtime: Runtime[RunContext]):
    ctx, write = runtime.context, _writer(runtime)
    n = state.get("iterations", 0)
    if n >= ctx.max_iterations:
        text = (f"I stopped after {n} reasoning steps without reaching an answer. "
                "Try a narrower question.")
        write({"type": "token", "text": text, "t": now_ms()})
        return {"status": "capped", "messages": [{"role": "assistant", "content": text}]}
    write({"type": "stage", "stage": "agent", "iteration": n + 1, "t": now_ms()})

    def on_token(text):
        write({"type": "token", "text": text, "t": now_ms()})

    turn = await ctx.llm.complete([{"role": "system", "content": SYSTEM_PROMPT}] + state["messages"],
                                  state.get("tool_specs") or [], on_token=on_token)
    if turn.usage:
        write({"type": "usage", "prompt_tokens": turn.usage.get("prompt_tokens", 0),
               "completion_tokens": turn.usage.get("completion_tokens", 0), "t": now_ms()})
    return {"messages": [turn.as_message()], "iterations": n + 1}


def route_after_agent(state: AgentState):
    if state.get("status") == "capped":
        return "finalize"
    calls = _last_tool_calls(state)
    if not calls:
        return "finalize"
    destructive = set(state.get("destructive") or [])
    if any(c["function"]["name"] in destructive for c in calls):
        return "approval"
    return "tools"


async def approval(state: AgentState, runtime: Runtime[RunContext]):
    destructive = set(state.get("destructive") or [])
    decided = state.get("decisions") or {}
    pending = [c for c in _last_tool_calls(state)
               if c["function"]["name"] in destructive and c["id"] not in decided]
    if not pending:
        return {}
    # Pauses the graph. The checkpoint already holds everything up to here, so the run can wait
    # indefinitely (and the process can restart) before the user answers.
    answer = interrupt({"type": "approval", "calls": [
        {"id": c["id"], "name": c["function"]["name"], "arguments": c["function"]["arguments"]} for c in pending]})
    answer = answer if isinstance(answer, dict) else {}
    return {"decisions": {**decided, **{c["id"]: bool(answer.get(c["id"], False)) for c in pending}}}


async def _run_one(call, state, ctx, write):
    name = call["function"]["name"]
    allowed = {s["function"]["name"] for s in state.get("tool_specs") or []}
    destructive = set(state.get("destructive") or [])
    t0 = time.perf_counter()
    write({"type": "tool_start", "id": call["id"], "name": name, "arguments": call["function"]["arguments"],
           "t": now_ms()})
    if name not in allowed:
        ok, text = False, json.dumps({"error": f"tool {name!r} is not available to you"})
    elif name in destructive and not (state.get("decisions") or {}).get(call["id"]):
        ok, text = False, json.dumps({"error": "the user declined this action"})
    else:
        try:
            args = json.loads(call["function"]["arguments"] or "{}")
            if not isinstance(args, dict):
                raise ValueError("arguments must be a JSON object")
        except ValueError as exc:
            ok, text = False, json.dumps({"error": f"invalid JSON arguments: {exc}"})
        else:
            ok, text = await ctx.tools.call(name, args, ctx.caller_token,
                                            idempotency_key=f"{ctx.thread_id}:{call['id']}")
    ms = (time.perf_counter() - t0) * 1000
    write({"type": "tool_end", "id": call["id"], "name": name, "ok": ok, "result": text[:2000],
           "ms": round(ms, 1), "t": now_ms()})
    if ctx.telemetry:
        ctx.telemetry.emit("agent_tool_call", ctx.user, int(ms), not ok)
    if ctx.on_tool_done:
        ctx.on_tool_done(name)
    return {"role": "tool", "tool_call_id": call["id"], "content": text}


async def tools(state: AgentState, runtime: Runtime[RunContext]):
    ctx, write = runtime.context, _writer(runtime)
    write({"type": "stage", "stage": "tools", "t": now_ms()})
    # Independent calls from one LLM turn run concurrently; results keep the call order.
    results = await asyncio.gather(*[_run_one(c, state, ctx, write) for c in _last_tool_calls(state)])
    return {"messages": list(results)}


async def finalize(state: AgentState, runtime: Runtime[RunContext]):
    ctx = runtime.context
    status = "capped" if state.get("status") == "capped" else "done"
    duration = now_ms() - state.get("turn_started_ms", now_ms())
    _writer(runtime)({"type": "done", "status": status, "iterations": state.get("iterations", 0),
                      "duration_ms": duration, "t": now_ms()})
    if ctx.telemetry:
        ctx.telemetry.emit("agent_run", ctx.user, duration, status != "done")
    return {"status": status}


def build_graph(checkpointer):
    g = StateGraph(AgentState, context_schema=RunContext)
    g.add_node("intake", intake)
    g.add_node("agent", agent)
    g.add_node("approval", approval)
    g.add_node("tools", tools)
    g.add_node("finalize", finalize)
    g.add_edge(START, "intake")
    g.add_edge("intake", "agent")
    g.add_conditional_edges("agent", route_after_agent, ["approval", "tools", "finalize"])
    g.add_edge("approval", "tools")
    g.add_edge("tools", "agent")
    g.add_edge("finalize", END)
    return g.compile(checkpointer=checkpointer)


def config_for(thread_id):
    # recursion_limit is LangGraph's own backstop; the 10-iteration cap in `agent` is the real limit
    # (each iteration is about 2-3 steps).
    return {"configurable": {"thread_id": thread_id}, "recursion_limit": 60}


async def run_turn(graph, inp, ctx: RunContext, on_event=None):
    """Drive the graph until it finishes or pauses. `inp` is {"messages": [...]}, None (resume
    after a crash) or a Command (answer to an approval). Returns ("done"|"capped"|"approval", payload)."""
    async for mode, chunk in graph.astream(inp, config_for(ctx.thread_id), context=ctx,
                                           stream_mode=["custom", "updates"], durability="sync"):
        if mode == "custom":
            if on_event:
                await on_event(chunk)
        elif "__interrupt__" in chunk:
            payload = chunk["__interrupt__"][0].value
            if on_event:
                await on_event({**payload, "type": "approval_required", "t": now_ms()})
            return "approval", payload
    state = await graph.aget_state(config_for(ctx.thread_id))
    return state.values.get("status", "done"), None
