"""Kill the agent process mid-run, then resume the same thread from Postgres in a new process.

The write executed before the crash must not execute again: the tools stage is re-run on resume
(its checkpoint was never written), and the idempotency key "<thread>:<tool_call_id>" makes the
MCP server return the recorded result instead of inserting a second alert rule.
"""
import os
import pathlib
import subprocess
import sys
import uuid

import psycopg

from agentplat.authz import StaticAuthorizer
from agentplat.checkpoint import PooledPostgresSaver, make_pool
from agentplat.graph import RunContext, build_graph, config_for, run_turn
from agentplat.llm import ScriptedLLM, Turn
from agentplat.mcp_client import ToolGateway
from agentplat.tokens import caller_token
from agentplat.tools_server import build_server

from .helpers import SETTINGS

BACKEND = pathlib.Path(__file__).resolve().parents[1]


def rule_count(dsn):
    with psycopg.connect(dsn) as conn:
        return conn.execute("SELECT count(*) FROM alert_rules").fetchone()[0]


async def test_killed_process_resumes_and_write_happens_once(repo, pg_dsn):
    thread = "crash-" + uuid.uuid4().hex[:8]
    child = subprocess.run([sys.executable, "-m", "tests.crash_child", pg_dsn, thread], cwd=BACKEND,
                           env={**os.environ, "PYTHONPATH": str(BACKEND)}, capture_output=True, text=True,
                           timeout=60)
    assert child.returncode == 17, child.stderr[-2000:]
    assert rule_count(pg_dsn) == 1  # the write happened before the crash

    pool = make_pool(pg_dsn)
    await pool.open()
    try:
        graph = build_graph(PooledPostgresSaver(pool))
        st = await graph.aget_state(config_for(thread))
        assert st.next == ("tools",)  # last durable step: the agent asked for the tool

        llm = ScriptedLLM([Turn(content="Created the alert rule.")])
        events = []

        async def on_event(e):
            events.append(e)
        tools = ToolGateway(build_server(repo, StaticAuthorizer(), SETTINGS))
        ctx = RunContext(user="bob", caller_token=caller_token("bob", "default", thread, SETTINGS.caller_secret),
                         thread_id=thread, llm=llm, tools=tools)
        status, _ = await run_turn(graph, None, ctx, on_event)
        await tools.aclose()

        assert status == "done"
        assert rule_count(pg_dsn) == 1  # not 2
        tool_end = [e for e in events if e["type"] == "tool_end"][0]
        assert '"replayed": true' in tool_end["result"]
        final = (await graph.aget_state(config_for(thread))).values["messages"]
        assert [m["role"] for m in final] == ["user", "assistant", "tool", "assistant"]
    finally:
        await pool.close()
