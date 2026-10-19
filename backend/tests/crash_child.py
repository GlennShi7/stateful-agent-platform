"""Child process for test_crash_resume: runs one turn and dies (os._exit) right after a write tool
has executed but before LangGraph checkpoints the tools stage. Not a test module."""
import asyncio
import os
import sys


from agentplat.authz import StaticAuthorizer
from agentplat.checkpoint import PooledPostgresSaver, make_pool
from agentplat.graph import RunContext, build_graph, run_turn
from agentplat.llm import ScriptedLLM, Turn, tool_call
from agentplat.mcp_client import ToolGateway
from agentplat.repo import Repo
from agentplat.tokens import caller_token
from agentplat.tools_server import build_server
from tests.helpers import SETTINGS

ALERT = {"event_type": "contract_upload", "metric": "error_rate", "threshold": 0.05, "window_minutes": 15}


async def main(dsn, thread):
    repo = Repo(dsn, dsn)
    await repo.open()
    pool = make_pool(dsn)
    await pool.open()
    saver = PooledPostgresSaver(pool)
    await saver.setup()
    graph = build_graph(saver)
    llm = ScriptedLLM([Turn(tool_calls=[tool_call("create_alert_rule", ALERT, "w1")])])
    tools = ToolGateway(build_server(repo, StaticAuthorizer(), SETTINGS))
    ctx = RunContext(user="bob", caller_token=caller_token("bob", "default", thread, SETTINGS.caller_secret),
                     thread_id=thread, llm=llm, tools=tools,
                     on_tool_done=lambda name: os._exit(17))  # simulated crash: no cleanup, no checkpoint
    await run_turn(graph, {"messages": [{"role": "user", "content": "alert me on upload errors"}]}, ctx)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1], sys.argv[2]))
