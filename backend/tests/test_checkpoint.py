"""PooledPostgresSaver: same behaviour as AsyncPostgresSaver, without serialising unrelated threads."""
import asyncio
import uuid
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver

from agentplat.checkpoint import PooledPostgresSaver, make_pool
from agentplat.graph import RunContext, build_graph, config_for, run_turn
from agentplat.llm import ScriptedLLM, Turn, tool_call

from .helpers import FakeTools


def policy(messages, tools):
    if messages[-1]["role"] == "user":
        return Turn(tool_calls=[tool_call("get_error_rate", {}, f"c{len(messages)}")])
    return Turn(content="ok")


async def max_concurrent_checkouts(saver_cls, pool, n=20):
    saver = saver_cls(pool)
    await saver.setup()
    active, peak = 0, 0
    original = pool.connection

    @asynccontextmanager
    async def counting(*a, **kw):
        nonlocal active, peak
        async with original(*a, **kw) as conn:
            active += 1
            peak = max(peak, active)
            try:
                yield conn
            finally:
                active -= 1
    pool.connection = counting
    graph, llm, tools = build_graph(saver), ScriptedLLM(policy), FakeTools()
    ids = [uuid.uuid4().hex for _ in range(n)]
    await asyncio.gather(*[run_turn(graph, {"messages": [{"role": "user", "content": "x"}]},
                                    RunContext("bob", "tok", t, llm, tools)) for t in ids])
    pool.connection = original
    states = [await graph.aget_state(config_for(t)) for t in ids]
    assert all(s.values["status"] == "done" and len(s.values["messages"]) == 4 for s in states)
    return peak


async def test_pooled_saver_overlaps_threads_and_stays_correct(pg_dsn):
    pool = make_pool(pg_dsn)
    await pool.open()
    try:
        assert await max_concurrent_checkouts(AsyncPostgresSaver, pool) == 1  # the library's global lock
        assert await max_concurrent_checkouts(PooledPostgresSaver, pool) > 1
    finally:
        await pool.close()
