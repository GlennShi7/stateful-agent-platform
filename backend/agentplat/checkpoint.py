"""Postgres checkpointer without the global lock.

`AsyncPostgresSaver` (langgraph-checkpoint-postgres 3.1.2) wraps every operation in one
`asyncio.Lock`, even when it is given a connection pool. The lock exists for the single-connection
mode, where a psycopg connection cannot run two statements at once. With a pool, each operation
already checks out its own connection, so the lock only serialises checkpoint I/O of unrelated
threads: under the 50-session load test, sessions spent most of their time queued on it.

Order within one thread is still guaranteed by LangGraph itself, which awaits each step's
checkpoint write (durability="sync") before starting the next step.

This overrides a private method; the version is pinned and tests/test_checkpoint.py checks both
correctness and that two threads' writes really overlap.
"""
from contextlib import asynccontextmanager

from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from psycopg.rows import dict_row
from psycopg_pool import AsyncConnectionPool


class PooledPostgresSaver(AsyncPostgresSaver):
    def __init__(self, pool: AsyncConnectionPool, **kw):
        if not isinstance(pool, AsyncConnectionPool):
            raise TypeError("PooledPostgresSaver needs an AsyncConnectionPool")
        super().__init__(pool, **kw)

    @asynccontextmanager
    async def _cursor(self, *, pipeline: bool = False):
        async with self.conn.connection() as conn:
            if pipeline and self.supports_pipeline:
                async with conn.pipeline(), conn.cursor(binary=True, row_factory=dict_row) as cur:
                    yield cur
            elif pipeline:
                async with conn.transaction(), conn.cursor(binary=True, row_factory=dict_row) as cur:
                    yield cur
            else:
                async with conn.cursor(binary=True, row_factory=dict_row) as cur:
                    yield cur


def make_pool(dsn, max_size=20):
    # The checkpointer manages its own transactions and needs autocommit connections.
    return AsyncConnectionPool(dsn, min_size=2, max_size=max_size, open=False,
                               kwargs={"autocommit": True, "prepare_threshold": 0, "row_factory": dict_row})
