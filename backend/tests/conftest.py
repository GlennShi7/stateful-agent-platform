import asyncio
import os

import psycopg
import pytest

from agentplat.repo import Repo
from agentplat.schema import apply_sql
from agentplat.seed import seed

TEST_DSN = os.environ.get("TEST_DATABASE_URL", "postgresql://agent:agent@localhost:5432/agent")
APP_TABLES = "alert_rules, annotations, api_keys, tool_invocations, audit_log, threads"


def _reachable(dsn):
    try:
        psycopg.connect(dsn, connect_timeout=2).close()
        return True
    except psycopg.OperationalError:
        return False


@pytest.fixture(scope="session")
def pg_dsn():
    if not _reachable(TEST_DSN):
        pytest.skip(f"Postgres not reachable at {TEST_DSN}")

    async def prepare():
        await apply_sql(TEST_DSN, ["001_analytics.sql", "002_app.sql"])
        await seed(TEST_DSN)
    asyncio.run(prepare())
    return TEST_DSN


@pytest.fixture
async def repo(pg_dsn):
    async with await psycopg.AsyncConnection.connect(pg_dsn, autocommit=True) as conn:
        await conn.execute(f"TRUNCATE {APP_TABLES} RESTART IDENTITY")
    r = Repo(pg_dsn, pg_dsn, max_size=5)
    await r.open()
    yield r
    await r.close()
