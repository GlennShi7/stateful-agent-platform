"""Run the evaluation tasks against the real agent: same graph, same MCP server and middleware,
real SpiceDB and Postgres, real DeepSeek. Each task starts from a clean state: analytics data
re-seeded, app tables emptied, and the workspace's roles reset to the demo assignments.

The eval uses its own SpiceDB workspace ("eval"), so it can run while the demo stack is in use.

Usage (inside the backend container, see `make eval`):
    python -m evals.run                    # all tasks once
    python -m evals.run --repeat 3         # each task 3 times (completion rate = passed / runs)
    python -m evals.run --only create_api_key,grant_read_only
"""
import argparse
import asyncio
import dataclasses
import json
import logging
import pathlib
import time
import uuid
from dataclasses import dataclass, field

import psycopg
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from agentplat import tokens
from agentplat.authz import PERMISSIONS, SpiceDBAuthorizer, StaticAuthorizer, bootstrap
from agentplat.config import get_settings
from agentplat.graph import RunContext, build_graph, run_turn
from agentplat.llm import make_llm
from agentplat.mcp_client import ToolGateway
from agentplat.repo import Repo
from agentplat.schema import apply_sql
from agentplat.seed import seed
from agentplat.tools_server import build_server

from .tasks import TASKS

APP_TABLES = "alert_rules, annotations, api_keys, tool_invocations, audit_log"
EVAL_WORKSPACE = "eval"
# deepseek-flash list prices, USD per 1M tokens (off-peak cache miss input / output); see README.
PRICE_IN, PRICE_OUT = 0.15, 0.60


@dataclass
class Result:
    text: str = ""
    tool_calls: list = field(default_factory=list)  # (name, ok)
    approvals: int = 0
    unexpected_approval: bool = False
    status: str = ""
    iterations: int = 0
    ms: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    error: str | None = None


class Env:
    def __init__(self, dsn, authz, workspace):
        self.dsn, self.authz, self.workspace = dsn, authz, workspace

    async def rows(self, sql, *args):
        async with await psycopg.AsyncConnection.connect(self.dsn) as conn:
            return await (await conn.execute(sql, args or None)).fetchall()

    async def scalar(self, sql, *args):
        rows = await self.rows(sql, *args)
        return rows[0][0] if rows else None

    async def execute(self, sql, *args):
        async with await psycopg.AsyncConnection.connect(self.dsn, autocommit=True) as conn:
            await conn.execute(sql, args or None)

    async def perms(self, user):
        got = await self.authz.check_many(user, self.workspace, list(PERMISSIONS))
        return {p for p, ok in got.items() if ok}

    async def reset(self):
        await seed(self.dsn, force=True)
        await self.execute(f"TRUNCATE {APP_TABLES} RESTART IDENTITY")
        if isinstance(self.authz, SpiceDBAuthorizer):
            await self.authz.reset_workspace(self.workspace)
        else:
            self.authz.rel = set()
        await bootstrap(self.authz, self.workspace)


async def run_task(task, env, graph, llm, tools, settings):
    thread = f"eval-{task.id}-{uuid.uuid4().hex[:6]}"
    ctx = RunContext(user=task.user, thread_id=thread, llm=llm, tools=tools, max_iterations=settings.max_iterations,
                     caller_token=tokens.caller_token(task.user, env.workspace, thread, settings.caller_secret))
    r = Result()
    prompt = task.prompt if isinstance(task.prompt, str) else await task.prompt(env)
    pieces = []

    async def on_event(e):
        if e["type"] == "token":
            pieces.append(e["text"])
        elif e["type"] == "tool_end":
            r.tool_calls.append((e["name"], e["ok"]))
        elif e["type"] == "usage":
            r.prompt_tokens += e["prompt_tokens"]
            r.completion_tokens += e["completion_tokens"]
        elif e["type"] == "done":
            r.iterations = e["iterations"]

    t0 = time.perf_counter()
    inp = {"messages": [{"role": "user", "content": prompt}]}
    try:
        for _ in range(4):  # at most a few approval round trips
            status, payload = await run_turn(graph, inp, ctx, on_event)
            if status != "approval":
                r.status = status
                break
            r.approvals += 1
            answer = task.approve if task.approve is not None else False
            r.unexpected_approval |= task.approve is None
            inp = Command(resume={c["id"]: answer for c in payload["calls"]})
    except Exception as exc:  # noqa: BLE001 - a crash is a failed task, not a failed eval
        r.error = f"{type(exc).__name__}: {exc}"[:300]
    r.ms = int((time.perf_counter() - t0) * 1000)
    r.text = "".join(pieces)
    return prompt, r


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeat", type=int, default=1)
    ap.add_argument("--only", default="")
    ap.add_argument("--static-authz", action="store_true", help="in-memory authorization instead of SpiceDB")
    a = ap.parse_args()
    logging.basicConfig(level=logging.WARNING)

    settings = dataclasses.replace(get_settings(), workspace=EVAL_WORKSPACE)
    dsn = settings.app_database_url
    if settings.analytics_database_url != dsn:
        raise SystemExit("The eval re-seeds analytics data, so it must run against the standalone database "
                         "(ANALYTICS_DATABASE_URL unset or equal to APP_DATABASE_URL).")
    await apply_sql(dsn, ["001_analytics.sql", "002_app.sql"])
    authz = StaticAuthorizer([], workspace=EVAL_WORKSPACE) if a.static_authz else SpiceDBAuthorizer(
        settings.spicedb_endpoint, settings.spicedb_key)
    repo = Repo(dsn, dsn)
    await repo.open()
    env = Env(dsn, authz, EVAL_WORKSPACE)
    tools = ToolGateway(build_server(repo, authz, settings))
    llm = make_llm(settings)
    graph = build_graph(InMemorySaver())

    tasks = [t for t in TASKS if not a.only or t.id in a.only.split(",")]
    records = []
    model = settings.deepseek_model if settings.llm_provider == "deepseek" else settings.llm_provider
    print(f"{len(tasks)} tasks x {a.repeat}, model {model}\n")
    for task in tasks:
        for rep in range(a.repeat):
            await env.reset()
            if task.setup:
                await task.setup(env)
            prompt, r = await run_task(task, env, graph, llm, tools, settings)
            if r.error:
                ok, detail = False, r.error
            else:
                ok, detail = await task.check(env, r)
                ok = bool(ok) and r.status in ("done", "capped") and not r.unexpected_approval
            records.append({"task": task.id, "category": task.category, "user": task.user, "repeat": rep,
                            "passed": ok, "detail": detail, "prompt": prompt, **dataclasses.asdict(r)})
            print(f"{'PASS' if ok else 'FAIL'}  {task.id:32} {r.ms:6d} ms  tools={len(r.tool_calls)}  {detail}")
            if not ok:
                print(f"      answer: {r.text[:300]!r}")
    await tools.aclose()
    await repo.close()
    if hasattr(llm, "aclose"):
        await llm.aclose()

    passed = sum(x["passed"] for x in records)
    by_cat = {}
    for x in records:
        c = by_cat.setdefault(x["category"], [0, 0])
        c[0] += x["passed"]
        c[1] += 1
    tok_in = sum(x["prompt_tokens"] for x in records)
    tok_out = sum(x["completion_tokens"] for x in records)
    summary = {
        "model": model, "tasks": len(tasks), "runs": len(records), "passed": passed,
        "completion_rate": round(passed / len(records), 4) if records else None,
        "by_category": {k: f"{v[0]}/{v[1]}" for k, v in by_cat.items()},
        "median_ms": sorted(x["ms"] for x in records)[len(records) // 2] if records else None,
        "avg_tool_calls": round(sum(len(x["tool_calls"]) for x in records) / max(1, len(records)), 2),
        "prompt_tokens": tok_in, "completion_tokens": tok_out,
        "approx_cost_usd": round(tok_in / 1e6 * PRICE_IN + tok_out / 1e6 * PRICE_OUT, 4),
    }
    print("\n" + json.dumps(summary, indent=2))
    out = pathlib.Path(__file__).parent / "results" / f"eval_{time.strftime('%Y%m%d-%H%M%S')}.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"summary": summary, "runs": records}, indent=2, default=str))
    print(f"saved {out}")


if __name__ == "__main__":
    asyncio.run(main())
