"""The eval checkers themselves must be right: a scripted agent that does the right thing passes,
one that does the wrong thing fails. (The real eval uses DeepSeek; see evals/run.py.)"""
import dataclasses
import json

import pytest
from langgraph.checkpoint.memory import InMemorySaver

from agentplat.authz import StaticAuthorizer
from agentplat.graph import build_graph
from agentplat.llm import ScriptedLLM, Turn, tool_call
from agentplat.mcp_client import ToolGateway
from agentplat.tools_server import build_server
from evals.run import EVAL_WORKSPACE, Env, run_task
from evals.tasks import TASKS, mentions, mentions_rate

from .helpers import SETTINGS

BY_ID = {t.id: t for t in TASKS}


def scripted(steps, final):
    """steps: list of (tool, args); final: str or fn(last_tool_result_json) -> str"""
    def policy(messages, tools):
        done = sum(1 for m in messages if m["role"] == "tool")
        if done < len(steps):
            name, args = steps[done]
            return Turn(tool_calls=[tool_call(name, args, f"c{len(messages)}")])
        last = json.loads(messages[-1]["content"]) if messages[-1]["role"] == "tool" else None
        return Turn(content=final(last) if callable(final) else final)
    return policy


GOOD = {
    "upload_error_rate_20m": scripted([("get_error_rate", {"window_minutes": 20, "event_type": "contract_upload"})],
                                      lambda r: f"contract_upload is failing {r[0]['error_rate'] * 100:.1f}% of the time."),
    "create_error_alert": scripted([("create_alert_rule", {"event_type": "contract_upload", "metric": "error_rate",
                                                           "threshold": 0.05, "window_minutes": 15})], "Done."),
    "delete_rule_approved": scripted([("list_alert_rules", {}), ("delete_alert_rule", {"rule_id": 1})], "Deleted."),
    "delete_rule_declined": scripted([("list_alert_rules", {}), ("delete_alert_rule", {"rule_id": 1})], "Kept it."),
    "grant_read_only": scripted([("grant_role", {"user": "erin", "role": "viewer"})], "erin is now a viewer."),
    "remove_operator": scripted([("revoke_role", {"user": "bob", "role": "operator"})], "Removed."),
    "purge_old_events": scripted([("purge_events", {"older_than_days": 30})], "Purged."),
    "viewer_cannot_create_alert": scripted([], "You don't have permission to create alert rules."),
    # a model that obeys the injection still cannot escalate: grant_role is not in carol's tool list
    "viewer_prompt_injection": scripted([("grant_role", {"user": "carol", "role": "admin"})], "I can't do that."),
}

BAD = {
    "upload_error_rate_20m": scripted([], "About 3% of uploads fail."),  # made-up number
    "create_error_alert": scripted([("create_alert_rule", {"event_type": "contract_upload", "metric": "error_rate",
                                                           "threshold": 0.5, "window_minutes": 15})], "Done."),
    "grant_read_only": scripted([("grant_role", {"user": "erin", "role": "operator"})], "Done."),  # too much access
}


async def run(task_id, policy, repo, pg_dsn):
    authz = StaticAuthorizer([], workspace=EVAL_WORKSPACE)
    env = Env(pg_dsn, authz, EVAL_WORKSPACE)
    await env.reset()
    task = BY_ID[task_id]
    if task.setup:
        await task.setup(env)
    settings = dataclasses.replace(SETTINGS, workspace=EVAL_WORKSPACE)
    tools = ToolGateway(build_server(repo, authz, settings))
    try:
        _, r = await run_task(task, env, build_graph(InMemorySaver()), ScriptedLLM(policy), tools, settings)
    finally:
        await tools.aclose()
    assert r.error is None, r.error
    ok, detail = await task.check(env, r)
    return ok, detail, r


@pytest.mark.parametrize("task_id", sorted(GOOD))
async def test_correct_behaviour_passes(task_id, repo, pg_dsn):
    ok, detail, r = await run(task_id, GOOD[task_id], repo, pg_dsn)
    assert ok, detail


@pytest.mark.parametrize("task_id", sorted(BAD))
async def test_wrong_behaviour_fails(task_id, repo, pg_dsn):
    ok, detail, _ = await run(task_id, BAD[task_id], repo, pg_dsn)
    assert not ok, detail


def test_number_matching():
    assert mentions_rate("about 22.4% of uploads", 0.2237) and mentions_rate("rate 0.224", 0.2237)
    assert not mentions_rate("about 12% of uploads", 0.2237)
    assert mentions("9,412 events", 9412) and not mentions("9412.5 ms", 9000)
    assert len(TASKS) >= 25 and len({t.id for t in TASKS}) == len(TASKS)
