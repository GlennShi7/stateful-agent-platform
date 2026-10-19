"""The MCP server's authorization layer, end to end through a real MCP client (in-process)."""
import json

import pytest
from mcp import Client

from agentplat import tokens
from agentplat.authz import TOOL_PERMISSIONS, StaticAuthorizer
from agentplat.tools_server import CALLER_META, GATED, IDEMPOTENCY_META, build_server

from .helpers import SETTINGS


class LedgerRepo:
    """Enough of Repo for the authorization tests: audit log, alert rules, idempotency ledger."""

    def __init__(self):
        self.audits, self.rules, self.ledger = [], [], {}

    async def audit(self, user, tool, permission, allowed, reason=None):
        self.audits.append((user, tool, permission, allowed))

    async def list_alert_rules(self):
        return self.rules

    async def insert_alert_rule(self, cur, event_type, metric, op, threshold, window_minutes, created_by):
        self.rules.append({"id": len(self.rules) + 1, "event_type": event_type, "created_by": created_by})
        return len(self.rules)

    async def run_idempotent(self, key, tool, caller, write):
        if key in self.ledger:
            return {**self.ledger[key], "replayed": True}
        result = await write(None)
        if key:
            self.ledger[key] = result
        return result


def caller(user, secret=SETTINGS.caller_secret):
    return {CALLER_META: tokens.caller_token(user, "default", "t", secret)}


@pytest.fixture
def server():
    # The client is opened inside each test: MCP's anyio task groups must be entered and exited by
    # the same task, and pytest-asyncio runs fixture setup and teardown in different tasks.
    repo = LedgerRepo()
    return build_server(repo, StaticAuthorizer(), SETTINGS), repo


async def names(client, meta):
    return {t.name for t in (await client.list_tools(meta=meta)).tools}


async def test_listing_is_filtered_by_permission(server):
    srv, repo = server
    async with Client(srv) as client:
        every = await names(client, caller("alice"))
        assert every == set(TOOL_PERMISSIONS) and len(every) == 20
        carol = await names(client, caller("carol"))
        assert carol == {t for t, p in TOOL_PERMISSIONS.items() if p == "view"}
        dave = await names(client, caller("dave"))
        assert {"create_api_key", "revoke_api_key"} <= dave and "create_alert_rule" not in dave
        assert await names(client, caller("mallory")) == set()  # no relationship at all
        assert await names(client, {}) == set()  # no caller context


async def test_destructive_tools_are_annotated(server):
    srv, repo = server
    async with Client(srv) as client:
        tools = (await client.list_tools(meta=caller("alice"))).tools
        destructive = {t.name for t in tools if t.annotations and t.annotations.destructive_hint}
        assert destructive == {"delete_alert_rule", "revoke_api_key", "revoke_role", "purge_events"}
        # The model must be told the platform does the confirming, or it asks in chat instead of calling.
        for t in tools:
            gated = bool(t.annotations and t.annotations.destructive_hint)
            assert t.description.endswith(GATED) == gated, t.name
            assert "requires the user's approval" not in t.description.lower(), t.name


async def test_call_is_checked_again_at_execution(server):
    srv, repo = server
    async with Client(srv) as client:
        args = {"event_type": "api_call", "metric": "error_rate", "threshold": 0.1, "window_minutes": 10}
        # carol was never shown create_alert_rule, but calls it anyway (a buggy or malicious client)
        r = await client.call_tool("create_alert_rule", args, meta=caller("carol"))
        assert r.is_error and "permission denied" in r.content[0].text and "manage_alerts" in r.content[0].text
        assert repo.rules == []
        assert repo.audits[-1] == ("carol", "create_alert_rule", "manage_alerts", False)

        r = await client.call_tool("create_alert_rule", args, meta=caller("bob"))
        assert not r.is_error and json.loads(r.content[0].text)["rule_id"] == 1
        assert repo.rules[0]["created_by"] == "bob"  # identity from the signed token, not from arguments


async def test_forged_or_missing_caller_is_rejected(server):
    srv, repo = server
    async with Client(srv) as client:
        r = await client.call_tool("list_alert_rules", {}, meta=caller("alice", secret="wrong"))
        assert r.is_error and "unauthenticated" in r.content[0].text
        r = await client.call_tool("list_alert_rules", {})
        assert r.is_error and "unauthenticated" in r.content[0].text
        session = tokens.session_token("alice", SETTINGS.caller_secret)  # right secret, wrong audience
        r = await client.call_tool("list_alert_rules", {}, meta={CALLER_META: session})
        assert r.is_error


async def test_write_with_same_idempotency_key_runs_once(server):
    srv, repo = server
    async with Client(srv) as client:
        args = {"event_type": "api_call", "metric": "event_count", "threshold": 5, "window_minutes": 10, "op": "<"}
        meta = {**caller("bob"), IDEMPOTENCY_META: "thread-1:call-7"}
        first = json.loads((await client.call_tool("create_alert_rule", args, meta=meta)).content[0].text)
        second = json.loads((await client.call_tool("create_alert_rule", args, meta=meta)).content[0].text)
        assert len(repo.rules) == 1 and second == {**first, "replayed": True}


async def test_argument_validation_reaches_the_model_as_an_error(server):
    srv, repo = server
    async with Client(srv) as client:
        r = await client.call_tool("create_alert_rule", {"event_type": "x", "metric": "error_rate", "threshold": 5,
                                                         "window_minutes": 10}, meta=caller("bob"))
        assert "fraction" in r.content[0].text  # 5 instead of 0.05 gets a corrective message
        r = await client.call_tool("create_alert_rule", {"event_type": "x", "metric": "bogus", "threshold": 1,
                                                         "window_minutes": 10}, meta=caller("bob"))
        assert r.is_error
