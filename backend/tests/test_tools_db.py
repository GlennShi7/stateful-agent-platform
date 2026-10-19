"""Every tool against real Postgres (seeded demo data), through an in-process MCP client."""
import json

import psycopg
from mcp import Client

from agentplat import tokens
from agentplat.authz import StaticAuthorizer
from agentplat.tools_server import CALLER_META, IDEMPOTENCY_META, build_server

from .helpers import SETTINGS


def meta(user, key=None):
    m = {CALLER_META: tokens.caller_token(user, "default", "t", SETTINGS.caller_secret)}
    if key:
        m[IDEMPOTENCY_META] = key
    return m


async def call(client, name, args, user="alice", key=None):
    r = await client.call_tool(name, args, meta=meta(user, key))
    assert not r.is_error, r.content[0].text
    return json.loads(r.content[0].text)


async def test_read_tools_find_the_planted_incidents(repo):
    authz = StaticAuthorizer()
    async with Client(build_server(repo, authz, SETTINGS)) as c:
        types = await call(c, "list_event_types", {"window_minutes": 60}, "carol")
        assert {t["event_type"] for t in types} >= {"page_view", "contract_upload", "api_call"}

        rates = {r["event_type"]: r["error_rate"] for r in await call(c, "get_error_rate", {"window_minutes": 20}, "carol")}
        assert rates["contract_upload"] > 0.15 and rates["page_view"] < 0.02
        one = await call(c, "get_error_rate", {"window_minutes": 20, "event_type": "contract_upload"}, "carol")
        assert len(one) == 1

        cmp = await call(c, "compare_windows", {"event_type": "contract_upload", "window_minutes": 20}, "carol")
        assert cmp["current"]["error_rate"] > 5 * cmp["previous"]["error_rate"]

        lat_now = await call(c, "get_latency_percentiles", {"event_type": "api_call", "window_minutes": 30}, "carol")
        cmp_lat = await call(c, "compare_windows", {"event_type": "api_call", "window_minutes": 30}, "carol")
        assert lat_now["p95_ms"] > 2 * cmp_lat["previous"]["p95_latency_ms"]

        top = await call(c, "top_users", {"window_minutes": 360, "limit": 3}, "carol")
        assert len(top) == 3 and top[0]["events"] >= top[1]["events"] >= top[2]["events"]
        act = await call(c, "get_user_activity", {"user_id": top[0]["user_id"], "window_minutes": 360}, "carol")
        assert act["total_events"] == top[0]["events"]

        minutes = await call(c, "get_metrics", {"event_type": "page_view", "minutes": 10}, "carol")
        assert 9 <= len(minutes) <= 11 and all(m["events"] > 0 for m in minutes)


async def test_alert_rule_lifecycle_and_evaluation(repo):
    async with Client(build_server(repo, StaticAuthorizer(), SETTINGS)) as c:
        rule = await call(c, "create_alert_rule", {"event_type": "contract_upload", "metric": "error_rate",
                                                   "threshold": 0.05, "window_minutes": 15}, "bob")
        quiet = await call(c, "create_alert_rule", {"event_type": "page_view", "metric": "error_rate",
                                                    "threshold": 0.5, "window_minutes": 15}, "bob")
        assert (await call(c, "evaluate_alert_rule", {"rule_id": rule["rule_id"]}, "carol"))["firing"] is True
        assert (await call(c, "evaluate_alert_rule", {"rule_id": quiet["rule_id"]}, "carol"))["firing"] is False
        assert len(await call(c, "list_alert_rules", {}, "carol")) == 2
        assert (await call(c, "delete_alert_rule", {"rule_id": quiet["rule_id"]}, "bob"))["deleted"] is True
        assert [r["id"] for r in await call(c, "list_alert_rules", {}, "carol")] == [rule["rule_id"]]


async def test_idempotent_write_in_the_database(repo, pg_dsn):
    async with Client(build_server(repo, StaticAuthorizer(), SETTINGS)) as c:
        args = {"event_type": "api_call", "metric": "p95_latency_ms", "threshold": 800, "window_minutes": 10}
        a = await call(c, "create_alert_rule", args, "bob", key="th:call1")
        b = await call(c, "create_alert_rule", args, "bob", key="th:call1")
        assert b["replayed"] and b["rule_id"] == a["rule_id"]
        assert len(await call(c, "list_alert_rules", {}, "bob")) == 1


async def test_api_key_plaintext_is_returned_once_and_never_stored(repo, pg_dsn):
    async with Client(build_server(repo, StaticAuthorizer(), SETTINGS)) as c:
        created = await call(c, "create_api_key", {"name": "ci-pipeline"}, "dave", key="th:k1")
        assert created["api_key"].startswith("ak_")
        replay = await call(c, "create_api_key", {"name": "ci-pipeline"}, "dave", key="th:k1")
        assert replay["replayed"] and replay["api_key"] is None
        with psycopg.connect(pg_dsn) as conn:
            dump = json.dumps(conn.execute("SELECT * FROM api_keys").fetchall(), default=str)
            dump += json.dumps(conn.execute("SELECT result FROM tool_invocations").fetchall(), default=str)
        assert created["api_key"] not in dump
        listed = await call(c, "list_api_keys", {}, "dave")
        assert listed[0]["name"] == "ci-pipeline" and "key_hash" not in listed[0]
        revoked = await call(c, "revoke_api_key", {"name": "ci-pipeline"}, "dave")
        assert revoked["revoked"] and (await call(c, "list_api_keys", {}, "dave"))[0]["revoked_at"]


async def test_access_control_tools_change_what_a_user_can_do(repo):
    authz = StaticAuthorizer()
    async with Client(build_server(repo, authz, SETTINGS)) as c:
        assert {"user": "carol", "role": "viewer"} in await call(c, "list_members", {})
        assert not (await c.list_tools(meta=meta("erin"))).tools
        await call(c, "grant_role", {"user": "erin", "role": "operator"}, key="th:g1")
        erin_tools = {t.name for t in (await c.list_tools(meta=meta("erin"))).tools}
        assert "create_alert_rule" in erin_tools and "grant_role" not in erin_tools
        await call(c, "revoke_role", {"user": "erin", "role": "operator"})
        assert not (await c.list_tools(meta=meta("erin"))).tools
        refused = await call(c, "revoke_role", {"user": "alice", "role": "admin"})
        assert "refusing" in refused["error"]


async def test_purge_only_removes_old_rows(repo, pg_dsn):
    with psycopg.connect(pg_dsn) as conn:
        before = conn.execute("SELECT count(*) FROM events").fetchone()[0]
        old = conn.execute("SELECT count(*) FROM events WHERE ts < now() - interval '30 days'").fetchone()[0]
    assert old > 0
    async with Client(build_server(repo, StaticAuthorizer(), SETTINGS)) as c:
        r = await call(c, "purge_events", {"older_than_days": 30})
    assert r["events_deleted"] == old
    with psycopg.connect(pg_dsn) as conn:
        assert conn.execute("SELECT count(*) FROM events").fetchone()[0] == before - old
    # put the old rows back for other tests
    from agentplat.seed import seed
    await seed(pg_dsn)


async def test_audit_log_records_allow_and_deny(repo, pg_dsn):
    async with Client(build_server(repo, StaticAuthorizer(), SETTINGS)) as c:
        await c.call_tool("purge_events", {"older_than_days": 400}, meta=meta("carol"))
        await call(c, "list_alert_rules", {}, "carol")
    with psycopg.connect(pg_dsn) as conn:
        rows = conn.execute("SELECT user_id, tool, permission, allowed FROM audit_log ORDER BY id").fetchall()
    assert rows == [("carol", "purge_events", "purge_data", False), ("carol", "list_alert_rules", "view", True)]
