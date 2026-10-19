import json

import httpx

from agentplat.telemetry import Telemetry


async def test_events_match_the_dashboard_schema_and_failures_are_swallowed():
    sent = []

    def handler(request):
        sent.append(json.loads(request.content))
        return httpx.Response(202)

    t = Telemetry("http://dash/events", transport=httpx.MockTransport(handler))
    t.emit("agent_run", "bob", 1234, False)
    t.emit("agent_tool_call", "bob", 10_000_000, True)  # clamped to the dashboard's 600000 ms limit
    await t.aclose()
    assert [e["type"] for e in sent] == ["agent_run", "agent_tool_call"]
    assert sent[1]["latency_ms"] == 600_000 and sent[1]["is_error"] is True
    assert set(sent[0]) == {"event_id", "type", "user_id", "latency_ms", "is_error"}

    def down(request):
        raise httpx.ConnectError("dashboard down")

    t = Telemetry("http://dash/events", transport=httpx.MockTransport(down))
    t.emit("agent_run", "bob", 5, False)
    await t.aclose()  # no exception
