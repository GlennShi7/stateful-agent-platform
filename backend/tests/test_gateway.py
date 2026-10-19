import pytest
from langgraph.checkpoint.memory import InMemorySaver
from starlette.testclient import TestClient
from starlette.websockets import WebSocketDisconnect

from agentplat.gateway import Services, create_app
from agentplat.graph import build_graph
from agentplat.llm import ScriptedLLM, Turn, tool_call

from .helpers import SETTINGS, FakeRepo, FakeTools


def echo_policy(messages, tools):
    last = messages[-1]
    if last["role"] == "user" and last["content"].startswith("delete"):
        return Turn(tool_calls=[tool_call("delete_alert_rule", {"rule_id": 1}, f"d{len(messages)}")])
    if last["role"] == "tool":
        return Turn(content="finished")
    return Turn(content=f"answer to {last['content']}")


@pytest.fixture
def client():
    svc = Services(SETTINGS, FakeRepo(), build_graph(InMemorySaver()), ScriptedLLM(echo_policy, token_delay_ms=1),
                   FakeTools())
    with TestClient(create_app(svc)) as c:
        yield c


def login(c, user):
    r = c.post("/api/login", json={"user": user})
    assert r.status_code == 200
    return r.json()["token"]


def until(ws, pred, limit=500):
    frames = []
    for _ in range(limit):
        f = ws.receive_json()
        frames.append(f)
        if pred(f):
            return frames
    raise AssertionError("condition not reached")


def test_login_only_for_demo_users(client):
    assert client.post("/api/login", json={"user": "mallory"}).status_code == 403
    assert login(client, "carol")


def test_websocket_requires_a_valid_session_token(client):
    with pytest.raises(WebSocketDisconnect) as exc:
        with client.websocket_connect("/ws?token=forged") as ws:
            ws.receive_json()
    assert exc.value.code == 4401


def test_two_sessions_stream_concurrently_over_one_socket(client):
    tok = login(client, "bob")
    with client.websocket_connect(f"/ws?token={tok}") as ws:
        ws.send_json({"op": "send", "session": "s1", "text": "alpha beta gamma"})
        ws.send_json({"op": "send", "session": "s2", "text": "one two three"})
        done = set()
        frames = until(ws, lambda f: f.get("type") == "done" and (done.add(f["session"]) or len(done) == 2))
    text = {s: "".join(f["text"] for f in frames if f.get("type") == "token" and f["session"] == s)
            for s in ("s1", "s2")}
    assert text == {"s1": "answer to alpha beta gamma", "s2": "answer to one two three"}
    assert all("t" in f for f in frames)


def test_sessions_belong_to_their_creator(client):
    with client.websocket_connect(f"/ws?token={login(client, 'bob')}") as ws:
        ws.send_json({"op": "open", "session": "private"})
        assert ws.receive_json()["type"] == "state"
    with client.websocket_connect(f"/ws?token={login(client, 'carol')}") as ws:
        ws.send_json({"op": "open", "session": "private"})
        f = ws.receive_json()
        assert f["type"] == "error" and "another user" in f["message"]


def test_approval_round_trip(client):
    with client.websocket_connect(f"/ws?token={login(client, 'bob')}") as ws:
        ws.send_json({"op": "send", "session": "a1", "text": "delete rule 1"})
        frames = until(ws, lambda f: f.get("type") == "approval_required")
        call_id = frames[-1]["calls"][0]["id"]
        ws.send_json({"op": "send", "session": "a1", "text": "something else"})
        assert "pending approval" in until(ws, lambda f: f.get("type") == "error")[-1]["message"]
        ws.send_json({"op": "open", "session": "a1"})
        state = until(ws, lambda f: f.get("type") == "state")[-1]
        assert state["approval"]["calls"][0]["id"] == call_id and not state["resumable"]
        ws.send_json({"op": "approve", "session": "a1", "decisions": {call_id: True}})
        frames = until(ws, lambda f: f.get("type") == "done")
        assert any(f.get("type") == "tool_end" and f["ok"] for f in frames)
        ws.send_json({"op": "open", "session": "a1"})
        state = until(ws, lambda f: f.get("type") == "state")[-1]
        assert state["status"] == "done" and state["messages"][-1] == {"role": "assistant", "text": "finished"}


def test_ping_pong_and_bad_input(client):
    with client.websocket_connect(f"/ws?token={login(client, 'bob')}") as ws:
        ws.send_json({"op": "ping", "client_t": 123})
        assert ws.receive_json()["client_t"] == 123
        ws.send_text("not json")
        assert ws.receive_json()["type"] == "error"
        ws.send_json({"op": "send", "session": "../etc", "text": "x"})
        assert ws.receive_json()["message"] == "bad session id"


def test_second_connection_picks_up_a_running_session(client):
    tok = login(client, "bob")
    with client.websocket_connect(f"/ws?token={tok}") as a, client.websocket_connect(f"/ws?token={tok}") as b:
        a.send_json({"op": "send", "session": "shared", "text": " ".join(f"w{i}" for i in range(300))})
        until(a, lambda f: f.get("type") == "token")
        b.send_json({"op": "open", "session": "shared"})  # e.g. the same user reconnecting
        frames_b = until(b, lambda f: f.get("type") == "done")
        until(a, lambda f: f.get("type") == "done")
    assert frames_b[0]["type"] == "state" and frames_b[0]["running"] is True
    assert any(f.get("type") == "token" for f in frames_b)
