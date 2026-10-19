"""Test doubles shared by the unit tests."""
import json

from agentplat.config import Settings

SETTINGS = Settings(caller_secret="test-caller", session_secret="test-session", max_iterations=10)


def spec(name, destructive=False):
    return {"type": "function", "function": {"name": name, "description": name,
                                             "parameters": {"type": "object", "properties": {}}}}


class FakeTools:
    """Stands in for ToolGateway: a fixed tool list and canned results, with an idempotency ledger
    so tests can check that a replayed write is not executed twice."""

    def __init__(self, names=("get_error_rate", "create_alert_rule", "delete_alert_rule"),
                 destructive=("delete_alert_rule",), results=None):
        self.names, self.destructive_names = list(names), list(destructive)
        self.results = results or {}
        self.executed = []        # (name, args) actually executed
        self.ledger = {}
        self.listed_with = []

    async def list_for(self, caller_token):
        self.listed_with.append(caller_token)
        return [spec(n) for n in self.names], list(self.destructive_names)

    async def call(self, name, arguments, caller_token, idempotency_key):
        if idempotency_key in self.ledger:
            return True, json.dumps({**self.ledger[idempotency_key], "replayed": True})
        self.executed.append((name, arguments))
        result = self.results.get(name, {"ok": True, "tool": name})
        if idempotency_key:
            self.ledger[idempotency_key] = result
        return True, json.dumps(result)


class FakeRepo:
    """Only what the gateway needs: thread ownership."""

    def __init__(self):
        self.threads = {}

    async def claim_thread(self, thread_id, user, title=""):
        return self.threads.setdefault(thread_id, user)

    async def set_thread_title(self, thread_id, title):
        pass

    async def list_threads(self, user, limit=50):
        return [{"thread_id": t, "title": "", "created_at": ""} for t, u in self.threads.items() if u == user]
