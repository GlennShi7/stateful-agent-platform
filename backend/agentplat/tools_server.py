"""MCP server exposing 20 analytics-admin tools, with SpiceDB authorization on every request.

Authorization is enforced in one middleware, at two points:
  * tools/list - the caller only sees tools their permissions allow, so the LLM is never even
    offered a tool it cannot use;
  * tools/call - checked again at execution time. The listing is a convenience; this check is the
    security boundary (a client could call a tool it was never shown, and permissions can change
    between listing and calling).

The caller's identity comes from a signed token in the request `_meta`, minted by the gateway for
the authenticated browser session. Tool arguments written by the LLM never carry identity.

Run: python -m agentplat.tools_server   (streamable HTTP on :8001/mcp)
"""
import json
import logging
import os
from contextlib import asynccontextmanager
from typing import Annotated, Literal

import mcp_types
from mcp.server.mcpserver import Context, MCPServer
from mcp.server.transport_security import TransportSecuritySettings
from pydantic import Field
from starlette.responses import JSONResponse

from . import tokens
from .authz import PERMISSIONS, ROLES, TOOL_PERMISSIONS, check

log = logging.getLogger(__name__)

CALLER_META = "agentplat/caller"
IDEMPOTENCY_META = "agentplat/idempotency_key"

Window = Annotated[int, Field(ge=1, le=10080, description="Look-back window in minutes (max 7 days)")]
EventType = Annotated[str, Field(min_length=1, max_length=40, description="Event type, e.g. contract_upload")]

DESTRUCTIVE = mcp_types.ToolAnnotations(destructive_hint=True, read_only_hint=False)
WRITE = mcp_types.ToolAnnotations(destructive_hint=False, read_only_hint=False)
READ = mcp_types.ToolAnnotations(read_only_hint=True)
# Appended to every destructive tool's description. Without it the model read "requires approval" as
# "ask the user first" and asked in chat instead of calling the tool (eval task purge_old_events).
GATED = " Calling it pauses for the user's approval; it runs only if they approve."


def _denied(text):
    return mcp_types.CallToolResult(content=[mcp_types.TextContent(type="text", text=text)], is_error=True)


class AuthzMiddleware:
    def __init__(self, authz, repo, settings):
        self.authz, self.repo, self.settings = authz, repo, settings

    def _claims(self, meta):
        return tokens.verify((meta or {}).get(CALLER_META), self.settings.caller_secret, aud=tokens.CALLER_AUD)

    async def _audit(self, *args):
        try:
            await self.repo.audit(*args)
        except Exception:  # noqa: BLE001 - auditing must not take the tool down
            log.exception("audit write failed")

    async def __call__(self, ctx, call_next):
        if ctx.method == "tools/list":
            try:
                claims = self._claims(ctx.meta)
            except tokens.TokenError:
                result = await call_next(ctx)
                return {**result, "tools": []} if isinstance(result, dict) else result
            granted = await self.authz.check_many(claims["sub"], claims["ws"], list(PERMISSIONS))
            result = await call_next(ctx)
            if isinstance(result, dict):
                result = {**result, "tools": [t for t in result.get("tools", [])
                                              if granted.get(TOOL_PERMISSIONS.get(t["name"]), False)]}
            return result

        if ctx.method == "tools/call":
            name = (ctx.params or {}).get("name")
            try:
                claims = self._claims(ctx.meta)
            except tokens.TokenError as exc:
                await self._audit(None, name or "?", None, False, f"caller token: {exc}")
                return _denied(f"unauthenticated caller ({exc})")
            permission = TOOL_PERMISSIONS.get(name)
            if permission is None:  # deny by default: a tool without a mapping is never callable
                await self._audit(claims["sub"], name or "?", None, False, "unmapped tool")
                return _denied(f"unknown tool {name!r}")
            if not await check(self.authz, claims["sub"], claims["ws"], permission):
                await self._audit(claims["sub"], name, permission, False, "permission denied")
                return _denied(f"permission denied: {claims['sub']} lacks '{permission}' on workspace "
                               f"{claims['ws']}, which {name} requires")
            await self._audit(claims["sub"], name, permission, True, None)
        return await call_next(ctx)


def build_server(repo, authz, settings, *, lifespan=None) -> MCPServer:
    srv = MCPServer(
        "analytics-admin",
        instructions="Admin tools for the real-time analytics platform. Times are UTC.",
        middleware=[AuthzMiddleware(authz, repo, settings)],
        lifespan=lifespan,
    )

    def meta_of(ctx: Context) -> dict:
        rc = ctx.request_context
        return (rc.meta or {}) if rc is not None else {}

    def caller(ctx: Context) -> dict:
        # The middleware already rejected bad tokens; verifying again keeps each tool safe on its own.
        return tokens.verify(meta_of(ctx).get(CALLER_META), settings.caller_secret, aud=tokens.CALLER_AUD)

    def idem(ctx: Context):
        return meta_of(ctx).get(IDEMPOTENCY_META)

    def out(obj) -> str:
        return json.dumps(obj, default=str)

    # ------------------------------------------------------------------ read-only analytics
    @srv.tool(annotations=READ)
    async def list_event_types(window_minutes: Window = 60) -> str:
        """List event types seen in the window, with event and error counts, busiest first."""
        return out(await repo.event_types(window_minutes))

    @srv.tool(annotations=READ)
    async def get_metrics(event_type: EventType,
                          minutes: Annotated[int, Field(ge=1, le=240)] = 30) -> str:
        """Per-minute rollup for one event type: events, average latency and errors per minute."""
        return out(await repo.metrics(event_type, minutes))

    @srv.tool(annotations=READ)
    async def get_error_rate(window_minutes: Window = 60, event_type: EventType | None = None) -> str:
        """Error rate (errors / events, a fraction between 0 and 1) per event type over the window.
        Omit event_type to get every type."""
        return out(await repo.error_rate(event_type, window_minutes))

    @srv.tool(annotations=READ)
    async def get_latency_percentiles(event_type: EventType, window_minutes: Window = 60) -> str:
        """p50 / p95 / p99 latency in milliseconds for one event type over the window."""
        return out(await repo.latency_percentiles(event_type, window_minutes))

    @srv.tool(annotations=READ)
    async def top_users(window_minutes: Window = 60, event_type: EventType | None = None,
                        limit: Annotated[int, Field(ge=1, le=50)] = 5) -> str:
        """Users with the most events in the window (optionally of one type), with their error counts."""
        return out(await repo.top_users(event_type, window_minutes, limit))

    @srv.tool(annotations=READ)
    async def compare_windows(event_type: EventType, window_minutes: Window = 30) -> str:
        """Compare the last `window_minutes` with the window of the same length just before it:
        events, errors, error rate and p95 latency for each."""
        return out(await repo.compare_windows(event_type, window_minutes))

    @srv.tool(annotations=READ)
    async def get_user_activity(user_id: Annotated[str, Field(min_length=1, max_length=64)],
                                window_minutes: Window = 60) -> str:
        """One user's events and errors by event type over the window."""
        return out(await repo.user_activity(user_id, window_minutes))

    @srv.tool(annotations=READ)
    async def list_alert_rules() -> str:
        """All configured alert rules."""
        return out(await repo.list_alert_rules())

    @srv.tool(annotations=READ)
    async def evaluate_alert_rule(rule_id: int) -> str:
        """Compute an alert rule's metric right now and report whether it is firing."""
        rule = await repo.get_alert_rule(rule_id)
        if rule is None:
            return out({"error": f"no alert rule with id {rule_id}"})
        value = await repo.metric_value(rule["event_type"], rule["metric"], rule["window_minutes"])
        firing = value is not None and (value > rule["threshold"] if rule["op"] == ">" else value < rule["threshold"])
        return out({"rule_id": rule_id, "event_type": rule["event_type"], "metric": rule["metric"],
                    "op": rule["op"], "threshold": rule["threshold"], "window_minutes": rule["window_minutes"],
                    "current_value": value, "firing": firing})

    @srv.tool(annotations=READ)
    async def list_annotations(window_hours: Annotated[int, Field(ge=1, le=720)] = 24) -> str:
        """Timeline annotations (deploys, incidents, notes) from the last `window_hours` hours."""
        return out(await repo.list_annotations(window_hours))

    # ------------------------------------------------------------------ operator writes
    @srv.tool(annotations=WRITE)
    async def add_annotation(note: Annotated[str, Field(min_length=1, max_length=500)], ctx: Context,
                             event_type: EventType | None = None) -> str:
        """Add a note to the metrics timeline at the current time, e.g. 'deployed v2.3'."""
        user = caller(ctx)["sub"]

        async def write(cur):
            r = await repo.insert_annotation(cur, note, event_type, user)
            return {"annotation_id": r["id"], "ts": r["ts"], "note": note, "event_type": event_type}
        return out(await repo.run_idempotent(idem(ctx), "add_annotation", user, write))

    @srv.tool(annotations=WRITE)
    async def create_alert_rule(event_type: EventType,
                                metric: Literal["error_rate", "p95_latency_ms", "event_count"],
                                threshold: float,
                                window_minutes: Annotated[int, Field(ge=1, le=1440)],
                                ctx: Context,
                                op: Literal[">", "<"] = ">") -> str:
        """Create an alert rule that fires when `metric` over the last `window_minutes` crosses
        `threshold`. error_rate is a fraction: 5% is 0.05. p95_latency_ms is in milliseconds."""
        if metric == "error_rate" and not 0 <= threshold <= 1:
            return out({"error": "error_rate threshold must be a fraction between 0 and 1 (5% is 0.05)"})
        user = caller(ctx)["sub"]

        async def write(cur):
            rid = await repo.insert_alert_rule(cur, event_type, metric, op, threshold, window_minutes, user)
            return {"rule_id": rid, "event_type": event_type, "metric": metric, "op": op,
                    "threshold": threshold, "window_minutes": window_minutes}
        return out(await repo.run_idempotent(idem(ctx), "create_alert_rule", user, write))

    @srv.tool(annotations=DESTRUCTIVE, description="Delete an alert rule." + GATED)
    async def delete_alert_rule(rule_id: int, ctx: Context) -> str:
        user = caller(ctx)["sub"]

        async def write(cur):
            r = await repo.delete_alert_rule(cur, rule_id)
            return {"deleted": r is not None, "rule_id": rule_id,
                    **({"event_type": r["event_type"], "metric": r["metric"]} if r else {})}
        return out(await repo.run_idempotent(idem(ctx), "delete_alert_rule", user, write))

    # ------------------------------------------------------------------ key management
    @srv.tool(annotations=READ)
    async def list_api_keys() -> str:
        """List API keys (name, prefix, creator, revoked or not). Never shows the secret."""
        return out(await repo.list_api_keys())

    @srv.tool(annotations=WRITE)
    async def create_api_key(name: Annotated[str, Field(min_length=1, max_length=60)], ctx: Context) -> str:
        """Create an ingestion API key. The plaintext key is returned once and never stored."""
        user = caller(ctx)["sub"]

        async def write(cur):
            kid, plaintext = await repo.insert_api_key(cur, name, user)
            return {"key_id": kid, "name": name, "api_key": plaintext}
        return out(await repo.run_idempotent(idem(ctx), "create_api_key", user, write))

    @srv.tool(annotations=DESTRUCTIVE, description="Revoke an API key by id or by name." + GATED)
    async def revoke_api_key(ctx: Context, key_id: int | None = None,
                             name: Annotated[str, Field(max_length=60)] | None = None) -> str:
        user = caller(ctx)["sub"]
        if key_id is None:
            if not name:
                return out({"error": "give key_id or name"})
            found = await repo.find_api_key(name)
            if found is None:
                return out({"error": f"no API key named {name!r}"})
            key_id = found["id"]

        async def write(cur):
            r = await repo.revoke_api_key(cur, key_id)
            if r is None:
                return {"error": f"no API key with id {key_id}"}
            return {"revoked": True, "key_id": r["id"], "name": r["name"], "revoked_at": r["revoked_at"].isoformat()}
        return out(await repo.run_idempotent(idem(ctx), "revoke_api_key", user, write))

    # ------------------------------------------------------------------ access control
    @srv.tool(annotations=READ)
    async def list_members() -> str:
        """Who has which role on this workspace."""
        return out([{"user": u, "role": r} for u, r in await authz.members(settings.workspace)])

    Role = Annotated[Literal[ROLES], Field(description="admin, operator, viewer or key_manager")]
    Username = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_-]{0,31}$")]

    @srv.tool(annotations=WRITE)
    async def grant_role(user: Username, role: Role, ctx: Context) -> str:
        """Give a user a role on this workspace."""
        me = caller(ctx)["sub"]

        async def op():
            await authz.grant(settings.workspace, user, role)
            return {"granted": True, "user": user, "role": role}
        return out(await repo.run_idempotent_external(idem(ctx), "grant_role", me, op))

    @srv.tool(annotations=DESTRUCTIVE, description="Remove a role from a user." + GATED)
    async def revoke_role(user: Username, role: Role, ctx: Context) -> str:
        me = caller(ctx)["sub"]
        if user == me and role == "admin":
            return out({"error": "refusing to remove your own admin role"})

        async def op():
            await authz.revoke(settings.workspace, user, role)
            return {"revoked": True, "user": user, "role": role}
        return out(await repo.run_idempotent_external(idem(ctx), "revoke_role", me, op))

    # ------------------------------------------------------------------ data lifecycle
    @srv.tool(annotations=DESTRUCTIVE,
              description="Permanently delete raw events and rollup rows older than N days." + GATED)
    async def purge_events(older_than_days: Annotated[int, Field(ge=1, le=3650)], ctx: Context) -> str:
        me = caller(ctx)["sub"]

        async def op():
            return await repo.purge_events(older_than_days)
        return out(await repo.run_idempotent_external(idem(ctx), "purge_events", me, op))

    @srv.custom_route("/healthz", methods=["GET"])
    async def healthz(request):
        return JSONResponse({"ok": True, "tools": len(TOOL_PERMISSIONS)})

    return srv


def main():  # pragma: no cover - exercised by docker compose
    import uvicorn

    from .authz import SpiceDBAuthorizer, bootstrap
    from .config import get_settings
    from .repo import Repo

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
    for noisy in ("httpx", "httpx2", "mcp"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
    settings = get_settings()
    repo = Repo(settings.app_database_url, settings.analytics_database_url)
    authz = SpiceDBAuthorizer(settings.spicedb_endpoint, settings.spicedb_key)

    @asynccontextmanager
    async def lifespan(_srv):
        from .schema import apply_sql

        await apply_sql(settings.app_database_url, ["002_app.sql"])
        await repo.open()
        await bootstrap(authz, settings.workspace)
        log.info("MCP server ready with %d tools", len(TOOL_PERMISSIONS))
        yield {}
        await repo.close()

    srv = build_server(repo, authz, settings, lifespan=lifespan)
    hosts = os.environ.get("MCP_ALLOWED_HOSTS", "mcp-server:*,localhost:*,127.0.0.1:*").split(",")
    app = srv.streamable_http_app(transport_security=TransportSecuritySettings(
        enable_dns_rebinding_protection=True, allowed_hosts=hosts, allowed_origins=[]))
    uvicorn.run(app, host="0.0.0.0", port=int(os.environ.get("MCP_PORT", "8001")), log_level="info")


if __name__ == "__main__":
    main()
