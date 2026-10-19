"""Authorization: tool -> permission mapping, a SpiceDB-backed authorizer, and an in-memory twin.

Least privilege here means each tool needs exactly one permission, the narrowest that covers it,
and an unmapped tool is denied. `StaticAuthorizer` implements the same rules as schema.zed for
unit tests; tests/test_integration_spicedb.py checks that the two agree for every user and permission.
"""
import asyncio
import logging
import pathlib
from typing import Protocol

log = logging.getLogger(__name__)

SCHEMA = (pathlib.Path(__file__).parent / "schema.zed").read_text()

ROLES = ("admin", "operator", "viewer", "key_manager")

# Mirrors the permission expressions in schema.zed.
PERMISSION_RELATIONS = {
    "view": {"viewer", "operator", "admin"},
    "annotate": {"operator", "admin"},
    "manage_alerts": {"operator", "admin"},
    "manage_keys": {"key_manager", "admin"},
    "manage_access": {"admin"},
    "purge_data": {"admin"},
}
PERMISSIONS = tuple(PERMISSION_RELATIONS)

TOOL_PERMISSIONS = {
    # analytics, read-only
    "list_event_types": "view",
    "get_metrics": "view",
    "get_error_rate": "view",
    "get_latency_percentiles": "view",
    "top_users": "view",
    "compare_windows": "view",
    "get_user_activity": "view",
    "list_alert_rules": "view",
    "evaluate_alert_rule": "view",
    "list_annotations": "view",
    # operator
    "add_annotation": "annotate",
    "create_alert_rule": "manage_alerts",
    "delete_alert_rule": "manage_alerts",
    # key management
    "list_api_keys": "manage_keys",
    "create_api_key": "manage_keys",
    "revoke_api_key": "manage_keys",
    # access control
    "list_members": "manage_access",
    "grant_role": "manage_access",
    "revoke_role": "manage_access",
    # data lifecycle
    "purge_events": "purge_data",
}

# Default role assignments for the demo workspace.
DEMO_RELATIONSHIPS = [
    ("alice", "admin"),
    ("bob", "operator"),
    ("carol", "viewer"),
    ("dave", "viewer"),
    ("dave", "key_manager"),
]


class Authorizer(Protocol):
    async def check_many(self, user: str, workspace: str, permissions: list[str]) -> dict[str, bool]: ...
    async def grant(self, workspace: str, user: str, role: str) -> None: ...
    async def revoke(self, workspace: str, user: str, role: str) -> None: ...
    async def members(self, workspace: str) -> list[tuple[str, str]]: ...


async def check(authz: Authorizer, user: str, workspace: str, permission: str) -> bool:
    return (await authz.check_many(user, workspace, [permission]))[permission]


class StaticAuthorizer:
    """In-memory implementation of schema.zed, for tests and for running without SpiceDB."""

    def __init__(self, relationships=None, workspace="default"):
        self.rel = {(workspace, u, r) for u, r in (relationships or DEMO_RELATIONSHIPS)}

    async def check_many(self, user, workspace, permissions):
        roles = {r for (ws, u, r) in self.rel if ws == workspace and u == user}
        return {p: bool(roles & PERMISSION_RELATIONS.get(p, set())) for p in permissions}

    async def grant(self, workspace, user, role):
        self.rel.add((workspace, user, role))

    async def revoke(self, workspace, user, role):
        self.rel.discard((workspace, user, role))

    async def members(self, workspace):
        return sorted((u, r) for (ws, u, r) in self.rel if ws == workspace)


class SpiceDBAuthorizer:
    def __init__(self, endpoint: str, key: str):
        self._endpoint, self._key, self._grpc = endpoint, key, None
        # ZedToken of our latest write. Checks ask for data at least that fresh, so a role granted
        # by one tool call is visible to the next check without paying for full consistency always.
        self._last_write = None

    @property
    def _client(self):
        # Created on first use: a grpc.aio channel binds to the event loop that is running when it is
        # made, and this object is constructed before the server's loop starts.
        if self._grpc is None:
            from authzed.api.v1 import AsyncClient
            from grpcutil import insecure_bearer_token_credentials

            self._grpc = AsyncClient(self._endpoint, insecure_bearer_token_credentials(self._key))
        return self._grpc

    def _consistency(self):
        from authzed.api.v1 import Consistency

        if self._last_write is not None:
            return Consistency(at_least_as_fresh=self._last_write)
        return Consistency(minimize_latency=True)

    @staticmethod
    def _refs(workspace, user):
        from authzed.api.v1 import ObjectReference, SubjectReference

        return (ObjectReference(object_type="workspace", object_id=workspace),
                SubjectReference(object=ObjectReference(object_type="user", object_id=user)))

    async def check_many(self, user, workspace, permissions):
        from authzed.api.v1 import CheckBulkPermissionsRequest, CheckBulkPermissionsRequestItem, CheckPermissionResponse

        resource, subject = self._refs(workspace, user)
        resp = await self._client.CheckBulkPermissions(CheckBulkPermissionsRequest(
            consistency=self._consistency(),
            items=[CheckBulkPermissionsRequestItem(resource=resource, permission=p, subject=subject)
                   for p in permissions]))
        out = {}
        for pair in resp.pairs:
            ok = (pair.WhichOneof("response") == "item"
                  and pair.item.permissionship == CheckPermissionResponse.PERMISSIONSHIP_HAS_PERMISSION)
            out[pair.request.permission] = ok
        return {p: out.get(p, False) for p in permissions}

    async def _write(self, op, workspace, user, role):
        from authzed.api.v1 import Relationship, RelationshipUpdate, WriteRelationshipsRequest

        resource, subject = self._refs(workspace, user)
        resp = await self._client.WriteRelationships(WriteRelationshipsRequest(updates=[RelationshipUpdate(
            operation=op, relationship=Relationship(resource=resource, relation=role, subject=subject))]))
        self._last_write = resp.written_at

    async def grant(self, workspace, user, role):
        from authzed.api.v1 import RelationshipUpdate

        await self._write(RelationshipUpdate.Operation.OPERATION_TOUCH, workspace, user, role)

    async def revoke(self, workspace, user, role):
        from authzed.api.v1 import RelationshipUpdate

        await self._write(RelationshipUpdate.Operation.OPERATION_DELETE, workspace, user, role)

    async def members(self, workspace):
        from authzed.api.v1 import Consistency, ReadRelationshipsRequest, RelationshipFilter

        req = ReadRelationshipsRequest(
            consistency=Consistency(fully_consistent=True),
            relationship_filter=RelationshipFilter(resource_type="workspace", optional_resource_id=workspace))
        out = []
        async for r in self._client.ReadRelationships(req):
            out.append((r.relationship.subject.object.object_id, r.relationship.relation))
        return sorted(out)

    async def write_schema(self):
        from authzed.api.v1 import WriteSchemaRequest

        await self._client.WriteSchema(WriteSchemaRequest(schema=SCHEMA))

    async def reset_workspace(self, workspace):
        """Delete every relationship on the workspace (used by the eval harness between tasks)."""
        from authzed.api.v1 import DeleteRelationshipsRequest, RelationshipFilter

        resp = await self._client.DeleteRelationships(DeleteRelationshipsRequest(
            relationship_filter=RelationshipFilter(resource_type="workspace", optional_resource_id=workspace)))
        self._last_write = resp.deleted_at


async def bootstrap(authz, workspace, *, attempts=30):
    """Write the schema and demo role assignments. Retries while SpiceDB is still starting."""
    for i in range(attempts):
        try:
            if isinstance(authz, SpiceDBAuthorizer):
                await authz.write_schema()
            for user, role in DEMO_RELATIONSHIPS:
                await authz.grant(workspace, user, role)
            return
        except Exception as exc:  # noqa: BLE001 - grpc raises several types while the server boots
            if i == attempts - 1:
                raise
            log.warning("SpiceDB not ready (%s: %s); retrying", type(exc).__name__, str(exc)[:120])
            await asyncio.sleep(1)
