"""SpiceDB must make exactly the same decisions as the in-memory model of schema.zed."""
import os
import socket
import uuid

import pytest

from agentplat.authz import DEMO_RELATIONSHIPS, PERMISSIONS, SpiceDBAuthorizer, StaticAuthorizer, bootstrap

ENDPOINT = os.environ.get("SPICEDB_ENDPOINT", "localhost:50051")


def _reachable():
    host, port = ENDPOINT.rsplit(":", 1)
    try:
        socket.create_connection((host, int(port)), timeout=1).close()
        return True
    except OSError:
        return False


pytestmark = pytest.mark.skipif(not _reachable(), reason=f"SpiceDB not reachable at {ENDPOINT}")


async def test_spicedb_matches_the_static_model():
    ws = "test-" + uuid.uuid4().hex[:8]
    spice = SpiceDBAuthorizer(ENDPOINT, os.environ.get("SPICEDB_KEY", "devkey"))
    await bootstrap(spice, ws, attempts=3)
    static = StaticAuthorizer(DEMO_RELATIONSHIPS, workspace=ws)
    for user in ("alice", "bob", "carol", "dave", "mallory"):
        assert await spice.check_many(user, ws, list(PERMISSIONS)) == await static.check_many(user, ws, list(PERMISSIONS)), user
    assert await spice.members(ws) == await static.members(ws)


async def test_grant_is_visible_to_the_next_check():
    ws = "test-" + uuid.uuid4().hex[:8]
    spice = SpiceDBAuthorizer(ENDPOINT, os.environ.get("SPICEDB_KEY", "devkey"))
    await bootstrap(spice, ws, attempts=3)
    assert not (await spice.check_many("erin", ws, ["manage_alerts"]))["manage_alerts"]
    await spice.grant(ws, "erin", "operator")
    assert (await spice.check_many("erin", ws, ["manage_alerts"]))["manage_alerts"]  # read-your-writes via ZedToken
    await spice.revoke(ws, "erin", "operator")
    assert not (await spice.check_many("erin", ws, ["manage_alerts"]))["manage_alerts"]
    await spice.reset_workspace(ws)
    assert await spice.members(ws) == []
