"""The SpiceDB client must work across a container network and send the key on every kind of call.

Regression: grpcutil.insecure_bearer_token_credentials only connects to loopback addresses, so the
MCP server worked against localhost and failed to start in docker compose (spicedb:50051).
"""
import socket

import grpc
import pytest
from authzed.api.v1 import (CheckBulkPermissionsRequest, CheckBulkPermissionsResponse,
                            ReadRelationshipsRequest)
from authzed.api.v1.permission_service_pb2_grpc import (PermissionsServiceServicer,
                                                        add_PermissionsServiceServicer_to_server)

from agentplat.authz import _plaintext_client


class Recorder(PermissionsServiceServicer):
    def __init__(self):
        self.auth = []

    def _record(self, context):
        self.auth.append(dict(context.invocation_metadata()).get("authorization"))

    async def CheckBulkPermissions(self, request, context):  # unary
        self._record(context)
        return CheckBulkPermissionsResponse()

    async def ReadRelationships(self, request, context):  # server streaming
        self._record(context)
        return
        yield


def _non_loopback_ip():
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("192.0.2.1", 9))  # no packet is sent; this just picks the outbound interface
            ip = s.getsockname()[0]
    except OSError:
        return None
    return None if ip.startswith("127.") else ip


@pytest.fixture
async def server():
    srv = grpc.aio.server()
    rec = Recorder()
    add_PermissionsServiceServicer_to_server(rec, srv)
    port = srv.add_insecure_port("0.0.0.0:0")
    await srv.start()
    yield rec, port
    await srv.stop(None)


@pytest.mark.parametrize("where", ["loopback", "network"])
async def test_token_sent_on_unary_and_streaming_calls(server, where, monkeypatch):
    for var in ("http_proxy", "https_proxy", "grpc_proxy", "HTTP_PROXY", "HTTPS_PROXY", "all_proxy", "ALL_PROXY"):
        monkeypatch.delenv(var, raising=False)
    rec, port = server
    host = "127.0.0.1" if where == "loopback" else _non_loopback_ip()
    if host is None:
        pytest.skip("no non-loopback interface")
    client = _plaintext_client(f"{host}:{port}", "sekrit")
    await client.CheckBulkPermissions(CheckBulkPermissionsRequest(), timeout=5)
    assert [r async for r in client.ReadRelationships(ReadRelationshipsRequest(), timeout=5)] == []
    assert rec.auth == ["Bearer sekrit", "Bearer sekrit"]
