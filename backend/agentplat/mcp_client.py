"""The agent runtime's side of MCP: one shared connection, caller identity per request.

Identity travels in each request's `_meta` rather than in connection headers, so a single MCP
connection serves every user and session concurrently.

The connection lives in its own background task. The MCP client is built on anyio task groups,
which must be exited by the task that entered them; tying the connection to whichever WebSocket
session happened to open it first would break when that session ends or is cancelled.
"""
import asyncio
import json
import logging

from mcp import Client

from .tools_server import CALLER_META, IDEMPOTENCY_META

log = logging.getLogger(__name__)


def to_openai_tool(tool) -> dict:
    return {"type": "function", "function": {
        "name": tool.name, "description": (tool.description or "").strip(),
        "parameters": tool.input_schema or {"type": "object", "properties": {}}}}


class ToolGateway:
    def __init__(self, server, read_timeout_s=60):
        """`server` is a URL (streamable HTTP) or an MCPServer instance (in-process, for tests/eval)."""
        self._server, self._timeout = server, read_timeout_s
        self._client = None
        self._task = None
        self._stop = None
        self._lock = asyncio.Lock()

    async def _hold(self, ready: asyncio.Future, stop: asyncio.Event):
        try:
            async with Client(self._server, read_timeout_seconds=self._timeout) as client:
                self._client = client
                ready.set_result(client)
                await stop.wait()
        except BaseException as exc:  # noqa: BLE001 - report startup failure to the waiter
            if not ready.done():
                ready.set_exception(exc)
            elif not isinstance(exc, asyncio.CancelledError):
                log.warning("MCP connection closed: %r", exc)
        finally:
            self._client = None

    async def _get(self):
        async with self._lock:
            if self._client is not None and self._task is not None and not self._task.done():
                return self._client
            ready = asyncio.get_running_loop().create_future()
            self._stop = asyncio.Event()
            self._task = asyncio.create_task(self._hold(ready, self._stop), name="mcp-connection")
            return await ready

    async def _reset(self):
        async with self._lock:
            task, stop = self._task, self._stop
            self._task = self._stop = self._client = None
        if task is not None:
            stop.set()
            try:
                await asyncio.wait_for(task, 5)
            except Exception:  # noqa: BLE001
                task.cancel()

    async def _with_retry(self, fn):
        try:
            return await fn(await self._get())
        except (ConnectionError, OSError, asyncio.TimeoutError):
            # Connection-level failure: reconnect once. Tool errors come back as results, not here.
            log.warning("MCP connection failed; reconnecting", exc_info=True)
            await self._reset()
            return await fn(await self._get())

    async def list_for(self, caller_token: str):
        """Tools this caller may use, in OpenAI function format, plus the names that need approval."""
        result = await self._with_retry(lambda c: c.list_tools(meta={CALLER_META: caller_token}))
        specs = [to_openai_tool(t) for t in result.tools]
        destructive = sorted(t.name for t in result.tools if t.annotations and t.annotations.destructive_hint)
        return specs, destructive

    async def call(self, name: str, arguments: dict, caller_token: str, idempotency_key: str | None):
        meta = {CALLER_META: caller_token}
        if idempotency_key:
            meta[IDEMPOTENCY_META] = idempotency_key
        result = await self._with_retry(lambda c: c.call_tool(name, arguments, meta=meta))
        text = "\n".join(getattr(c, "text", "") for c in result.content) or json.dumps(result.structured_content)
        return (not result.is_error), text

    async def aclose(self):
        await self._reset()
