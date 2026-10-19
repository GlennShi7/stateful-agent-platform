"""Optional integration with the Real-Time Analytics Dashboard: report agent runs and tool calls
as events to its ingest API (POST /events), so agent usage shows up on the dashboard.

Fire-and-forget with a short timeout: the dashboard being down must never slow down or fail a run.
"""
import asyncio
import logging
import uuid

import httpx

log = logging.getLogger(__name__)


class Telemetry:
    def __init__(self, events_url: str, transport=None):
        self.url = events_url
        self._http = httpx.AsyncClient(timeout=1.0, transport=transport)
        self._pending: set[asyncio.Task] = set()

    def emit(self, event_type: str, user: str, latency_ms: int, is_error: bool):
        event = {"event_id": str(uuid.uuid4()), "type": event_type, "user_id": user,
                 "latency_ms": max(0, min(int(latency_ms), 600_000)), "is_error": bool(is_error)}
        task = asyncio.create_task(self._send(event))
        self._pending.add(task)  # keep a reference so the task is not garbage-collected mid-flight
        task.add_done_callback(self._pending.discard)

    async def _send(self, event):
        try:
            await self._http.post(self.url, json=event)
        except httpx.HTTPError as exc:
            log.debug("telemetry dropped: %s", exc)

    async def aclose(self):
        if self._pending:
            await asyncio.wait(self._pending, timeout=2)
        await self._http.aclose()
