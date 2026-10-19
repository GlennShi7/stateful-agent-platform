"""Gateway: dev login, REST helpers, and one multiplexed WebSocket per browser tab.

Several conversations ("sessions", one LangGraph thread each) share one WebSocket. Every frame
carries its session id; each run is its own asyncio task, and one writer task per connection
serialises frames onto the socket (a WebSocket send is not safe to call concurrently).

Client -> server: {"op": "open"|"send"|"approve"|"resume"|"cancel"|"ping", "session": id, ...}
Server -> client: {"session": id, "type": "state"|"stage"|"token"|"tool_start"|"tool_end"|
                   "approval_required"|"done"|"error"|"cancelled"|"pong", "t": server_ms, ...}

Run: uvicorn agentplat.gateway:app --host 0.0.0.0 --port 8000
"""
import asyncio
import json
import logging
import re
from contextlib import asynccontextmanager

from fastapi import FastAPI, Header, HTTPException, WebSocket, WebSocketDisconnect
from langgraph.types import Command
from pydantic import BaseModel

from . import tokens
from .config import get_settings
from .graph import RunContext, build_graph, config_for, now_ms, run_turn

log = logging.getLogger(__name__)
SESSION_RE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")


def transcript(values: dict) -> list[dict]:
    out = []
    for m in values.get("messages") or []:
        if m["role"] == "user":
            out.append({"role": "user", "text": m["content"]})
        elif m["role"] == "assistant":
            if m.get("content"):
                out.append({"role": "assistant", "text": m["content"]})
            for c in m.get("tool_calls") or []:
                out.append({"role": "tool_call", "id": c["id"], "name": c["function"]["name"],
                            "arguments": c["function"]["arguments"]})
        elif m["role"] == "tool":
            out.append({"role": "tool_result", "id": m["tool_call_id"], "result": m["content"][:2000]})
    return out


class Services:
    """Everything a run needs. Built by the app lifespan, or directly by tests."""

    def __init__(self, settings, repo, graph, llm, tools, telemetry=None):
        self.settings, self.repo, self.graph, self.llm, self.tools, self.telemetry = (
            settings, repo, graph, llm, tools, telemetry)
        self.runs: dict[str, asyncio.Task] = {}
        self._locks: dict[str, asyncio.Lock] = {}
        # Connections currently showing each session. Run events go to all of them, so a client that
        # reconnects (or a second tab) picks up a run that is already streaming.
        self.subs: dict[str, set] = {}

    def subscribe(self, sid, conn):
        self.subs.setdefault(sid, set()).add(conn)
        conn.sessions.add(sid)

    def unsubscribe_all(self, conn):
        for sid in conn.sessions:
            subs = self.subs.get(sid)
            if subs is not None:
                subs.discard(conn)
                if not subs:
                    del self.subs[sid]

    async def broadcast(self, sid, frame):
        for conn in list(self.subs.get(sid, ())):
            await conn.send({"session": sid, **frame})

    def lock(self, thread_id) -> asyncio.Lock:
        return self._locks.setdefault(thread_id, asyncio.Lock())

    def context(self, user, thread_id):
        s = self.settings
        return RunContext(user=user, thread_id=thread_id, llm=self.llm, tools=self.tools,
                          caller_token=tokens.caller_token(user, s.workspace, thread_id, s.caller_secret),
                          max_iterations=s.max_iterations, telemetry=self.telemetry)


class Connection:
    def __init__(self, ws: WebSocket, user: str):
        self.ws, self.user = ws, user
        self.queue: asyncio.Queue = asyncio.Queue(maxsize=5000)
        self.closed = False
        self.sessions: set[str] = set()

    async def send(self, frame: dict):
        if not self.closed:
            await self.queue.put(frame)  # bounded: a slow client back-pressures only its own runs

    async def writer(self):
        try:
            while True:
                frame = await self.queue.get()
                await self.ws.send_text(json.dumps(frame, default=str))
        except (WebSocketDisconnect, RuntimeError):
            self.closed = True


async def _state_frame(svc: Services, sid: str) -> dict:
    st = await svc.graph.aget_state(config_for(sid))
    values = st.values or {}
    approval = st.interrupts[0].value if st.interrupts else None
    running = sid in svc.runs
    return {"session": sid, "type": "state", "messages": transcript(values), "status": values.get("status"),
            "approval": approval, "running": running,
            # next nodes but no interrupt and nothing running = the process died mid-run
            "resumable": bool(st.next) and not st.interrupts and not running, "t": now_ms()}


async def _start_run(svc: Services, conn: Connection, sid: str, inp):
    held = []

    async def on_event(ev):
        if ev.get("type") == "done":
            held.append(ev)  # sent after the run is deregistered, so a client may send again on receipt
            return
        await svc.broadcast(sid, ev)

    async def body():
        try:
            await run_turn(svc.graph, inp, svc.context(conn.user, sid), on_event)
        except asyncio.CancelledError:
            await svc.broadcast(sid, {"type": "cancelled", "t": now_ms()})
            raise
        except Exception as exc:  # noqa: BLE001 - surface any run failure to the client
            log.exception("run failed for session %s", sid)
            await svc.broadcast(sid, {"type": "error", "message": str(exc)[:300], "t": now_ms()})
        finally:
            svc.runs.pop(sid, None)
            for ev in held:
                await svc.broadcast(sid, ev)

    svc.runs[sid] = asyncio.create_task(body(), name=f"run-{sid}")


async def handle_op(svc: Services, conn: Connection, msg: dict):
    op, sid = msg.get("op"), msg.get("session")
    if op == "ping":
        return await conn.send({"type": "pong", "client_t": msg.get("client_t"), "t": now_ms()})
    if not isinstance(sid, str) or not SESSION_RE.match(sid):
        return await conn.send({"type": "error", "message": "bad session id", "t": now_ms()})

    def err(text):
        return conn.send({"session": sid, "type": "error", "message": text, "t": now_ms()})

    # Ops on one session apply in arrival order (asyncio.Lock wakes waiters FIFO, and taking the
    # lock is the first await); ops on different sessions proceed in parallel.
    async with svc.lock(sid):
        owner = await svc.repo.claim_thread(sid, conn.user)
        if owner != conn.user:
            return await err("this session belongs to another user")
        svc.subscribe(sid, conn)
        if op == "open":
            return await conn.send(await _state_frame(svc, sid))
        if op == "cancel":
            task = svc.runs.get(sid)
            if task:
                task.cancel()
            return None
        return await _start_op(svc, conn, sid, op, msg, err)


async def _start_op(svc, conn, sid, op, msg, err):
    if sid in svc.runs:
        return await err("a run is already in progress in this session")
    st = await svc.graph.aget_state(config_for(sid))
    if op == "send":
        text = (msg.get("text") or "").strip()
        if not text or len(text) > 4000:
            return await err("message must be 1-4000 characters")
        if st.interrupts:
            return await err("answer the pending approval first")
        if st.next:
            return await err("the previous run did not finish; resume it first")
        await svc.repo.set_thread_title(sid, text)
        return await _start_run(svc, conn, sid, {"messages": [{"role": "user", "content": text}]})
    if op == "approve":
        if not st.interrupts:
            return await err("nothing is waiting for approval")
        decisions = msg.get("decisions") or {}
        return await _start_run(svc, conn, sid, Command(resume={k: bool(v) for k, v in decisions.items()}))
    if op == "resume":
        if not st.next or st.interrupts:
            return await err("nothing to resume")
        return await _start_run(svc, conn, sid, None)
    return await err(f"unknown op {op!r}")


def create_app(services: Services | None = None) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if services is not None:
            app.state.svc = services
            yield
            return
        svc, cleanup = await build_services()
        app.state.svc = svc
        try:
            yield
        finally:
            for task in list(svc.runs.values()):
                task.cancel()
            await cleanup()

    app = FastAPI(title="Stateful Agent Platform", lifespan=lifespan)

    class Login(BaseModel):
        user: str

    def _user(authorization):
        if not authorization or not authorization.startswith("Bearer "):
            raise HTTPException(401, "missing bearer token")
        try:
            return tokens.verify(authorization[7:], app.state.svc.settings.session_secret,
                                 aud=tokens.SESSION_AUD)["sub"]
        except tokens.TokenError as exc:
            raise HTTPException(401, str(exc)) from exc

    @app.get("/healthz")
    async def healthz():
        return {"ok": True}

    @app.post("/api/login")
    async def login(body: Login):
        # Development login: no password, a fixed list of demo users. Real deployments would put an
        # identity provider here; everything downstream only consumes the signed session token.
        svc = app.state.svc
        if body.user not in svc.settings.dev_users:
            raise HTTPException(403, "unknown user")
        return {"user": body.user, "token": tokens.session_token(body.user, svc.settings.session_secret)}

    @app.get("/api/me")
    async def me(authorization: str | None = Header(default=None)):
        user = _user(authorization)
        svc = app.state.svc
        specs, destructive = await svc.tools.list_for(
            tokens.caller_token(user, svc.settings.workspace, "-", svc.settings.caller_secret))
        return {"user": user, "tools": [s["function"]["name"] for s in specs], "needs_approval": destructive}

    @app.get("/api/threads")
    async def threads(authorization: str | None = Header(default=None)):
        return await app.state.svc.repo.list_threads(_user(authorization))

    @app.websocket("/ws")
    async def ws_endpoint(ws: WebSocket):
        svc: Services = app.state.svc
        try:
            user = tokens.verify(ws.query_params.get("token", ""), svc.settings.session_secret,
                                 aud=tokens.SESSION_AUD)["sub"]
        except tokens.TokenError:
            await ws.close(code=4401)
            return
        await ws.accept()
        conn = Connection(ws, user)
        writer = asyncio.create_task(conn.writer())
        ops: set[asyncio.Task] = set()

        async def guarded(msg):
            try:
                await handle_op(svc, conn, msg)
            except Exception:  # noqa: BLE001
                log.exception("op failed")
                await conn.send({"session": msg.get("session"), "type": "error", "message": "internal error",
                                 "t": now_ms()})
        try:
            while True:
                raw = await ws.receive_text()
                try:
                    msg = json.loads(raw)
                except ValueError:
                    await conn.send({"type": "error", "message": "invalid JSON", "t": now_ms()})
                    continue
                # Each op is its own task. Awaiting ops inline here made every session on a socket
                # wait for the previous op's database round trips (head-of-line blocking).
                task = asyncio.create_task(guarded(msg))
                ops.add(task)
                task.add_done_callback(ops.discard)
        except WebSocketDisconnect:
            pass
        finally:
            # Runs keep going after a disconnect; their state is checkpointed and the client can
            # reopen the session later. Only this connection's frames are dropped.
            conn.closed = True
            svc.unsubscribe_all(conn)
            writer.cancel()

    return app


async def build_services():
    from .checkpoint import PooledPostgresSaver, make_pool
    from .llm import make_llm
    from .mcp_client import ToolGateway
    from .repo import Repo
    from .schema import apply_sql
    from .telemetry import Telemetry

    settings = get_settings()
    await apply_sql(settings.app_database_url, ["002_app.sql"])
    repo = Repo(settings.app_database_url, settings.analytics_database_url)
    await repo.open()
    ckpt_pool = make_pool(settings.app_database_url)
    await ckpt_pool.open(wait=True, timeout=60)
    saver = PooledPostgresSaver(ckpt_pool)
    await saver.setup()
    llm = make_llm(settings)
    tools = ToolGateway(settings.mcp_url)
    telemetry = Telemetry(settings.dashboard_events_url) if settings.dashboard_events_url else None
    svc = Services(settings, repo, build_graph(saver), llm, tools, telemetry)
    log.info("gateway ready (llm=%s, mcp=%s, telemetry=%s)", settings.llm_provider, settings.mcp_url,
             bool(telemetry))

    async def cleanup():
        await tools.aclose()
        if telemetry:
            await telemetry.aclose()
        if hasattr(llm, "aclose"):
            await llm.aclose()
        await ckpt_pool.close()
        await repo.close()

    return svc, cleanup


logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
for _noisy in ("httpx", "httpx2", "mcp"):
    logging.getLogger(_noisy).setLevel(logging.WARNING)
app = create_app()  # services are built in the lifespan, so importing this module is cheap
