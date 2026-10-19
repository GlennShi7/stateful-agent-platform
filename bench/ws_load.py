"""Load test: N concurrent agent sessions multiplexed over a few WebSocket connections.

Start the stack with the scripted LLM (`make up-fake`), so the numbers measure the platform
(gateway, LangGraph + Postgres checkpoints, MCP, SpiceDB, Postgres queries) rather than a model
provider. Each run = intake (tools/list through SpiceDB) -> LLM turn requesting a tool -> MCP tool
call against Postgres -> LLM turn streaming the answer.

Reports, over all runs:
  token_delivery_ms  server stamp on each token frame -> received by the client
  ttft_ms            client sends the message -> first answer token arrives
  run_ms             client sends the message -> `done`

Usage: python bench/ws_load.py --connections 10 --sessions 5 --rounds 3
"""
import argparse
import asyncio
import json
import pathlib
import statistics
import time
import uuid

import httpx
from websockets.asyncio.client import connect


def pct(values, p):
    if not values:
        return None
    s = sorted(values)
    return round(s[min(len(s) - 1, int(round(p / 100 * (len(s) - 1))))], 2)


def summary(values):
    return {"n": len(values), "p50": pct(values, 50), "p95": pct(values, 95), "p99": pct(values, 99),
            "max": round(max(values), 2) if values else None}


def now_ms():
    return time.time() * 1000


async def run_connection(url, token, n_sessions, rounds, stats, prefix, ramp_s=0.0):
    async with connect(f"{url}?token={token}", max_size=None) as ws:
        waiters: dict[str, asyncio.Queue] = {}

        async def reader():
            async for raw in ws:
                f = json.loads(raw)
                f["_recv"] = now_ms()
                if f.get("session") in waiters:
                    waiters[f["session"]].put_nowait(f)

        async def session(i):
            sid = f"{prefix}-{i}"
            q = waiters[sid] = asyncio.Queue()
            await asyncio.sleep(ramp_s * i / max(1, n_sessions))  # 0 = every session starts at once
            for r in range(rounds):
                t_send = now_ms()
                await ws.send(json.dumps({"op": "send", "session": sid, "text": f"error rates please ({r})"}))
                first = None
                while True:
                    f = await asyncio.wait_for(q.get(), timeout=60)
                    if f["type"] == "token":
                        stats["token_delivery_ms"].append(f["_recv"] - f["t"])
                        if first is None:
                            first = f["_recv"]
                            stats["ttft_ms"].append(first - t_send)
                    elif f["type"] == "done":
                        stats["run_ms"].append(f["_recv"] - t_send)
                        stats["runs"] += 1
                        break
                    elif f["type"] in ("error", "approval_required"):
                        stats["errors"].append(f.get("message", f["type"]))
                        break

        rtask = asyncio.create_task(reader())
        try:
            await asyncio.gather(*[session(i) for i in range(n_sessions)])
        finally:
            rtask.cancel()


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--ws", default="ws://localhost:8000/ws")
    ap.add_argument("--user", default="bob")
    ap.add_argument("--connections", type=int, default=10)
    ap.add_argument("--sessions", type=int, default=5, help="sessions per connection")
    ap.add_argument("--rounds", type=int, default=3, help="messages per session")
    ap.add_argument("--ramp-s", type=float, default=0.0,
                    help="spread session starts over this many seconds (default 0: all at once, a burst)")
    a = ap.parse_args()

    token = httpx.post(f"{a.api}/api/login", json={"user": a.user}).json()["token"]
    stats = {"token_delivery_ms": [], "ttft_ms": [], "run_ms": [], "runs": 0, "errors": []}
    run_id = uuid.uuid4().hex[:6]
    t0 = time.perf_counter()
    await asyncio.gather(*[run_connection(a.ws, token, a.sessions, a.rounds, stats, f"bench-{run_id}-{c}", a.ramp_s)
                           for c in range(a.connections)])
    wall = time.perf_counter() - t0
    out = {
        "concurrent_sessions": a.connections * a.sessions, "connections": a.connections, "ramp_s": a.ramp_s,
        "runs_completed": stats["runs"], "errors": len(stats["errors"]), "error_samples": stats["errors"][:5],
        "wall_s": round(wall, 2), "runs_per_s": round(stats["runs"] / wall, 2),
        "token_delivery_ms": summary(stats["token_delivery_ms"]),
        "ttft_ms": summary(stats["ttft_ms"]), "run_ms": summary(stats["run_ms"]),
    }
    print(json.dumps(out, indent=2))
    path = pathlib.Path(__file__).parent / "results" / f"ws_load_{time.strftime('%Y%m%d-%H%M%S')}.json"
    path.parent.mkdir(exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    print(f"saved {path}")


if __name__ == "__main__":
    asyncio.run(main())
