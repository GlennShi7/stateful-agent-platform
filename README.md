# Stateful AI Agent Platform

An operations agent for a real-time analytics platform. Users ask questions in a browser ("why did
upload errors jump?", "alert me when p95 latency passes 800 ms"); a LangGraph agent answers by calling
20 admin tools over MCP. What each user may do is decided by SpiceDB. Every step of a run is
checkpointed to Postgres, so a run survives a process crash, and destructive actions wait for a human
to approve them.

```
 browser (React)                gateway (FastAPI)                         MCP server                  data
 ───────────────                ─────────────────                         ──────────                  ────
 several sessions  ◄── one ──►  per-op tasks, per-session lock            authz middleware ──► SpiceDB
 on one WebSocket     socket    writer task per connection                  tools/list: filter     (roles, perms)
                                      │                                     tools/call: check
                                      ▼                                          │
                                LangGraph run (one per session)                  ▼
                                intake → agent ⇄ tools → finalize   ── MCP ──►  20 tools ──────────► Postgres
                                           └► approval (interrupt)  signed       idempotency ledger    events, rollups,
                                checkpoint after every step ──────► caller       audit log             alert rules, keys
                                                    Postgres        context
                                agent ──► DeepSeek (SSE)            in _meta
```

| Component | Code | Role |
|---|---|---|
| Agent graph | `backend/agentplat/graph.py` | Five stages, ReAct loop capped at 10 iterations, approval interrupt, checkpointed every step |
| Gateway | `backend/agentplat/gateway.py` | Dev login, REST helpers, multiplexed WebSocket streaming of tokens and tool events |
| MCP server | `backend/agentplat/tools_server.py` | 20 tools over the analytics data, authorization middleware, idempotent writes |
| Authorization | `backend/agentplat/authz.py`, `schema.zed` | Tool → permission map, SpiceDB client, an in-memory twin for tests |
| LLM | `backend/agentplat/llm.py` | DeepSeek over raw SSE (tool-call reassembly, reasoning_content round trip); a scripted model for tests and load tests |
| Checkpointer | `backend/agentplat/checkpoint.py` | Postgres checkpointer without the library's global lock (see below) |
| Frontend | `frontend/src` | Session list, streaming chat, tool cards, approval and resume controls |
| Eval | `backend/evals` | 28 tasks with automatic checks against the database and SpiceDB |

## Design decisions

**Five stages, each a checkpoint boundary.** `intake` loads the caller's tools, `agent` makes one LLM
call, `approval` pauses for a human, `tools` executes the requested calls, `finalize` records the
outcome. LangGraph writes the state to Postgres after every node (`durability="sync"`), so a run that
dies anywhere resumes from the last completed step (`tests/test_crash_resume.py` kills the process
with `os._exit` mid-run and resumes it in another process).

**Resuming must not repeat writes.** If the process dies after a tool executed but before the tools
stage was checkpointed, resuming re-runs that stage. Every call carries the idempotency key
`<thread>:<tool_call_id>`; the MCP server records the key in the same transaction as the write and
returns the recorded result on replay. Writes outside that database (SpiceDB relationships, purging
analytics rows) are idempotent by construction (TOUCH/DELETE, delete-older-than).

**Bounded loop.** The agent stage counts iterations and stops at 10 with an explicit message;
LangGraph's `recursion_limit` is only a backstop.

**Identity never comes from the model.** The gateway mints a signed caller token (HMAC, audience,
expiry) for the authenticated user on every run and sends it in each MCP request's `_meta`. The
MCP server derives the user from that signature only. The token is passed through LangGraph's runtime
context, not the graph state, so it is never checkpointed, and a resumed run acts with the identity
of whoever resumed it.

**Two authorization checks.** `tools/list` returns only the tools the caller's permissions allow, so
the model is never offered what the user cannot do. `tools/call` checks again: the listing is a
convenience, the call-time check is the boundary (a client can call a tool it was never shown, and
permissions can change in between). Tools without a mapping are denied; every decision goes to
`audit_log`. Least privilege: each tool needs exactly one, the narrowest, permission; `dave` is a
viewer with only the extra `key_manager` relation, so he can rotate API keys but not touch alerts.

**Approval is driven by MCP metadata.** Tools annotated `destructiveHint` (delete rule, revoke key,
revoke role, purge events) make the graph `interrupt()` before the tools stage. The pause is a
checkpoint: the decision can arrive minutes later, after a page reload or a server restart.

**One MCP connection, many users.** Because identity is per request, a single connection serves
every session concurrently. It lives in its own task, since the MCP client's anyio task groups
must be exited by the task that entered them.

**Multiplexed WebSocket.** One socket per browser tab carries all of its sessions; every frame is
tagged with its session. Each op runs as its own task; a per-session lock keeps ops on one session
in arrival order; one writer task per connection serialises sends. Runs are not tied to the socket
that started them: their events go to every connection currently showing the session, so a client
that reconnects (or a second tab) picks up a run that is still streaming.

**Two bottlenecks found with the load test.**
1. Ops were first awaited inline in the socket's receive loop, so every session on a connection
   waited for the previous op's database round trips (head-of-line blocking). Now each op is a task.
2. `AsyncPostgresSaver` guards every operation with one `asyncio.Lock`, even when given a connection
   pool, which serialises checkpoint I/O across all threads. `PooledPostgresSaver` removes the lock
   (each operation already has its own pooled connection; LangGraph orders a thread's writes itself).
   `tests/test_checkpoint.py` shows the stock saver never has more than one checkout in flight.

**DeepSeek specifics.** `deepseek-flash` thinks by default, and with tools enabled the API then
requires all earlier `reasoning_content` to be sent back. Thinking is off by default (lower time to
first token); when enabled, the client keeps `reasoning_content` in the message history.

## Run it

Prerequisites: Docker (Docker Desktop with WSL2 on Windows) and `make`.

```bash
cp .env.example .env        # put your DeepSeek API key in it
make up                     # Postgres, SpiceDB, MCP server, gateway, frontend; seeds demo data
open http://localhost:8081
make up-fake                # same stack with a scripted model, no API key needed
make down                   # stop (make clean also deletes the data)
```

Demo users (dev login, no password):

| User | SpiceDB relations | Tools visible |
|---|---|---|
| alice | admin | all 20 |
| bob | operator | 13: analytics, annotations, alert rules |
| carol | viewer | 10: read-only analytics |
| dave | viewer + key_manager | 13: read-only analytics + API keys |

The demo data has two planted incidents relative to "now": `contract_upload` errors jump to about
22% in the last 20 minutes, and `api_call` latency is about 3.5x higher in the last 30 minutes.
`make seed` refreshes it.

### With the Real-Time Analytics Dashboard

Start the [dashboard](https://github.com/GlennShi7/realtime-analytics-dashboard) first, then
`make up-integrated`. The tools then read the dashboard's live `events` and `metrics_minute` tables,
and every agent run and tool call is posted to the dashboard's `POST /events`, so agent usage appears
on its charts (`agent_run`, `agent_tool_call`). Ports do not overlap (this stack uses 5433, 8000,
8081, 50051).

## Tests

`make test` runs 58 tests: unit tests with in-memory fakes, plus integration tests against the
stack's Postgres (scratch database) and SpiceDB. CI runs all of them on every push.

| What | Where |
|---|---|
| ReAct loop, iteration cap, approval approve/decline, bad tool arguments, refusing tools outside the caller's list | `tests/test_graph.py` |
| Kill the process mid-run, resume in a new process, write executed once | `tests/test_crash_resume.py` |
| Listing filtered per user, call-time re-check, forged/missing/wrong-audience caller tokens, idempotency | `tests/test_tools_server.py` |
| Every tool against seeded Postgres, API key never stored in plaintext, audit log | `tests/test_tools_db.py` |
| SpiceDB decisions equal the in-memory model for every user and permission; read-your-writes | `tests/test_integration_spicedb.py` |
| Two sessions streaming over one socket, a second connection joining a running session, session ownership, approval over the WebSocket | `tests/test_gateway.py` |
| The eval checkers pass correct behaviour and fail wrong behaviour | `tests/test_evals.py` |

## Measure it

Every number quoted for this project should come from one of these runs. Results are written to
`bench/results/` and `backend/evals/results/` (not committed).

| Claim | Command | What it measures |
|---|---|---|
| Task completion rate | `make eval` (`ARGS="--repeat 3"` for variance) | 28 tasks run by DeepSeek through the full stack: 10 analytics questions checked against SQL, 12 changes checked in the database or SpiceDB, 6 authorization cases that must leave state unchanged. Also reports latency, tool calls and token cost |
| Streaming under concurrency | `make up-fake`, then `make bench-load` | 50 sessions over 10 WebSockets, 3 messages each. Reports token delivery latency (server stamp → client), time to first answer token, and run time, p50/p95/p99. The scripted model isolates what the platform adds; `ARGS="--ramp-s 5"` spreads the start instead of a burst |
| Model latency | `make bench-llm` | DeepSeek time to first token and total latency, without the platform |
| Crash recovery | `make test-crash` | Kill mid-run, resume, exactly one write |

Load-test numbers depend on the machine; state the hardware and that the model was scripted when
quoting them.

## Layout

```
backend/agentplat/   graph, gateway, MCP server, authz, LLM clients, checkpointer, repo, seed
backend/evals/       eval tasks and runner
backend/tests/       pytest suite
bench/               WebSocket load test, DeepSeek latency
db/                  SQL schema (analytics tables, app tables)
frontend/            React + Vite console
```

## How this project was built

Written with an AI coding assistant (Claude). I specified the architecture and requirements, reviewed
the code, and ran every test, evaluation and benchmark on my own machine; the numbers I quote come
from those runs.
