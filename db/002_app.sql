-- State owned by the agent platform. LangGraph's checkpoint tables are created separately by
-- AsyncPostgresSaver.setup() when the gateway starts.

-- Which user owns which conversation thread. Every gateway operation checks this, so one user
-- cannot resume or read another user's run.
CREATE TABLE IF NOT EXISTS threads (
    thread_id   TEXT PRIMARY KEY,
    user_id     TEXT        NOT NULL,
    title       TEXT        NOT NULL DEFAULT '',
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS idx_threads_user ON threads (user_id, created_at DESC);

CREATE TABLE IF NOT EXISTS alert_rules (
    id              SERIAL PRIMARY KEY,
    event_type      TEXT        NOT NULL,
    metric          TEXT        NOT NULL CHECK (metric IN ('error_rate', 'p95_latency_ms', 'event_count')),
    op              TEXT        NOT NULL CHECK (op IN ('>', '<')),
    threshold       DOUBLE PRECISION NOT NULL,
    window_minutes  INTEGER     NOT NULL CHECK (window_minutes BETWEEN 1 AND 1440),
    created_by      TEXT        NOT NULL,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS annotations (
    id          SERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    event_type  TEXT,
    note        TEXT        NOT NULL,
    created_by  TEXT        NOT NULL
);

-- Only a hash of each key is stored; the plaintext is returned once, at creation.
CREATE TABLE IF NOT EXISTS api_keys (
    id          SERIAL PRIMARY KEY,
    name        TEXT        NOT NULL,
    key_prefix  TEXT        NOT NULL,
    key_hash    TEXT        NOT NULL,
    created_by  TEXT        NOT NULL,
    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
    revoked_at  TIMESTAMPTZ
);

-- Idempotency ledger for write tools. The agent runtime sends "<thread>:<tool_call_id>" as the key,
-- so when a crashed run is resumed and the tools stage re-executes, a write that already happened
-- returns its recorded result instead of running twice.
CREATE TABLE IF NOT EXISTS tool_invocations (
    idempotency_key TEXT PRIMARY KEY,
    tool            TEXT        NOT NULL,
    caller          TEXT        NOT NULL,
    result          JSONB,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Every authorization decision the MCP server makes.
CREATE TABLE IF NOT EXISTS audit_log (
    id          BIGSERIAL PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL DEFAULT now(),
    user_id     TEXT,
    tool        TEXT        NOT NULL,
    permission  TEXT,
    allowed     BOOLEAN     NOT NULL,
    reason      TEXT
);
