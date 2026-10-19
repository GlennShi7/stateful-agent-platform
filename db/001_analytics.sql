-- Same tables as the Real-Time Analytics Dashboard. In standalone mode the agent platform keeps its
-- own copy filled by `python -m agentplat.seed`; in integrated mode ANALYTICS_DATABASE_URL points
-- at the dashboard's Postgres instead and this file is unused.
CREATE TABLE IF NOT EXISTS events (
    id          BIGSERIAL PRIMARY KEY,
    event_id    TEXT        NOT NULL UNIQUE,
    ts          TIMESTAMPTZ NOT NULL,
    event_type  TEXT        NOT NULL,
    user_id     TEXT,
    latency_ms  INTEGER     NOT NULL DEFAULT 0,
    is_error    BOOLEAN     NOT NULL DEFAULT FALSE,
    simulated   BOOLEAN     NOT NULL DEFAULT FALSE
);
CREATE INDEX IF NOT EXISTS idx_events_type_ts ON events (event_type, ts);
CREATE INDEX IF NOT EXISTS idx_events_ts_brin ON events USING brin (ts);

CREATE TABLE IF NOT EXISTS metrics_minute (
    bucket          TIMESTAMPTZ NOT NULL,
    event_type      TEXT        NOT NULL,
    event_count     BIGINT      NOT NULL,
    latency_sum_ms  BIGINT      NOT NULL,
    error_count     BIGINT      NOT NULL,
    PRIMARY KEY (bucket, event_type)
);
