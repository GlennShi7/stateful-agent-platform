"""Postgres access for the MCP tools and the gateway.

Two pools: `analytics` (events + metrics_minute; the dashboard's database in integrated mode) and
`app` (alert rules, keys, threads, idempotency ledger, audit log). In standalone mode both URLs
point at the same database.
"""
import hashlib
import secrets

from psycopg.rows import dict_row
from psycopg.types.json import Jsonb
from psycopg_pool import AsyncConnectionPool

METRICS = ("error_rate", "p95_latency_ms", "event_count")


def _round(x, nd=4):
    return None if x is None else round(float(x), nd)


def _redact(result: dict) -> dict:
    """What the idempotency ledger stores: never a plaintext secret."""
    if "api_key" in result:
        return {**result, "api_key": None,
                "note": "the key was shown once when it was created; it cannot be displayed again"}
    return result


class Repo:
    def __init__(self, app_url: str, analytics_url: str, min_size=1, max_size=10):
        kw = dict(min_size=min_size, max_size=max_size, open=False, kwargs={"row_factory": dict_row})
        self.app = AsyncConnectionPool(app_url, **kw)
        self.analytics = self.app if analytics_url == app_url else AsyncConnectionPool(analytics_url, **kw)

    async def open(self):
        await self.app.open(wait=True, timeout=60)
        if self.analytics is not self.app:
            await self.analytics.open(wait=True, timeout=60)

    async def close(self):
        await self.app.close()
        if self.analytics is not self.app:
            await self.analytics.close()

    async def _all(self, pool, sql, args=()):
        async with pool.connection() as conn, conn.cursor() as cur:
            await cur.execute(sql, args)
            return await cur.fetchall()

    async def _one(self, pool, sql, args=()):
        rows = await self._all(pool, sql, args)
        return rows[0] if rows else None

    # ---------------------------------------------------------------- analytics (read)
    async def event_types(self, window_minutes):
        rows = await self._all(self.analytics, """
            SELECT event_type, count(*) AS events, sum(is_error::int) AS errors
            FROM events WHERE ts >= now() - make_interval(mins => %s)
            GROUP BY event_type ORDER BY events DESC""", (window_minutes,))
        return [{**r, "errors": int(r["errors"])} for r in rows]

    async def metrics(self, event_type, minutes):
        rows = await self._all(self.analytics, """
            SELECT bucket, event_count, latency_sum_ms, error_count FROM metrics_minute
            WHERE event_type = %s AND bucket >= date_trunc('minute', now()) - make_interval(mins => %s)
            ORDER BY bucket""", (event_type, minutes))
        return [{"minute": r["bucket"].strftime("%Y-%m-%dT%H:%MZ"), "events": r["event_count"],
                 "avg_latency_ms": round(r["latency_sum_ms"] / r["event_count"], 1) if r["event_count"] else None,
                 "errors": r["error_count"]} for r in rows]

    async def error_rate(self, event_type, window_minutes):
        rows = await self._all(self.analytics, """
            SELECT event_type, count(*) AS events, sum(is_error::int) AS errors
            FROM events WHERE ts >= now() - make_interval(mins => %s)
              AND (%s::text IS NULL OR event_type = %s)
            GROUP BY event_type ORDER BY event_type""", (window_minutes, event_type, event_type))
        return [{"event_type": r["event_type"], "events": r["events"], "errors": int(r["errors"]),
                 "error_rate": _round(int(r["errors"]) / r["events"])} for r in rows]

    async def latency_percentiles(self, event_type, window_minutes):
        r = await self._one(self.analytics, """
            SELECT count(*) AS events,
                   percentile_cont(0.5)  WITHIN GROUP (ORDER BY latency_ms) AS p50,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95,
                   percentile_cont(0.99) WITHIN GROUP (ORDER BY latency_ms) AS p99
            FROM events WHERE event_type = %s AND ts >= now() - make_interval(mins => %s)""",
                            (event_type, window_minutes))
        return {"event_type": event_type, "window_minutes": window_minutes, "events": r["events"],
                "p50_ms": _round(r["p50"], 1), "p95_ms": _round(r["p95"], 1), "p99_ms": _round(r["p99"], 1)}

    async def top_users(self, event_type, window_minutes, limit):
        rows = await self._all(self.analytics, """
            SELECT user_id, count(*) AS events, sum(is_error::int) AS errors
            FROM events WHERE ts >= now() - make_interval(mins => %s) AND user_id IS NOT NULL
              AND (%s::text IS NULL OR event_type = %s)
            GROUP BY user_id ORDER BY events DESC, user_id LIMIT %s""",
                               (window_minutes, event_type, event_type, limit))
        return [{**r, "errors": int(r["errors"])} for r in rows]

    async def _window_stats(self, event_type, start_min, end_min):
        r = await self._one(self.analytics, """
            SELECT count(*) AS events, coalesce(sum(is_error::int), 0) AS errors,
                   percentile_cont(0.95) WITHIN GROUP (ORDER BY latency_ms) AS p95
            FROM events WHERE event_type = %s
              AND ts >= now() - make_interval(mins => %s) AND ts < now() - make_interval(mins => %s)""",
                            (event_type, start_min, end_min))
        events = r["events"]
        return {"events": events, "errors": int(r["errors"]),
                "error_rate": _round(int(r["errors"]) / events) if events else None,
                "p95_latency_ms": _round(r["p95"], 1)}

    async def compare_windows(self, event_type, window_minutes):
        cur = await self._window_stats(event_type, window_minutes, 0)
        prev = await self._window_stats(event_type, 2 * window_minutes, window_minutes)
        return {"event_type": event_type, "window_minutes": window_minutes,
                "current": cur, "previous": prev}

    async def user_activity(self, user_id, window_minutes):
        rows = await self._all(self.analytics, """
            SELECT event_type, count(*) AS events, sum(is_error::int) AS errors
            FROM events WHERE user_id = %s AND ts >= now() - make_interval(mins => %s)
            GROUP BY event_type ORDER BY events DESC""", (user_id, window_minutes))
        return {"user_id": user_id, "window_minutes": window_minutes,
                "by_type": [{**r, "errors": int(r["errors"])} for r in rows],
                "total_events": sum(r["events"] for r in rows),
                "total_errors": sum(int(r["errors"]) for r in rows)}

    async def metric_value(self, event_type, metric, window_minutes):
        s = await self._window_stats(event_type, window_minutes, 0)
        return {"error_rate": s["error_rate"], "p95_latency_ms": s["p95_latency_ms"],
                "event_count": s["events"]}[metric]

    # ---------------------------------------------------------------- analytics (write)
    async def purge_events(self, older_than_days):
        async with self.analytics.connection() as conn, conn.transaction(), conn.cursor() as cur:
            await cur.execute("SELECT now() - make_interval(days => %s) AS cutoff", (older_than_days,))
            cutoff = (await cur.fetchone())["cutoff"]
            await cur.execute("DELETE FROM events WHERE ts < %s", (cutoff,))
            n_events = cur.rowcount
            await cur.execute("DELETE FROM metrics_minute WHERE bucket < %s", (cutoff,))
            n_rollup = cur.rowcount
        return {"cutoff": cutoff.isoformat(), "events_deleted": n_events, "rollup_rows_deleted": n_rollup}

    # ---------------------------------------------------------------- alert rules
    async def list_alert_rules(self):
        rows = await self._all(self.app, "SELECT * FROM alert_rules ORDER BY id")
        return [{**r, "created_at": r["created_at"].isoformat()} for r in rows]

    async def get_alert_rule(self, rule_id):
        return await self._one(self.app, "SELECT * FROM alert_rules WHERE id = %s", (rule_id,))

    @staticmethod
    async def insert_alert_rule(cur, event_type, metric, op, threshold, window_minutes, created_by):
        await cur.execute("""
            INSERT INTO alert_rules (event_type, metric, op, threshold, window_minutes, created_by)
            VALUES (%s, %s, %s, %s, %s, %s) RETURNING id""",
                          (event_type, metric, op, threshold, window_minutes, created_by))
        return (await cur.fetchone())["id"]

    @staticmethod
    async def delete_alert_rule(cur, rule_id):
        await cur.execute("DELETE FROM alert_rules WHERE id = %s RETURNING event_type, metric, threshold", (rule_id,))
        return await cur.fetchone()

    # ---------------------------------------------------------------- annotations
    async def list_annotations(self, window_hours):
        rows = await self._all(self.app, """
            SELECT id, ts, event_type, note, created_by FROM annotations
            WHERE ts >= now() - make_interval(hours => %s) ORDER BY ts""", (window_hours,))
        return [{**r, "ts": r["ts"].isoformat()} for r in rows]

    @staticmethod
    async def insert_annotation(cur, note, event_type, created_by):
        await cur.execute("INSERT INTO annotations (note, event_type, created_by) VALUES (%s, %s, %s) RETURNING id, ts",
                          (note, event_type, created_by))
        r = await cur.fetchone()
        return {"id": r["id"], "ts": r["ts"].isoformat()}

    # ---------------------------------------------------------------- api keys
    async def list_api_keys(self):
        rows = await self._all(self.app, """
            SELECT id, name, key_prefix, created_by, created_at, revoked_at FROM api_keys ORDER BY id""")
        return [{**r, "created_at": r["created_at"].isoformat(),
                 "revoked_at": r["revoked_at"].isoformat() if r["revoked_at"] else None} for r in rows]

    @staticmethod
    async def insert_api_key(cur, name, created_by):
        plaintext = "ak_" + secrets.token_urlsafe(24)
        await cur.execute("""
            INSERT INTO api_keys (name, key_prefix, key_hash, created_by) VALUES (%s, %s, %s, %s) RETURNING id""",
                          (name, plaintext[:10], hashlib.sha256(plaintext.encode()).hexdigest(), created_by))
        return (await cur.fetchone())["id"], plaintext

    @staticmethod
    async def revoke_api_key(cur, key_id):
        await cur.execute("""
            UPDATE api_keys SET revoked_at = coalesce(revoked_at, now()) WHERE id = %s
            RETURNING id, name, revoked_at""", (key_id,))
        return await cur.fetchone()

    async def find_api_key(self, name):
        return await self._one(self.app, "SELECT id, name, revoked_at FROM api_keys WHERE name = %s ORDER BY id DESC LIMIT 1",
                               (name,))

    # ---------------------------------------------------------------- idempotency
    async def run_idempotent(self, key, tool, caller, write):
        """Run `write(cur)` in an app-DB transaction at most once per idempotency key.

        The ledger row is inserted in the same transaction as the write, so either both commit or
        neither does. A concurrent duplicate blocks on the primary key until the first commits,
        then sees the conflict and returns the recorded result.
        """
        async with self.app.connection() as conn, conn.transaction(), conn.cursor() as cur:
            if key:
                await cur.execute("""
                    INSERT INTO tool_invocations (idempotency_key, tool, caller) VALUES (%s, %s, %s)
                    ON CONFLICT (idempotency_key) DO NOTHING RETURNING 1""", (key, tool, caller))
                if await cur.fetchone() is None:
                    await cur.execute("SELECT result FROM tool_invocations WHERE idempotency_key = %s", (key,))
                    return {**((await cur.fetchone())["result"] or {}), "replayed": True}
            result = await write(cur)
            if key:
                await cur.execute("UPDATE tool_invocations SET result = %s WHERE idempotency_key = %s",
                                  (Jsonb(_redact(result)), key))
            return result

    async def run_idempotent_external(self, key, tool, caller, op):
        """For writes outside the app DB (SpiceDB, the analytics DB). They cannot share a
        transaction with the ledger, so `op` itself must be safe to repeat (TOUCH / DELETE
        relationships, delete-older-than). The ledger only stops the repeat from being visible."""
        if key:
            r = await self._one(self.app, "SELECT result FROM tool_invocations WHERE idempotency_key = %s", (key,))
            if r is not None:
                return {**(r["result"] or {}), "replayed": True}
        result = await op()
        if key:
            async with self.app.connection() as conn, conn.cursor() as cur:
                await cur.execute("""
                    INSERT INTO tool_invocations (idempotency_key, tool, caller, result) VALUES (%s, %s, %s, %s)
                    ON CONFLICT (idempotency_key) DO NOTHING""", (key, tool, caller, Jsonb(_redact(result))))
        return result

    # ---------------------------------------------------------------- audit + threads
    async def audit(self, user, tool, permission, allowed, reason=None):
        async with self.app.connection() as conn, conn.cursor() as cur:
            await cur.execute("""
                INSERT INTO audit_log (user_id, tool, permission, allowed, reason) VALUES (%s, %s, %s, %s, %s)""",
                              (user, tool, permission, allowed, reason))

    async def claim_thread(self, thread_id, user, title=""):
        """Create the thread for `user` if new; return its owner either way."""
        async with self.app.connection() as conn, conn.cursor() as cur:
            await cur.execute("""
                INSERT INTO threads (thread_id, user_id, title) VALUES (%s, %s, %s)
                ON CONFLICT (thread_id) DO NOTHING""", (thread_id, user, title[:80]))
            await cur.execute("SELECT user_id FROM threads WHERE thread_id = %s", (thread_id,))
            return (await cur.fetchone())["user_id"]

    async def set_thread_title(self, thread_id, title):
        async with self.app.connection() as conn, conn.cursor() as cur:
            await cur.execute("UPDATE threads SET title = %s WHERE thread_id = %s AND title = ''",
                              (title[:80], thread_id))

    async def list_threads(self, user, limit=50):
        rows = await self._all(self.app, """
            SELECT thread_id, title, created_at FROM threads WHERE user_id = %s
            ORDER BY created_at DESC LIMIT %s""", (user, limit))
        return [{**r, "created_at": r["created_at"].isoformat()} for r in rows]
