"""Fill the standalone analytics tables with deterministic demo data, relative to now.

Six hours of traffic with two planted incidents the agent can find:
  * contract_upload error rate jumps from ~1% to ~22% in the last 20 minutes;
  * api_call latency is ~3.5x higher in the last 30 minutes.
Plus a few hundred events from 40 days ago, so "purge events older than 30 days" has work to do.

Refuses to touch a table that holds non-seed events (for example the dashboard's database in
integrated mode) unless --force is given.

Usage: python -m agentplat.seed [--hours 6] [--rate 2.5]
"""
import argparse
import asyncio
import io
import random
import time

import psycopg

from .config import get_settings
from .schema import apply_sql

TYPES = [("page_view", 0.55, 40, 0.002), ("contract_upload", 0.15, 180, 0.01),
         ("analysis_complete", 0.15, 2200, 0.0), ("analysis_error", 0.02, 900, 1.0), ("api_call", 0.13, 120, 0.01)]
N_USERS = 500


def generate(hours, rate, now, seed=7):
    rng = random.Random(seed)
    weights = [1 / (i + 1) ** 0.8 for i in range(N_USERS)]  # a few heavy users, a long tail
    users = [f"u{i}" for i in range(N_USERS)]
    rng.shuffle(users)
    n = int(hours * 3600 * rate)
    start = now - hours * 3600
    rows = []
    for i in range(n):
        ts = start + (i + rng.random()) / rate
        etype, _, median, perr = rng.choices(TYPES, weights=[t[1] for t in TYPES])[0]
        age_min = (now - ts) / 60
        if etype == "contract_upload" and age_min <= 20:
            perr = 0.22
        if etype == "api_call" and age_min <= 30:
            median *= 3.5
        latency = int(min(rng.lognormvariate(0, 0.5) * median, 600_000))
        rows.append((f"seed-{i}", ts, etype, rng.choices(users, weights)[0], latency, rng.random() < perr))
    for j in range(300):
        ts = now - 40 * 86400 + j * 10
        rows.append((f"seed-old-{j}", ts, "page_view", rng.choice(users), 40, False))
    return rows


async def seed(dsn, hours=6.0, rate=2.5, force=False):
    await apply_sql(dsn, ["001_analytics.sql"])
    rows = generate(hours, rate, time.time())
    async with await psycopg.AsyncConnection.connect(dsn) as conn:
        async with conn.cursor() as cur:
            await cur.execute("SELECT count(*) FROM events WHERE event_id NOT LIKE 'seed-%%'")
            foreign = (await cur.fetchone())[0]
            if foreign and not force:
                raise SystemExit(f"events holds {foreign} non-seed rows (is this the dashboard database?). "
                                 "Refusing to overwrite; pass --force if you mean it.")
            await cur.execute("TRUNCATE events, metrics_minute")
            buf = io.StringIO()
            for eid, ts, etype, user, lat, err in rows:
                buf.write(f"{eid}\t{time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime(ts))}.{int(ts % 1 * 1e6):06d}+00"
                          f"\t{etype}\t{user}\t{lat}\t{'t' if err else 'f'}\tt\n")
            async with cur.copy("COPY events (event_id, ts, event_type, user_id, latency_ms, is_error, simulated) "
                                "FROM STDIN") as cp:
                await cp.write(buf.getvalue())
            await cur.execute("""
                INSERT INTO metrics_minute (bucket, event_type, event_count, latency_sum_ms, error_count)
                SELECT date_trunc('minute', ts), event_type, count(*), sum(latency_ms), sum(is_error::int)
                FROM events GROUP BY 1, 2""")
            await cur.execute("ANALYZE events")
        await conn.commit()
    return len(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hours", type=float, default=6)
    ap.add_argument("--rate", type=float, default=2.5, help="events per second")
    ap.add_argument("--force", action="store_true")
    a = ap.parse_args()
    n = asyncio.run(seed(get_settings().analytics_database_url, a.hours, a.rate, a.force))
    print(f"seeded {n:,} events")


if __name__ == "__main__":
    main()
