"""Apply the SQL files in db/ (all statements are idempotent: CREATE ... IF NOT EXISTS)."""
import os
import pathlib

import psycopg


def sql_dir() -> pathlib.Path:
    if os.environ.get("SQL_DIR"):
        return pathlib.Path(os.environ["SQL_DIR"])
    here = pathlib.Path(__file__).resolve()
    for cand in (here.parents[1] / "db", here.parents[2] / "db"):  # image layout, repo layout
        if cand.is_dir():
            return cand
    raise FileNotFoundError("db/ directory with the SQL files not found; set SQL_DIR")


async def apply_sql(dsn: str, files: list[str]):
    async with await psycopg.AsyncConnection.connect(dsn, autocommit=True) as conn:
        for name in files:
            await conn.execute((sql_dir() / name).read_text())
