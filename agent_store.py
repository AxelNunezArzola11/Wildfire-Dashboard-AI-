"""
agent_store.py — SQLite persistence layer for autonomous agent runs.

Schema
------
Table ``agent_runs`` in the same wildfire_cache.db used by ingestor.py.

Columns (PostgreSQL-compatible names; stored in SQLite):
    run_id          TEXT  PRIMARY KEY  (uuid4 hex)
    country         TEXT
    min_frp         REAL
    started_at      TEXT  (ISO-8601 UTC)
    finished_at     TEXT  (ISO-8601 UTC, nullable)
    status          TEXT  ('running' | 'success' | 'failed' | 'partial')
    risk_metrics    TEXT  (JSON)
    forecast_top10  TEXT  (JSON list of top-10 GridCell dicts)
    summary_text    TEXT
    forecast_text   TEXT
    guardrail_verdict TEXT ('pass' | 'corrected' | 'unverified' | 'n/a')
    error_message   TEXT  (nullable)
    latency_seconds REAL  (nullable)
    artifacts_dir   TEXT  (nullable — path to agent_artifacts/{run_id}/ folder)

Note: PostgreSQL was specified but is not available in this environment;
SQLite provides an identical interface (same column names, same JSON fields).
To migrate to PostgreSQL: swap _get_conn() for a psycopg2 connection and
change CREATE TABLE IF NOT EXISTS syntax — the rest of the code is unchanged.

Public API
----------
init_schema()                          — create table if absent
insert_run(run_id, country, min_frp, started_at) -> None
update_run(run_id, **fields)           -> None
get_latest_run(country) -> dict | None
get_recent_runs(n=20)  -> list[dict]

Migration note: ``artifacts_dir`` was added after the initial schema.  A
``ALTER TABLE`` guard in ``init_schema()`` adds it to any existing DB that
pre-dates this column so callers never need to run a separate migration script.
"""

import json
import logging
import sqlite3
import uuid
from datetime import datetime, timezone
from typing import Any

import config

logger = logging.getLogger(__name__)

_CREATE_SQL = """
CREATE TABLE IF NOT EXISTS agent_runs (
    run_id            TEXT PRIMARY KEY,
    country           TEXT NOT NULL,
    min_frp           REAL NOT NULL,
    started_at        TEXT NOT NULL,
    finished_at       TEXT,
    status            TEXT NOT NULL DEFAULT 'running',
    risk_metrics      TEXT,
    forecast_top10    TEXT,
    summary_text      TEXT,
    forecast_text     TEXT,
    guardrail_verdict TEXT,
    error_message     TEXT,
    latency_seconds   REAL,
    artifacts_dir     TEXT
)
"""

# ALTER TABLE guard — adds artifacts_dir to databases created before this column
# was introduced.  SQLite silently ignores 'duplicate column' errors so this is
# safe to run on fresh DBs too.
_MIGRATE_SQL = "ALTER TABLE agent_runs ADD COLUMN artifacts_dir TEXT"

_IDX_SQL = (
    "CREATE INDEX IF NOT EXISTS idx_agent_runs_country_started "
    "ON agent_runs (country, started_at DESC)"
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def _row_to_dict(row: sqlite3.Row) -> dict:
    d = dict(row)
    for jf in ("risk_metrics", "forecast_top10"):
        if d.get(jf):
            try:
                d[jf] = json.loads(d[jf])
            except (json.JSONDecodeError, TypeError):
                pass
    return d


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_schema() -> None:
    """Create the agent_runs table and index if they don't exist yet.

    Also runs a one-time column-addition migration for ``artifacts_dir`` so
    that existing databases created before this column was added continue to
    work without any manual intervention.
    """
    with _conn() as c:
        c.execute(_CREATE_SQL)
        c.execute(_IDX_SQL)
        # Best-effort migration: ignore OperationalError if column already exists.
        try:
            c.execute(_MIGRATE_SQL)
        except sqlite3.OperationalError:
            pass  # column already present
        c.commit()
    logger.debug("agent_store: schema initialised.")


def new_run_id() -> str:
    return uuid.uuid4().hex


def insert_run(run_id: str, country: str, min_frp: float, started_at: str) -> None:
    """Insert a new row with status='running'."""
    with _conn() as c:
        c.execute(
            """INSERT INTO agent_runs (run_id, country, min_frp, started_at, status)
               VALUES (?, ?, ?, ?, 'running')""",
            (run_id, country, min_frp, started_at),
        )
        c.commit()


def update_run(run_id: str, **fields: Any) -> None:
    """
    Update arbitrary columns on an existing row.

    JSON-serialisable objects (dict/list) are automatically serialised.
    """
    if not fields:
        return
    serialised = {}
    for k, v in fields.items():
        if isinstance(v, (dict, list)):
            serialised[k] = json.dumps(v, default=str)
        else:
            serialised[k] = v

    set_clause = ", ".join(f"{k} = ?" for k in serialised)
    values = list(serialised.values()) + [run_id]
    with _conn() as c:
        c.execute(f"UPDATE agent_runs SET {set_clause} WHERE run_id = ?", values)
        c.commit()


def get_latest_run(country: str) -> dict | None:
    """Return the most recent *successful* run for *country*, or None."""
    with _conn() as c:
        row = c.execute(
            """SELECT * FROM agent_runs
               WHERE country = ? AND status = 'success'
               ORDER BY started_at DESC LIMIT 1""",
            (country,),
        ).fetchone()
    return _row_to_dict(row) if row else None


def get_recent_runs(n: int = 20) -> list[dict]:
    """Return the *n* most recent runs across all countries, newest first."""
    with _conn() as c:
        rows = c.execute(
            "SELECT * FROM agent_runs ORDER BY started_at DESC LIMIT ?", (n,)
        ).fetchall()
    return [_row_to_dict(r) for r in rows]
