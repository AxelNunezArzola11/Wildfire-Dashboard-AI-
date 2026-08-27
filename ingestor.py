"""
ingestor.py — NASA FIRMS Area API client + SQLite TTL cache.

Public API
----------
get_fire_data(country, days, min_frp, force_refresh=False) -> pd.DataFrame

The returned DataFrame always contains these columns (even when empty):
    latitude, longitude, brightness, frp, acq_date, acq_time, confidence, instrument
"""

import io
import logging
import sqlite3
import time
from datetime import date, datetime, timedelta, timezone

import pandas as pd
import requests

import config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
PRIMARY_SOURCE = "VIIRS_SNPP_NRT"

# FIRMS NRT area API caps at 5 days per request for the area CSV endpoint.
# For longer windows we issue multiple chunked requests and deduplicate.
FIRMS_MAX_DAYS = 5

# Canonical output columns — rename from FIRMS CSV names where they differ.
OUTPUT_COLUMNS = [
    "latitude",
    "longitude",
    "brightness",
    "frp",
    "acq_date",
    "acq_time",
    "confidence",
    "instrument",
]

# FIRMS CSV column → output column name (only entries that differ).
_FIRMS_RENAME = {
    "bright_ti4": "brightness",   # VIIRS primary thermal band
    "bright_t31": "brightness",   # MODIS thermal band
}

# ---------------------------------------------------------------------------
# SQLite helpers
# ---------------------------------------------------------------------------

def _get_conn() -> sqlite3.Connection:
    """Open (and return) a connection to the shared SQLite cache database."""
    conn = sqlite3.connect(config.DB_PATH)
    return conn


def _init_schema(conn: sqlite3.Connection) -> None:
    """Create the fire_cache table if it doesn't exist yet."""
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS fire_cache (
            latitude    REAL,
            longitude   REAL,
            brightness  REAL,
            frp         REAL,
            acq_date    TEXT,
            acq_time    TEXT,
            confidence  TEXT,
            instrument  TEXT,
            fetched_at  TEXT,
            cache_key   TEXT
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_fire_cache_key ON fire_cache (cache_key)"
    )
    conn.commit()


def _is_cache_fresh(conn: sqlite3.Connection, cache_key: str) -> bool:
    """Return True if the most recent fetched_at for cache_key is within the TTL."""
    row = conn.execute(
        "SELECT MAX(fetched_at) FROM fire_cache WHERE cache_key = ?",
        (cache_key,),
    ).fetchone()

    if row is None or row[0] is None:
        return False

    fetched_at = datetime.fromisoformat(row[0])
    # Ensure comparison is timezone-aware.
    if fetched_at.tzinfo is None:
        fetched_at = fetched_at.replace(tzinfo=timezone.utc)

    age_minutes = (datetime.now(timezone.utc) - fetched_at).total_seconds() / 60
    return age_minutes < config.CACHE_TTL_MINUTES


def _read_from_cache(conn: sqlite3.Connection, cache_key: str) -> pd.DataFrame:
    """Load cached rows for cache_key and return as a DataFrame."""
    df = pd.read_sql_query(
        "SELECT * FROM fire_cache WHERE cache_key = ?",
        conn,
        params=(cache_key,),
    )
    # Drop internal cache columns before returning.
    return df.drop(columns=["fetched_at", "cache_key"], errors="ignore")


def _write_to_cache(conn: sqlite3.Connection, cache_key: str, df: pd.DataFrame) -> None:
    """Replace existing rows for cache_key with df, tagging each row with now()."""
    conn.execute("DELETE FROM fire_cache WHERE cache_key = ?", (cache_key,))

    if df.empty:
        conn.commit()
        return

    now = datetime.now(timezone.utc).isoformat()
    rows = df[OUTPUT_COLUMNS].copy()
    rows["fetched_at"] = now
    rows["cache_key"] = cache_key
    rows.to_sql("fire_cache", conn, if_exists="append", index=False)
    conn.commit()


# ---------------------------------------------------------------------------
# FIRMS fetch
# ---------------------------------------------------------------------------

def _fetch_single_window(
    country: str, days: int, source: str, start_date: date | None = None
) -> pd.DataFrame:
    """
    Issue one FIRMS Area API request for exactly `days` days (must be ≤ FIRMS_MAX_DAYS).

    When *start_date* is given the URL takes the form ``…/{days}/{YYYY-MM-DD}``,
    which anchors the window to [start_date … start_date+days-1].
    Without a date the API defaults to the most-recent `days` days.

    NOTE: The FIRMS Area API date parameter is a START date, not an end date.
    Verified empirically: /5/2026-08-01 returns 2026-08-01 → 2026-08-05.

    Returns an empty DataFrame with OUTPUT_COLUMNS on any API or parse error.
    Does NOT expose the key in log messages.
    """
    bbox = config.COUNTRY_BBOX[country]          # already validated by caller
    if start_date is not None:
        url = (
            f"{FIRMS_BASE_URL}/{config.FIRMS_MAP_KEY}/{source}/{bbox}"
            f"/{days}/{start_date.strftime('%Y-%m-%d')}"
        )
    else:
        url = f"{FIRMS_BASE_URL}/{config.FIRMS_MAP_KEY}/{source}/{bbox}/{days}"

    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException as exc:
        logger.warning("FIRMS network error for %s (days=%d): %s", country, days, exc)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    if resp.status_code != 200:
        body = resp.text[:200]
        if "MAP_KEY" in body or resp.status_code == 400:
            # Likely a key problem — surface clearly without echoing the key.
            logger.error(
                "FIRMS rejected the request for %s (HTTP %d). "
                "Check that FIRMS_MAP_KEY in your .env is correct and not a placeholder. "
                "Response: %s",
                country, resp.status_code, body,
            )
        else:
            logger.warning(
                "FIRMS returned HTTP %d for %s (days=%d): %s",
                resp.status_code, country, days, body,
            )
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    try:
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as exc:  # noqa: BLE001
        logger.warning("Failed to parse FIRMS CSV for %s (days=%d): %s", country, days, exc)
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    if df.empty:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    # Normalise column names to OUTPUT_COLUMNS.
    df.rename(columns={k: v for k, v in _FIRMS_RENAME.items() if k in df.columns}, inplace=True)

    # Ensure all expected columns are present; fill missing with pd.NA.
    for col in OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA

    return df[OUTPUT_COLUMNS].reset_index(drop=True)


def _fetch_from_firms(country: str, days: int, source: str) -> pd.DataFrame:
    """
    Fetch fire data for `country` over the last `days` days.

    FIRMS NRT area API caps each request at FIRMS_MAX_DAYS (5) days.
    When `days` exceeds that limit we slice the total window into
    non-overlapping chunks using the dated URL form
    ``…/{chunk_days}/{YYYY-MM-DD}``, which anchors each request to a
    specific start date.  The chunks are then concatenated and deduplicated
    on (acq_date, acq_time, latitude, longitude) as a safety net for any
    satellite overpass that straddles a chunk boundary.

    FIRMS API semantics (verified empirically):
        ``…/{days}/{YYYY-MM-DD}`` returns [date … date+days-1]
        The supplied date is a START date, not an end date.

    Chunk layout for days=30 (FIRMS_MAX_DAYS=5), today=T:
        chunk 0: start=T-4,  5 days  → [T-4 … T]
        chunk 1: start=T-9,  5 days  → [T-9 … T-5]
        …
        chunk 5: start=T-29, 5 days  → [T-29 … T-25]

    Formula: chunk_start = today - timedelta(days=days_fetched + window - 1)

    Returns an empty DataFrame with OUTPUT_COLUMNS on any error.
    """
    if days <= FIRMS_MAX_DAYS:
        return _fetch_single_window(country, days, source)

    # Build non-overlapping date windows from newest → oldest.
    # The FIRMS date parameter is a start date: /days/YYYY-MM-DD → [date … date+days-1].
    # To cover [today-N+1 … today] we compute each chunk's start as:
    #   chunk_start = today - (days_fetched + window - 1)
    today = date.today()
    chunks: list[pd.DataFrame] = []
    days_fetched = 0
    while days_fetched < days:
        remaining = days - days_fetched
        window = min(remaining, FIRMS_MAX_DAYS)
        chunk_start = today - timedelta(days=days_fetched + window - 1)
        chunk = _fetch_single_window(country, window, source, start_date=chunk_start)
        logger.debug(
            "_fetch_from_firms chunk: start=%s window=%d rows=%d",
            chunk_start, window, len(chunk),
        )
        if not chunk.empty:
            chunks.append(chunk)
        days_fetched += window

    if not chunks:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    combined = pd.concat(chunks, ignore_index=True)
    pre_dedup = len(combined)

    # Deduplicate on the natural key: same detection at same time and place.
    # This is a safety net; with dated URLs there should be no overlap, but
    # satellite overpasses that span a midnight boundary can appear in both
    # adjacent chunks.
    dedup_cols = ["acq_date", "acq_time", "latitude", "longitude"]
    present = [c for c in dedup_cols if c in combined.columns]
    if present:
        combined = combined.drop_duplicates(subset=present)

    post_dedup = len(combined)
    if pre_dedup != post_dedup:
        logger.info(
            "_fetch_from_firms dedup removed %d duplicate rows (country=%s days=%d)",
            pre_dedup - post_dedup, country, days,
        )

    return combined.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_fire_data(
    country: str,
    days: int,
    min_frp: float,
    force_refresh: bool = False,
) -> tuple[pd.DataFrame, float | None]:
    """
    Return fire detections for *country* over the last *days* days plus timing info.

    Parameters
    ----------
    country       : Must be a key in config.COUNTRY_BBOX.
    days          : Positive integer. FIRMS caps single requests at 5 days; larger
                    values (e.g. 30) are satisfied by chunked requests with deduplication.
    min_frp       : Minimum Fire Radiative Power (MW) filter — applied after cache read.
    force_refresh : When True, bypass the TTL check and re-fetch from FIRMS.

    Returns
    -------
    (df, ingest_seconds)
        df             — pd.DataFrame with columns: latitude, longitude, brightness,
                         frp, acq_date, acq_time, confidence, instrument
        ingest_seconds — wall-clock seconds for the cold-cache FIRMS fetch, or None
                         when the result was served from cache.
    """
    if not config.FIRMS_MAP_KEY:
        raise RuntimeError(
            "FIRMS_MAP_KEY is not set. Register at "
            "https://firms.modaps.eosdis.nasa.gov/api/area/ and add it to your .env file."
        )

    if country not in config.COUNTRY_BBOX:
        raise ValueError(
            f"Country '{country}' is not in config.COUNTRY_BBOX. "
            f"Available countries: {sorted(config.COUNTRY_BBOX.keys())}"
        )

    source = PRIMARY_SOURCE
    cache_key = f"{country}_{days}_{source}"

    ingest_seconds: float | None = None

    conn = _get_conn()
    try:
        _init_schema(conn)

        if not force_refresh and _is_cache_fresh(conn, cache_key):
            df = _read_from_cache(conn, cache_key)
            logger.debug("Cache hit for key '%s'.", cache_key)
        else:
            t0 = time.monotonic()
            df = _fetch_from_firms(country, days, source)
            ingest_seconds = time.monotonic() - t0
            logger.info(
                "Cold-cache FIRMS ingest: country=%s days=%d rows=%d time=%.2fs",
                country, days, len(df), ingest_seconds,
            )
            _write_to_cache(conn, cache_key, df)
    finally:
        conn.close()

    # Apply min_frp filter after cache retrieval — never affects stored data.
    if not df.empty and "frp" in df.columns:
        df = df[pd.to_numeric(df["frp"], errors="coerce").fillna(0) >= min_frp]

    return df.reset_index(drop=True), ingest_seconds
