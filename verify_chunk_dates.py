"""
verify_chunk_dates.py
=====================
Audit the per-chunk URL, actual returned date range, and row count for a
multi-day FIRMS pull, then verify the merged dataset covers all requested
days with no gaps.

FIRMS API semantics (verified empirically):
    …/{days}/{YYYY-MM-DD}  →  returns [date … date+days-1]
    The supplied date is a START date, not an end date.

Run:
    python verify_chunk_dates.py [--country Angola] [--days 30]
"""

import argparse
import io
import os
import sys
from datetime import date, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv(override=True)

FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
PRIMARY_SOURCE = "VIIRS_SNPP_NRT"
FIRMS_MAX_DAYS = 5

try:
    import config as _cfg
    COUNTRY_BBOX = _cfg.COUNTRY_BBOX
    FIRMS_MAP_KEY = _cfg.FIRMS_MAP_KEY
except Exception:
    FIRMS_MAP_KEY = os.environ.get("FIRMS_MAP_KEY", "")
    COUNTRY_BBOX = {}


def fetch_chunk(bbox: str, window: int, start_date: date, key: str) -> tuple[str, pd.DataFrame]:
    """Return (exact_url, dataframe) for one dated chunk request."""
    url = (
        f"{FIRMS_BASE_URL}/{key}/{PRIMARY_SOURCE}/{bbox}"
        f"/{window}/{start_date.strftime('%Y-%m-%d')}"
    )
    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException as exc:
        print(f"  [NETWORK ERROR] {exc}", file=sys.stderr)
        return url, pd.DataFrame()

    if resp.status_code != 200:
        print(f"  [HTTP {resp.status_code}] {resp.text[:200]}", file=sys.stderr)
        return url, pd.DataFrame()

    try:
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as exc:
        print(f"  [PARSE ERROR] {exc}", file=sys.stderr)
        return url, pd.DataFrame()

    for src, dst in [("bright_ti4", "brightness"), ("bright_t31", "brightness")]:
        if src in df.columns:
            df.rename(columns={src: dst}, inplace=True)

    return url, df


def run_audit(country: str, days: int) -> None:
    if not FIRMS_MAP_KEY:
        print("ERROR: FIRMS_MAP_KEY is not set. Check your .env file.", file=sys.stderr)
        sys.exit(1)

    if country not in COUNTRY_BBOX:
        print(
            f"ERROR: '{country}' not in COUNTRY_BBOX.\n"
            f"Available: {sorted(COUNTRY_BBOX.keys())}",
            file=sys.stderr,
        )
        sys.exit(1)

    bbox = COUNTRY_BBOX[country]
    today = date.today()
    overall_start = today - timedelta(days=days - 1)

    # Chunk formula (start-date semantics):
    #   chunk_start = today - (days_fetched + window - 1)
    #   chunk_end   = chunk_start + window - 1
    # This gives non-overlapping windows covering [today-days+1 … today].

    print("=" * 72)
    print(f"FIRMS chunk audit  country={country}  days={days}  today={today}")
    print(f"Expected total window: {overall_start} → {today}  ({days} days)")
    print(f"FIRMS date param semantics: START date (window = [date … date+N-1])")
    print("=" * 72)
    print()

    all_chunks: list[pd.DataFrame] = []
    days_fetched = 0
    chunk_idx = 0

    while days_fetched < days:
        remaining = days - days_fetched
        window = min(remaining, FIRMS_MAX_DAYS)
        chunk_start = today - timedelta(days=days_fetched + window - 1)
        chunk_end_expected = chunk_start + timedelta(days=window - 1)

        url, df = fetch_chunk(bbox, window, chunk_start, FIRMS_MAP_KEY)
        masked_url = url.replace(FIRMS_MAP_KEY, "<MAP_KEY>")

        print(f"── Chunk {chunk_idx} ──────────────────────────────────────────────────────")
        print(f"  (a) URL sent:            {masked_url}")
        print(f"      Stated query window: {chunk_start} → {chunk_end_expected}  ({window} days)")

        if df.empty or "acq_date" not in df.columns:
            print(f"  (b) acq_date min/max:    [NO DATA RETURNED]")
            print(f"  (c) Row count:           0")
        else:
            df["acq_date"] = pd.to_datetime(df["acq_date"], errors="coerce")
            actual_min = df["acq_date"].min().date()
            actual_max = df["acq_date"].max().date()
            row_count = len(df)
            print(f"  (b) acq_date min/max:    {actual_min} → {actual_max}")
            print(f"  (c) Row count:           {row_count}")

            outside_low  = actual_min < chunk_start
            outside_high = actual_max > chunk_end_expected
            if outside_low or outside_high:
                print(f"  *** MISMATCH: actual dates fall outside stated window!")
                if outside_low:
                    print(f"      actual_min={actual_min} < expected_start={chunk_start}")
                if outside_high:
                    print(f"      actual_max={actual_max} > expected_end={chunk_end_expected}")
            else:
                print(f"  ✓  Actual dates are inside the stated window.")

            all_chunks.append(df)

        print()
        days_fetched += window
        chunk_idx += 1

    # ── Merged dataset analysis ──────────────────────────────────────────────
    print("=" * 72)
    print("MERGED DATASET ANALYSIS")
    print("=" * 72)

    if not all_chunks:
        print("No data returned for any chunk — cannot verify coverage.")
        return

    combined = pd.concat(all_chunks, ignore_index=True)
    pre_dedup = len(combined)

    dedup_cols = ["acq_date", "acq_time", "latitude", "longitude"]
    present = [c for c in dedup_cols if c in combined.columns]
    if present:
        combined = combined.drop_duplicates(subset=present)
    post_dedup = len(combined)

    combined["acq_date"] = pd.to_datetime(combined["acq_date"], errors="coerce")
    merged_min = combined["acq_date"].min().date()
    merged_max = combined["acq_date"].max().date()
    dates_present = sorted(combined["acq_date"].dropna().dt.date.unique())

    print(f"  Rows before dedup:   {pre_dedup}")
    print(f"  Rows after dedup:    {post_dedup}")
    print(f"  Duplicates removed:  {pre_dedup - post_dedup}")
    print(f"  Merged min acq_date: {merged_min}")
    print(f"  Merged max acq_date: {merged_max}")
    print()

    expected_dates = set()
    d = overall_start
    while d <= today:
        expected_dates.add(d)
        d += timedelta(days=1)

    dates_present_set = set(dates_present)
    missing_dates = sorted(expected_dates - dates_present_set)
    extra_dates   = sorted(dates_present_set - expected_dates)

    if missing_dates:
        print(f"  *** COVERAGE GAPS — dates expected but NOT present in merged data:")
        for d in missing_dates:
            print(f"      {d}")
    else:
        print(f"  ✓  All {days} expected dates are present in merged data.")

    if extra_dates:
        print(f"  *** EXTRA DATES — dates present but OUTSIDE the {days}-day window:")
        for d in extra_dates:
            print(f"      {d}")
    else:
        print(f"  ✓  No dates outside the {days}-day window.")

    print()
    print(f"  Dates present ({len(dates_present)} unique):")
    for d in dates_present:
        row_count_d = (combined["acq_date"].dt.date == d).sum()
        marker = " ← OUTSIDE WINDOW" if d not in expected_dates else ""
        print(f"      {d}  ({row_count_d} rows){marker}")

    print()
    print("=" * 72)
    if not missing_dates and not extra_dates and (pre_dedup - post_dedup) == 0:
        print("VERDICT: ✓ Code is correct. Zero gaps, zero extra dates, zero duplicates.")
    elif not missing_dates and not extra_dates:
        dupes = pre_dedup - post_dedup
        print(
            f"VERDICT: ✓ {dupes} duplicate(s) removed (midnight-straddling overpasses).\n"
            "         Date coverage is complete. No gaps or extra dates."
        )
    else:
        print(
            "VERDICT: ✗ REAL ISSUE DETECTED — see COVERAGE GAPS / EXTRA DATES above.\n"
            "         Review the URL output per chunk to identify the broken window."
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Audit FIRMS chunk date windows")
    parser.add_argument("--country", default="Angola", help="Country key (default: Angola)")
    parser.add_argument("--days", type=int, default=30, help="Days window (default: 30)")
    args = parser.parse_args()
    run_audit(args.country, args.days)
