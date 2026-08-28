"""
wildfire_model_export.py — Standalone Wildfire Fire-Probability Model
======================================================================

This script is a self-contained reproduction of the XGBoost fire-probability
model used by the Wildfire Dashboard AI.  You can run it
independently to reproduce training, inspect feature engineering, and generate
24-hour fire-probability forecasts without the Streamlit UI.

Requirements
------------
    pip install pandas numpy scikit-learn xgboost shap requests python-dotenv

Credentials
-----------
Set these environment variables (or add them to a local .env file) before
running.  DO NOT hardcode real keys here.

    FIRMS_MAP_KEY=<your NASA FIRMS key>
        Register for free at https://firms.modaps.eosdis.nasa.gov/api/area/

The watsonx / LLM keys (WATSONX_API_KEY, WATSONX_PROJECT_ID) are NOT needed
by this script — the model logic is fully local.

Usage
-----
    python wildfire_model_export.py --country Angola --days 7 --min-frp 10
    python wildfire_model_export.py --country Brazil --days 30 --min-frp 10 --output brazil_30d.csv

Outputs a CSV of per-cell fire probabilities to stdout (or redirect to a file).
"""

from __future__ import annotations

import argparse
import io
import logging
import math
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd
import requests
from xgboost import XGBClassifier

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("wildfire_model")

# ---------------------------------------------------------------------------
# Credentials — read from environment variables only, never hardcoded
# ---------------------------------------------------------------------------

# Register for a free FIRMS key at:
#   https://firms.modaps.eosdis.nasa.gov/api/area/
# Then run:  export FIRMS_MAP_KEY=<your_key>
FIRMS_MAP_KEY: str = os.environ.get("FIRMS_MAP_KEY", "")

# Optionally load from a .env file in the current directory
try:
    from dotenv import load_dotenv  # type: ignore

    load_dotenv()
    FIRMS_MAP_KEY = FIRMS_MAP_KEY or os.environ.get("FIRMS_MAP_KEY", "")
except ImportError:
    pass  # python-dotenv not installed — rely on real env vars

# ---------------------------------------------------------------------------
# Country bounding boxes  (W, S, E, N)
# Imported from the shared, dependency-free registry — no config.py needed.
# ---------------------------------------------------------------------------
from country_bboxes import COUNTRY_BBOX

# ---------------------------------------------------------------------------
# Model / forecast constants
# ---------------------------------------------------------------------------

FORECAST_HORIZON_HOURS = 24  # default; overridden per-call via horizon_days
FORECAST_GRID_DEG = 0.25          # degrees per grid cell edge (~28 km at equator)
MAX_GRID_CELLS = 200               # cap before uniform subsampling
MIN_LABELLED_SAMPLES = 10         # minimum pseudo-labelled samples to use GBT

# All feature columns in the feature matrix
FEATURE_COLS = [
    "hist_fire_count_7d",
    "hist_frp_mean_7d",
    "hist_fire_count_24h",
    "days_since_last_fire",
    "temp_24h_mean",
    "humidity_24h_mean",
    "wind_24h_max",
    "precip_24h_sum",
    "soil_moisture_now",
]
FINAL_COLUMNS = ["lat", "lon"] + FEATURE_COLS

# Columns fed to the classifier — excludes columns that are collinear with
# pseudo-labels (would give the model trivially-perfect splits).
MODEL_FEATURE_COLS = [
    "temp_24h_mean",
    "humidity_24h_mean",
    "wind_24h_max",
    "precip_24h_sum",
    "soil_moisture_now",
]

# FIRMS API
FIRMS_BASE_URL = "https://firms.modaps.eosdis.nasa.gov/api/area/csv"
PRIMARY_SOURCE = "VIIRS_SNPP_NRT"
FIRMS_MAX_DAYS = 5

# Output columns from FIRMS
FIRMS_OUTPUT_COLUMNS = [
    "latitude", "longitude", "brightness", "frp",
    "acq_date", "acq_time", "confidence", "instrument",
]
_FIRMS_RENAME = {
    "bright_ti4": "brightness",
    "bright_t31": "brightness",
}

# Open-Meteo
OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"
_WEATHER_WORKERS = 10

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GridCell:
    lat_center: float
    lon_center: float
    fire_prob: float            # 0.0 – 1.0
    risk_band: str              # LOW | MEDIUM | HIGH | EXTREME
    feature_snapshot: dict
    historical_fire_count: int
    # SHAP weather-feature contributions for this cell (XGBoost path only).
    # Each entry: {"feature": str, "label": str, "shap": float, "pct": float}
    # sorted by abs(shap) descending.  Empty list when model_used=="Deterministic".
    shap_contribs: list = field(default_factory=list)


@dataclass
class ForecastResult:
    cells: list                 # list[GridCell], sorted descending by fire_prob
    top_risk_cells: list        # list[GridCell], top 5
    forecast_horizon_hours: int
    generated_at: str           # ISO UTC timestamp
    model_used: str             # "XGBoost" | "Deterministic"
    country: str


# ===========================================================================
# FIRMS data ingestion
# ===========================================================================


def _fetch_single_window(country: str, days: int, start_date: date | None = None) -> pd.DataFrame:
    """Issue one FIRMS Area API request (days must be ≤ FIRMS_MAX_DAYS).

    When *start_date* is supplied the URL takes the form …/{days}/{YYYY-MM-DD},
    anchoring the window to [start_date … start_date+days-1].

    NOTE: The FIRMS Area API date parameter is a START date, not an end date.
    Verified empirically: /5/2026-08-01 returns 2026-08-01 → 2026-08-05.

    Without a date the API defaults to the most-recent `days` days.
    """
    if not FIRMS_MAP_KEY:
        raise RuntimeError(
            "FIRMS_MAP_KEY is not set.\n"
            "Register at https://firms.modaps.eosdis.nasa.gov/api/area/ "
            "and export FIRMS_MAP_KEY=<your_key> before running."
        )
    bbox = COUNTRY_BBOX[country]
    if start_date is not None:
        url = (
            f"{FIRMS_BASE_URL}/{FIRMS_MAP_KEY}/{PRIMARY_SOURCE}/{bbox}"
            f"/{days}/{start_date.strftime('%Y-%m-%d')}"
        )
    else:
        url = f"{FIRMS_BASE_URL}/{FIRMS_MAP_KEY}/{PRIMARY_SOURCE}/{bbox}/{days}"
    logger.debug("Fetching FIRMS: days=%d start_date=%s country=%s", days, start_date, country)

    try:
        resp = requests.get(url, timeout=30)
    except requests.RequestException as exc:
        logger.warning("FIRMS network error: %s", exc)
        return pd.DataFrame(columns=FIRMS_OUTPUT_COLUMNS)

    if resp.status_code != 200:
        logger.warning("FIRMS HTTP %d for %s (days=%d)", resp.status_code, country, days)
        return pd.DataFrame(columns=FIRMS_OUTPUT_COLUMNS)

    try:
        df = pd.read_csv(io.StringIO(resp.text))
    except Exception as exc:
        logger.warning("Failed to parse FIRMS CSV: %s", exc)
        return pd.DataFrame(columns=FIRMS_OUTPUT_COLUMNS)

    if df.empty:
        return pd.DataFrame(columns=FIRMS_OUTPUT_COLUMNS)

    df.rename(
        columns={k: v for k, v in _FIRMS_RENAME.items() if k in df.columns},
        inplace=True,
    )
    for col in FIRMS_OUTPUT_COLUMNS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[FIRMS_OUTPUT_COLUMNS].reset_index(drop=True)


def fetch_fire_data(country: str, days: int, min_frp: float = 10.0) -> pd.DataFrame:
    """
    Fetch fire detections from NASA FIRMS for *country* over the last *days* days.

    Parameters
    ----------
    country  : Must be a key in COUNTRY_BBOX.
    days     : Positive integer (> 5 triggers multiple requests).
    min_frp  : Minimum Fire Radiative Power (MW) to include in results.

    Returns
    -------
    pd.DataFrame with columns:
        latitude, longitude, brightness, frp, acq_date, acq_time,
        confidence, instrument
    """
    if country not in COUNTRY_BBOX:
        raise ValueError(f"Unknown country '{country}'. Available: {sorted(COUNTRY_BBOX)}")

    if days <= FIRMS_MAX_DAYS:
        df = _fetch_single_window(country, days)
    else:
        # Build non-overlapping date windows from newest → oldest using dated URLs.
        # The FIRMS date parameter is a start date: /days/YYYY-MM-DD → [date … date+days-1].
        # Formula: chunk_start = today - (days_fetched + window - 1)
        today = date.today()
        chunks = []
        days_fetched = 0
        while days_fetched < days:
            remaining = days - days_fetched
            window = min(remaining, FIRMS_MAX_DAYS)
            chunk_start = today - timedelta(days=days_fetched + window - 1)
            chunk = _fetch_single_window(country, window, start_date=chunk_start)
            logger.debug(
                "fetch_fire_data chunk: start=%s window=%d rows=%d",
                chunk_start, window, len(chunk),
            )
            if not chunk.empty:
                chunks.append(chunk)
            days_fetched += window

        if not chunks:
            return pd.DataFrame(columns=FIRMS_OUTPUT_COLUMNS)

        df = pd.concat(chunks, ignore_index=True)
        dedup_cols = ["acq_date", "acq_time", "latitude", "longitude"]
        present = [c for c in dedup_cols if c in df.columns]
        if present:
            df = df.drop_duplicates(subset=present)
        df = df.reset_index(drop=True)

    if not df.empty and "frp" in df.columns:
        df = df[pd.to_numeric(df["frp"], errors="coerce").fillna(0) >= min_frp]

    return df.reset_index(drop=True)


# ===========================================================================
# Weather feature fetching (Open-Meteo — free, no key required)
# ===========================================================================


def _fetch_weather_point(lat: float, lon: float, horizon_hours: int = 24) -> dict:
    """Fetch weather features for (lat, lon) from Open-Meteo.

    Parameters
    ----------
    horizon_hours : Aggregation window — 24 (next 24 h) or 168 (next 7 days).
        Controls both the Open-Meteo forecast_days request and the slice length.
        Feature column names stay the same regardless of horizon.
    """
    nan_row = {
        "lat": lat, "lon": lon,
        "temp_24h_mean": float("nan"), "humidity_24h_mean": float("nan"),
        "wind_24h_max": float("nan"), "precip_24h_sum": float("nan"),
        "soil_moisture_now": float("nan"),
    }
    # Add 1 day of headroom so the slice is never truncated by API rounding.
    api_forecast_days = max(2, horizon_hours // 24 + 1)
    try:
        resp = requests.get(
            OPEN_METEO_URL,
            params={
                "latitude": lat,
                "longitude": lon,
                "hourly": (
                    "temperature_2m,relativehumidity_2m,"
                    "precipitation,windspeed_10m,soil_moisture_0_1cm"
                ),
                "forecast_days": api_forecast_days,
                "timezone": "auto",
            },
            timeout=10,
        )
        if resp.status_code != 200:
            return nan_row
        payload = resp.json()
    except Exception as exc:
        logger.warning("Open-Meteo error for (%.3f, %.3f): %s", lat, lon, exc)
        return nan_row

    try:
        hourly = payload["hourly"]
        times = hourly["time"]
        temp_arr = hourly["temperature_2m"]
        hum_arr = hourly["relativehumidity_2m"]
        prec_arr = hourly["precipitation"]
        wind_arr = hourly["windspeed_10m"]
        soil_arr = hourly["soil_moisture_0_1cm"]

        now_str = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:00")
        now_idx = 0
        for i, t in enumerate(times):
            if t <= now_str:
                now_idx = i
            else:
                break

        sl = slice(now_idx, min(now_idx + horizon_hours, len(times)))

        def _mean(arr):
            vals = [v for v in arr[sl] if v is not None]
            return sum(vals) / len(vals) if vals else float("nan")

        def _max(arr):
            vals = [v for v in arr[sl] if v is not None]
            return max(vals) if vals else float("nan")

        def _sum(arr):
            vals = [v for v in arr[sl] if v is not None]
            return sum(vals) if vals else float("nan")

        def _now(arr):
            v = arr[now_idx] if now_idx < len(arr) else None
            return float(v) if v is not None else float("nan")

        return {
            "lat": lat, "lon": lon,
            "temp_24h_mean": _mean(temp_arr),
            "humidity_24h_mean": _mean(hum_arr),
            "wind_24h_max": _max(wind_arr),
            "precip_24h_sum": _sum(prec_arr),
            "soil_moisture_now": _now(soil_arr),
        }
    except Exception as exc:
        logger.warning("Open-Meteo parse failed for (%.3f, %.3f): %s", lat, lon, exc)
        return nan_row


def get_weather_for_points(
    points: list[tuple[float, float]],
    horizon_hours: int = 24,
) -> pd.DataFrame:
    """Parallel-fetch weather features for all (lat, lon) grid points.

    Parameters
    ----------
    horizon_hours : Aggregation window — 24 (next 24 h) or 168 (next 7 days).
    """
    if not points:
        return pd.DataFrame()

    results: dict[int, dict] = {}

    def _fetch_and_tag(item):
        idx, lat, lon = item
        return idx, _fetch_weather_point(lat, lon, horizon_hours)

    tagged = [(i, lat, lon) for i, (lat, lon) in enumerate(points)]
    with ThreadPoolExecutor(max_workers=_WEATHER_WORKERS) as pool:
        futures = {pool.submit(_fetch_and_tag, item): item for item in tagged}
        for future in as_completed(futures):
            try:
                idx, data = future.result()
                results[idx] = data
            except Exception as exc:
                orig = futures[future]
                idx = orig[0]
                results[idx] = {"lat": orig[1], "lon": orig[2]}

    rows = [results[i] for i in range(len(points)) if i in results]
    return pd.DataFrame(rows) if rows else pd.DataFrame()


# ===========================================================================
# Grid builder
# ===========================================================================


def _build_grid(bbox_str: str) -> list[tuple[float, float]]:
    """
    Parse "W,S,E,N" and return (lat, lon) centroids at FORECAST_GRID_DEG intervals.
    Capped at MAX_GRID_CELLS via uniform subsampling.
    """
    w, s, e, n = (float(v) for v in bbox_str.split(","))
    step = FORECAST_GRID_DEG
    half = step / 2.0

    lats, lat = [], s + half
    while lat <= n - half + 1e-9:
        lats.append(round(lat, 6))
        lat += step

    lons, lon = [], w + half
    while lon <= e - half + 1e-9:
        lons.append(round(lon, 6))
        lon += step

    points = [(lat, lon) for lat in lats for lon in lons]

    if len(points) > MAX_GRID_CELLS:
        indices = [
            round(i * (len(points) - 1) / (MAX_GRID_CELLS - 1))
            for i in range(MAX_GRID_CELLS)
        ]
        points = [points[i] for i in indices]

    return points


# ===========================================================================
# Fire feature engineering
# ===========================================================================


def _compute_fire_features(
    lat: float,
    lon: float,
    fire_7d_df: pd.DataFrame,
    fire_short_df: pd.DataFrame,
) -> dict:
    """Return per-cell fire history features.

    Parameters
    ----------
    lat, lon      : Grid cell centre coordinates.
    fire_7d_df    : Always a 7-day FIRMS window (independent of --days).
                    Used for hist_fire_count_7d and hist_frp_mean_7d.
    fire_short_df : The sidebar/CLI-selected window (2, 7, or 30 days).
                    Used for hist_fire_count_24h and days_since_last_fire
                    (these describe recency, which should reflect the most
                    recent data regardless of window length).
    """
    half = FORECAST_GRID_DEG / 2.0
    empty_result = {
        "hist_fire_count_7d": 0,
        "hist_frp_mean_7d": 0.0,
        "hist_fire_count_24h": 0,
        "days_since_last_fire": 30,
    }

    # ── 7-day features (always from the fixed 7d window) ────────────────────
    if fire_7d_df.empty:
        hist_fire_count_7d = 0
        hist_frp_mean_7d = 0.0
    else:
        mask_7d = (
            (fire_7d_df["latitude"] >= lat - half) & (fire_7d_df["latitude"] < lat + half)
            & (fire_7d_df["longitude"] >= lon - half) & (fire_7d_df["longitude"] < lon + half)
        )
        cell_7d = fire_7d_df[mask_7d]
        hist_fire_count_7d = len(cell_7d)
        hist_frp_mean_7d = float(cell_7d["frp"].mean()) if hist_fire_count_7d > 0 else 0.0

    # ── Recency features (from the short/CLI-selected window) ────────────────
    if fire_short_df.empty:
        return {**empty_result,
                "hist_fire_count_7d": hist_fire_count_7d,
                "hist_frp_mean_7d": hist_frp_mean_7d}

    mask_short = (
        (fire_short_df["latitude"] >= lat - half) & (fire_short_df["latitude"] < lat + half)
        & (fire_short_df["longitude"] >= lon - half) & (fire_short_df["longitude"] < lon + half)
    )
    cell_short = fire_short_df[mask_short]

    if len(cell_short) > 0:
        dates = pd.to_datetime(cell_short["acq_date"], errors="coerce")
        latest = dates.max()
        if not pd.isna(latest):
            latest_str = latest.normalize().strftime("%Y-%m-%d")
            hist_fire_count_24h = int((cell_short["acq_date"] == latest_str).sum())
        else:
            hist_fire_count_24h = 0
        if pd.isna(latest):
            days_since_last_fire = 30
        else:
            today_dt = pd.Timestamp(datetime.now(timezone.utc).date())
            days_since_last_fire = int(min((today_dt - latest.normalize()).days, 30))
    else:
        hist_fire_count_24h = 0
        days_since_last_fire = 30

    return {
        "hist_fire_count_7d": hist_fire_count_7d,
        "hist_frp_mean_7d": hist_frp_mean_7d,
        "hist_fire_count_24h": hist_fire_count_24h,
        "days_since_last_fire": days_since_last_fire,
    }


# ===========================================================================
# Feature matrix builder
# ===========================================================================


def _build_feature_matrix(
    grid_points: list[tuple[float, float]],
    fire_7d_df: pd.DataFrame,
    fire_short_df: pd.DataFrame,
    horizon_hours: int = 24,
) -> pd.DataFrame:
    """Build the full (fire + weather) feature DataFrame, one row per grid cell.

    Parameters
    ----------
    grid_points   : List of (lat, lon) cell centroids.
    fire_7d_df    : Independent 7-day FIRMS window — used for 7d history features.
    fire_short_df : CLI/sidebar-selected window — used for recency features.
    horizon_hours : Weather aggregation window (24 or 168) passed to weather fetch.
    """
    logger.info("Fetching weather for %d grid points (horizon=%dh) ...",
                len(grid_points), horizon_hours)
    weather_df = get_weather_for_points(grid_points, horizon_hours)

    fire_rows = []
    for lat, lon in grid_points:
        feats = _compute_fire_features(lat, lon, fire_7d_df, fire_short_df)
        feats["lat"] = lat
        feats["lon"] = lon
        fire_rows.append(feats)
    fire_feat_df = pd.DataFrame(fire_rows)

    weather_cols = [
        "lat", "lon", "temp_24h_mean", "humidity_24h_mean",
        "wind_24h_max", "precip_24h_sum", "soil_moisture_now",
    ]
    if weather_df.empty:
        weather_df = pd.DataFrame([{"lat": lat, "lon": lon} for lat, lon in grid_points])
        for col in weather_cols[2:]:
            weather_df[col] = float("nan")

    for col in weather_cols:
        if col not in weather_df.columns:
            weather_df[col] = float("nan")

    merged = pd.merge(fire_feat_df, weather_df[weather_cols], on=["lat", "lon"], how="left")

    for col in FINAL_COLUMNS:
        if col not in merged.columns:
            merged[col] = float("nan")
    merged = merged[FINAL_COLUMNS]

    for col in merged.columns:
        if merged[col].isna().any():
            median = merged[col].median()
            merged[col] = merged[col].fillna(0.0 if pd.isna(median) else median)

    return merged


# ===========================================================================
# Risk band
# ===========================================================================


def _prob_to_risk_band(prob: float) -> str:
    if prob < 0.20:
        return "LOW"
    elif prob < 0.50:
        return "MEDIUM"
    elif prob < 0.75:
        return "HIGH"
    return "EXTREME"


# ===========================================================================
# Deterministic fallback scorer
# ===========================================================================


def _deterministic_score(row: pd.Series) -> float:
    """
    Weighted deterministic fire-risk score when insufficient pseudo-labels exist.
    Each term is normalised and clamped to [0, 1].  Returns a value in [0.0, 1.0].
    """

    def safe(val):
        try:
            v = float(val)
            return 0.0 if math.isnan(v) else v
        except (TypeError, ValueError):
            return 0.0

    score = (
        0.30 * min(safe(row.hist_fire_count_7d) / 10.0, 1.0)
        + 0.25 * max(1.0 - safe(row.humidity_24h_mean) / 100.0, 0.0)
        + 0.20 * min(max(safe(row.temp_24h_mean) - 20.0, 0.0) / 30.0, 1.0)
        + 0.15 * min(safe(row.wind_24h_max) / 60.0, 1.0)
        + 0.10 * max(1.0 - safe(row.precip_24h_sum) / 10.0, 0.0)
    )
    return float(np.clip(score, 0.0, 1.0))


# ===========================================================================
# Model training + inference
# ===========================================================================

# Human-readable labels for the 5 MODEL_FEATURE_COLS used in SHAP breakdowns.
_FEATURE_LABELS = {
    "temp_24h_mean":       "Temperature",
    "humidity_24h_mean":   "Humidity",
    "wind_24h_max":        "Wind speed",
    "precip_24h_sum":      "Precipitation",
    "soil_moisture_now":   "Soil moisture",
}


def _compute_shap_contribs(
    clf,
    X_all: np.ndarray,
    cell_row_indices: list,
) -> list:
    """
    Compute SHAP feature contributions for a subset of cells.

    Returns a list of length len(cell_row_indices).  Each element is a list of
    dicts: [{"feature": str, "label": str, "shap": float, "pct": float}, ...]
    sorted by abs(shap) descending, limited to the top 5 contributors.

    SHAP values are in log-odds space.  ``pct`` is each feature's share of the
    total absolute push, as a percentage.
    """
    import shap as _shap  # lazy import

    X_subset = X_all[cell_row_indices]
    explainer = _shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_subset)   # ndarray (n_cells, n_feats)

    result = []
    for row_sv in sv:
        total_abs = float(np.abs(row_sv).sum())
        contribs = []
        for j, col in enumerate(MODEL_FEATURE_COLS):
            shap_val = float(row_sv[j])
            pct = (shap_val / total_abs * 100.0) if total_abs > 1e-9 else 0.0
            contribs.append({
                "feature": col,
                "label":   _FEATURE_LABELS.get(col, col),
                "shap":    shap_val,
                "pct":     pct,
            })
        contribs.sort(key=lambda d: abs(d["shap"]), reverse=True)
        result.append(contribs[:5])
    return result


def _get_model_and_predictions(
    feature_df: pd.DataFrame,
) -> tuple:
    """
    Train (or fall back from) an XGBClassifier using pseudo-labels derived
    from historical fire activity, then predict fire probabilities for all cells.

    Pseudo-label strategy
    ---------------------
    - label 1  if hist_fire_count_24h > 0  (fire detected on the most recent day)
    - label 0  if hist_fire_count_7d  == 0  (no fire in the last 7 days)
    - discard  ambiguous rows

    The majority class is downsampled to at most 4× the minority class to
    prevent probability saturation on fire-dense regions.

    Falls back to _deterministic_score when:
    - fewer than MIN_LABELLED_SAMPLES pseudo-labelled rows exist, OR
    - only one class is represented, OR
    - fewer than MIN_POSITIVE_SAMPLES positive examples remain after downsampling
      (XGBoost cannot learn a meaningful split from 1–2 fire examples and returns
      a constant equal to the training base rate instead of a useful gradient).

    Returns
    -------
    (probabilities_array, model_used_str, fitted_clf_or_None, X_all_or_None)
    """
    MIN_POSITIVE_SAMPLES = 3

    label_1 = feature_df["hist_fire_count_24h"] > 0
    label_0 = feature_df["hist_fire_count_7d"] == 0

    labelled_idx = feature_df.index[label_1 | label_0]
    labels = np.where(label_1[labelled_idx], 1, 0)
    unique_labels = np.unique(labels)

    logger.info(
        "Pseudo-labels: %d rows, distribution: %s",
        len(labelled_idx),
        {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
    )

    if len(labelled_idx) < MIN_LABELLED_SAMPLES or len(unique_labels) < 2:
        logger.warning(
            "Using deterministic fallback: only %d pseudo-labelled samples "
            "(need >=%d with both classes).",
            len(labelled_idx),
            MIN_LABELLED_SAMPLES,
        )
        probs = np.array([_deterministic_score(row) for _, row in feature_df.iterrows()])
        return probs, "Deterministic", None, None

    MAX_IMBALANCE_RATIO = 4
    idx_pos = labelled_idx[labels == 1]
    idx_neg = labelled_idx[labels == 0]
    minority_n = min(len(idx_pos), len(idx_neg))
    majority_cap = minority_n * MAX_IMBALANCE_RATIO
    rng = np.random.default_rng(42)
    if len(idx_pos) > majority_cap:
        idx_pos = rng.choice(idx_pos, size=majority_cap, replace=False)
    if len(idx_neg) > majority_cap:
        idx_neg = rng.choice(idx_neg, size=majority_cap, replace=False)

    # Guard: if positive class is too sparse after downsampling, the tree
    # cannot learn a split and returns a constant equal to the training base
    # rate (e.g. 2/10 = 0.20 for all cells).  Fall back instead.
    if len(idx_pos) < MIN_POSITIVE_SAMPLES:
        logger.warning(
            "Using deterministic fallback: only %d positive training samples "
            "after downsampling (need >=%d). XGBoost would return a flat "
            "base-rate constant.",
            len(idx_pos),
            MIN_POSITIVE_SAMPLES,
        )
        probs = np.array([_deterministic_score(row) for _, row in feature_df.iterrows()])
        return probs, "Deterministic", None, None

    balanced_idx = np.concatenate([idx_pos, idx_neg])
    balanced_labels = np.concatenate([
        np.ones(len(idx_pos), dtype=int),
        np.zeros(len(idx_neg), dtype=int),
    ])

    X_train = feature_df.loc[balanced_idx, MODEL_FEATURE_COLS].values
    X_all = feature_df[MODEL_FEATURE_COLS].values

    # n_estimators=50 / max_depth=2 / learning_rate=0.1 mirror the former
    # GradientBoosting setup tuned to avoid probability saturation on small
    # pseudo-labelled sets.
    clf = XGBClassifier(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    clf.fit(X_train, balanced_labels)

    probs = clf.predict_proba(X_all)[:, 1]
    logger.info(
        "XGBoost predict_proba: min=%.4f  max=%.4f  mean=%.4f",
        probs.min(), probs.max(), probs.mean(),
    )
    return probs, "XGBoost", clf, X_all


# ===========================================================================
# Public orchestrator
# ===========================================================================


def run_forecast(
    fire_df: pd.DataFrame,
    country: str,
    min_frp: float = 10.0,
    horizon_days: int = 1,
) -> ForecastResult:
    """
    Build and return a ForecastResult for *country* using *fire_df* as history.

    Steps
    -----
    1. Look up bounding box from COUNTRY_BBOX.
    2. Fetch an independent 7-day FIRMS window for hist_fire_count_7d /
       hist_frp_mean_7d features.  This is always 7 days regardless of
       whatever --days was passed on the CLI — mirroring the fix already
       verified in forecast_engine._get_fire_window.
    3. Build the 0.25° grid (capped at MAX_GRID_CELLS).
    4. Build the feature matrix (fire history + weather).
    5. Run XGBoost model (or deterministic fallback).
    6. Assemble GridCell list, sort descending by fire_prob.
    7. Return ForecastResult.

    Parameters
    ----------
    fire_df      : Caller-supplied (CLI-selected) fire detections — used only for
                   recency features (hist_fire_count_24h, days_since_last_fire).
    country      : Must be a key in COUNTRY_BBOX.
    min_frp      : FRP threshold applied to the independent 7-day fetch so it
                   matches the filter already applied to fire_df.
    horizon_days : 1 (24-hour window, default) or 7 (7-day window).  Controls
                   weather aggregation window and ForecastResult.forecast_horizon_hours.
    """
    if horizon_days not in (1, 7):
        raise ValueError(f"horizon_days must be 1 or 7, got {horizon_days}")
    horizon_hours = horizon_days * 24  # 24 or 168

    generated_at = datetime.now(timezone.utc).isoformat()
    bbox_str = COUNTRY_BBOX.get(country)

    if not bbox_str:
        logger.warning("No bounding box for '%s'.", country)
        return ForecastResult(
            cells=[], top_risk_cells=[],
            forecast_horizon_hours=horizon_hours,
            generated_at=generated_at,
            model_used="Deterministic",
            country=country,
        )

    grid_points = _build_grid(bbox_str)
    if not grid_points:
        return ForecastResult(
            cells=[], top_risk_cells=[],
            forecast_horizon_hours=horizon_hours,
            generated_at=generated_at,
            model_used="Deterministic",
            country=country,
        )

    # Always fetch a fixed 7-day window for the historical features,
    # regardless of what --days the caller chose.
    logger.info(
        "run_forecast: fetching independent 7-day window for %s min_frp=%.1f horizon=%dh",
        country, min_frp, horizon_hours,
    )
    fire_7d_df = fetch_fire_data(country, days=7, min_frp=min_frp)
    logger.info(
        "run_forecast: fire_7d_df=%d rows  fire_short_df=%d rows  country=%s",
        len(fire_7d_df), len(fire_df), country,
    )

    logger.info("Grid: %d cells for %s", len(grid_points), country)
    feature_df = _build_feature_matrix(grid_points, fire_7d_df, fire_df, horizon_hours)
    probabilities, model_used, clf, X_all = _get_model_and_predictions(feature_df)
    feature_df = feature_df.reset_index(drop=True)

    cells = []
    for i, row in feature_df.iterrows():
        prob = float(np.clip(probabilities[i], 0.0, 1.0))
        cells.append(
            GridCell(
                lat_center=float(row["lat"]),
                lon_center=float(row["lon"]),
                fire_prob=prob,
                risk_band=_prob_to_risk_band(prob),
                feature_snapshot={col: row[col] for col in FEATURE_COLS},
                historical_fire_count=int(row["hist_fire_count_7d"]),
            )
        )

    cells.sort(key=lambda c: c.fire_prob, reverse=True)

    # Attach SHAP contributions to the top-10 cells (XGBoost path only).
    if clf is not None and X_all is not None:
        try:
            top10_row_indices = [
                feature_df.index[
                    (feature_df["lat"].round(4) == round(c.lat_center, 4)) &
                    (feature_df["lon"].round(4) == round(c.lon_center, 4))
                ].tolist()[0]
                for c in cells[:10]
            ]
            shap_lists = _compute_shap_contribs(clf, X_all, top10_row_indices)
            for cell, contribs in zip(cells[:10], shap_lists):
                cell.shap_contribs = contribs
            logger.info("SHAP computed for top-10 cells.")
        except Exception as shap_err:
            logger.warning("SHAP computation failed (non-fatal): %s", shap_err)

    return ForecastResult(
        cells=cells,
        top_risk_cells=cells[:5],
        forecast_horizon_hours=horizon_hours,
        generated_at=generated_at,
        model_used=model_used,
        country=country,
    )


# ===========================================================================
# CLI entry point
# ===========================================================================


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the wildfire fire-probability model for a country and print results as CSV."
    )
    parser.add_argument(
        "--country",
        default="Angola",
        choices=sorted(COUNTRY_BBOX.keys()),
        help="Country to analyse (default: Angola)",
    )
    parser.add_argument(
        "--days",
        type=int,
        default=7,
        choices=[2, 7, 30],
        help=(
            "Days of FIRMS history to ingest (2, 7, or 30; default: 7). "
            "Note: hist_fire_count_7d and hist_frp_mean_7d are always computed "
            "from an independent 7-day fetch, regardless of this value."
        ),
    )
    parser.add_argument(
        "--min-frp",
        type=float,
        default=10.0,
        help="Minimum FRP threshold in MW (default: 10.0)",
    )
    parser.add_argument(
        "--horizon-days",
        type=int,
        default=1,
        choices=[1, 7],
        help="Forecast horizon in days: 1 = 24-hour window, 7 = 7-day window (default: 1)",
    )
    parser.add_argument(
        "--output",
        default="-",
        help="Output CSV file path (default: stdout, i.e. '-')",
    )
    args = parser.parse_args()

    logger.info("Fetching FIRMS fire data: country=%s  days=%d  min_frp=%.1f",
                args.country, args.days, args.min_frp)
    fire_df = fetch_fire_data(args.country, args.days, args.min_frp)
    logger.info("Loaded %d fire detections.", len(fire_df))

    result = run_forecast(fire_df, args.country, args.min_frp, args.horizon_days)
    logger.info(
        "Forecast complete: %d cells, model=%s, generated_at=%s",
        len(result.cells), result.model_used, result.generated_at,
    )

    # Flatten cells to a DataFrame and emit as CSV
    rows = []
    for c in result.cells:
        row = {
            "lat_center": c.lat_center,
            "lon_center": c.lon_center,
            "fire_prob": round(c.fire_prob, 4),
            "risk_band": c.risk_band,
            "historical_fire_count_7d": c.historical_fire_count,
        }
        row.update(c.feature_snapshot)
        rows.append(row)

    out_df = pd.DataFrame(rows)

    if args.output == "-":
        out_df.to_csv(sys.stdout, index=False)
    else:
        out_df.to_csv(args.output, index=False)
        logger.info("Results written to %s", args.output)


if __name__ == "__main__":
    main()
