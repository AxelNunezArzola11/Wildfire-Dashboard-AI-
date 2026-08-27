"""
forecast_engine.py — Grid-Cell Fire Probability Model (Sub-Task 5).

Public API
----------
run_forecast(fire_df, country, min_frp, horizon_days=1) -> ForecastResult

Builds a 0.25° grid over the country bounding box, computes per-cell fire-risk
features from FIRMS history and Open-Meteo weather, trains (or scores with) an
XGBClassifier, returns per-cell fire probabilities, and attaches SHAP
feature-contribution breakdowns to the top-10 highest-risk cells.

horizon_days
------------
Supported values: 1 (24-hour window, default) or 7 (7-day window).
Controls both the Open-Meteo weather aggregation window and the
ForecastResult.forecast_horizon_hours field.  Longer horizons carry more
uncertainty; anything beyond 16 days is outside Open-Meteo's reliable
forecast range and is not offered.

Feature correctness guarantee
------------------------------
hist_fire_count_7d and hist_frp_mean_7d are always computed from an independent
7-day FIRMS fetch, regardless of the Time Range the user selected in the UI.
The sidebar-selected fire_df is used only for hist_fire_count_24h (the positive
pseudo-label) and days_since_last_fire.  This prevents silent understatement of
the 7-day features when "Last 48 hours" is selected.
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from xgboost import XGBClassifier

import config
import weather_client

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

FORECAST_HORIZON_HOURS = 24  # default; overridden per-call via horizon_days
MAX_GRID_CELLS = 200
MIN_LABELLED_SAMPLES = 10
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

# Columns used as predictors when fitting/scoring the GBT model.
# Four columns are deliberately excluded from training features:
#   - hist_fire_count_7d:  defines label 0 directly (== 0 → label 0)
#   - hist_fire_count_24h: defines label 1 directly (> 0 → label 1)
#   - hist_frp_mean_7d:    always exactly 0.0 when hist_fire_count_7d == 0, so
#                          perfectly collinear with the label boundary
#   - days_since_last_fire: always the sentinel value 30 when hist_fire_count_7d == 0,
#                           perfectly collinear with the label boundary
# Including any of these gives the classifier a trivially-perfect split that
# generalises to saturated probabilities (0.0 / 1.0) on all cells.
MODEL_FEATURE_COLS = [
    "temp_24h_mean",
    "humidity_24h_mean",
    "wind_24h_max",
    "precip_24h_sum",
    "soil_moisture_now",
]

# ---------------------------------------------------------------------------
# Dataclasses
# ---------------------------------------------------------------------------


@dataclass
class GridCell:
    lat_center: float
    lon_center: float
    fire_prob: float            # 0.0 – 1.0
    risk_band: str              # "LOW" | "MEDIUM" | "HIGH" | "EXTREME"
    feature_snapshot: dict      # all input features used for this cell
    historical_fire_count: int  # fires in cell over last 7 days
    # SHAP weather-feature contributions for this cell (XGBoost path only).
    # Each entry: {"feature": str, "label": str, "shap": float, "pct": float}
    # sorted by abs(shap) descending.  Empty list when model_used=="Deterministic".
    shap_contribs: list = field(default_factory=list)


@dataclass
class ForecastResult:
    cells: list                    # list[GridCell], all cells
    top_risk_cells: list           # list[GridCell], top 5 by fire_prob
    forecast_horizon_hours: int    # 24 (1-day) or 168 (7-day)
    generated_at: str              # ISO timestamp (UTC)
    model_used: str                # "XGBoost" | "Deterministic"
    country: str
    # Fitted XGBClassifier — present only on the XGBoost path, None otherwise.
    # Stored here so agent_runner.py can persist the exact booster to disk for
    # reproducibility without re-training.  Not serialised to the DB.
    _clf: object = field(default=None, repr=False, compare=False)
    # The 7-day FIRMS DataFrame used as training input for XGBoost feature
    # engineering (_build_feature_matrix → _get_model_and_predictions).
    # Populated only when _clf is not None so the artifact writer can persist
    # it as dataset_forecast_window.csv.gz alongside the booster.
    # None on the deterministic path (no training occurred).  Not serialised.
    _fire_7d_df: object = field(default=None, repr=False, compare=False)


# ---------------------------------------------------------------------------
# Grid builder
# ---------------------------------------------------------------------------


def _build_grid(bbox_str: str) -> list[tuple[float, float]]:
    """
    Parse "W,S,E,N" and return (lat, lon) centroids at FORECAST_GRID_DEG intervals.

    First centroid: W + step/2 (lon), S + step/2 (lat).
    Last centroid:  ≤ E - step/2 (lon), ≤ N - step/2 (lat).
    Capped at MAX_GRID_CELLS via uniform subsampling.
    """
    w, s, e, n = (float(v) for v in bbox_str.split(","))
    step = config.FORECAST_GRID_DEG
    half = step / 2.0

    lats = []
    lat = s + half
    while lat <= n - half + 1e-9:
        lats.append(round(lat, 6))
        lat += step

    lons = []
    lon = w + half
    while lon <= e - half + 1e-9:
        lons.append(round(lon, 6))
        lon += step

    points = [(lat, lon) for lat in lats for lon in lons]

    if len(points) > MAX_GRID_CELLS:
        indices = [round(i * (len(points) - 1) / (MAX_GRID_CELLS - 1))
                   for i in range(MAX_GRID_CELLS)]
        points = [points[i] for i in indices]

    return points


# ---------------------------------------------------------------------------
# Windowed FIRMS fetch — decoupled from the sidebar Time Range
# ---------------------------------------------------------------------------

# Module-level in-memory TTL cache for windowed fire DataFrames.
# Key: (country, min_frp, days)  →  (fetched_at: datetime, df: pd.DataFrame)
_window_cache: dict[tuple, tuple] = {}
_WINDOW_CACHE_TTL_SECONDS = config.CACHE_TTL_MINUTES * 60


def _get_fire_window(country: str, min_frp: float, days: int = 7) -> pd.DataFrame:
    """
    Return a *days*-day FIRMS DataFrame for *country* filtered to *min_frp*.

    Results are cached in-process for CACHE_TTL_MINUTES to prevent a
    redundant FIRMS round-trip on every Streamlit re-render.  The
    underlying ingestor also caches in SQLite, so cache misses here still
    hit the SQLite layer before going to the network.

    For days > FIRMS_MAX_DAYS (5) the ingestor automatically chunks the
    request into ≤5-day slices, concatenates, and deduplicates.

    This fetch is completely independent of the sidebar's Time Range.
    """
    # Import here to satisfy the module docstring constraint
    # ("No imports from ingestor at module level") while still reusing
    # its SQLite cache and deduplication logic.
    from ingestor import get_fire_data  # noqa: PLC0415

    key = (country, min_frp, days)
    now = datetime.now(timezone.utc)

    cached = _window_cache.get(key)
    if cached is not None:
        fetched_at, df = cached
        age = (now - fetched_at).total_seconds()
        if age < _WINDOW_CACHE_TTL_SECONDS:
            logger.debug(
                "_get_fire_window cache hit for %s min_frp=%.1f days=%d (age %.0fs)",
                country, min_frp, days, age,
            )
            return df

    logger.info(
        "_get_fire_window fetching %d-day FIRMS window for %s min_frp=%.1f",
        days, country, min_frp,
    )
    df, _ingest_secs = get_fire_data(country, days=days, min_frp=min_frp)
    if _ingest_secs is not None:
        logger.info(
            "_get_fire_window cold-cache ingest: country=%s days=%d rows=%d time=%.2fs",
            country, days, len(df), _ingest_secs,
        )
    _window_cache[key] = (now, df)
    return df


# ---------------------------------------------------------------------------
# Fire feature extractor
# ---------------------------------------------------------------------------


def _compute_fire_features(
    lat: float,
    lon: float,
    fire_7d_df: pd.DataFrame,
    fire_short_df: pd.DataFrame,
) -> dict:
    """
    Compute historical fire features for the cell centred at (lat, lon).

    Cell bounds: [lat ± step/2, lon ± step/2].

    Parameters
    ----------
    fire_7d_df   : Always a 7-day FIRMS window — used for hist_fire_count_7d,
                   hist_frp_mean_7d, and days_since_last_fire.
    fire_short_df: The sidebar-selected window (48 h or 7 d) — used only for
                   hist_fire_count_24h (the positive pseudo-label), which
                   reflects fires on the most-recent available reporting day
                   in whatever window the user selected.
    """
    half = config.FORECAST_GRID_DEG / 2.0

    def _cell_mask(df: pd.DataFrame) -> pd.Series:
        return (
            (df["latitude"] >= lat - half) & (df["latitude"] < lat + half)
            & (df["longitude"] >= lon - half) & (df["longitude"] < lon + half)
        )

    # ── 7-day features ───────────────────────────────────────────────────────
    if fire_7d_df.empty:
        hist_fire_count_7d = 0
        hist_frp_mean_7d = 0.0
        days_since_last_fire = 30
    else:
        cell_7d = fire_7d_df[_cell_mask(fire_7d_df)]
        hist_fire_count_7d = len(cell_7d)
        hist_frp_mean_7d = float(cell_7d["frp"].mean()) if hist_fire_count_7d > 0 else 0.0

        if hist_fire_count_7d > 0:
            dates_7d = pd.to_datetime(cell_7d["acq_date"], errors="coerce")
            latest_7d = dates_7d.max()
            if pd.isna(latest_7d):
                days_since_last_fire = 30
            else:
                today_dt = pd.Timestamp(datetime.now(timezone.utc).date())
                days_since_last_fire = int(
                    min((today_dt - latest_7d.normalize()).days, 30)
                )
        else:
            days_since_last_fire = 30

    # ── Short-window pseudo-label feature ────────────────────────────────────
    # hist_fire_count_24h = fires on the most recent reporting day in
    # fire_short_df (the sidebar-selected window).  Using the short window
    # here is correct: the positive pseudo-label should reflect fires that
    # were active in the most recent observation period, not the full 7 days.
    if fire_short_df.empty or hist_fire_count_7d == 0:
        # If there's no fire anywhere in the 7-day window, there can't be
        # a positive label from the short window either.
        hist_fire_count_24h = 0
    else:
        cell_short = fire_short_df[_cell_mask(fire_short_df)]
        if cell_short.empty:
            hist_fire_count_24h = 0
        else:
            # FIRMS data arrives with a 1-2 day lag, so "acq_date == today"
            # is always zero.  Use the most recent acquisition date in the
            # short-window cell as the "latest day" boundary.
            dates_short = pd.to_datetime(cell_short["acq_date"], errors="coerce")
            latest_short = dates_short.max()
            if pd.isna(latest_short):
                hist_fire_count_24h = 0
            else:
                latest_str = latest_short.normalize().strftime("%Y-%m-%d")
                hist_fire_count_24h = int(
                    (cell_short["acq_date"] == latest_str).sum()
                )

    return {
        "hist_fire_count_7d": hist_fire_count_7d,
        "hist_frp_mean_7d": hist_frp_mean_7d,
        "hist_fire_count_24h": hist_fire_count_24h,
        "days_since_last_fire": days_since_last_fire,
    }


# ---------------------------------------------------------------------------
# Feature matrix builder
# ---------------------------------------------------------------------------


def _build_feature_matrix(
    grid_points: list[tuple[float, float]],
    fire_7d_df: pd.DataFrame,
    fire_short_df: pd.DataFrame,
    horizon_hours: int = 24,
) -> pd.DataFrame:
    """
    Build the full feature DataFrame (one row per grid cell).

    Calls weather_client.get_weather_for_points(), merges fire features,
    fills remaining NaNs with column medians.

    Parameters
    ----------
    fire_7d_df   : Independent 7-day FIRMS window for hist_*_7d features.
    fire_short_df: Sidebar-selected window for hist_fire_count_24h.
    horizon_hours: Weather aggregation window (24 or 168) passed to the
                   weather client.
    """
    weather_df = weather_client.get_weather_for_points(grid_points, horizon_hours)

    fire_rows = []
    for lat, lon in grid_points:
        feats = _compute_fire_features(lat, lon, fire_7d_df, fire_short_df)
        feats["lat"] = lat
        feats["lon"] = lon
        fire_rows.append(feats)

    fire_feat_df = pd.DataFrame(fire_rows)

    if weather_df.empty:
        # Build a NaN weather frame aligned to grid_points
        weather_df = pd.DataFrame(
            [{"lat": lat, "lon": lon} for lat, lon in grid_points]
        )
        for col in ["temp_24h_mean", "humidity_24h_mean", "wind_24h_max",
                    "precip_24h_sum", "soil_moisture_now"]:
            weather_df[col] = float("nan")

    merged = pd.merge(
        fire_feat_df,
        weather_df[["lat", "lon", "temp_24h_mean", "humidity_24h_mean",
                    "wind_24h_max", "precip_24h_sum", "soil_moisture_now"]],
        on=["lat", "lon"],
        how="left",
    )

    # Ensure column order and fill NaNs with medians
    for col in FINAL_COLUMNS:
        if col not in merged.columns:
            merged[col] = float("nan")
    merged = merged[FINAL_COLUMNS]

    for col in merged.columns:
        if merged[col].isna().any():
            median = merged[col].median()
            merged[col] = merged[col].fillna(0.0 if pd.isna(median) else median)

    return merged


# ---------------------------------------------------------------------------
# Risk band classifier
# ---------------------------------------------------------------------------


def _prob_to_risk_band(prob: float) -> str:
    """Map a fire probability float to a named risk band."""
    if prob < 0.20:
        return "LOW"
    elif prob < 0.50:
        return "MEDIUM"
    elif prob < 0.75:
        return "HIGH"
    return "EXTREME"


# ---------------------------------------------------------------------------
# Deterministic fallback scorer
# ---------------------------------------------------------------------------


def _deterministic_score(row: pd.Series) -> float:
    """
    Weighted deterministic fire-risk score.

    Each term is normalised and clamped to [0, 1]; NaN inputs are treated as 0.
    Returns a value in [0.0, 1.0].
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


# ---------------------------------------------------------------------------
# Model selector
# ---------------------------------------------------------------------------


# Human-readable labels for the 5 MODEL_FEATURE_COLS.
# Used in SHAP breakdowns shown in the UI.
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
    cell_row_indices: list[int],
) -> list[list[dict]]:
    """
    Compute SHAP feature contributions for a subset of cells.

    Parameters
    ----------
    clf:
        Fitted XGBClassifier.
    X_all:
        Full feature matrix (n_cells × len(MODEL_FEATURE_COLS)).
    cell_row_indices:
        Row indices (into X_all) for the cells we want explanations for.
        Typically the top-10 cells sorted by descending fire_prob.

    Returns
    -------
    List of length len(cell_row_indices).  Each element is a list of dicts:
        [{"feature": str, "label": str, "shap": float, "pct": float}, ...]
    sorted by abs(shap) descending, limited to the top 5 contributors.

    SHAP values are in log-odds space (XGBoost default).  ``pct`` is each
    feature's share of the total absolute push, as a percentage — useful for
    a ranked display even when the raw log-odds values are unfamiliar to users.
    """
    import shap as _shap  # lazy import — not needed on deterministic fallback

    X_subset = X_all[cell_row_indices]                        # (n_cells, n_feats)
    explainer = _shap.TreeExplainer(clf)
    sv = explainer.shap_values(X_subset)                      # ndarray (n_cells, n_feats)

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
        result.append(contribs[:5])                           # top-5 contributors
    return result


def _get_model_and_predictions(
    feature_df: pd.DataFrame,
) -> tuple[np.ndarray, str, object, np.ndarray | None]:
    """
    Return (probabilities_array, model_used_str, fitted_clf_or_None, X_all_or_None).

    The fitted clf and X_all are returned so the caller can compute SHAP values
    for whichever cells it needs without re-fitting.  Both are None when the
    deterministic fallback is used (SHAP not applicable in that case).

    Pseudo-label strategy:
    - label 1  if hist_fire_count_24h > 0
    - label 0  if hist_fire_count_7d  == 0
    - discard ambiguous rows

    To prevent probability saturation on fire-dense regions (where nearly all
    cells are class-1), the majority class is downsampled to at most 4x the
    minority class before fitting.

    Falls back to _deterministic_score when:
    - fewer than MIN_LABELLED_SAMPLES labelled rows exist, OR
    - only one class is represented, OR
    - fewer than MIN_POSITIVE_SAMPLES positive examples exist after downsampling
      (XGBoost/GBT cannot learn a meaningful split from 1–2 fire examples and
      will return a constant equal to the training base rate, which is
      indistinguishable from a flat/broken output — the deterministic scorer
      produces more informative differentiation in this regime).
    """
    # Minimum positive-class examples required after downsampling.
    # Below this the tree cannot find a meaningful split and returns a constant.
    MIN_POSITIVE_SAMPLES = 3

    label_1 = feature_df["hist_fire_count_24h"] > 0
    label_0 = feature_df["hist_fire_count_7d"] == 0

    labelled_idx = feature_df.index[label_1 | label_0]
    labels = np.where(label_1[labelled_idx], 1, 0)

    unique_labels = np.unique(labels)
    logger.debug(
        "_get_model_and_predictions: %d labelled rows, label distribution: %s",
        len(labelled_idx),
        {int(k): int(v) for k, v in zip(*np.unique(labels, return_counts=True))},
    )

    if len(labelled_idx) < MIN_LABELLED_SAMPLES or len(unique_labels) < 2:
        logger.warning(
            "Falling back to deterministic scorer: %d pseudo-labelled samples "
            "(need >=%d with both classes). unique_labels=%s",
            len(labelled_idx),
            MIN_LABELLED_SAMPLES,
            unique_labels.tolist(),
        )
        probabilities = np.array(
            [_deterministic_score(row) for _, row in feature_df.iterrows()]
        )
        return probabilities, "Deterministic", None, None

    # Downsample majority class to at most MAX_IMBALANCE_RATIO × minority count
    # to prevent the classifier from saturating all predictions at 1.0.
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
    # cannot learn a split and will return a constant equal to the training
    # base rate (e.g. 2/10 = 0.20 for all cells).  Fall back instead.
    if len(idx_pos) < MIN_POSITIVE_SAMPLES:
        logger.warning(
            "Falling back to deterministic scorer: only %d positive training "
            "samples after downsampling (need >=%d). "
            "XGBoost would return a flat base-rate constant.",
            len(idx_pos),
            MIN_POSITIVE_SAMPLES,
        )
        probabilities = np.array(
            [_deterministic_score(row) for _, row in feature_df.iterrows()]
        )
        return probabilities, "Deterministic", None, None

    balanced_idx = np.concatenate([idx_pos, idx_neg])
    balanced_labels = np.concatenate([
        np.ones(len(idx_pos), dtype=int),
        np.zeros(len(idx_neg), dtype=int),
    ])
    logger.debug(
        "After downsampling: %d pos, %d neg training samples.",
        len(idx_pos), len(idx_neg),
    )

    X_labelled = feature_df.loc[balanced_idx, MODEL_FEATURE_COLS].values
    X_all = feature_df[MODEL_FEATURE_COLS].values

    # n_estimators=50 / max_depth=2 / learning_rate=0.1 match the former
    # GradientBoosting setup that was tuned to avoid probability saturation on
    # small pseudo-labelled sets.  eval_metric='logloss' suppresses the default
    # XGBoost verbosity; use_label_encoder=False is not needed in XGBoost ≥1.6.
    clf = XGBClassifier(
        n_estimators=50,
        max_depth=2,
        learning_rate=0.1,
        random_state=42,
        eval_metric="logloss",
        verbosity=0,
    )
    clf.fit(X_labelled, balanced_labels)

    probabilities = clf.predict_proba(X_all)[:, 1]
    logger.debug(
        "predict_proba raw range: min=%.4f max=%.4f mean=%.4f",
        probabilities.min(), probabilities.max(), probabilities.mean(),
    )
    return probabilities, "XGBoost", clf, X_all


# ---------------------------------------------------------------------------
# Public orchestrator
# ---------------------------------------------------------------------------


def run_forecast(
    fire_df: pd.DataFrame,
    country: str,
    min_frp: float = 0.0,
    horizon_days: int = 1,
) -> ForecastResult:
    """
    Build and return a ForecastResult for *country* using *fire_df* as history.

    Steps:
    1. Look up bounding box from config.COUNTRY_BBOX.
    2. Fetch an independent 7-day FIRMS window (cached; decoupled from sidebar).
    3. Build the 0.25° grid.
    4. Build the feature matrix (fire + weather), splitting 7d / short-window.
    5. Run model / deterministic scorer.
    6. Assemble GridCell list, sort descending by fire_prob.
    7. Return ForecastResult.

    Parameters
    ----------
    fire_df      : Sidebar-selected fire detections (any days value).  Used only
                   for hist_fire_count_24h (positive pseudo-label).
    country      : Must be a key in config.COUNTRY_BBOX.
    min_frp      : FRP threshold applied to the independent 7-day fetch so that
                   it matches the filter already applied to fire_df.
    horizon_days : 1 (24-hour window, default) or 7 (7-day window).  Controls
                   the Open-Meteo weather aggregation and ForecastResult.forecast_horizon_hours.
    """
    if horizon_days not in (1, 7):
        raise ValueError(f"horizon_days must be 1 or 7, got {horizon_days}")
    horizon_hours = horizon_days * 24  # 24 or 168

    generated_at = datetime.now(timezone.utc).isoformat()

    bbox_str = config.COUNTRY_BBOX.get(country)
    if not bbox_str:
        logger.warning("No bounding box found for country '%s'.", country)
        return ForecastResult(
            cells=[],
            top_risk_cells=[],
            forecast_horizon_hours=horizon_hours,
            generated_at=generated_at,
            model_used="Deterministic",
            country=country,
        )

    grid_points = _build_grid(bbox_str)

    if not grid_points:
        return ForecastResult(
            cells=[],
            top_risk_cells=[],
            forecast_horizon_hours=horizon_hours,
            generated_at=generated_at,
            model_used="Deterministic",
            country=country,
        )

    # Always fetch a full 7-day window regardless of the sidebar selection.
    # _get_fire_window is a no-op on subsequent renders within the cache TTL.
    fire_7d_df = _get_fire_window(country, min_frp, days=7)
    logger.info(
        "run_forecast: fire_7d_df=%d rows  fire_short_df=%d rows  "
        "country=%s  horizon=%dh",
        len(fire_7d_df), len(fire_df), country, horizon_hours,
    )

    feature_df = _build_feature_matrix(grid_points, fire_7d_df, fire_df, horizon_hours)
    probabilities, model_used, clf, X_all = _get_model_and_predictions(feature_df)

    # Reset index so positional integers align with probabilities array indices.
    feature_df = feature_df.reset_index(drop=True)

    cells = []
    for i, row in feature_df.iterrows():
        prob = float(np.clip(probabilities[i], 0.0, 1.0))
        snapshot = {col: row[col] for col in FEATURE_COLS}
        cells.append(
            GridCell(
                lat_center=float(row["lat"]),
                lon_center=float(row["lon"]),
                fire_prob=prob,
                risk_band=_prob_to_risk_band(prob),
                feature_snapshot=snapshot,
                historical_fire_count=int(row["hist_fire_count_7d"]),
            )
        )

    cells.sort(key=lambda c: c.fire_prob, reverse=True)

    # Attach SHAP contributions to the top-10 cells (XGBoost path only).
    # SHAP on 200 cells takes ~80ms; we compute all at once and slice to top-10.
    if clf is not None and X_all is not None:
        try:
            # Map each top-10 cell back to its row index in X_all (feature_df
            # was reset_index'd above, so cell index i == feature_df row i).
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
            logger.debug("SHAP computed for top-10 cells.")
        except Exception as shap_err:
            logger.warning("SHAP computation failed (non-fatal): %s", shap_err)

    top_risk_cells = cells[:5]

    return ForecastResult(
        cells=cells,
        top_risk_cells=top_risk_cells,
        forecast_horizon_hours=horizon_hours,
        generated_at=generated_at,
        model_used=model_used,
        country=country,
        _clf=clf,            # None when deterministic path was used
        _fire_7d_df=fire_7d_df if clf is not None else None,
    )
