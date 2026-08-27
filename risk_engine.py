"""
risk_engine.py — Pure-Python fire risk metric computation.

No imports from ingestor.py, llm_gateway.py, weather_client.py, or
forecast_engine.py.  All functions are deterministic and unit-testable
without network access.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import List

import pandas as pd

import config


# ---------------------------------------------------------------------------
# RiskContext dataclass
# ---------------------------------------------------------------------------

@dataclass
class RiskContext:
    """Structured risk summary for a country/time-window query.

    Consumed by both the Streamlit UI (metric cards) and the LLM Gateway
    (prompt construction).  All numeric fields default to 0 / 0.0 so the
    UI can safely render the "NO DATA" state without None-checks everywhere.
    """

    country: str
    time_window_days: int
    fire_count: int
    total_frp: float          # Sum of Fire Radiative Power (MW)
    max_frp: float
    mean_frp: float
    hotspot_density: float    # Fires per 10,000 km²
    high_confidence_pct: float  # % of detections rated high-confidence
    spread_index: float       # Bounding-box area of detections (km²)
    risk_level: str           # "NO DATA" | "LOW" | "MEDIUM" | "HIGH" | "EXTREME"
    top_hotspots: List[dict] = field(default_factory=list)
    # Each dict: {"lat": float, "lon": float, "frp": float, "acq_date": str}


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _score_risk(fire_count: int, total_frp: float, spread_index: float) -> str:
    """Derive a risk level string from three key metrics.

    Scoring formula (weighted sum, clamped to [0, 1]):
        score = 0.4 * (fire_count / 500)
              + 0.4 * (total_frp   / 5_000)
              + 0.2 * (spread_index / 500_000)

    Thresholds:
        fire_count == 0              → "NO DATA"
        score <  0.15                → "LOW"
        0.15 ≤ score <  0.40         → "MEDIUM"
        0.40 ≤ score <  0.70         → "HIGH"
        score ≥  0.70                → "EXTREME"

    Reference baselines (used to set the normalisation denominators):
        500     fires  — roughly one busy fire day in a large country
        5 000   MW FRP — moderate-to-high aggregate intensity
        500 000 km²    — roughly 1/6 of Brazil's land area
    """
    if fire_count == 0:
        return "NO DATA"

    score = (
        0.4 * (fire_count  / 500)
        + 0.4 * (total_frp   / 5_000)
        + 0.2 * (spread_index / 500_000)
    )
    score = max(0.0, min(1.0, score))  # clamp to [0, 1]

    if score < 0.15:
        return "LOW"
    if score < 0.40:
        return "MEDIUM"
    if score < 0.70:
        return "HIGH"
    return "EXTREME"


def _top_hotspots(df: pd.DataFrame) -> List[dict]:
    """Return the top-5 fire detections by FRP.

    Args:
        df: DataFrame with at least columns latitude, longitude, frp, acq_date.

    Returns:
        List of up to 5 dicts: {lat, lon, frp, acq_date}.
    """
    if df.empty:
        return []

    top = df.nlargest(5, "frp")[["latitude", "longitude", "frp", "acq_date"]]
    return [
        {
            "lat": float(row.latitude),
            "lon": float(row.longitude),
            "frp": float(row.frp),
            "acq_date": str(row.acq_date),
        }
        for row in top.itertuples(index=False)
    ]


def _compute_hotspot_density(fire_count: int, country: str) -> float:
    """Return fires per 10,000 km² using the country's bounding box.

    Area is approximated with a flat-earth formula:
        area_km² = (E - W) * (N - S) * (111.32 * cos(mid_lat))²

    where 111.32 km/° is the mean meridional degree length.
    """
    if fire_count == 0:
        return 0.0

    bbox_str = config.COUNTRY_BBOX.get(country, "")
    if not bbox_str:
        return 0.0

    w, s, e, n = (float(v) for v in bbox_str.split(","))
    mid_lat = math.radians((s + n) / 2.0)
    km_per_deg = 111.32 * math.cos(mid_lat)
    area_km2 = (e - w) * (n - s) * (km_per_deg ** 2)

    if area_km2 <= 0:
        return 0.0

    return fire_count / area_km2 * 10_000


def _compute_spread_index(df: pd.DataFrame) -> float:
    """Approximate the geographic extent of fire detections in km².

    Spread index = bounding-box area of actual detection lat/lon coordinates:
        (lat_max - lat_min) * (lon_max - lon_min) * 111.32²

    This is a flat-earth approximation adequate for risk-scoring purposes.
    Returns 0.0 for empty DataFrames.
    """
    if df.empty:
        return 0.0

    lat_range = float(df["latitude"].max() - df["latitude"].min())
    lon_range = float(df["longitude"].max() - df["longitude"].min())
    return lat_range * lon_range * (111.32 ** 2)


def _compute_high_confidence_pct(df: pd.DataFrame) -> float:
    """Return the percentage of detections rated high-confidence.

    Handles both sensor types:
    - VIIRS: single-letter codes 'l' (low) | 'n' (nominal) | 'h' (high)
      The FIRMS CSV API returns the abbreviated form; 'high' (full word) is
      accepted as well for robustness.
    - MODIS: integer string 0–100  → high-confidence = value >= 80
    """
    if df.empty:
        return 0.0

    def _is_high(val) -> bool:
        try:
            return int(val) >= 80  # MODIS integer path
        except (ValueError, TypeError):
            v = str(val).strip().lower()
            return v in ("h", "high")  # VIIRS: abbreviated or full word

    high_count = df["confidence"].apply(_is_high).sum()
    return float(high_count) / len(df) * 100.0


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def compute_risk(
    df: pd.DataFrame,
    country: str,
    time_window_days: int,
) -> RiskContext:
    """Compute a full RiskContext from a fire detection DataFrame.

    Args:
        df:               Fire detection DataFrame (columns: latitude, longitude,
                          brightness, frp, acq_date, acq_time, confidence, instrument).
        country:          Country name matching a key in config.COUNTRY_BBOX.
        time_window_days: Number of days the DataFrame covers (stored verbatim).

    Returns:
        RiskContext with all metrics populated.  If df is empty, returns a
        zeroed RiskContext with risk_level="NO DATA".
    """
    if df.empty:
        return RiskContext(
            country=country,
            time_window_days=time_window_days,
            fire_count=0,
            total_frp=0.0,
            max_frp=0.0,
            mean_frp=0.0,
            hotspot_density=0.0,
            high_confidence_pct=0.0,
            spread_index=0.0,
            risk_level="NO DATA",
            top_hotspots=[],
        )

    fire_count = len(df)
    total_frp = float(df["frp"].sum())
    max_frp = float(df["frp"].max())
    mean_frp = float(df["frp"].mean())

    spread_index = _compute_spread_index(df)
    hotspot_density = _compute_hotspot_density(fire_count, country)
    high_confidence_pct = _compute_high_confidence_pct(df)
    risk_level = _score_risk(fire_count, total_frp, spread_index)
    top_hotspots = _top_hotspots(df)

    return RiskContext(
        country=country,
        time_window_days=time_window_days,
        fire_count=fire_count,
        total_frp=total_frp,
        max_frp=max_frp,
        mean_frp=mean_frp,
        hotspot_density=hotspot_density,
        high_confidence_pct=high_confidence_pct,
        spread_index=spread_index,
        risk_level=risk_level,
        top_hotspots=top_hotspots,
    )
