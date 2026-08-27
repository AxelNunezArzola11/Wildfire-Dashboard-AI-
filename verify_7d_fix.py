"""
verify_7d_fix.py — Before/after verification of the hist_fire_count_7d fix.

Runs entirely offline (no network calls) by constructing synthetic fire
DataFrames that mimic what FIRMS would return for a 2-day vs 7-day window.

Tests:
  1. hist_fire_count_7d and hist_frp_mean_7d are larger when the 7-day window
     is used (old bug: these were silently truncated to 2 days).
  2. hist_fire_count_24h still reflects only the short window's latest day.
  3. Both pseudo-label classes (0 and 1) are present in the feature matrix
     after the fix — the degenerate single-class regression is not reintroduced.

Run with:
    python3 verify_7d_fix.py
"""
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

# ── Build a synthetic fire DataFrame ────────────────────────────────────────

def _make_fire_df(n_days: int, lat_center=0.125, lon_center=0.125) -> pd.DataFrame:
    """
    Generate one detection per day for *n_days* within a 0.25° cell centred
    at (lat_center, lon_center), with FRP values 50, 60, 70, … (incremental).
    """
    today = datetime.now(timezone.utc).date()
    rows = []
    for i in range(n_days):
        day = today - timedelta(days=i + 2)  # +2 simulates FIRMS lag
        rows.append({
            "latitude":   lat_center,
            "longitude":  lon_center,
            "frp":        50.0 + i * 10.0,
            "acq_date":   day.strftime("%Y-%m-%d"),
            "acq_time":   "1200",
            "brightness": 320.0,
            "confidence": "nominal",
            "instrument": "VIIRS",
        })
    return pd.DataFrame(rows)


# ── Import the feature extractor directly ───────────────────────────────────

sys.path.insert(0, ".")
from forecast_engine import _compute_fire_features  # noqa: E402

# ── Build test DataFrames ────────────────────────────────────────────────────

fire_2d = _make_fire_df(2)   # what the sidebar returns for "Last 48 hours"
fire_7d = _make_fire_df(7)   # what the independent 7-day fetch returns

LAT, LON = 0.125, 0.125      # centroid of the cell both DataFrames sit in

# ── OLD behaviour: same DataFrame used for everything ───────────────────────
# Before fix: _compute_fire_features(lat, lon, fire_2d) — fire_2d for all features.
# We simulate this by passing fire_2d as both arguments.
old_feats = _compute_fire_features(LAT, LON, fire_2d, fire_2d)

# ── NEW behaviour: split DataFrames ─────────────────────────────────────────
new_feats = _compute_fire_features(LAT, LON, fire_7d, fire_2d)

print("\n=== Feature comparison: OLD (2d as 7d) vs NEW (proper 7d) ===\n")
print(f"{'Feature':<30} {'OLD (buggy)':>14} {'NEW (fixed)':>14}  {'Match expected?':>16}")
print("-" * 80)

checks = []

def check(name, old_val, new_val, expected_new, compare="eq"):
    ok = (new_val == expected_new) if compare == "eq" else (new_val > old_val)
    status = "✅ PASS" if ok else "❌ FAIL"
    print(f"{name:<30} {str(old_val):>14} {str(new_val):>14}  {status:>16}")
    checks.append(ok)

check("hist_fire_count_7d",  old_feats["hist_fire_count_7d"],
                             new_feats["hist_fire_count_7d"],
                             expected_new=7,          compare="eq")
check("hist_frp_mean_7d",   round(old_feats["hist_frp_mean_7d"], 1),
                             round(new_feats["hist_frp_mean_7d"], 1),
                             expected_new=None,        compare="gt")  # new > old
check("hist_fire_count_24h", old_feats["hist_fire_count_24h"],
                             new_feats["hist_fire_count_24h"],
                             expected_new=old_feats["hist_fire_count_24h"],  # unchanged
                             compare="eq")

# days_since_last_fire should be the same (both DFs share the same lag)
check("days_since_last_fire", old_feats["days_since_last_fire"],
                              new_feats["days_since_last_fire"],
                              expected_new=old_feats["days_since_last_fire"],
                              compare="eq")

# ── Understatement ratio ─────────────────────────────────────────────────────
old_count = old_feats["hist_fire_count_7d"]
new_count = new_feats["hist_fire_count_7d"]
ratio = new_count / old_count if old_count > 0 else float("inf")
print(f"\nhist_fire_count_7d ratio (new/old): {ratio:.2f}x  (expected ~3.5x for 7d vs 2d)")

# ── Pseudo-label class check ─────────────────────────────────────────────────
print("\n=== Pseudo-label class distribution check ===\n")

from forecast_engine import _build_feature_matrix, _get_model_and_predictions  # noqa: E402

# Build a realistic multi-cell grid: half the cells should have fires (class 1),
# half should be fire-free (class 0) so both classes are represented.
def _make_grid_dfs(n_fire_cells=15, n_empty_cells=15):
    """
    n_fire_cells cells: each has detections in both 7d and 2d windows.
    n_empty_cells cells: no detections anywhere.
    """
    rows_7d, rows_2d = [], []
    for i in range(n_fire_cells):
        base_lat = 0.125 + i * 0.25
        base_lon = 0.125
        for day_offset in range(7):
            day = (datetime.now(timezone.utc).date() - timedelta(days=day_offset + 2))
            rows_7d.append({
                "latitude": base_lat, "longitude": base_lon,
                "frp": 60.0, "acq_date": day.strftime("%Y-%m-%d"),
                "acq_time": "1200", "brightness": 320.0,
                "confidence": "nominal", "instrument": "VIIRS",
            })
        for day_offset in range(2):
            day = (datetime.now(timezone.utc).date() - timedelta(days=day_offset + 2))
            rows_2d.append({
                "latitude": base_lat, "longitude": base_lon,
                "frp": 60.0, "acq_date": day.strftime("%Y-%m-%d"),
                "acq_time": "1200", "brightness": 320.0,
                "confidence": "nominal", "instrument": "VIIRS",
            })
    grid_7d = pd.DataFrame(rows_7d) if rows_7d else pd.DataFrame(columns=["latitude","longitude","frp","acq_date","acq_time","brightness","confidence","instrument"])
    grid_2d = pd.DataFrame(rows_2d) if rows_2d else pd.DataFrame(columns=["latitude","longitude","frp","acq_date","acq_time","brightness","confidence","instrument"])
    # Grid points: fire cells + empty cells
    grid_points = [(0.125 + i * 0.25, 0.125) for i in range(n_fire_cells + n_empty_cells)]
    return grid_points, grid_7d, grid_2d


grid_points, grid_7d, grid_2d = _make_grid_dfs(n_fire_cells=15, n_empty_cells=15)

# Use an empty weather df — _build_feature_matrix handles this with NaN-fill.
import unittest.mock as mock
empty_weather = pd.DataFrame([{"lat": lat, "lon": lon,
                               "temp_24h_mean": float("nan"),
                               "humidity_24h_mean": float("nan"),
                               "wind_24h_max": float("nan"),
                               "precip_24h_sum": float("nan"),
                               "soil_moisture_now": float("nan")}
                              for lat, lon in grid_points])

import weather_client as wc
with mock.patch.object(wc, "get_weather_for_points", return_value=empty_weather):
    feat_df = _build_feature_matrix(grid_points, grid_7d, grid_2d)

label_1 = (feat_df["hist_fire_count_24h"] > 0)
label_0 = (feat_df["hist_fire_count_7d"] == 0)
labelled = feat_df.index[label_1 | label_0]
labels = np.where(label_1[labelled], 1, 0)
unique = np.unique(labels)

print(f"Total grid cells:     {len(feat_df)}")
print(f"Pseudo-label class 1: {(labels == 1).sum()} (fire active on last reporting day)")
print(f"Pseudo-label class 0: {(labels == 0).sum()} (no fires in 7-day window)")
print(f"Unique classes:       {unique.tolist()}")

both_classes_ok = len(unique) == 2
print(f"\nBoth classes present: {'✅ PASS' if both_classes_ok else '❌ FAIL'}")
checks.append(both_classes_ok)

# ── Final result ─────────────────────────────────────────────────────────────
print("\n" + "=" * 80)
all_passed = all(checks)
print(f"Overall: {'✅ ALL CHECKS PASSED' if all_passed else '❌ SOME CHECKS FAILED'}")
sys.exit(0 if all_passed else 1)
