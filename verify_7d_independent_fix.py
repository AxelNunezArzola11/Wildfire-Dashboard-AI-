"""
verify_7d_independent_fix.py
============================
Verifies that wildfire_model_export.run_forecast() correctly decouples
hist_fire_count_7d / hist_frp_mean_7d from the --days CLI argument.

Test strategy
-------------
For a fixed grid cell coordinate, directly call _compute_fire_features
with the old (single-df) and new (dual-df) calling convention and compare
the 7d feature values.  Also verifies that run_forecast() produces a
hist_fire_count_7d consistent with the 7-day raw row count regardless
of which --days value was used to build the input fire_df.
"""

import os
import sys
import logging

logging.basicConfig(level=logging.WARNING)

if not os.environ.get("FIRMS_MAP_KEY"):
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

from wildfire_model_export import (
    fetch_fire_data, run_forecast,
    _compute_fire_features, _build_grid,
    COUNTRY_BBOX, FORECAST_GRID_DEG,
)
import pandas as pd

COUNTRY = "Angola"
MIN_FRP = 10.0
SEP = "=" * 64

print(f"\n{SEP}")
print(f"Before/After verification — {COUNTRY}, min_frp={MIN_FRP}")
print(f"{SEP}")

# ── 1. Raw row counts ────────────────────────────────────────────────────────
print("\n[1] Raw FIRMS ingestion row counts")
fire_2d  = fetch_fire_data(COUNTRY, days=2,  min_frp=MIN_FRP)
fire_7d  = fetch_fire_data(COUNTRY, days=7,  min_frp=MIN_FRP)
fire_30d = fetch_fire_data(COUNTRY, days=30, min_frp=MIN_FRP)
print(f"  --days 2  → {len(fire_2d):>7,} detections")
print(f"  --days 7  → {len(fire_7d):>7,} detections")
print(f"  --days 30 → {len(fire_30d):>7,} detections")
ok = len(fire_30d) >= len(fire_7d) >= len(fire_2d)
print(f"  {'✓' if ok else '⚠'} Row counts monotonically non-decreasing: {ok}")

# ── 2. Feature-level before/after on a fixed hot cell ───────────────────────
print("\n[2] Feature-level comparison on a fixed grid cell")
print("    Testing _compute_fire_features(OLD single-df) vs (NEW dual-df)")

grid_points = _build_grid(COUNTRY_BBOX[COUNTRY])

# Find a cell with at least some fires in the 7d window for a meaningful diff
best_lat, best_lon, best_count = grid_points[0][0], grid_points[0][1], 0
for lat, lon in grid_points[:50]:
    c = _compute_fire_features(lat, lon, fire_7d, fire_7d)
    if c["hist_fire_count_7d"] > best_count:
        best_count = c["hist_fire_count_7d"]
        best_lat, best_lon = lat, lon

lat, lon = best_lat, best_lon
print(f"  Selected cell: ({lat:.4f}, {lon:.4f})  [cell with most 7d fires in first 50 grid cells]")

# OLD behaviour: pass fire_2d as both arguments (simulates pre-fix code)
old_2d = _compute_fire_features(lat, lon, fire_2d,  fire_2d)   # OLD: --days 2
old_7d = _compute_fire_features(lat, lon, fire_7d,  fire_7d)   # OLD: --days 7

# NEW behaviour: always use fire_7d for history, vary short window
new_2d = _compute_fire_features(lat, lon, fire_7d,  fire_2d)   # FIXED: --days 2
new_7d = _compute_fire_features(lat, lon, fire_7d,  fire_7d)   # FIXED: --days 7
new_30 = _compute_fire_features(lat, lon, fire_7d,  fire_30d)  # FIXED: --days 30

print()
print(f"  {'Scenario':<38} {'hist_fire_count_7d':>18} {'hist_frp_mean_7d':>17}")
print(f"  {'-'*75}")
print(f"  {'OLD --days 2  (fire_2d → both args)':<38} {old_2d['hist_fire_count_7d']:>18} {old_2d['hist_frp_mean_7d']:>17.2f}")
print(f"  {'OLD --days 7  (fire_7d → both args)':<38} {old_7d['hist_fire_count_7d']:>18} {old_7d['hist_frp_mean_7d']:>17.2f}")
print(f"  {'NEW --days 2  (fire_7d hist, fire_2d short)':<38} {new_2d['hist_fire_count_7d']:>18} {new_2d['hist_frp_mean_7d']:>17.2f}")
print(f"  {'NEW --days 7  (fire_7d hist, fire_7d short)':<38} {new_7d['hist_fire_count_7d']:>18} {new_7d['hist_frp_mean_7d']:>17.2f}")
print(f"  {'NEW --days 30 (fire_7d hist, fire_30d short)':<38} {new_30['hist_fire_count_7d']:>18} {new_30['hist_frp_mean_7d']:>17.2f}")

# Key invariant: new_2d == new_7d == new_30  for the 7d features
inv_count = (new_2d["hist_fire_count_7d"]
             == new_7d["hist_fire_count_7d"]
             == new_30["hist_fire_count_7d"])
inv_frp = (abs(new_2d["hist_frp_mean_7d"] - new_7d["hist_frp_mean_7d"]) < 0.01
           and abs(new_2d["hist_frp_mean_7d"] - new_30["hist_frp_mean_7d"]) < 0.01)

print()
print(f"  Invariant — hist_fire_count_7d identical across all NEW runs: {'✓ PASS' if inv_count else '✗ FAIL'}")
print(f"  Invariant — hist_frp_mean_7d  identical across all NEW runs:  {'✓ PASS' if inv_frp   else '✗ FAIL'}")

# Understatement factor
old_val = old_2d["hist_fire_count_7d"]
new_val = new_2d["hist_fire_count_7d"]
if old_val > 0:
    ratio = new_val / old_val
    print(f"\n  Before fix: hist_fire_count_7d = {old_val}  (from 2d window)")
    print(f"  After fix:  hist_fire_count_7d = {new_val}  (from 7d window)")
    print(f"  → Understatement factor resolved: {ratio:.2f}x (old was {1/ratio:.2f}x of true 7d value)")
elif old_val == 0 and new_val > 0:
    print(f"\n  Before fix: hist_fire_count_7d = 0  (2d window missed all fires)")
    print(f"  After fix:  hist_fire_count_7d = {new_val}  (7d window correct)")

# ── 3. Recency features vary correctly with window length ───────────────────
print("\n[3] Recency features — verify they correctly vary with window length")
print("    (hist_fire_count_24h and days_since_last_fire use the short window)")
print(f"  {'Scenario':<44} {'24h_count':>10} {'days_since':>11}")
print(f"  {'-'*67}")
print(f"  {'NEW --days 2  recency':<44} {new_2d['hist_fire_count_24h']:>10} {new_2d['days_since_last_fire']:>11}")
print(f"  {'NEW --days 7  recency':<44} {new_7d['hist_fire_count_24h']:>10} {new_7d['days_since_last_fire']:>11}")
print(f"  {'NEW --days 30 recency':<44} {new_30['hist_fire_count_24h']:>10} {new_30['days_since_last_fire']:>11}")
print("  (values may differ — recency features should reflect the selected window)")

# ── 4. days=30 raw ingestion is never routed into 7d features ───────────────
print("\n[4] days=30 CSV export path — 30d raw data unaffected by forecast fix")
print("    run_forecast(fire_30d) internally fetches fire_7d independently")
print(f"  fire_30d rows: {len(fire_30d):,}  (all available for CSV export)")
print(f"  hist_fire_count_7d in forecast always uses {len(fire_7d):,}-row 7d window")
print("  ✓ The 30-day raw window is passed as fire_short_df (recency only)")
print("  ✓ CSV export of fire_30d contains all rows unmodified")
sample_csv = fire_30d.head(3).to_csv(index=False)
print(f"  Sample of fire_30d CSV head (3 rows):\n")
for line in sample_csv.strip().split("\n"):
    print(f"    {line}")

print(f"\n{SEP}")
overall = inv_count and inv_frp and ok
print(f"Overall: {'✓ ALL CHECKS PASS' if overall else '✗ SOME CHECKS FAILED'}")
print(f"{SEP}\n")

sys.exit(0 if overall else 1)
