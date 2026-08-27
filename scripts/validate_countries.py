"""
scripts/validate_countries.py — Lightweight multi-country validation (Task 4).

Fetches real Sentinel-2/HLS tiles for 4 countries not previously validated,
runs classify_tile() with calibrated temperature, and checks NDVI consistency.

Countries chosen for biome diversity:
  Brazil    — tropical Amazon rainforest  (expected Forest_Vegetation / high NDVI)
  India     — mixed Indo-Gangetic agriculture + monsoon-season crops
  Australia — arid/semi-arid scrub + desert (expected Bare_Sparse / low NDVI)
  Mexico    — mixed subtropical/semi-arid (montane forests + cropland)

Each country: 5 tiles spread across the country bbox in a coarse grid,
different sub-windows to maximise scene diversity.

Usage:
    python3 scripts/validate_countries.py

Output: JSON report in reports/country_validation_task4.json
"""

from __future__ import annotations

import json
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import config
import sentinel_fetch as sf
import landcover_classifier as lc

# ── Countries to validate ─────────────────────────────────────────────────────
# Each entry: (name, rationale, expected_biome, bbox_key_in_config)
VALIDATION_COUNTRIES = [
    ("Brazil",    "Amazon tropical rainforest — should show Forest_Vegetation + NDVI > 0.5",   "Forest_Vegetation"),
    ("India",     "Indo-Gangetic mixed agriculture — should show Cropland + moderate NDVI",     "Cropland"),
    ("Australia", "Central/NW arid scrub — should show Bare_Sparse + NDVI < 0.2",              "Bare_Sparse"),
    ("Mexico",    "Mixed subtropical/semi-arid — diverse classes expected",                      "Forest_Vegetation"),
]

# Use a recent date range that has good HLS coverage (recent 90 days)
TODAY     = datetime.utcnow()
DATE_END  = TODAY.strftime("%Y-%m-%d")
DATE_START = (TODAY - timedelta(days=90)).strftime("%Y-%m-%d")

# Tiles per country: we sample 5 sub-bboxes spread across the country bbox
TILES_PER_COUNTRY = 5

# ── Sub-bbox sampling ─────────────────────────────────────────────────────────

def sample_subbboxes(bbox_str: str, count: int) -> list[str]:
    """
    Divide country bbox into count roughly equal sub-bboxes arranged in a coarse grid.
    Returns count sub-bbox strings in 'W,S,E,N' format.
    """
    import math
    bb_w, bb_s, bb_e, bb_n = [float(x) for x in bbox_str.split(",")]
    lon_span = bb_e - bb_w
    lat_span = bb_n - bb_s
    # Arrange in ~sqrt(count) × sqrt(count) grid (1×5, 2×3, etc.)
    cols = max(2, round(math.sqrt(count)))
    rows = int(math.ceil(count / cols))
    sub_w = lon_span / cols
    sub_h = lat_span / rows
    bboxes = []
    idx = 0
    for r in range(rows):
        for c in range(cols):
            if idx >= count:
                break
            sw = bb_w + c * sub_w
            se = sw + sub_w
            ss = bb_s + r * sub_h
            sn = ss + sub_h
            # Use the centre ± a modest margin (not the whole sub-cell — keep tiles focused)
            cx = (sw + se) / 2
            cy = (ss + sn) / 2
            margin = min(sub_w, sub_h) * 0.35
            bboxes.append(f"{cx-margin:.4f},{cy-margin:.4f},{cx+margin:.4f},{cy+margin:.4f}")
            idx += 1
    return bboxes[:count]


# ── NDVI helper ───────────────────────────────────────────────────────────────

def compute_ndvi_stats(B4: np.ndarray, B8: np.ndarray) -> dict:
    """Compute NDVI stats for the valid land pixels in a band pair."""
    B4f = np.where(np.isnan(B4), np.nan, B4.astype(float))
    B8f = np.where(np.isnan(B8), np.nan, B8.astype(float))
    denom = B8f + B4f
    ndvi  = np.where(denom > 0, (B8f - B4f) / (denom + 1e-9), np.nan)
    ndvi  = np.clip(ndvi, -1.0, 1.0)
    valid = ndvi[~np.isnan(ndvi)]
    if len(valid) == 0:
        return {"mean": float("nan"), "veg_pct": float("nan")}
    return {
        "mean":    round(float(np.nanmean(ndvi)), 3),
        "veg_pct": round(float(100 * np.nanmean(ndvi > 0.3)), 1),
    }


# ── Crop selection (same logic as app.py, with Fmask cloud filter) ───────────

def best_crop(B4, B3, B2, B8, Fmask=None, crop_half=512):
    """
    Select the 1024×1024 window with most valid cloud-free land pixels.

    Mirrors app.py's crop selection logic, extended with the Fmask cloud
    filter (HLS HLSS30 v2.0 bits 1/2/3: cloud, adjacent-to-cloud,
    cloud shadow).  When Fmask is None, falls back to the B4-only criterion.
    """
    import sentinel_fetch as _sf_mod
    h, w = B4.shape
    _LAND_MIN_REFL = 0.001

    # Build cloud mask once for the full tile
    _cloud_mask = _sf_mod.is_cloud_contaminated(Fmask) if Fmask is not None else None

    def _valid_win(b4_win, y0, x0, y1, x1):
        refl_ok = b4_win > _LAND_MIN_REFL
        if _cloud_mask is not None:
            return refl_ok & (~_cloud_mask[y0:y1, x0:x1])
        return refl_ok

    best_y0, best_x0, best_valid = None, None, 0.0
    for _gy in range(1, 6):
        for _gx in range(1, 6):
            cy = int(h * _gy / 6); cx = int(w * _gx / 6)
            y0 = max(0, cy - crop_half); y1 = min(h, cy + crop_half)
            x0 = max(0, cx - crop_half); x1 = min(w, cx + crop_half)
            if (y1 - y0) < 64 or (x1 - x0) < 64:
                continue
            win = B4[y0:y1, x0:x1]
            vf  = float(np.nanmean(_valid_win(win, y0, x0, y1, x1)))
            if vf > best_valid:
                best_valid, best_y0, best_x0 = vf, y0, x0

    if best_y0 is None:
        if _cloud_mask is not None:
            land_clear = (B4 > _LAND_MIN_REFL) & (~_cloud_mask)
        else:
            land_clear = B4 > _LAND_MIN_REFL
        row_v = land_clear.sum(axis=1)
        col_v = land_clear.sum(axis=0)
        if row_v.max() == 0:
            return None, None, 0.0  # fully cloud-covered — no valid crop
        best_y0 = max(0, int(row_v.argmax()) - crop_half)
        best_x0 = max(0, int(col_v.argmax()) - crop_half)

    y0_c = best_y0
    y1_c = min(h, y0_c + 2 * crop_half)
    x0_c = best_x0
    x1_c = min(w, x0_c + 2 * crop_half)

    # Recompute valid fraction over the chosen window with cloud filter applied
    _fb_b4 = B4[y0_c:y1_c, x0_c:x1_c]
    best_valid = float(np.nanmean(_valid_win(_fb_b4, y0_c, x0_c, y1_c, x1_c)))

    tile_r = np.clip(np.nan_to_num(B4[y0_c:y1_c, x0_c:x1_c], nan=0.0), 0, None)
    tile_g = np.clip(np.nan_to_num(B3[y0_c:y1_c, x0_c:x1_c], nan=0.0), 0, None)
    tile_b = np.clip(np.nan_to_num(B2[y0_c:y1_c, x0_c:x1_c], nan=0.0), 0, None)
    tile_b8 = np.clip(np.nan_to_num(B8[y0_c:y1_c, x0_c:x1_c], nan=0.0), 0, None)
    tile_rgb = np.stack([tile_r, tile_g, tile_b], axis=-1)
    return tile_rgb, tile_b8, best_valid


# ── Consistency check ─────────────────────────────────────────────────────────

def is_prediction_plausible(pred_class: str, ndvi_mean: float, ndvi_veg_pct: float) -> tuple[bool, str]:
    """
    Check whether the model prediction is physically consistent with NDVI.
    Returns (plausible: bool, reason: str).
    """
    if np.isnan(ndvi_mean):
        return None, "NDVI unavailable"
    # Vegetation classes should have NDVI > 0.3 on a meaningful fraction of pixels
    veg_classes   = {"Forest_Vegetation"}
    crop_classes  = {"Cropland"}
    water_classes = {"Water", "Wetland"}
    bare_classes  = {"Bare_Sparse"}
    built_classes = {"Built_up"}

    if pred_class in veg_classes:
        if ndvi_mean > 0.35:
            return True,  f"Forest_Vegetation predicted, NDVI={ndvi_mean:.3f} (consistent)"
        elif ndvi_mean > 0.15:
            return None,  f"Forest_Vegetation predicted, NDVI={ndvi_mean:.3f} (borderline)"
        else:
            return False, f"Forest_Vegetation predicted but NDVI={ndvi_mean:.3f} — likely cloud/bare"
    elif pred_class in crop_classes:
        if 0.10 < ndvi_mean < 0.65:
            return True,  f"Cropland predicted, NDVI={ndvi_mean:.3f} (consistent)"
        elif ndvi_mean > 0.65:
            return None,  f"Cropland predicted, NDVI={ndvi_mean:.3f} (high — may be dense vegetation)"
        else:
            return False, f"Cropland predicted, NDVI={ndvi_mean:.3f} (too low for cropland)"
    elif pred_class in water_classes:
        if ndvi_mean < 0.15:
            return True,  f"{pred_class} predicted, NDVI={ndvi_mean:.3f} (consistent)"
        else:
            return None,  f"{pred_class} predicted, NDVI={ndvi_mean:.3f} (borderline)"
    elif pred_class in bare_classes:
        if ndvi_mean < 0.25:
            return True,  f"Bare_Sparse predicted, NDVI={ndvi_mean:.3f} (consistent)"
        elif ndvi_mean < 0.40:
            return None,  f"Bare_Sparse predicted, NDVI={ndvi_mean:.3f} (borderline)"
        else:
            return False, f"Bare_Sparse predicted but NDVI={ndvi_mean:.3f} — likely vegetated scene"
    elif pred_class in built_classes:
        if ndvi_mean < 0.40:
            return True,  f"Built_up predicted, NDVI={ndvi_mean:.3f} (consistent)"
        elif ndvi_mean < 0.55:
            return None,  f"Built_up predicted, NDVI={ndvi_mean:.3f} (borderline)"
        else:
            return False, f"Built_up predicted but NDVI={ndvi_mean:.3f} — dense vegetation present"
    return None, f"Unknown class {pred_class}"


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("Task 4 — Lightweight multi-country validation")
    print(f"Date range: {DATE_START} → {DATE_END}")
    print("=" * 70)

    # Load model once
    model = lc.load_landcover_model()
    if model is None:
        print("[ERROR] Could not load land cover model")
        sys.exit(1)
    T = lc._TEMPERATURE
    print(f"Model: {lc._model_version}  T={T:.4f}")

    report = {
        "generated_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M UTC"),
        "model_version": lc._model_version,
        "temperature": T,
        "date_range": [DATE_START, DATE_END],
        "countries": [],
    }

    for country_name, rationale, expected_class in VALIDATION_COUNTRIES:
        print(f"\n{'─'*60}")
        print(f"Country: {country_name}")
        print(f"Rationale: {rationale}")
        bbox_str = config.COUNTRY_BBOX.get(country_name)
        if not bbox_str:
            print(f"  [SKIP] {country_name} not in config.COUNTRY_BBOX")
            continue

        subbboxes = sample_subbboxes(bbox_str, TILES_PER_COUNTRY)
        print(f"  {len(subbboxes)} sub-bboxes sampled")

        country_results = []
        country_t_start = time.monotonic()

        for ti, sub_bbox in enumerate(subbboxes):
            print(f"\n  Tile {ti+1}/{len(subbboxes)}  bbox={sub_bbox}")
            tile_t_start = time.monotonic()
            try:
                result = sf.fetch_sentinel2_tile(
                    bbox=sub_bbox,
                    date_range=(DATE_START, DATE_END),
                    max_cloud_cover=35.0,
                    timeout_seconds=120.0,
                )
            except Exception as exc:
                elapsed = round(time.monotonic() - tile_t_start, 1)
                print(f"  [FETCH ERROR in {elapsed}s] {exc}")
                country_results.append({
                    "tile": ti + 1,
                    "bbox": sub_bbox,
                    "status": "fetch_error",
                    "error": str(exc),
                    "fetch_seconds": elapsed,
                })
                continue

            elapsed_fetch = round(time.monotonic() - tile_t_start, 1)
            print(f"  Fetch OK in {elapsed_fetch}s — "
                  f"granule={result['granule_id']}  "
                  f"date={result['acquisition_date']}  "
                  f"cloud={result['cloud_cover']}%")

            B2, B3, B4, B8 = result["B2"], result["B3"], result["B4"], result["B8"]
            Fmask = result.get("Fmask")  # may be None for older fetches
            if B4 is None or B8 is None or B3 is None or B2 is None:
                missing = [k for k,v in [("B2",B2),("B3",B3),("B4",B4),("B8",B8)] if v is None]
                print(f"  [SKIP] Missing bands: {missing}")
                country_results.append({
                    "tile": ti + 1,
                    "bbox": sub_bbox,
                    "status": "missing_bands",
                    "missing": missing,
                    "fetch_seconds": elapsed_fetch,
                })
                continue

            fmask_loaded = Fmask is not None
            print(f"  Fmask: {'loaded' if fmask_loaded else 'unavailable (no cloud filter)'}")

            # Crop selection with cloud filter
            result_crop = best_crop(B4, B3, B2, B8, Fmask=Fmask)
            tile_rgb, tile_b8, valid_frac = result_crop

            if tile_rgb is None:
                # Fully cloud-covered — no valid cloud-free crop
                print(f"  [SKIP] No cloud-free land pixels in granule (Fmask filtered all)")
                country_results.append({
                    "tile": ti + 1,
                    "bbox": sub_bbox,
                    "status": "fully_clouded",
                    "granule_id": result["granule_id"],
                    "acquisition_date": result["acquisition_date"],
                    "cloud_cover": result["cloud_cover"],
                    "fmask_loaded": fmask_loaded,
                    "fetch_seconds": elapsed_fetch,
                })
                continue

            # Crop NDVI
            tile_r = tile_rgb[:,:,0]
            ndvi_stats = compute_ndvi_stats(tile_r, tile_b8)
            print(f"  crop valid={valid_frac*100:.0f}%  NDVI mean={ndvi_stats['mean']:.3f}  "
                  f"veg>0.3={ndvi_stats['veg_pct']:.1f}%  "
                  f"fmask={'active' if fmask_loaded else 'off'}")

            # Classify
            classify_t = time.monotonic()
            lc_pred = lc.classify_tile(model, tile_rgb)
            classify_elapsed = round(time.monotonic() - classify_t, 2)

            pred_class = lc_pred["class"]
            confidence = round(lc_pred["confidence"] * 100, 1)
            print(f"  → {pred_class}  {confidence}%  (classify in {classify_elapsed}s)")

            # Consistency check
            plausible, reason = is_prediction_plausible(
                pred_class, ndvi_stats["mean"], ndvi_stats["veg_pct"]
            )
            plausible_label = {True: "plausible", False: "implausible", None: "borderline"}.get(plausible, "unknown")
            print(f"  consistency: {plausible_label} — {reason}")

            country_results.append({
                "tile": ti + 1,
                "bbox": sub_bbox,
                "status": "ok",
                "granule_id": result["granule_id"],
                "acquisition_date": result["acquisition_date"],
                "cloud_cover": result["cloud_cover"],
                "fmask_loaded": fmask_loaded,
                "predicted_class": pred_class,
                "confidence_pct": confidence,
                "ndvi_mean": ndvi_stats["mean"],
                "ndvi_veg_pct": ndvi_stats["veg_pct"],
                "valid_land_frac_pct": round(valid_frac * 100, 1),
                "ndvi_consistency": plausible_label,
                "consistency_reason": reason,
                "fetch_seconds": elapsed_fetch,
                "classify_seconds": classify_elapsed,
            })

        # Summarise country
        total_elapsed = round(time.monotonic() - country_t_start, 1)
        ok_tiles = [r for r in country_results if r["status"] == "ok"]
        if ok_tiles:
            plausible_n   = sum(1 for r in ok_tiles if r["ndvi_consistency"] == "plausible")
            borderline_n  = sum(1 for r in ok_tiles if r["ndvi_consistency"] == "borderline")
            implausible_n = sum(1 for r in ok_tiles if r["ndvi_consistency"] == "implausible")
            # Assign country bucket
            if implausible_n > len(ok_tiles) / 2:
                bucket = "Unreliable"
            elif plausible_n >= len(ok_tiles) * 0.7:
                bucket = "Plausible"
            else:
                bucket = "Mixed"
        else:
            bucket = "no_data"
            plausible_n = borderline_n = implausible_n = 0

        print(f"\n  SUMMARY — {country_name}: {bucket}")
        print(f"    OK tiles: {len(ok_tiles)}/{len(subbboxes)}  "
              f"plausible={plausible_n}  borderline={borderline_n}  implausible={implausible_n}")
        print(f"    Total elapsed: {total_elapsed}s")

        report["countries"].append({
            "name": country_name,
            "rationale": rationale,
            "expected_class": expected_class,
            "bbox": bbox_str,
            "tiles_attempted": len(subbboxes),
            "tiles_ok": len(ok_tiles),
            "bucket": bucket,
            "plausible_tiles": plausible_n,
            "borderline_tiles": borderline_n,
            "implausible_tiles": implausible_n,
            "total_fetch_seconds": total_elapsed,
            "tile_results": country_results,
        })

    # Save report
    report_path = ROOT / "reports" / "country_validation_task4.json"
    report_path.parent.mkdir(exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"\n{'='*70}")
    print(f"Report saved → {report_path}")
    print(f"{'='*70}")

    # Final summary table
    print("\nFINAL BUCKETS:")
    for c in report["countries"]:
        print(f"  {c['name']:12s} → {c['bucket']:12s}  "
              f"({c['tiles_ok']}/{c['tiles_attempted']} tiles ok, "
              f"{c['plausible_tiles']}P/{c['borderline_tiles']}B/{c['implausible_tiles']}I)")


if __name__ == "__main__":
    main()
