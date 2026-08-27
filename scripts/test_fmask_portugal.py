"""
scripts/test_fmask_portugal.py — Verify Fmask cloud filter against the known
bad Portugal case documented in JUDGE.md (26% cloud cover, NDVI=0.235,
previously predicted Water at high confidence due to cloud contamination).

Fetches the Portugal bbox with same date range used in the prior bad run
(2026-05-03 ±7 days), confirms:
  1. Fmask is loaded and has real unique values
  2. The previous "best crop" window now has a lower valid-pixel fraction
     when cloud pixels are removed
  3. The new best crop (if any) has plausible NDVI and classification
  4. If no cloud-free crop exists, the "no valid land crop" path fires

Usage:
    python3 scripts/test_fmask_portugal.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import sentinel_fetch as sf
import landcover_classifier as lc

PORTUGAL_BBOX = "-9.53,36.96,-6.19,42.15"

# The bad granule was from 2026-05-03 (26% cloud cover).
# Search ±14 days around that date to find it again.
DATE_START = "2026-04-19"
DATE_END   = "2026-05-17"

# Previous bad crop coordinates from JUDGE.md
PREV_Y0, PREV_Y1 = 98, 1122
PREV_X0, PREV_X1 = 2538, 3562

_LAND_MIN_REFL = 0.001
CROP_HALF = 512


def compute_valid_frac_old(B4_win: np.ndarray) -> float:
    """Old validity: only positive reflectance (no cloud filter)."""
    return float(np.nanmean(B4_win > _LAND_MIN_REFL))


def compute_valid_frac_new(B4_win: np.ndarray, fmask_win: np.ndarray | None) -> float:
    """New validity: positive reflectance AND not cloud-contaminated."""
    refl_ok = B4_win > _LAND_MIN_REFL
    if fmask_win is not None:
        cloud_ok = ~sf.is_cloud_contaminated(fmask_win)
        return float(np.nanmean(refl_ok & cloud_ok))
    return float(np.nanmean(refl_ok))


def best_crop_new(B4, B3, B2, B8, fmask, crop_half=512):
    """Crop selection with Fmask cloud filter (mirrors updated app.py)."""
    h, w = B4.shape
    cloud_mask = sf.is_cloud_contaminated(fmask) if fmask is not None else None

    def valid_win(b4_win, y0, x0, y1, x1):
        refl = b4_win > _LAND_MIN_REFL
        if cloud_mask is not None:
            return refl & (~cloud_mask[y0:y1, x0:x1])
        return refl

    best_y0, best_x0, best_vf = None, None, 0.0
    for gy in range(1, 6):
        for gx in range(1, 6):
            cy = int(h * gy / 6); cx = int(w * gx / 6)
            y0 = max(0, cy - crop_half); y1 = min(h, cy + crop_half)
            x0 = max(0, cx - crop_half); x1 = min(w, cx + crop_half)
            if (y1 - y0) < 64 or (x1 - x0) < 64:
                continue
            vf = float(np.nanmean(valid_win(B4[y0:y1, x0:x1], y0, x0, y1, x1)))
            if vf > best_vf:
                best_vf, best_y0, best_x0 = vf, y0, x0

    if best_y0 is None or best_vf == 0.0:
        if cloud_mask is not None:
            land_clear = (B4 > _LAND_MIN_REFL) & (~cloud_mask)
        else:
            land_clear = B4 > _LAND_MIN_REFL
        rsum = land_clear.sum(axis=1)
        csum = land_clear.sum(axis=0)
        if rsum.max() == 0:
            return None, None, None, 0.0, "no_cloud_free_land"
        best_y0 = max(0, int(rsum.argmax()) - crop_half)
        best_x0 = max(0, int(csum.argmax()) - crop_half)

    y0_c = best_y0
    y1_c = min(h, y0_c + 2 * crop_half)
    x0_c = best_x0
    x1_c = min(w, x0_c + 2 * crop_half)
    final_vf = float(np.nanmean(valid_win(B4[y0_c:y1_c, x0_c:x1_c], y0_c, x0_c, y1_c, x1_c)))

    tile_r = np.clip(np.nan_to_num(B4[y0_c:y1_c, x0_c:x1_c], nan=0.0), 0, None)
    tile_g = np.clip(np.nan_to_num(B3[y0_c:y1_c, x0_c:x1_c], nan=0.0), 0, None)
    tile_b = np.clip(np.nan_to_num(B2[y0_c:y1_c, x0_c:x1_c], nan=0.0), 0, None)
    tile_b8 = np.clip(np.nan_to_num(B8[y0_c:y1_c, x0_c:x1_c], nan=0.0), 0, None)
    tile_rgb = np.stack([tile_r, tile_g, tile_b], axis=-1)
    return tile_rgb, tile_b8, (y0_c, y1_c, x0_c, x1_c), final_vf, "ok"


def main():
    print("=" * 70)
    print("Portugal Fmask cloud-filter test")
    print(f"  bbox      : {PORTUGAL_BBOX}")
    print(f"  date range: {DATE_START} → {DATE_END}")
    print(f"  known bad crop (old): y=[{PREV_Y0},{PREV_Y1}] x=[{PREV_X0},{PREV_X1}]  valid=68% (B4-only)")
    print("=" * 70)

    t0 = time.monotonic()
    try:
        result = sf.fetch_sentinel2_tile(
            bbox=PORTUGAL_BBOX,
            date_range=(DATE_START, DATE_END),
            max_cloud_cover=50.0,  # allow the bad granule through
            timeout_seconds=180.0,
        )
    except Exception as exc:
        print(f"[FETCH ERROR] {exc}")
        sys.exit(1)

    elapsed = time.monotonic() - t0
    print(f"\nFetch OK in {elapsed:.1f}s")
    print(f"  granule_id     : {result['granule_id']}")
    print(f"  acquisition_date: {result['acquisition_date']}")
    print(f"  cloud_cover    : {result['cloud_cover']:.1f}%")
    print(f"  bands loaded   : {[k for k,v in result.items() if isinstance(v, np.ndarray)]}")

    B4, B8, B3, B2 = result["B4"], result["B8"], result["B3"], result["B2"]
    fmask = result.get("Fmask")

    if fmask is not None:
        fm_valid = fmask[~np.isnan(fmask)]
        print(f"\nFmask array: shape={fmask.shape}  dtype={fmask.dtype}")
        print(f"  unique values (first 500 px): {np.unique(fm_valid[:500].astype(np.uint8)).tolist()}")
        cloud_mask = sf.is_cloud_contaminated(fmask)
        cloud_pct = float(100 * cloud_mask.mean())
        print(f"  cloud-contaminated pixels: {cloud_pct:.1f}%")
    else:
        print("\nFmask: NOT loaded — cloud filter unavailable")

    # ── Compare old vs new valid fraction on the previous bad crop ──────────
    if B4 is not None and fmask is not None:
        prev_b4_win  = B4[PREV_Y0:PREV_Y1, PREV_X0:PREV_X1]
        prev_fm_win  = fmask[PREV_Y0:PREV_Y1, PREV_X0:PREV_X1]
        old_vf = compute_valid_frac_old(prev_b4_win)
        new_vf = compute_valid_frac_new(prev_b4_win, prev_fm_win)
        print(f"\nPrevious bad crop y=[{PREV_Y0},{PREV_Y1}] x=[{PREV_X0},{PREV_X1}]:")
        print(f"  Old valid frac (B4-only):             {old_vf*100:.1f}%")
        print(f"  New valid frac (B4 + Fmask filter):   {new_vf*100:.1f}%")
        delta = old_vf - new_vf
        if delta > 0.05:
            print(f"  → Cloud filter removed {delta*100:.1f}pp of pixels  ✅ confirms contamination")
        else:
            print(f"  → Cloud filter changed valid frac by only {delta*100:.1f}pp  (scene may be clear)")

    # ── Find new best crop with cloud filter active ──────────────────────────
    if B4 is not None:
        model = lc.load_landcover_model()
        tile_rgb, tile_b8, coords, final_vf, status = best_crop_new(B4, B3, B2, B8, fmask)

        if status == "no_cloud_free_land":
            print("\n[EXPECTED FALLBACK] No cloud-free land pixels in this granule.")
            print("  → System would report: 'no valid land crop — try a different date range'")
            print("  → Behaviour: honest refusal (no silent fallback to cloud crop)  ✅")
        else:
            y0_c, y1_c, x0_c, x1_c = coords
            tile_r = tile_rgb[:, :, 0]
            # NDVI
            crop_b8 = tile_b8.astype(float)
            crop_r  = tile_r.astype(float)
            ndvi_arr = np.clip((crop_b8 - crop_r) / (crop_b8 + crop_r + 1e-9), -1, 1)
            ndvi_mean = float(np.mean(ndvi_arr))
            veg_pct   = float(100 * np.mean(ndvi_arr > 0.3))

            lc_pred   = lc.classify_tile(model, tile_rgb)
            pred_class = lc_pred["class"]
            conf       = lc_pred["confidence"]

            print(f"\nNew best crop (Fmask-filtered):")
            print(f"  y=[{y0_c},{y1_c}] x=[{x0_c},{x1_c}]")
            print(f"  valid-pixel fraction: {final_vf*100:.1f}%")
            print(f"  NDVI mean: {ndvi_mean:.3f}   veg>0.3: {veg_pct:.1f}%")
            print(f"  Predicted: {pred_class}  confidence={conf*100:.1f}%")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
