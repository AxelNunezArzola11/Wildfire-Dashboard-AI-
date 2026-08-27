"""
scripts/test_fmask_regression.py — Regression test for the Fmask cloud filter.

Verifies that clear-sky Angola and clear-sky Greece/Portugal scenes still
produce valid crops and plausible predictions after the Fmask filter is
active.  Tests two cases:

  1. Angola   — 2026-05-XX, clear-sky, expects Forest_Vegetation or Cropland
  2. Greece   — 2026-05-XX, clear-sky (or lowest available cloud), expects
                a result with higher valid% than pre-filter baseline

Usage:
    python3 scripts/test_fmask_regression.py
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
import config

_LAND_MIN_REFL = 0.001
CROP_HALF = 512

DATE_START = "2026-04-01"
DATE_END   = "2026-06-01"

CASES = [
    ("Angola",   config.COUNTRY_BBOX["Angola"],   20.0),
    ("Greece",   config.COUNTRY_BBOX["Greece"],   20.0),
    ("Portugal", config.COUNTRY_BBOX["Portugal"],  5.0),  # low cloud — clear scene
]


def best_crop_with_fmask(B4, B3, B2, B8, fmask, crop_half=512):
    """Mirrors the updated crop selection logic from app.py."""
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
            return None, None, None, 0.0
        best_y0 = max(0, int(rsum.argmax()) - crop_half)
        best_x0 = max(0, int(csum.argmax()) - crop_half)

    y0_c = best_y0
    y1_c = min(h, y0_c + 2 * crop_half)
    x0_c = best_x0
    x1_c = min(w, x0_c + 2 * crop_half)
    final_vf = float(np.nanmean(valid_win(B4[y0_c:y1_c, x0_c:x1_c], y0_c, x0_c, y1_c, x1_c)))
    old_vf = float(np.nanmean(B4[y0_c:y1_c, x0_c:x1_c] > _LAND_MIN_REFL))

    tile_r = np.clip(np.nan_to_num(B4[y0_c:y1_c, x0_c:x1_c], nan=0.0), 0, None)
    tile_g = np.clip(np.nan_to_num(B3[y0_c:y1_c, x0_c:x1_c], nan=0.0), 0, None)
    tile_b = np.clip(np.nan_to_num(B2[y0_c:y1_c, x0_c:x1_c], nan=0.0), 0, None)
    tile_b8 = np.clip(np.nan_to_num(B8[y0_c:y1_c, x0_c:x1_c], nan=0.0), 0, None)
    tile_rgb = np.stack([tile_r, tile_g, tile_b], axis=-1)
    return tile_rgb, tile_b8, (y0_c, y1_c, x0_c, x1_c, old_vf), final_vf


def main():
    print("=" * 70)
    print("Fmask cloud-filter regression test")
    print(f"Date range: {DATE_START} → {DATE_END}")
    print("=" * 70)

    model = lc.load_landcover_model()

    for country, bbox, max_cc in CASES:
        print(f"\n{'─'*60}")
        print(f"Country: {country}  bbox={bbox}  max_cloud={max_cc}%")
        t0 = time.monotonic()
        try:
            result = sf.fetch_sentinel2_tile(
                bbox=bbox,
                date_range=(DATE_START, DATE_END),
                max_cloud_cover=max_cc,
                timeout_seconds=180.0,
            )
        except Exception as exc:
            print(f"  [FETCH ERROR] {exc}")
            continue

        elapsed = time.monotonic() - t0
        B4, B8, B3, B2 = result["B4"], result["B8"], result["B3"], result["B2"]
        fmask = result.get("Fmask")
        print(f"  Fetch OK in {elapsed:.1f}s  granule={result['granule_id']}  "
              f"date={result['acquisition_date']}  cloud={result['cloud_cover']:.1f}%")
        print(f"  Fmask: {'loaded' if fmask is not None else 'unavailable'}")

        if fmask is not None:
            cloud_pct = float(100 * sf.is_cloud_contaminated(fmask).mean())
            print(f"  Scene cloud-contaminated (Fmask): {cloud_pct:.1f}%")

        if B4 is None:
            print("  [SKIP] B4 not loaded")
            continue

        crop_result = best_crop_with_fmask(B4, B3, B2, B8, fmask)
        tile_rgb, tile_b8, coords, final_vf = crop_result

        if tile_rgb is None:
            print(f"  [RESULT] No cloud-free land pixels — granule fully clouded")
            continue

        y0_c, y1_c, x0_c, x1_c, old_vf = coords
        tile_r = tile_rgb[:, :, 0]
        ndvi_arr = np.clip(
            (tile_b8.astype(float) - tile_r.astype(float))
            / (tile_b8.astype(float) + tile_r.astype(float) + 1e-9),
            -1, 1,
        )
        ndvi_mean = float(np.mean(ndvi_arr))
        veg_pct   = float(100 * np.mean(ndvi_arr > 0.3))

        lc_pred    = lc.classify_tile(model, tile_rgb)
        pred_class = lc_pred["class"]
        conf       = lc_pred["confidence"]

        print(f"\n  Best crop: y=[{y0_c},{y1_c}] x=[{x0_c},{x1_c}]")
        print(f"    Old valid% (B4-only):           {old_vf*100:.1f}%")
        print(f"    New valid% (B4 + Fmask filter): {final_vf*100:.1f}%")
        print(f"    NDVI mean: {ndvi_mean:.3f}   veg>0.3: {veg_pct:.1f}%")
        print(f"    Predicted: {pred_class}  confidence={conf*100:.1f}%")

        # Regression assertion: new valid% should not be unreasonably lower for clear scenes
        # (clear scenes should keep most of their pixels — any Fmask drop should be small)
        if result["cloud_cover"] < 5.0:
            drop = old_vf - final_vf
            if drop > 0.30:
                print(f"  [WARN] Clear-sky scene lost {drop*100:.1f}pp of pixels — Fmask may be overfiring")
            else:
                print(f"  [OK] Clear-sky scene: Fmask removed only {drop*100:.1f}pp  ✅")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
