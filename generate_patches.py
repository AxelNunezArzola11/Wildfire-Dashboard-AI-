"""
generate_patches.py — Track 3: stratified Sentinel-2 patch generation for Angola.

Strategy
--------
  1. Load the already-fetched WorldCover raster for Angola (via worldcover_fetch).
  2. Scan it in 0.05° windows; select minority-class-dominant windows using
     per-class purity thresholds (≥40% for Built_up, ≥50% for all others).
  3. For each selected window, fetch the matching HLS S30 Sentinel-2 tile via
     sentinel_fetch.fetch_sentinel2_tile (NASA Earthdata, no re-auth needed).
  4. Crop the fetched granule to the window bbox, resize to 64×64, save as .npy.
  5. Assign label via landcover_schema.WORLDCOVER_TO_GLOBAL6 (majority class in window).
  6. Maintain a resume file (data/patches_angola/progress.json) so the job can be
     interrupted and restarted without re-fetching already-saved patches.
  7. After all fetches complete, perform an 80/20 stratified split and write
     data/patches_angola/train_manifest.csv and data/patches_angola/val_manifest.csv.

Usage
-----
    python3 generate_patches.py                  # run with defaults
    python3 generate_patches.py --workers 3      # parallel workers (default: 3)
    python3 generate_patches.py --dry-run        # print allocation, no fetches
    python3 generate_patches.py --split-only     # skip fetching, just re-split

Output layout
-------------
    data/patches_angola/
        Forest_Vegetation/patch_<lat>_<lon>.npy
        Cropland/patch_<lat>_<lon>.npy
        ...
        progress.json         — {patch_id: "ok"|"error"|"skip"}
        train_manifest.csv    — filepath, global6_class, global6_index, split
        val_manifest.csv

Each .npy is a float32 array of shape (3, 64, 64) — RGB channels (B4, B3, B2)
normalised to [0, 1] reflectance.  NODATA pixels (NaN) are filled with 0.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import os
import random
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np

from config import COUNTRY_BBOX
from landcover_schema import GLOBAL6_CLASSES, GLOBAL6_LABEL, WORLDCOVER_TO_GLOBAL6
from worldcover_fetch import fetch_worldcover_tile

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-7s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

COUNTRY = "Angola"
COUNTRY_BBOX_STR = COUNTRY_BBOX[COUNTRY]
OUTPUT_DIR = Path("data/patches_angola")
PROGRESS_FILE = OUTPUT_DIR / "progress.json"
PATCH_SIZE = 64  # pixels, both H and W

WINDOW_DEG = 0.05          # 0.05° ≈ 5.5 km at equator
PURITY_DEFAULT = 0.50
PURITY_BUILDUP = 0.40      # relaxed — Built_up is rare in Angola

# Per-class fetch caps (how many patches to attempt per class)
CLASS_CAPS: dict[str, int] = {
    "Forest_Vegetation":  50,
    "Cropland":          100,
    "Water":             100,
    "Built_up":           43,   # take all available (43 windows found)
    "Bare_Sparse":       150,
    "Wetland":           150,
}
TOTAL_TARGET = sum(CLASS_CAPS.values())  # 593

# Sentinel-2 fetch settings
DATE_RANGE = ("2023-01-01", "2023-12-31")  # full year for cloud-free coverage
MAX_CLOUD_COVER = 40.0
FETCH_TIMEOUT = 150.0

# Parallel workers — 3 gives ~1.6× speedup without saturating CMR rate limits
DEFAULT_WORKERS = 3

# ---------------------------------------------------------------------------
# Window scanning
# ---------------------------------------------------------------------------

def scan_windows(arr: np.ndarray, bbox_str: str) -> dict[str, list[dict]]:
    """
    Return stratified candidate windows, one dict per class.
    Each window dict: {lat, lon, purity, bbox_str}
    """
    W, S, E, N = (float(v) for v in bbox_str.split(","))
    H, W_px = arr.shape
    deg_per_row = (N - S) / H
    deg_per_col = (E - W) / W_px
    win_rows = max(1, int(round(WINDOW_DEG / deg_per_row)))
    win_cols = max(1, int(round(WINDOW_DEG / deg_per_col)))

    logger.info(
        "Scanning %s at %.2f° windows (%dr × %dc pixels)…",
        COUNTRY, WINDOW_DEG, win_rows, win_cols,
    )

    windows_by_g6: dict[str, list[dict]] = {c: [] for c in GLOBAL6_CLASSES}

    for row in range(0, H - win_rows, win_rows):
        for col in range(0, W_px - win_cols, win_cols):
            patch = arr[row : row + win_rows, col : col + win_cols]
            flat = patch[patch != 255]
            if flat.size == 0:
                continue

            g6_counts: dict[str, int] = {}
            for code in np.unique(flat):
                g6 = WORLDCOVER_TO_GLOBAL6.get(int(code))
                if g6:
                    g6_counts[g6] = g6_counts.get(g6, 0) + int((flat == code).sum())
            if not g6_counts:
                continue

            total = sum(g6_counts.values())
            dominant = max(g6_counts, key=g6_counts.get)
            purity = g6_counts[dominant] / total

            threshold = PURITY_BUILDUP if dominant == "Built_up" else PURITY_DEFAULT
            if purity < threshold:
                continue

            lat_c = N - (row + win_rows / 2) * deg_per_row
            lon_c = W + (col + win_cols / 2) * deg_per_col
            half = WINDOW_DEG / 2
            win_bbox = f"{lon_c - half:.5f},{lat_c - half:.5f},{lon_c + half:.5f},{lat_c + half:.5f}"
            windows_by_g6[dominant].append({
                "lat": round(lat_c, 5),
                "lon": round(lon_c, 5),
                "purity": round(purity, 3),
                "bbox": win_bbox,
            })

    for cls in GLOBAL6_CLASSES:
        n = len(windows_by_g6[cls])
        logger.info("  %-24s %d windows", cls, n)

    return windows_by_g6


def select_candidates(
    windows_by_g6: dict[str, list[dict]],
    rng: random.Random,
) -> list[dict]:
    """
    Apply CLASS_CAPS, sort minority classes by purity (highest first) to
    maximise label quality, shuffle Forest/Crop/Water for spatial diversity.
    Returns a flat list of {cls, lat, lon, bbox, purity} dicts.
    """
    selected: list[dict] = []
    for cls in GLOBAL6_CLASSES:
        wins = windows_by_g6[cls]
        cap = CLASS_CAPS.get(cls, 0)
        if not wins or cap == 0:
            continue

        # Minority classes: prioritise highest purity
        # Majority classes (Forest, Crop, Water): random spatial sample
        if cls in {"Built_up", "Bare_Sparse", "Wetland"}:
            wins_sorted = sorted(wins, key=lambda w: w["purity"], reverse=True)
            chosen = wins_sorted[:cap]
        else:
            rng.shuffle(wins)
            chosen = wins[:cap]

        for w in chosen:
            selected.append({"cls": cls, **w})

    rng.shuffle(selected)   # shuffle fetch order for even per-class rate
    return selected


# ---------------------------------------------------------------------------
# Patch extraction
# ---------------------------------------------------------------------------

def _crop_and_resize(band: np.ndarray | None, size: int = PATCH_SIZE) -> np.ndarray:
    """Crop centre of a band array and resize to (size, size)."""
    if band is None:
        return np.zeros((size, size), dtype=np.float32)

    arr = band.astype(np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=1.0, neginf=0.0)
    arr = np.clip(arr, 0.0, 1.0)

    H, W = arr.shape
    if H == 0 or W == 0:
        return np.zeros((size, size), dtype=np.float32)

    # Use simple numpy resize (nearest-neighbour-equivalent via linspace indexing)
    row_idx = np.linspace(0, H - 1, size).astype(int)
    col_idx = np.linspace(0, W - 1, size).astype(int)
    return arr[np.ix_(row_idx, col_idx)]


def make_patch(result: dict) -> np.ndarray:
    """
    Build a (3, 64, 64) float32 RGB array from a fetch_sentinel2_tile result.
    Channel order: [R=B4, G=B3, B=B2] to match EuroSAT RGB convention.
    """
    r = _crop_and_resize(result.get("B4"))
    g = _crop_and_resize(result.get("B3"))
    b = _crop_and_resize(result.get("B2"))
    return np.stack([r, g, b], axis=0).astype(np.float32)


# ---------------------------------------------------------------------------
# Progress tracking (thread-safe)
# ---------------------------------------------------------------------------

_progress_lock = threading.Lock()


def _load_progress() -> dict[str, str]:
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {}


def _save_progress(progress: dict[str, str]) -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with _progress_lock:
        with open(PROGRESS_FILE, "w") as f:
            json.dump(progress, f, indent=2)


def _patch_id(window: dict) -> str:
    return f"{window['cls']}_{window['lat']:.5f}_{window['lon']:.5f}"


def _patch_path(window: dict) -> Path:
    cls_dir = OUTPUT_DIR / window["cls"]
    safe_lat = f"{window['lat']:.5f}".replace("-", "m")
    safe_lon = f"{window['lon']:.5f}".replace("-", "m")
    return cls_dir / f"patch_{safe_lat}_{safe_lon}.npy"


# ---------------------------------------------------------------------------
# Per-patch fetch worker
# ---------------------------------------------------------------------------

_tally_lock = threading.Lock()
_tally: dict[str, int] = defaultdict(int)
_tally_errors: dict[str, int] = defaultdict(int)


def fetch_and_save(window: dict, progress: dict[str, str]) -> tuple[str, str]:
    """
    Fetch one Sentinel-2 patch, save to disk, return (patch_id, status).
    Status: "ok" | "error:<msg>" | "skipped"
    """
    from sentinel_fetch import fetch_sentinel2_tile

    pid = _patch_id(window)
    out_path = _patch_path(window)
    cls = window["cls"]

    # Resume: skip if already done
    if pid in progress and progress[pid] == "ok" and out_path.exists():
        with _tally_lock:
            _tally[cls] += 1
        return pid, "skipped"

    out_path.parent.mkdir(parents=True, exist_ok=True)

    t0 = time.monotonic()
    try:
        result = fetch_sentinel2_tile(
            bbox=window["bbox"],
            date_range=DATE_RANGE,
            max_cloud_cover=MAX_CLOUD_COVER,
            timeout_seconds=FETCH_TIMEOUT,
        )
        patch = make_patch(result)
        np.save(out_path, patch)

        elapsed = time.monotonic() - t0
        with _tally_lock:
            _tally[cls] += 1
            tally_snapshot = dict(_tally)
        logger.info(
            "✓ %-24s  lat=%7.4f lon=%7.4f  purity=%.2f  %.0fs  "
            "| tally: Fv=%d Cr=%d Wa=%d Bu=%d Ba=%d We=%d",
            cls, window["lat"], window["lon"], window["purity"], elapsed,
            tally_snapshot.get("Forest_Vegetation", 0),
            tally_snapshot.get("Cropland", 0),
            tally_snapshot.get("Water", 0),
            tally_snapshot.get("Built_up", 0),
            tally_snapshot.get("Bare_Sparse", 0),
            tally_snapshot.get("Wetland", 0),
        )
        return pid, "ok"

    except Exception as exc:
        elapsed = time.monotonic() - t0
        msg = str(exc)[:120]
        with _tally_lock:
            _tally_errors[cls] += 1
        logger.warning("✗ %-24s  lat=%7.4f lon=%7.4f  %.0fs  %s",
                       cls, window["lat"], window["lon"], elapsed, msg)
        return pid, f"error:{msg}"


# ---------------------------------------------------------------------------
# Train / val split
# ---------------------------------------------------------------------------

def split_manifest(dry_run: bool = False) -> tuple[int, int]:
    """
    Build stratified 80/20 train/val split from saved patches.
    Writes train_manifest.csv and val_manifest.csv.
    Returns (n_train, n_val).
    """
    rng = random.Random(42)
    train_rows: list[dict] = []
    val_rows: list[dict] = []

    print(f"\n{'='*60}")
    print("  Train / Val split (80/20 stratified by class)")
    print(f"{'='*60}")
    print(f"  {'Class':<24}  {'Total':>6}  {'Train':>6}  {'Val':>5}")
    print("  " + "-"*44)

    for cls in GLOBAL6_CLASSES:
        cls_dir = OUTPUT_DIR / cls
        if not cls_dir.exists():
            print(f"  {cls:<24}  {'—':>6}")
            continue

        files = sorted(cls_dir.glob("patch_*.npy"))
        if not files:
            print(f"  {cls:<24}  {'0':>6}")
            continue

        rng.shuffle(files)
        n_val = max(1, int(round(len(files) * 0.20)))
        val_files = files[:n_val]
        train_files = files[n_val:]

        for f in train_files:
            train_rows.append({
                "filepath": str(f),
                "global6_class": cls,
                "global6_index": GLOBAL6_LABEL[cls],
                "split": "train",
                "country": COUNTRY,
            })
        for f in val_files:
            val_rows.append({
                "filepath": str(f),
                "global6_class": cls,
                "global6_index": GLOBAL6_LABEL[cls],
                "split": "val",
                "country": COUNTRY,
            })

        print(f"  {cls:<24}  {len(files):>6}  {len(train_files):>6}  {len(val_files):>5}")

    print(f"  {'TOTAL':<24}  {len(train_rows)+len(val_rows):>6}  {len(train_rows):>6}  {len(val_rows):>5}")

    if not dry_run:
        fields = ["filepath", "global6_class", "global6_index", "split", "country"]
        for rows, name in [(train_rows, "train_manifest.csv"), (val_rows, "val_manifest.csv")]:
            out = OUTPUT_DIR / name
            with open(out, "w", newline="") as f:
                w = csv.DictWriter(f, fieldnames=fields)
                w.writeheader()
                w.writerows(rows)
            logger.info("Saved %s  (%d rows)", out, len(rows))

    return len(train_rows), len(val_rows)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Generate Angola Sentinel-2 patches")
    parser.add_argument("--workers",    type=int,  default=DEFAULT_WORKERS,
                        help="Parallel fetch workers (default: 3)")
    parser.add_argument("--dry-run",    action="store_true",
                        help="Print allocation and time estimate, then exit")
    parser.add_argument("--split-only", action="store_true",
                        help="Skip fetching; only (re)build train/val manifests")
    args = parser.parse_args()

    rng = random.Random(42)

    # ── Step 1: scan WorldCover raster ───────────────────────────────────────
    logger.info("Loading WorldCover raster for %s…", COUNTRY)
    arr = fetch_worldcover_tile(COUNTRY_BBOX_STR, verbose=False)

    windows_by_g6 = scan_windows(arr, COUNTRY_BBOX_STR)
    candidates = select_candidates(windows_by_g6, rng)

    # ── Print allocation summary ──────────────────────────────────────────────
    per_cls = defaultdict(int)
    for c in candidates:
        per_cls[c["cls"]] += 1

    print(f"\n{'='*65}")
    print(f"  Patch allocation — {COUNTRY}  (Option D)")
    print(f"{'='*65}")
    print(f"  {'Class':<24}  {'Patches':>8}  {'Est. time':>10}  {'Note'}")
    print("  " + "-"*60)
    total_patches = 0
    for cls in GLOBAL6_CLASSES:
        n = per_cls.get(cls, 0)
        mins = n * 45.7 / 60 / args.workers
        note = "ALL available" if cls == "Built_up" else ""
        if n < 100 and cls not in {"Forest_Vegetation"}:
            note = "⚠ below 100 minimum" if not note else note
        print(f"  {cls:<24}  {n:>8}  {mins:>9.1f}m  {note}")
        total_patches += n
    total_min = total_patches * 45.7 / 60 / args.workers
    total_hr  = total_min / 60
    print(f"\n  TOTAL:  {total_patches} patches")
    print(f"  Workers: {args.workers}")
    print(f"  Est. wall time: {total_min:.0f} min = {total_hr:.1f} h")
    print(f"  (at 45.7s/patch average; real range 25-75s)")

    if args.dry_run:
        print("\n  [DRY RUN] Exiting without fetching.")
        return

    if args.split_only:
        split_manifest()
        return

    # ── Step 2: load progress, skip already-done patches ─────────────────────
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    progress = _load_progress()
    already_done = sum(1 for v in progress.values() if v == "ok")
    logger.info("Progress file: %d already fetched, %d candidates queued",
                already_done, len(candidates))

    # ── Step 3: parallel fetch loop ───────────────────────────────────────────
    wall_t0 = time.monotonic()
    completed = 0
    errors = 0
    SAVE_EVERY = 10   # flush progress.json every N completions

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch_and_save, window, progress): window
            for window in candidates
        }
        for future in as_completed(futures):
            window = futures[future]
            try:
                pid, status = future.result()
            except Exception as exc:
                pid = _patch_id(window)
                status = f"error:{exc}"
                errors += 1

            progress[pid] = status
            if status == "ok":
                completed += 1
            elif status.startswith("error"):
                errors += 1

            # Periodic progress save and summary
            n_done = completed + errors
            if n_done % SAVE_EVERY == 0 or n_done == len(candidates):
                _save_progress(progress)
                elapsed = time.monotonic() - wall_t0
                rate = n_done / elapsed if elapsed > 0 else 0
                remaining = (len(candidates) - n_done) / rate if rate > 0 else 0
                logger.info(
                    "Progress: %d/%d  ok=%d  err=%d  elapsed=%.0fm  eta=%.0fm",
                    n_done, len(candidates), completed, errors,
                    elapsed / 60, remaining / 60,
                )

    _save_progress(progress)
    wall_elapsed = time.monotonic() - wall_t0

    # ── Step 4: final tally ───────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"  Fetch complete — {COUNTRY}")
    print(f"{'='*60}")
    print(f"  Wall time: {wall_elapsed/60:.1f} min = {wall_elapsed/3600:.1f} h")
    print(f"  OK: {completed}   Errors: {errors}   Total attempted: {len(candidates)}")
    print(f"\n  Final patch counts per class (saved on disk):")
    grand_total = 0
    for cls in GLOBAL6_CLASSES:
        cls_dir = OUTPUT_DIR / cls
        n = len(list(cls_dir.glob("patch_*.npy"))) if cls_dir.exists() else 0
        flag = " ⚠ below 100 minimum" if n < 100 and cls not in {"Forest_Vegetation", "Built_up"} else ""
        print(f"    {cls:<24}: {n:>5}{flag}")
        grand_total += n
    print(f"    {'TOTAL':<24}: {grand_total:>5}")

    # ── Step 5: train/val split ───────────────────────────────────────────────
    n_train, n_val = split_manifest()
    print(f"\n  Train manifest: data/patches_angola/train_manifest.csv  ({n_train} rows)")
    print(f"  Val manifest:   data/patches_angola/val_manifest.csv    ({n_val} rows)")
    print(f"\n  Track 3 complete. DO NOT touch val set until Track 5 evaluation.")


if __name__ == "__main__":
    main()
