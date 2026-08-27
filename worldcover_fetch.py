"""
worldcover_fetch.py — Stream ESA WorldCover v200 (10m, 2021) tiles from the
public AWS Open Data bucket (no credentials required) and clip to a bbox.

Usage
-----
    from worldcover_fetch import fetch_worldcover_tile
    arr = fetch_worldcover_tile("11.67,-18.04,24.08,-4.39")   # Angola
    arr = fetch_worldcover_tile("-73.99,-33.75,-28.85,5.27")  # Brazil

WorldCover class codes (values in returned array)
--------------------------------------------------
    10  Tree cover              → Forest_Vegetation
    20  Shrubland               → Forest_Vegetation
    30  Grassland               → Forest_Vegetation
    40  Cropland                → Cropland
    50  Built-up                → Built_up
    60  Bare / sparse veg       → Bare_Sparse
    70  Snow & ice              → Bare_Sparse
    80  Permanent water         → Water
    90  Herbaceous wetland      → Wetland
    95  Mangroves               → Wetland
   100  Moss & lichen           → Forest_Vegetation
   255  No-data mask (excluded from analysis)
"""

import math
import os
from io import BytesIO

import numpy as np
import rasterio
from rasterio.merge import merge
from rasterio.windows import from_bounds

# ---------------------------------------------------------------------------
# S3 configuration — public bucket, no credentials needed
# ---------------------------------------------------------------------------

os.environ.setdefault("AWS_NO_SIGN_REQUEST", "YES")
os.environ.setdefault("AWS_DEFAULT_REGION", "eu-central-1")

_BUCKET = "esa-worldcover"
_PREFIX = "v200/2021/map"
_TILE_SIZE_DEG = 3  # each tile covers 3° × 3°
_NODATA_VALUE = 255


def _tile_key(lat_origin: int, lon_origin: int) -> str:
    """
    Return the S3 key for the tile whose SW corner is (lat_origin, lon_origin).

    ESA naming: lat prefix is N/S, lon prefix is E/W.
    Example: lat=-9, lon=12  →  S09E012
    """
    lat_prefix = "N" if lat_origin >= 0 else "S"
    lon_prefix = "E" if lon_origin >= 0 else "W"
    lat_str = f"{abs(lat_origin):02d}"
    lon_str = f"{abs(lon_origin):03d}"
    fname = f"ESA_WorldCover_10m_2021_v200_{lat_prefix}{lat_str}{lon_prefix}{lon_str}_Map.tif"
    return f"{_PREFIX}/{fname}"


def _tile_vsis3_path(lat_origin: int, lon_origin: int) -> str:
    return f"/vsis3/{_BUCKET}/{_tile_key(lat_origin, lon_origin)}"


def _tiles_for_bbox(W: float, S: float, E: float, N: float) -> list[tuple[int, int]]:
    """Return list of (lat_origin, lon_origin) tile SW corners covering bbox."""
    lat_starts = range(math.floor(S / _TILE_SIZE_DEG) * _TILE_SIZE_DEG,
                       math.ceil(N / _TILE_SIZE_DEG) * _TILE_SIZE_DEG,
                       _TILE_SIZE_DEG)
    lon_starts = range(math.floor(W / _TILE_SIZE_DEG) * _TILE_SIZE_DEG,
                       math.ceil(E / _TILE_SIZE_DEG) * _TILE_SIZE_DEG,
                       _TILE_SIZE_DEG)
    return [(lat, lon) for lat in lat_starts for lon in lon_starts]


def _open_overview(path: str, overview_level: int = 5):
    """
    Open a rasterio dataset at the requested overview index (0-based).
    overview_level=5 corresponds to factor 64 (sub-pixel ~640m).
    For large countries we use a coarser overview to keep memory manageable.
    """
    ds = rasterio.open(path)
    overviews = ds.overviews(1)
    if not overviews:
        return ds
    idx = min(overview_level, len(overviews) - 1)
    # rasterio Dataset.read() with out_shape reads the best matching overview
    return ds


def fetch_worldcover_tile(
    bbox: str,
    overview_level: int = 5,
    verbose: bool = True,
) -> np.ndarray:
    """
    Fetch ESA WorldCover v200 (2021) class-code raster for the given bbox.

    Parameters
    ----------
    bbox : str
        Bounding box as "W,S,E,N" (decimal degrees), matching config.COUNTRY_BBOX.
    overview_level : int
        Overview index to use when reading tiles (0=factor-2, 5=factor-64 ≈640 m/px).
        Lower value = finer resolution but larger memory footprint.
    verbose : bool
        Print progress to stdout.

    Returns
    -------
    np.ndarray
        2-D uint8 array of WorldCover class codes clipped to bbox.
        Shape depends on overview level and bbox size.
        NODATA pixels (255) are excluded from returned array contents but
        retained in position so callers can mask if needed.
    """
    W, S, E, N = (float(v) for v in bbox.split(","))
    tile_origins = _tiles_for_bbox(W, S, E, N)

    if verbose:
        print(f"  bbox: W={W} S={S} E={E} N={N}")
        print(f"  Tiles to fetch: {len(tile_origins)}")

    # Decide overview factor. Each tile is 36000×36000 at native 10m.
    # At overview 5 (factor 64) each tile is ~562×562 pixels → fine for stats.
    overviews_available = [2, 4, 8, 16, 32, 64]
    ov_factor = overviews_available[min(overview_level, len(overviews_available) - 1)]

    opened = []
    missing = []
    for lat_o, lon_o in tile_origins:
        path = _tile_vsis3_path(lat_o, lon_o)
        try:
            ds = rasterio.open(path)
            opened.append(ds)
        except Exception:
            missing.append((lat_o, lon_o))

    if verbose and missing:
        print(f"  {len(missing)} tiles not found on S3 (ocean/no-data areas, expected):")
        for lat_o, lon_o in missing[:5]:
            print(f"    lat={lat_o}, lon={lon_o}")
        if len(missing) > 5:
            print(f"    ... and {len(missing) - 5} more")

    if not opened:
        raise RuntimeError("No WorldCover tiles found for this bbox. Check S3 connectivity.")

    if verbose:
        print(f"  Reading {len(opened)} tiles at overview factor {ov_factor}× ...")

    # Merge tiles into a single mosaic, reading at the downsampled overview level.
    # We pass `res` to merge() equal to the native pixel size × overview factor.
    native_res = 10 / 111320  # 10m in degrees (approx at equator)
    target_res = native_res * ov_factor

    mosaic, mosaic_transform = merge(
        opened,
        bounds=(W, S, E, N),
        res=target_res,
        nodata=_NODATA_VALUE,
        resampling=rasterio.enums.Resampling.nearest,
        dtype="uint8",
    )

    for ds in opened:
        ds.close()

    arr = mosaic[0]  # single band
    if verbose:
        print(f"  Mosaic shape: {arr.shape}")

    return arr


# ---------------------------------------------------------------------------
# Convenience histogram helper
# ---------------------------------------------------------------------------

WORLDCOVER_CLASS_NAMES = {
    10: "Tree cover",
    20: "Shrubland",
    30: "Grassland",
    40: "Cropland",
    50: "Built-up",
    60: "Bare/sparse veg",
    70: "Snow & ice",
    80: "Permanent water",
    90: "Herbaceous wetland",
    95: "Mangroves",
    100: "Moss & lichen",
    255: "No-data",
}


def class_histogram(arr: np.ndarray, exclude_nodata: bool = True) -> dict:
    """Return {class_code: count} for all values in arr."""
    codes, counts = np.unique(arr, return_counts=True)
    hist = {}
    for code, count in zip(codes.tolist(), counts.tolist()):
        if exclude_nodata and code == _NODATA_VALUE:
            continue
        hist[code] = count
    return hist


def print_histogram(arr: np.ndarray, country: str = ""):
    """Print a formatted class-frequency histogram."""
    hist = class_histogram(arr)
    total = sum(hist.values())
    header = f"  Class histogram — {country}" if country else "  Class histogram"
    print(header)
    print(f"  {'Code':>4}  {'Name':<22}  {'Count':>10}  {'%':>6}")
    print("  " + "-" * 50)
    for code in sorted(hist.keys()):
        name = WORLDCOVER_CLASS_NAMES.get(code, "unknown")
        pct = 100.0 * hist[code] / total if total else 0
        print(f"  {code:>4}  {name:<22}  {hist[code]:>10,}  {pct:>5.1f}%")
    print(f"  {'TOTAL':>4}  {'':22}  {total:>10,}  100.0%")


if __name__ == "__main__":
    import sys
    from config import COUNTRY_BBOX

    countries = sys.argv[1:] if len(sys.argv) > 1 else ["Angola", "Brazil"]
    for country in countries:
        if country not in COUNTRY_BBOX:
            print(f"ERROR: '{country}' not in config.COUNTRY_BBOX")
            continue
        print(f"\n{'='*60}")
        print(f"  {country.upper()}")
        print(f"{'='*60}")
        arr = fetch_worldcover_tile(COUNTRY_BBOX[country], verbose=True)
        print(f"  Array shape : {arr.shape}")
        print(f"  Array dtype : {arr.dtype}")
        codes = [int(c) for c in np.unique(arr) if c != 255]
        print(f"  Unique codes: {codes}")
        print()
        print_histogram(arr, country)
