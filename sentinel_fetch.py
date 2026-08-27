"""
sentinel_fetch.py — Fetch Sentinel-2 Band data via NASA Earthdata (earthaccess).

Uses the HLSL30 (HLS Landsat/Sentinel-2) product, which provides Sentinel-2
harmonised bands at 30m resolution via NASA's Common Metadata Repository (CMR).

HLS band mapping (Sentinel-2 equivalent):
    B02  → Blue   (B2)
    B03  → Green  (B3)
    B04  → Red    (B4)
    B8A  → NIR    (B8 / NIR narrow)  — HLS uses B8A, close to S2 B8

Public API
----------
    fetch_sentinel2_tile(bbox, date_range) -> dict
"""

from __future__ import annotations

import logging
import os
import threading
import time
from datetime import datetime, timezone

# Load .env at import time so EARTHDATA_USERNAME/EARTHDATA_PASSWORD are available
# before earthaccess.login() is called.  Must happen at module level — calling
# load_dotenv() inside a function body fails inside Streamlit's render process
# because find_dotenv() walks up the call stack and hits a guard assertion.
try:
    from dotenv import load_dotenv as _load_dotenv
    _load_dotenv()
except Exception:
    pass  # dotenv not installed or .env not found — credentials may come from env directly

import numpy as np

# ---------------------------------------------------------------------------
# aiohttp session-level timeout for earthaccess.open()
# ---------------------------------------------------------------------------
# earthaccess.open() streams granule data over HTTPS using fsspec's async
# aiohttp backend.  Without explicit timeouts aiohttp will wait indefinitely
# on a stalled TCP connection, blocking the calling thread in an
# uninterruptible D-state that cannot be stopped with SIGTERM.
#
# The correct injection point is aiohttp.ClientSession(timeout=...), which
# sets a session-wide default for every request.  Two approaches that look
# correct but are NOT:
#
#  • fsspec.config.conf["https"]["client_kwargs"]["timeout"] — does not work
#    because apply_config() is a shallow merge and earthaccess's explicit
#    client_kwargs={headers, trust_env=False} overwrites the entire key.
#
#  • open_kwargs={"timeout": ClientTimeout(...)} passed to earthaccess.open()
#    — does not work because fsspec.asyn.sync() intercepts any kwarg named
#    "timeout" as its own asyncio.wait_for deadline and raises
#    TypeError when given a ClientTimeout instead of a float.
#
# What works: after earthaccess.login() creates a fresh Store instance, patch
# Store.get_fsspec_session on that instance so the next call builds the
# HTTPFileSystem with timeout= included in client_kwargs.  Since the method
# is @lru_cache on the instance, the patch is done before the first call and
# the cache then holds the session with our timeout.
#
# Scope: earthaccess.open() is the only earthaccess I/O call in this codebase.
# agent_runner.py never calls it.  The patch is applied per-login, so repeated
# calls to fetch_sentinel2_tile (which each call earthaccess.login()) each get
# a fresh, correctly-configured session.
try:
    import aiohttp as _aiohttp

    # Per-request timeout forwarded to every aiohttp session.get() / .head()
    # call made inside earthaccess.open().
    _AIOHTTP_TIMEOUT = _aiohttp.ClientTimeout(
        total=None,    # ThreadPoolExecutor handles the wall-clock budget
        connect=15,    # TCP + TLS handshake must finish within 15 s
        sock_read=60,  # each recv() must return data within 60 s
    )
    logging.getLogger(__name__).debug(
        "sentinel_fetch: aiohttp per-request timeout set — "
        "connect=15s  sock_read=60s"
    )
except Exception as _timeout_cfg_err:
    _AIOHTTP_TIMEOUT = None
    logging.getLogger(__name__).warning(
        "sentinel_fetch: could not build aiohttp timeout (%s); "
        "earthaccess.open() will have no read timeout.",
        _timeout_cfg_err,
    )

# ---------------------------------------------------------------------------
# Per-session timeout counter (visible in logs during bad-network events)
# ---------------------------------------------------------------------------
_timeout_counter_lock = threading.Lock()
_timeout_counter = 0  # incremented each time a fetch times out

logger = logging.getLogger(__name__)

# HLS Sentinel-2 short name in CMR
_HLS_S2_SHORT_NAME = "HLSS30"
_HLS_VERSION = "2.0"

# Band names in the HLS product that correspond to Sentinel-2 B2/B3/B4/B8
_BAND_MAP = {
    "B2": "B02",
    "B3": "B03",
    "B4": "B04",
    "B8": "B8A",   # NIR narrow — closest analogue to S2 B8 in HLS
}

# HLS fill/nodata value
_HLS_FILL = -9999
# HLS scale factor (reflectance = DN * 0.0001)
_HLS_SCALE = 0.0001

# ---------------------------------------------------------------------------
# HLS HLSS30 v2.0 Fmask QA band
# ---------------------------------------------------------------------------
# Source: HLS V2.0 User Guide (LP DAAC, 2023)
#   https://lpdaac.usgs.gov/documents/1698/HLS_User_Guide_V2.pdf  Table 7
#
# Fmask bit layout (each pixel is uint8):
#   Bit 0 — Cirrus
#   Bit 1 — Cloud
#   Bit 2 — Adjacent to cloud / cloud shadow
#   Bit 3 — Cloud shadow
#   Bit 4 — Snow / ice
#   Bit 5 — Water
#   Bit 6–7 — Aerosol level (00=climatology, 01=low, 10=moderate, 11=high)
#
# A pixel is considered cloud-contaminated if bit 1 (cloud) OR bit 3
# (cloud shadow) is set.  Bit 2 (adjacent to cloud/shadow) is also included
# because those pixels are frequently contaminated by cloud aureole.
#
# Mask value: bits 1, 2, 3  → 0b00001110 = 0x0E = 14
_FMASK_CLOUD_BITS = np.uint8(0b00001110)  # cloud | adjacent-to-cloud | cloud-shadow


def is_cloud_contaminated(fmask_array: np.ndarray) -> np.ndarray:
    """
    Return a boolean array: True where a pixel is cloud or cloud-shadow
    contaminated, per the HLS HLSS30 v2.0 Fmask QA band.

    Parameters
    ----------
    fmask_array : np.ndarray
        Raw Fmask values (uint8 or float32 from the HLS read path — values
        0–255 before any scaling, or NaN for nodata).  The reflectance
        scaling (*0.0001) applied to optical bands must NOT be applied to
        Fmask; we undo it here if the array appears to have been scaled.

    Returns
    -------
    np.ndarray (bool)
        Same shape as fmask_array.  NaN / nodata pixels are treated as
        contaminated (True) so they never count as valid land.
    """
    fa = np.asarray(fmask_array, dtype=np.float32)

    # If Fmask was read through the same _HLS_SCALE path as optical bands
    # its values would be in [0, 0.0255] instead of [0, 255].  Detect this
    # and reverse the scaling so bit-masking works correctly.
    _fa_valid = fa[~np.isnan(fa)]
    if len(_fa_valid) > 0 and _fa_valid.max() < 1.0:
        fa = fa / _HLS_SCALE  # undo the *0.0001 scaling

    # NaN (nodata) pixels are flagged as contaminated
    nodata_mask = np.isnan(fa)
    # Fill NaN before casting to uint8 (NaN→uint8 raises RuntimeWarning);
    # the nodata_mask union below ensures they are flagged contaminated.
    fa_filled = np.where(nodata_mask, 0.0, fa)
    uint8_vals = fa_filled.astype(np.uint8)
    cloud_mask = (uint8_vals & _FMASK_CLOUD_BITS) != 0
    # Also flag nodata as contaminated
    cloud_mask |= nodata_mask
    return cloud_mask


def _bbox_to_cmr(bbox_str: str) -> tuple[float, float, float, float]:
    """Parse COUNTRY_BBOX string 'W,S,E,N' into (west, south, east, north) floats."""
    parts = [float(x) for x in bbox_str.split(",")]
    w, s, e, n = parts
    return w, s, e, n


def _read_band_from_granule(fileobj, band_name: str) -> np.ndarray | None:
    """
    Read a single band from an open HLS granule file object.

    HLS granules are HDF4/HDF5-based GeoTIFFs opened via rasterio.
    The band is identified by a subdataset name matching *band_name*.
    """
    try:
        import rasterio
        from rasterio.io import MemoryFile
    except ImportError:
        raise RuntimeError("rasterio is required: pip install rasterio")

    try:
        data = fileobj.read()
        with MemoryFile(data) as mf:
            with mf.open() as ds:
                # HLS files often have subdatasets; try to find the matching band
                if ds.count >= 1:
                    # Single-band GeoTIFF per file (most common for HLS from earthaccess)
                    arr = ds.read(1).astype(np.int16)
                    # Apply fill mask and scale
                    arr = arr.astype(np.float32)
                    arr[arr == _HLS_FILL] = np.nan
                    arr = arr * _HLS_SCALE
                    return arr
    except Exception as exc:
        logger.debug("Could not read band %s: %s", band_name, exc)
    return None


def fetch_sentinel2_tile(
    bbox: str,
    date_range: tuple[str, str],
    max_cloud_cover: float = 30.0,
    timeout_seconds: float = 120.0,
) -> dict:
    """
    Fetch Sentinel-2-equivalent (HLS S30) band data for *bbox* within *date_range*.

    Parameters
    ----------
    bbox : str
        Bounding box in COUNTRY_BBOX format: "W,S,E,N" (lon/lat degrees).
    date_range : tuple[str, str]
        ISO date strings (YYYY-MM-DD): (start, end).
    max_cloud_cover : float
        Maximum acceptable cloud cover percentage. Defaults to 30%.
    timeout_seconds : float
        Give up on the download after this many seconds.

    Returns
    -------
    dict with keys:
        "B2"               : np.ndarray (float32, reflectance 0-1), or None
        "B3"               : np.ndarray
        "B4"               : np.ndarray
        "B8"               : np.ndarray
        "acquisition_date" : str  (ISO date of the selected scene)
        "cloud_cover"      : float (percentage)
        "granule_id"       : str
        "bbox"             : str  (echoed input)
        "source"           : str  ("HLSS30 v2.0 via NASA Earthdata")

    Raises
    ------
    RuntimeError  if earthaccess auth fails or no suitable granule is found.
    """
    try:
        import earthaccess
    except ImportError:
        raise RuntimeError("earthaccess is required: pip install earthaccess")

    t_start = time.monotonic()

    # ── Auth ─────────────────────────────────────────────────────────────────
    logger.info("fetch_sentinel2_tile: authenticating with NASA Earthdata...")
    auth = earthaccess.login(strategy="environment")
    if not auth.authenticated:
        raise RuntimeError(
            "NASA Earthdata authentication failed. "
            "Set EARTHDATA_USERNAME and EARTHDATA_PASSWORD in your .env file."
        )
    logger.info("fetch_sentinel2_tile: authenticated.")

    # ── Inject aiohttp timeout into the fsspec HTTPS session ─────────────────
    # earthaccess.login() creates a new Store instance on __store__; its
    # get_fsspec_session() is @lru_cache and builds an HTTPFileSystem with
    # client_kwargs = {Authorization, trust_env=False}.  That dict goes
    # to aiohttp.ClientSession(**client_kwargs) — so the right place for the
    # session-level default timeout is inside client_kwargs["timeout"].
    #
    # We cannot set this via fsspec.config.conf because apply_config() does a
    # shallow merge and earthaccess's explicit client_kwargs={} overwrites it.
    # We also cannot pass timeout= in open_kwargs because fsspec.asyn.sync()
    # captures any kwarg named "timeout" as its own asyncio.wait_for deadline.
    #
    # The minimal-invasive fix: after login, patch get_fsspec_session on the
    # live Store instance so the next call builds the HTTPFileSystem with
    # our timeout in client_kwargs.  Since @lru_cache is on the instance
    # method, we just clear it and replace it with a wrapper.
    if _AIOHTTP_TIMEOUT is not None:
        try:
            import functools as _functools
            _store = earthaccess.__store__
            _orig_get_session = _store.__class__.get_fsspec_session.__wrapped__

            @_functools.lru_cache
            def _patched_get_fsspec_session(self):  # type: ignore[override]
                import fsspec as _fsspec
                token = self.auth.token["access_token"]
                kw = {
                    "headers": {"Authorization": f"Bearer {token}"},
                    "trust_env": False,
                    "timeout": _AIOHTTP_TIMEOUT,
                }
                return _fsspec.filesystem("https", client_kwargs=kw)

            # Replace on the class so all calls on this instance use our version.
            # We bind it back as a bound method on the instance to avoid touching
            # other hypothetical Store instances.
            import types as _types
            _store.get_fsspec_session = _types.MethodType(_patched_get_fsspec_session, _store)
            logger.debug(
                "sentinel_fetch: patched get_fsspec_session with "
                "aiohttp timeout (connect=15s, sock_read=60s)"
            )
        except Exception as _patch_err:
            logger.warning(
                "sentinel_fetch: could not patch get_fsspec_session (%s); "
                "earthaccess.open() will have no session-level read timeout.",
                _patch_err,
            )

    # ── CMR granule search ────────────────────────────────────────────────────
    w, s, e, n = _bbox_to_cmr(bbox)
    logger.info(
        "fetch_sentinel2_tile: searching %s  bbox=(%g,%g,%g,%g)  dates=%s..%s",
        _HLS_S2_SHORT_NAME, w, s, e, n, *date_range,
    )

    results = earthaccess.search_data(
        short_name=_HLS_S2_SHORT_NAME,
        version=_HLS_VERSION,
        temporal=date_range,
        bounding_box=(w, s, e, n),
        count=50,
    )

    if not results:
        raise RuntimeError(
            f"No {_HLS_S2_SHORT_NAME} granules found for bbox={bbox!r} "
            f"dates={date_range}. "
            "Try a wider date range or lower max_cloud_cover."
        )

    logger.info("fetch_sentinel2_tile: found %d granules, selecting least-cloudy...", len(results))

    # ── Select least-cloudy granule ───────────────────────────────────────────
    def _cloud_cover(g) -> float:
        try:
            return float(g["umm"]["AdditionalAttributes"][0]["Values"][0])
        except Exception:
            pass
        # Try attributes dict directly
        try:
            for attr in g["umm"].get("AdditionalAttributes", []):
                if attr.get("Name") in ("cloud_coverage", "CloudCover",
                                        "CLOUD_COVERAGE", "HLS_CLOUD_COVERAGE"):
                    return float(attr["Values"][0])
        except Exception:
            pass
        return 100.0  # treat unknown as fully cloudy

    candidates = []
    for g in results:
        cc = _cloud_cover(g)
        if cc <= max_cloud_cover:
            candidates.append((cc, g))

    if not candidates:
        # Relax: just take the least-cloudy of what we have
        logger.warning(
            "No granule ≤%.0f%% cloud cover; taking least-cloudy available.", max_cloud_cover
        )
        for g in results:
            candidates.append((_cloud_cover(g), g))

    candidates.sort(key=lambda x: x[0])
    best_cc, best_granule = candidates[0]

    # Extract acquisition date from granule metadata
    try:
        acq_date = best_granule["umm"]["TemporalExtent"]["RangeDateTime"]["BeginningDateTime"][:10]
    except Exception:
        acq_date = date_range[0]

    granule_id = best_granule.get("meta", {}).get("concept-id", "unknown")
    logger.info(
        "fetch_sentinel2_tile: selected granule %s  cloud=%.1f%%  date=%s",
        granule_id, best_cc, acq_date,
    )

    # ── Download and read bands (with hard timeout) ───────────────────────────
    # earthaccess.open() and fobj.read() are blocking S3 streaming calls with
    # no native timeout.  We run the entire open+read block in a thread and
    # enforce timeout_seconds via Future.result(timeout=...) so a stalled
    # connection cannot hang the Streamlit process indefinitely.
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as _FutureTimeout
    import rasterio
    from rasterio.io import MemoryFile

    # Fine-grained fetch log — flushed after every call so a hang leaves
    # a precise last-checkpoint even if the process is later killed -9.
    _FETCH_LOG = "/tmp/sentinel_fetch_trace.log"

    def _flog(msg: str) -> None:
        ts = time.strftime("%H:%M:%S")
        line = f"[{ts}] {msg}\n"
        try:
            with open(_FETCH_LOG, "a") as _f:
                _f.write(line)
        except Exception:
            pass
        logger.debug(msg)

    def _download_and_read() -> dict[str, np.ndarray | None]:
        bands: dict[str, np.ndarray | None] = {k: None for k in ("B2", "B3", "B4", "B8", "Fmask")}

        _flog(f"_download_and_read: start  granule={granule_id}")
        _flog("calling earthaccess.open()...")
        try:
            files = earthaccess.open([best_granule])
        except Exception as exc:
            _flog(f"earthaccess.open() RAISED: {exc}")
            raise RuntimeError(f"earthaccess.open() failed: {exc}") from exc
        _flog(f"earthaccess.open() returned {len(files)} file objects")

        logger.info(
            "fetch_sentinel2_tile: opened %d file objects, reading bands...", len(files)
        )

        # Strategy A: single multi-band file
        if len(files) == 1:
            _flog("Strategy A: single file — calling files[0].read()...")
            try:
                data = files[0].read()
                _flog(f"files[0].read() complete — {len(data):,} bytes")
                with MemoryFile(data) as mf:
                    with mf.open() as ds:
                        _flog(f"rasterio open: {ds.count} band(s), shape=({ds.height},{ds.width})")
                        logger.info(
                            "fetch_sentinel2_tile: file has %d band(s), shape=%s",
                            ds.count, (ds.height, ds.width),
                        )
                        desc_to_key = {
                            "B02": "B2", "B2": "B2", "blue": "B2",
                            "B03": "B3", "B3": "B3", "green": "B3",
                            "B04": "B4", "B4": "B4", "red": "B4",
                            "B8A": "B8", "B08": "B8", "B8": "B8", "nir": "B8",
                            "Fmask": "Fmask", "FMASK": "Fmask",
                        }
                        descs = ds.descriptions or []
                        if descs and any(d for d in descs):
                            for bi, desc in enumerate(descs, start=1):
                                if desc:
                                    key = desc_to_key.get(desc.strip())
                                    if key:
                                        _flog(f"ds.read({bi}) for key={key}...")
                                        arr = ds.read(bi).astype(np.float32)
                                        arr[arr == _HLS_FILL] = np.nan
                                        arr = arr * _HLS_SCALE
                                        bands[key] = arr
                                        _flog(f"ds.read({bi}) done  shape={arr.shape}")
                        else:
                            if ds.count >= 4:
                                for bi, key in enumerate(["B2", "B3", "B4", "B8"], start=1):
                                    _flog(f"ds.read({bi}) for key={key}...")
                                    arr = ds.read(bi).astype(np.float32)
                                    arr[arr == _HLS_FILL] = np.nan
                                    arr = arr * _HLS_SCALE
                                    bands[key] = arr
                                    _flog(f"ds.read({bi}) done  shape={arr.shape}")
            except Exception as exc:
                _flog(f"Strategy A RAISED: {exc}")
                logger.warning("Strategy A (single file) failed: %s", exc)

        else:
            # Strategy B: one file per band, match by filename
            # HLS granules deliver one GeoTIFF per band (per earthaccess.open),
            # including a separate Fmask file whose name contains "Fmask".
            _flog(f"Strategy B: {len(files)} files — reading by filename match")
            hls_key_map = {
                "B02": "B2", "B03": "B3", "B04": "B4", "B8A": "B8", "B08": "B8",
                "FMASK": "Fmask",
            }
            for _fi, fobj in enumerate(files):
                fname = getattr(fobj, "path", "") or getattr(fobj, "name", "") or ""
                matched_key = None
                for hls_band, our_key in hls_key_map.items():
                    if hls_band in fname.upper():
                        matched_key = our_key
                        break
                if matched_key is None:
                    _flog(f"  file[{_fi}] {fname!r} — no band match, skipping")
                    continue
                _flog(f"  file[{_fi}] {fname!r} → {matched_key}  calling fobj.read()...")
                try:
                    data = fobj.read()
                    _flog(f"  fobj.read() complete — {len(data):,} bytes")
                    with MemoryFile(data) as mf:
                        with mf.open() as ds:
                            arr = ds.read(1).astype(np.float32)
                            arr[arr == _HLS_FILL] = np.nan
                            # Fmask is a QA integer band — do NOT apply reflectance
                            # scaling.  Store raw DN values (0–255) as float32.
                            if matched_key != "Fmask":
                                arr = arr * _HLS_SCALE
                            bands[matched_key] = arr
                            _flog(f"  {matched_key} loaded  shape={arr.shape}")
                            if matched_key == "Fmask":
                                valid = arr[~np.isnan(arr)]
                                _flog(
                                    f"  Fmask unique values (sample): "
                                    f"{np.unique(valid[:1000].astype(np.uint8)).tolist()}"
                                )
                            logger.debug(
                                "read %s from %s  shape=%s", matched_key, fname, arr.shape
                            )
                except Exception as exc:
                    _flog(f"  fobj.read() for {matched_key} RAISED: {exc}")
                    logger.warning(
                        "Could not read %s from %s: %s", matched_key, fname, exc
                    )

        _flog(f"_download_and_read: done  loaded={[k for k,v in bands.items() if v is not None]}")
        return bands

    # Remaining budget after auth + search
    remaining = timeout_seconds - (time.monotonic() - t_start)
    remaining = max(remaining, 30.0)  # always allow at least 30 s for the download

    try:
        with ThreadPoolExecutor(max_workers=1) as _pool:
            _future = _pool.submit(_download_and_read)
            bands = _future.result(timeout=remaining)
    except _FutureTimeout:
        global _timeout_counter
        with _timeout_counter_lock:
            _timeout_counter += 1
            _tc = _timeout_counter
        # NOTE: the worker thread may still be blocked inside aiohttp even after
        # this exception is caught.  The sock_read=60s timeout injected into the
        # aiohttp ClientSession (via the get_fsspec_session patch above) will
        # cause aiohttp to raise ServerTimeoutError inside that thread within
        # 60 s of the last received byte, so the orphaned thread will
        # self-terminate rather than accumulating indefinitely.  Without that
        # configuration the thread would block forever in D-state.
        logger.warning(
            "sentinel_fetch: fetch timed out after %.0f s "
            "(session total=%d timeout(s) so far).  "
            "The aiohttp sock_read=60s guard will unblock the worker thread. "
            "Granule: %s",
            timeout_seconds, _tc, granule_id,
        )
        raise RuntimeError(
            f"Sentinel-2 fetch timed out after {timeout_seconds:.0f} s "
            f"(session timeout #{_tc}). "
            "The NASA Earthdata HTTPS stream stalled. Try again — transient "
            "throttling usually resolves within a few minutes."
        )

    elapsed = time.monotonic() - t_start
    loaded_bands = [k for k, v in bands.items() if v is not None]
    logger.info(
        "fetch_sentinel2_tile: done in %.1fs. Loaded bands: %s",
        elapsed, loaded_bands,
    )

    # Log Fmask diagnostics when available
    if bands.get("Fmask") is not None:
        _fm = bands["Fmask"]
        _fm_valid = _fm[~np.isnan(_fm)]
        _cloud_pct = 0.0
        if len(_fm_valid) > 0:
            _cloud_mask = is_cloud_contaminated(_fm)
            _cloud_pct = float(100 * _cloud_mask.mean())
        logger.info(
            "fetch_sentinel2_tile: Fmask loaded — shape=%s dtype=%s "
            "unique=%s  cloud_contaminated=%.1f%%",
            _fm.shape, _fm.dtype,
            np.unique(_fm_valid[:500].astype(np.uint8)).tolist() if len(_fm_valid) > 0 else [],
            _cloud_pct,
        )

    return {
        "B2": bands["B2"],
        "B3": bands["B3"],
        "B4": bands["B4"],
        "B8": bands["B8"],
        "Fmask": bands.get("Fmask"),
        "acquisition_date": acq_date,
        "cloud_cover": round(best_cc, 2),
        "granule_id": granule_id,
        "bbox": bbox,
        "source": f"HLSS30 v{_HLS_VERSION} via NASA Earthdata",
        "elapsed_seconds": round(elapsed, 1),
    }
