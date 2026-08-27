"""
artifacts.py — Shared helpers for building and writing agent run artifacts.

Imported by both agent_runner.py (to produce artifacts on every run) and
app.py (to serve download buttons in the Agent Status tab).

All functions here are pure Python — no Streamlit imports.
"""

from __future__ import annotations

import gzip
import logging
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    # Avoid importing heavy forecast deps at import time when not needed.
    from forecast_engine import ForecastResult

logger = logging.getLogger(__name__)

# Root folder under which all per-run artifact directories are created.
ARTIFACTS_ROOT = Path("agent_artifacts")


# ---------------------------------------------------------------------------
# CSV builder (extracted from app.py._build_export_csv — identical logic)
# ---------------------------------------------------------------------------

def build_export_csv(
    fire_df: pd.DataFrame,
    forecast_result: "ForecastResult | None",
    country: str,
    days: int,
) -> bytes:
    """
    Return a UTF-8 encoded (gzip-compressed) CSV of fire detections enriched with:
    - All original FIRMS columns
    - query_country, query_days   — traceability metadata
    - nearest_cell_fire_prob      — fire probability of the closest forecast
      grid cell (NaN when forecast has no cells)

    The returned bytes are gzip-compressed (.csv.gz) to keep disk usage
    reasonable (~10-20× smaller than the uncompressed CSV for large detection
    sets).  The suffix used for the artifact file is .csv.gz.
    """
    if fire_df.empty:
        export_df = fire_df.copy()
        export_df["query_country"] = country
        export_df["query_days"] = days
        export_df["nearest_cell_fire_prob"] = pd.NA
        return gzip.compress(export_df.to_csv(index=False).encode("utf-8"))

    export_df = fire_df.copy()
    export_df["query_country"] = country
    export_df["query_days"] = days

    if forecast_result and forecast_result.cells:
        import numpy as np  # transitive dep via pandas

        cell_lats_arr = np.array(
            [c.lat_center for c in forecast_result.cells], dtype=float
        )
        cell_lons_arr = np.array(
            [c.lon_center for c in forecast_result.cells], dtype=float
        )
        cell_probs_arr = np.array(
            [c.fire_prob for c in forecast_result.cells], dtype=float
        )

        det_lats = export_df["latitude"].to_numpy(dtype=float)
        det_lons = export_df["longitude"].to_numpy(dtype=float)

        dists = (
            (det_lats[:, None] - cell_lats_arr[None, :]) ** 2
            + (det_lons[:, None] - cell_lons_arr[None, :]) ** 2
        )
        nearest_idx = dists.argmin(axis=1)
        export_df["nearest_cell_fire_prob"] = cell_probs_arr[nearest_idx]
    else:
        export_df["nearest_cell_fire_prob"] = float("nan")

    return gzip.compress(export_df.to_csv(index=False).encode("utf-8"))


# ---------------------------------------------------------------------------
# Artifact folder writer
# ---------------------------------------------------------------------------

def save_run_artifacts(
    run_id: str,
    country: str,
    started_at: str,
    guardrail_verdict: str,
    fire_df: pd.DataFrame,
    forecast_result: "ForecastResult",
    summary_text: str,
    forecast_text: str,
    dataset_days: int = 2,
) -> str:
    """
    Persist artifacts for a completed agent run.

    Writes into  agent_artifacts/{run_id}/  and returns the directory path
    as a string.

    Parameters
    ----------
    dataset_days : the ``days`` argument passed to ``get_fire_data`` when
        building *fire_df* (currently always 2, i.e. 48-hour window in
        agent_runner.py).  Used as ``query_days`` in dataset.csv.gz and
        stated explicitly in report.md.

    Files written
    -------------
    dataset.csv.gz
        The 48h risk-metrics FIRMS window (*fire_df*) — used for risk scoring
        and as the short-window positive-label source for XGBoost.
        query_days column = *dataset_days*.

    dataset_forecast_window.csv.gz  [XGBoost runs only]
        The 7-day FIRMS window fetched internally by run_forecast()
        (_get_fire_window).  This is the actual training-data input for
        _build_feature_matrix → _get_model_and_predictions, and therefore
        the data set that reproduces model.json.  Only written when _clf is
        not None (i.e., XGBoost was used); omitted for deterministic runs.

    model.json
        Trained XGBoost booster (XGBoost native JSON format), OR a sentinel
        JSON note on the deterministic fallback path.

    model_script.py
        Copy of wildfire_model_export.py.

    report.md
        Markdown report: metadata table + artifact inventory + AI text.
        XGBoost runs: both dataset files are listed with their roles.
        Deterministic runs: sentinel warning, no forecast-window file listed.
    """
    artifact_dir = ARTIFACTS_ROOT / run_id
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_used = getattr(forecast_result, "model_used", "Unknown") if forecast_result else "Unknown"
    is_deterministic = (model_used == "Deterministic")
    booster_clf = getattr(forecast_result, "_clf", None) if forecast_result else None
    fire_7d_df   = getattr(forecast_result, "_fire_7d_df", None) if forecast_result else None

    # ── 1a. dataset.csv.gz (48h risk-metrics window) ─────────────────────────
    csv_bytes = build_export_csv(fire_df, forecast_result, country, days=dataset_days)
    (artifact_dir / "dataset.csv.gz").write_bytes(csv_bytes)
    logger.debug(
        "[artifacts %s] dataset.csv.gz written (%d bytes, query_days=%d)",
        run_id[:8], len(csv_bytes), dataset_days,
    )

    # ── 1b. dataset_forecast_window.csv.gz (7-day XGBoost training window) ───
    # Written only when a real booster was trained so this file is the exact
    # data that reproduces model.json.  Omitted for deterministic runs.
    forecast_window_size: int | None = None
    if booster_clf is not None and fire_7d_df is not None:
        fw_bytes = build_export_csv(fire_7d_df, forecast_result, country, days=7)
        (artifact_dir / "dataset_forecast_window.csv.gz").write_bytes(fw_bytes)
        forecast_window_size = len(fire_7d_df)
        logger.debug(
            "[artifacts %s] dataset_forecast_window.csv.gz written "
            "(%d bytes, %d rows, query_days=7)",
            run_id[:8], len(fw_bytes), forecast_window_size,
        )

    # ── 2. model.json ────────────────────────────────────────────────────────
    if booster_clf is not None:
        model_path = str(artifact_dir / "model.json")
        booster_clf.get_booster().save_model(model_path)
        logger.debug("[artifacts %s] model.json written (XGBoost booster)", run_id[:8])
    else:
        # Deterministic fallback: no booster trained — write an explicit
        # sentinel so the file always exists and is clearly not a real model.
        import json as _json
        sentinel = {
            "model_used": "Deterministic",
            "note": (
                "XGBoost was not used for this run — the deterministic fallback "
                "scorer was applied because there were insufficient pseudo-labelled "
                "training samples (see forecast_engine.MIN_LABELLED_SAMPLES). "
                "This file does NOT contain a trained model."
            ),
        }
        (artifact_dir / "model.json").write_text(
            _json.dumps(sentinel, indent=2) + "\n", encoding="utf-8"
        )
        logger.debug(
            "[artifacts %s] model.json sentinel written (deterministic fallback)",
            run_id[:8],
        )

    # ── 3. model_script.py ───────────────────────────────────────────────────
    script_src = Path("wildfire_model_export.py")
    if script_src.exists():
        shutil.copy2(script_src, artifact_dir / "model_script.py")
        logger.debug("[artifacts %s] model_script.py copied", run_id[:8])
    else:
        logger.warning(
            "[artifacts %s] wildfire_model_export.py not found — skipping",
            run_id[:8],
        )

    # ── 4. report.md ─────────────────────────────────────────────────────────
    ts_human = started_at[:19].replace("T", " ") + " UTC"

    if is_deterministic:
        model_status = (
            "⚠️ **Deterministic fallback used** — insufficient pseudo-labelled "
            "samples for XGBoost training. `model.json` is a note file, "
            "NOT a trained model artifact."
        )
        artifact_inventory = (
            "## Artifact inventory\n"
            "\n"
            f"| File | Role |\n"
            f"|------|------|\n"
            f"| `dataset.csv.gz` | {dataset_days}-day (48h) FIRMS detections — risk-metrics window |\n"
            f"| `model.json` | **Sentinel only** — no booster trained (deterministic fallback) |\n"
            f"| `model_script.py` | Standalone training script (wildfire_model_export.py) |\n"
            f"| `report.md` | This file |\n"
        )
    else:
        fw_rows = f"{forecast_window_size:,}" if forecast_window_size is not None else "n/a"
        model_status = "✅ XGBoost booster trained and saved to `model.json`."
        artifact_inventory = (
            "## Artifact inventory\n"
            "\n"
            f"| File | Role |\n"
            f"|------|------|\n"
            f"| `dataset.csv.gz` | {dataset_days}-day (48h) FIRMS detections — risk-metrics window only |\n"
            f"| `dataset_forecast_window.csv.gz` | 7-day FIRMS window — **actual XGBoost training input** ({fw_rows} rows) |\n"
            f"| `model.json` | Trained XGBoost booster (native JSON format) |\n"
            f"| `model_script.py` | Standalone training script (wildfire_model_export.py) |\n"
            f"| `report.md` | This file |\n"
        )

    report_lines = [
        "# Wildfire Agent Run Report",
        "",
        "| Field | Value |",
        "|-------|-------|",
        f"| Run ID | `{run_id}` |",
        f"| Country | {country} |",
        f"| Started (UTC) | {ts_human} |",
        f"| Guardrail verdict | {guardrail_verdict} |",
        f"| Model used | {model_used} |",
        f"| Risk-metrics window | Last {dataset_days} days (48h) of FIRMS detections |",
        f"| Forecast training window | {'Last 7 days of FIRMS detections' if not is_deterministic else 'N/A (deterministic fallback)'} |",
        "",
        model_status,
        "",
        artifact_inventory,
        "---",
        "",
        "## Risk Summary",
        "",
        summary_text or "_No summary generated for this run._",
        "",
        "---",
        "",
        "## Forecast Interpretation",
        "",
        forecast_text or "_No forecast interpretation generated for this run._",
        "",
    ]
    (artifact_dir / "report.md").write_text(
        "\n".join(report_lines), encoding="utf-8"
    )
    logger.debug("[artifacts %s] report.md written", run_id[:8])

    return str(artifact_dir)


# ---------------------------------------------------------------------------
# Size helpers (used by the UI to display total artifact folder size)
# ---------------------------------------------------------------------------

def artifact_dir_size(artifacts_dir: str) -> int:
    """Return the total size in bytes of all files in *artifacts_dir*."""
    total = 0
    try:
        for entry in Path(artifacts_dir).iterdir():
            if entry.is_file():
                total += entry.stat().st_size
    except (FileNotFoundError, NotADirectoryError):
        pass
    return total


def read_artifact(artifacts_dir: str, filename: str) -> bytes | None:
    """
    Return the raw bytes of *filename* inside *artifacts_dir*, or None.

    For .csv.gz files, use read_artifact_csv() instead — this returns the
    compressed bytes, which will be double-compressed by Streamlit's gzip
    middleware and arrive corrupted on the client.
    """
    path = Path(artifacts_dir) / filename
    try:
        return path.read_bytes()
    except (FileNotFoundError, IsADirectoryError):
        return None


def read_artifact_csv(artifacts_dir: str, gz_filename: str) -> bytes | None:
    """
    Return decompressed UTF-8 CSV bytes from a .csv.gz artifact file.

    Streamlit's SelectiveGZipMiddleware re-compresses ALL HTTP responses
    whose Content-Type is not in a narrow exclusion list (text/event-stream,
    audio/, video/).  This means serving gzip bytes via st.download_button
    with mime="application/gzip" causes double-compression: the browser
    transparently decompresses the outer HTTP layer but the inner gzip stream
    is left as raw bytes, producing a corrupt file on the client side.

    The safe fix is to decompress here before handing bytes to Streamlit,
    then serve with mime="text/csv" and a .csv filename.  The on-disk file
    stays gzip-compressed for storage efficiency.

    Returns None if the file is missing.
    """
    path = Path(artifacts_dir) / gz_filename
    try:
        return gzip.decompress(path.read_bytes())
    except (FileNotFoundError, IsADirectoryError):
        return None
    except (OSError, gzip.BadGzipFile) as exc:
        logger.warning(
            "read_artifact_csv: failed to decompress %s: %s", path, exc
        )
        return None
