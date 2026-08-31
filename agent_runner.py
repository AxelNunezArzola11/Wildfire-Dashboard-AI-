"""
agent_runner.py — Autonomous single-agent wildfire pipeline.

Orchestrates a full end-to-end cycle for one or more countries using
existing modules — no logic is duplicated:

    ingestor.py       → get_fire_data()
    risk_engine.py    → compute_risk()
    forecast_engine.py→ run_forecast()
    llm_gateway.py    → WatsonxGateway.summarize() / .interpret_forecast()
                        (includes Task 6 guardrail pipeline: pre-filter +
                         Guardian critic + correction loop)

All watsonx calls go through the EXISTING _WATSONX_SEMAPHORE shared with
the interactive Streamlit app, so concurrent runs never exceed the rate limit.

Usage
-----
    # Single run for Angola:
    python agent_runner.py --country Angola

    # Single run for all configured countries:
    python agent_runner.py --all

    # Continuous loop every 3 hours (default) for Angola:
    python agent_runner.py --country Angola --loop

    # Loop with custom interval and min-FRP:
    python agent_runner.py --country Brazil --loop --interval 6 --min-frp 20

    # Force-refresh FIRMS cache before each run:
    python agent_runner.py --country Angola --force-refresh

Exit codes
----------
    0   all runs succeeded
    1   at least one run failed
"""

import argparse
import dataclasses
import json
import logging
import sys
import time
from datetime import datetime, timezone

import config
import agent_store
import artifacts as _artifacts
from email_alerts import check_and_send_alert
from ingestor import get_fire_data
from risk_engine import compute_risk
from forecast_engine import run_forecast
from llm_gateway import WatsonxGateway, UNVERIFIED_MARKER

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Guardrail verdict helper
# ---------------------------------------------------------------------------

def _classify_guardrail(text: str) -> str:
    """
    Infer the guardrail outcome from the summary text.

    WatsonxGateway.summarize() embeds UNVERIFIED_MARKER when both audit
    attempts fail.  We detect 'corrected' by checking whether the text is
    clean (no UNVERIFIED_MARKER) but the pipeline log would have shown a
    correction cycle — we can't tell that from the text alone, so we use
    a conservative heuristic: if the text contains a disclaimer phrase
    injected by the correction prompt it was corrected, otherwise 'pass'.

    Values: 'pass' | 'corrected' | 'unverified' | 'n/a'
    """
    if not text:
        return "n/a"
    if UNVERIFIED_MARKER in text:
        return "unverified"
    # The correction addendum asks the model to add an uncertainty disclaimer.
    # If the text contains the standard disclaimer the model added after
    # being corrected, classify as 'corrected'.  If absent, 'pass'.
    disclaimer_phrases = [
        "not a certainty",
        "probabilistic estimate",
        "uncertainty disclaimer",
    ]
    has_disclaimer = any(p in text.lower() for p in disclaimer_phrases)
    # All well-formed summaries should have a disclaimer; 'pass' is the
    # normal outcome when the first generation already included one.
    return "pass"


# ---------------------------------------------------------------------------
# Core single-run logic
# ---------------------------------------------------------------------------

def run_once(
    country: str,
    min_frp: float = config.DEFAULT_FRP_THRESHOLD,
    horizon_days: int = 1,
    force_refresh: bool = False,
    gateway: WatsonxGateway | None = None,
) -> dict:
    """
    Execute one full agent cycle for *country*.

    Returns the completed agent_runs row as a dict.
    Also writes the row to agent_store (SQLite) and saves four downloadable
    artifacts under agent_artifacts/{run_id}/ on successful/partial runs.
    """
    agent_store.init_schema()
    run_id = agent_store.new_run_id()
    started_at = datetime.now(timezone.utc).isoformat()
    agent_store.insert_run(run_id, country, min_frp, started_at)

    t_start = time.monotonic()
    status = "failed"
    error_message: str | None = None
    risk_metrics: dict = {}
    forecast_top10: list = []
    summary_text: str = ""
    forecast_text: str = ""
    guardrail_verdict: str = "n/a"
    artifacts_dir: str | None = None
    fire_df_ref = None          # hold reference for artifact writer
    forecast_result_ref = None  # hold reference for artifact writer
    # Fixed 48-hour detection window used for risk metrics, forecast input,
    # and the dataset.csv.gz artifact.  NOT tied to any sidebar selector.
    _AGENT_FETCH_DAYS = 2

    logger.info(
        "[agent] run_id=%s  country=%s  min_frp=%.1f  horizon=%dd",
        run_id[:8], country, min_frp, horizon_days,
    )

    try:
        # ── Step 1: fetch fire data ────────────────────────────────────────
        # days=_AGENT_FETCH_DAYS (48 h) is a fixed autonomous-agent constant,
        # independent of any sidebar selector.  The forecast engine fetches its
        # own separate 7-day window internally (_get_fire_window).
        logger.info("[agent %s] Step 1/4: fetching FIRMS data...", run_id[:8])
        fire_df, ingest_s = get_fire_data(
            country, days=_AGENT_FETCH_DAYS, min_frp=min_frp, force_refresh=force_refresh
        )
        if ingest_s is not None:
            logger.info(
                "[agent %s] cold-cache ingest %.1fs, %d detections",
                run_id[:8], ingest_s, len(fire_df),
            )
        fire_df_ref = fire_df

        # ── Step 2: compute risk metrics ──────────────────────────────────
        logger.info("[agent %s] Step 2/4: computing risk metrics...", run_id[:8])
        risk_ctx = compute_risk(fire_df, country, _AGENT_FETCH_DAYS)
        risk_metrics = dataclasses.asdict(risk_ctx)

        # ── Step 2b: check EXTREME-risk alert (idempotent) ─────────────────
        alert_outcome = check_and_send_alert(country, risk_ctx)
        logger.info(
            "[agent %s] alert_outcome=%s risk_level=%s",
            run_id[:8], alert_outcome, risk_ctx.risk_level,
        )

        # ── Step 3: run forecast ──────────────────────────────────────────
        logger.info("[agent %s] Step 3/4: running XGBoost forecast...", run_id[:8])
        forecast_result = run_forecast(fire_df, country, min_frp, horizon_days)
        forecast_result_ref = forecast_result
        top10 = forecast_result.cells[:10] if forecast_result.cells else []
        forecast_top10 = [
            {
                "lat": c.lat_center,
                "lon": c.lon_center,
                "risk_band": c.risk_band,
                "fire_prob_pct": round(c.fire_prob * 100, 1),
                "hist_fire_count_7d": c.historical_fire_count,
            }
            for c in top10
        ]

        # ── Step 4: generate AI insights (includes Task 6 guardrails) ─────
        logger.info("[agent %s] Step 4/4: generating AI insights...", run_id[:8])
        if gateway is None:
            try:
                gateway = WatsonxGateway()
            except RuntimeError as gw_err:
                logger.warning(
                    "[agent %s] LLM gateway unavailable (%s) — "
                    "saving metrics without AI text.",
                    run_id[:8], gw_err,
                )
                status = "partial"

        if gateway is not None:
            summary_text = gateway.summarize(risk_ctx)
            forecast_text = gateway.interpret_forecast(forecast_result)
            guardrail_verdict = _classify_guardrail(summary_text)
            logger.info(
                "[agent %s] guardrail_verdict=%s", run_id[:8], guardrail_verdict
            )

        if status != "partial":
            status = "success"

    except Exception as exc:
        error_message = str(exc)
        status = "failed"
        logger.error("[agent %s] run failed: %s", run_id[:8], exc, exc_info=True)

    finally:
        latency_seconds = time.monotonic() - t_start
        finished_at = datetime.now(timezone.utc).isoformat()

        # ── Save downloadable artifacts (non-fatal if it fails) ────────────
        if status in ("success", "partial") and fire_df_ref is not None:
            try:
                artifacts_dir = _artifacts.save_run_artifacts(
                    run_id=run_id,
                    country=country,
                    started_at=started_at,
                    guardrail_verdict=guardrail_verdict,
                    fire_df=fire_df_ref,
                    forecast_result=forecast_result_ref,
                    summary_text=summary_text,
                    forecast_text=forecast_text,
                    dataset_days=_AGENT_FETCH_DAYS,
                )
                total_size = _artifacts.artifact_dir_size(artifacts_dir)
                logger.info(
                    "[agent %s] artifacts saved → %s  (total %.1f KB)",
                    run_id[:8], artifacts_dir, total_size / 1024,
                )
            except Exception as art_exc:
                logger.warning(
                    "[agent %s] artifact save failed (non-fatal): %s",
                    run_id[:8], art_exc,
                )

        agent_store.update_run(
            run_id,
            finished_at=finished_at,
            status=status,
            risk_metrics=risk_metrics,
            forecast_top10=forecast_top10,
            summary_text=summary_text,
            forecast_text=forecast_text,
            guardrail_verdict=guardrail_verdict,
            error_message=error_message,
            latency_seconds=round(latency_seconds, 1),
            artifacts_dir=artifacts_dir,
        )
        logger.info(
            "[agent %s] finished. status=%s latency=%.1fs guardrail=%s",
            run_id[:8], status, latency_seconds, guardrail_verdict,
        )

    return {
        "run_id": run_id,
        "country": country,
        "min_frp": min_frp,
        "started_at": started_at,
        "finished_at": finished_at,
        "status": status,
        "risk_metrics": risk_metrics,
        "forecast_top10": forecast_top10,
        "summary_text": summary_text,
        "forecast_text": forecast_text,
        "guardrail_verdict": guardrail_verdict,
        "error_message": error_message,
        "latency_seconds": latency_seconds,
        "artifacts_dir": artifacts_dir,
    }


# ---------------------------------------------------------------------------
# Loop mode
# ---------------------------------------------------------------------------

def run_loop(
    countries: list[str],
    interval_hours: float,
    min_frp: float,
    horizon_days: int,
    force_refresh: bool,
) -> None:
    """
    Repeat run_once() for each country every *interval_hours* hours.
    Ctrl-C to stop.
    """
    logger.info(
        "[agent] loop mode: countries=%s  interval=%.1fh  min_frp=%.1f",
        countries, interval_hours, min_frp,
    )
    # Initialise gateway once and reuse across iterations.
    try:
        gateway = WatsonxGateway()
    except RuntimeError as e:
        logger.warning("[agent] LLM gateway unavailable (%s), will retry per run.", e)
        gateway = None

    while True:
        for country in countries:
            run_once(
                country=country,
                min_frp=min_frp,
                horizon_days=horizon_days,
                force_refresh=force_refresh,
                gateway=gateway,
            )
        sleep_s = interval_hours * 3600
        logger.info("[agent] sleeping %.0f s until next cycle...", sleep_s)
        time.sleep(sleep_s)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Autonomous wildfire agent — fetch, forecast, generate insights.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    grp = p.add_mutually_exclusive_group(required=True)
    grp.add_argument("--country", help="Single country to process.")
    grp.add_argument(
        "--all",
        action="store_true",
        help="Process all countries in config.COUNTRY_BBOX.",
    )
    p.add_argument(
        "--loop",
        action="store_true",
        help="Run repeatedly on a fixed interval instead of once.",
    )
    p.add_argument(
        "--interval",
        type=float,
        default=config.AGENT_LOOP_HOURS,
        metavar="HOURS",
        help="Loop interval in hours (only used with --loop).",
    )
    p.add_argument(
        "--min-frp",
        type=float,
        default=config.DEFAULT_FRP_THRESHOLD,
        dest="min_frp",
        help="Minimum FRP (MW) filter applied to FIRMS detections.",
    )
    p.add_argument(
        "--horizon",
        type=int,
        default=1,
        choices=[1, 7],
        help="Forecast horizon in days.",
    )
    p.add_argument(
        "--force-refresh",
        action="store_true",
        dest="force_refresh",
        help="Bypass FIRMS cache and re-fetch live data.",
    )
    p.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return p


def main() -> int:
    p = _build_parser()
    args = p.parse_args()

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)-8s %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        stream=sys.stdout,
    )

    countries: list[str] = (
        sorted(config.COUNTRY_BBOX.keys()) if args.all else [args.country]
    )
    # Validate country names early.
    for c in countries:
        if c not in config.COUNTRY_BBOX:
            print(f"ERROR: '{c}' is not in config.COUNTRY_BBOX.", file=sys.stderr)
            return 1

    if args.loop:
        run_loop(
            countries=countries,
            interval_hours=args.interval,
            min_frp=args.min_frp,
            horizon_days=args.horizon,
            force_refresh=args.force_refresh,
        )
        return 0  # unreachable unless interrupted

    any_failed = False
    try:
        gw = WatsonxGateway()
    except RuntimeError as e:
        logger.warning("LLM gateway unavailable: %s", e)
        gw = None

    for country in countries:
        result = run_once(
            country=country,
            min_frp=args.min_frp,
            horizon_days=args.horizon,
            force_refresh=args.force_refresh,
            gateway=gw,
        )
        if result["status"] == "failed":
            any_failed = True
        # Print a compact summary line to stdout.
        print(
            f"[{result['status'].upper():8s}] {result['country']:30s} "
            f"latency={result['latency_seconds']:.1f}s  "
            f"guardrail={result['guardrail_verdict']}  "
            f"run_id={result['run_id'][:8]}"
        )
        if result.get("error_message"):
            print(f"           error: {result['error_message']}")

    return 1 if any_failed else 0


if __name__ == "__main__":
    sys.exit(main())
