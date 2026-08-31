"""
email_alerts.py — Email alerting for EXTREME wildfire risk.

Public API
----------
check_and_send_alert(country, risk_ctx) -> str
    "sent" | "skipped-not-extreme" | "skipped-already-alerted" | "skipped-not-configured"

send_extreme_risk_alert(country, risk_ctx) -> bool
    Low-level send; does NOT check idempotency.  Prefer check_and_send_alert().

SQLite table: alert_state in the same wildfire_cache.db used by ingestor/agent_store.
Schema:
    country            TEXT PRIMARY KEY
    last_alerted_level TEXT   -- risk level string at last send
    last_alerted_at    TEXT   -- ISO-8601 UTC timestamp of last send

Idempotency rule: only send when the level TRANSITIONS INTO EXTREME
(previous last_alerted_level != "EXTREME" AND current risk_level == "EXTREME").
When the state returns to a non-EXTREME level the row is updated so that the
next EXTREME transition fires again.

Fail-open: if ALERT_EMAIL_ENABLED is not "true" or credentials are incomplete,
every call silently returns "skipped-not-configured" — no exception is raised.
This matches how the project treats missing watsonx credentials.
"""

import logging
import smtplib
import sqlite3
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

import config
from risk_engine import RiskContext

logger = logging.getLogger(__name__)

DASHBOARD_URL = "https://kwgzjbgdsdyd9epjepovex.streamlit.app/"

# ---------------------------------------------------------------------------
# SQLite helpers — alert_state table
# ---------------------------------------------------------------------------

_CREATE_ALERT_STATE_SQL = """
CREATE TABLE IF NOT EXISTS alert_state (
    country            TEXT PRIMARY KEY,
    last_alerted_level TEXT NOT NULL,
    last_alerted_at    TEXT NOT NULL
)
"""


def _conn() -> sqlite3.Connection:
    c = sqlite3.connect(config.DB_PATH, check_same_thread=False)
    c.row_factory = sqlite3.Row
    return c


def init_alert_schema() -> None:
    """Create alert_state table if it does not exist yet (idempotent)."""
    with _conn() as c:
        c.execute(_CREATE_ALERT_STATE_SQL)
        c.commit()
    logger.debug("email_alerts: alert_state schema initialised.")


def _get_last_alerted_level(country: str) -> str | None:
    """Return the last alerted risk level for *country*, or None if never alerted."""
    with _conn() as c:
        row = c.execute(
            "SELECT last_alerted_level FROM alert_state WHERE country = ?",
            (country,),
        ).fetchone()
    return row["last_alerted_level"] if row else None


def _record_alert(country: str, level: str) -> None:
    """Upsert the alert state for *country* to *level* with the current timestamp."""
    now = datetime.now(timezone.utc).isoformat()
    with _conn() as c:
        c.execute(
            """INSERT INTO alert_state (country, last_alerted_level, last_alerted_at)
               VALUES (?, ?, ?)
               ON CONFLICT(country) DO UPDATE SET
                   last_alerted_level = excluded.last_alerted_level,
                   last_alerted_at    = excluded.last_alerted_at""",
            (country, level, now),
        )
        c.commit()


# ---------------------------------------------------------------------------
# Configuration helpers
# ---------------------------------------------------------------------------

def _is_alert_enabled() -> bool:
    """Return True only when ALERT_EMAIL_ENABLED=true and all credentials present."""
    if config.ALERT_EMAIL_ENABLED.lower() != "true":
        return False
    return bool(
        config.ALERT_SMTP_USER
        and config.ALERT_SMTP_APP_PASSWORD
        and config.ALERT_EMAIL_TO
    )


# ---------------------------------------------------------------------------
# Email construction and send
# ---------------------------------------------------------------------------

def _build_message(country: str, risk_ctx: RiskContext) -> MIMEMultipart:
    """Construct the alert MIMEMultipart message."""
    msg = MIMEMultipart("alternative")
    msg["Subject"] = f"[Wildfire Alert] EXTREME risk detected — {country}"
    msg["From"] = config.ALERT_SMTP_USER
    msg["To"] = config.ALERT_EMAIL_TO

    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    body = (
        f"Wildfire risk alert — {country}\n"
        f"{'=' * 50}\n\n"
        f"Risk Level    : {risk_ctx.risk_level}\n"
        f"Fire Count    : {risk_ctx.fire_count:,} detections "
        f"(last {risk_ctx.time_window_days}d)\n"
        f"Total FRP     : {risk_ctx.total_frp:,.0f} MW\n"
        f"Max FRP       : {risk_ctx.max_frp:,.0f} MW\n"
        f"Spread Index  : {risk_ctx.spread_index:,.0f} km²\n"
        f"Generated     : {now_utc}\n\n"
        f"Live dashboard: {DASHBOARD_URL}\n\n"
        f"This is an automated alert from Wildfire Dashboard AI.\n"
        f"Alert fires on EXTREME risk transitions; no further email will be sent\n"
        f"until the risk level falls and then rises to EXTREME again.\n"
    )
    msg.attach(MIMEText(body, "plain"))
    return msg


def send_extreme_risk_alert(country: str, risk_ctx: RiskContext) -> bool:
    """
    Attempt to send an EXTREME-risk email alert.

    Does NOT check idempotency — callers should use check_and_send_alert()
    for the full transition-guarded path.

    Returns True on success, False on any failure or misconfiguration.
    Does NOT raise exceptions.
    """
    if not _is_alert_enabled():
        logger.info(
            "[alert] skipped-not-configured: ALERT_EMAIL_ENABLED not true "
            "or SMTP credentials incomplete."
        )
        return False

    msg = _build_message(country, risk_ctx)
    try:
        with smtplib.SMTP(config.ALERT_SMTP_HOST, int(config.ALERT_SMTP_PORT)) as smtp:
            smtp.ehlo()
            smtp.starttls()
            smtp.login(config.ALERT_SMTP_USER, config.ALERT_SMTP_APP_PASSWORD)
            smtp.sendmail(
                config.ALERT_SMTP_USER,
                [config.ALERT_EMAIL_TO],
                msg.as_string(),
            )
        logger.info(
            "[alert] sent EXTREME alert for %s to %s",
            country, config.ALERT_EMAIL_TO,
        )
        return True
    except Exception as exc:  # noqa: BLE001
        logger.error("[alert] failed to send alert for %s: %s", country, exc)
        return False


# ---------------------------------------------------------------------------
# Idempotent public entry point
# ---------------------------------------------------------------------------

def check_and_send_alert(country: str, risk_ctx: RiskContext) -> str:
    """
    Check the idempotent transition condition and send an alert if it fires.

    Transitions that trigger a send:
        previous last_alerted_level != "EXTREME"  AND  current == "EXTREME"

    Non-EXTREME levels are always recorded so the next EXTREME transition fires.

    Returns one of:
        "sent"                    — email was dispatched successfully
        "skipped-not-extreme"     — current risk level is not EXTREME
        "skipped-already-alerted" — risk is EXTREME but was already alerted
        "skipped-not-configured"  — alert disabled or credentials missing
        "failed"                  — configured and attempted, but SMTP call failed
                                    (state is NOT recorded; next cycle will retry)
    """
    init_alert_schema()

    if not _is_alert_enabled():
        logger.info("[alert] skipped-not-configured for %s", country)
        return "skipped-not-configured"

    current_level = risk_ctx.risk_level

    if current_level != "EXTREME":
        # Always update state for non-EXTREME so next EXTREME transition fires.
        _record_alert(country, current_level)
        logger.info(
            "[alert] skipped-not-extreme for %s (level=%s)", country, current_level
        )
        return "skipped-not-extreme"

    # current_level == "EXTREME"
    last_level = _get_last_alerted_level(country)
    if last_level == "EXTREME":
        logger.info(
            "[alert] skipped-already-alerted for %s (still EXTREME)", country
        )
        return "skipped-already-alerted"

    # Transition INTO EXTREME — send the alert.
    success = send_extreme_risk_alert(country, risk_ctx)
    if success:
        _record_alert(country, "EXTREME")
        return "sent"
    # Configured and attempted but SMTP call failed — do NOT record as alerted
    # so the next pipeline cycle retries.
    logger.warning("[alert] failed to send for %s — will retry next cycle", country)
    return "failed"
