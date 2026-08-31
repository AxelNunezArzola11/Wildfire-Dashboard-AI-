"""
config.py — Central configuration for the Wildfire Dashboard AI.

All tunable constants live here. Environment variables are loaded from a
local `.env` file via python-dotenv. Application code must import from this
module; no other module should call os.getenv directly.
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# Credentials (required at runtime; missing values are caught at call sites)
# ---------------------------------------------------------------------------

# Strip surrounding whitespace/quotes that text editors sometimes add to .env values.
_PLACEHOLDER_PREFIXES = ("your_", "change_me", "<", "TODO")


def _clean(value: str | None) -> str:
    """Strip whitespace and reject placeholder-looking values."""
    if not value:
        return ""
    value = value.strip().strip('"').strip("'")
    if any(value.lower().startswith(p.lower()) for p in _PLACEHOLDER_PREFIXES):
        return ""
    return value


FIRMS_MAP_KEY: str = _clean(os.getenv("FIRMS_MAP_KEY"))
CARTO_API_KEY: str = _clean(os.getenv("CARTO_API_KEY"))
WATSONX_API_KEY: str = _clean(os.getenv("WATSONX_API_KEY"))
WATSONX_PROJECT_ID: str = _clean(os.getenv("WATSONX_PROJECT_ID"))
WATSONX_URL: str = _clean(os.getenv("WATSONX_URL")) or "https://us-south.ml.cloud.ibm.com"

# ---------------------------------------------------------------------------
# Email alert settings (SMTP via Gmail app password or compatible server)
# ---------------------------------------------------------------------------

# Set to "true" to enable email alerts; any other value (or absent) disables them.
ALERT_EMAIL_ENABLED: str = _clean(os.getenv("ALERT_EMAIL_ENABLED", "false"))

# SMTP host — default is Gmail's submission endpoint (TLS on port 587).
ALERT_SMTP_HOST: str = _clean(os.getenv("ALERT_SMTP_HOST", "smtp.gmail.com"))

# SMTP port — 587 for STARTTLS (recommended); 465 for implicit TLS.
ALERT_SMTP_PORT: str = _clean(os.getenv("ALERT_SMTP_PORT", "587"))

# Gmail address used as the sending account.  Must match the account whose
# app password is set in ALERT_SMTP_APP_PASSWORD.
ALERT_SMTP_USER: str = _clean(os.getenv("ALERT_SMTP_USER"))

# 16-character Gmail app password (Google Account → Security → App passwords).
# NOT your regular Google password.  Requires 2FA to be enabled on the account.
ALERT_SMTP_APP_PASSWORD: str = _clean(os.getenv("ALERT_SMTP_APP_PASSWORD"))

# Recipient address for EXTREME-risk alerts (can differ from the sender).
ALERT_EMAIL_TO: str = _clean(os.getenv("ALERT_EMAIL_TO"))

# ---------------------------------------------------------------------------
# LLM settings
# ---------------------------------------------------------------------------

# Default model; overridable via WATSONX_MODEL_ID env var.
GRANITE_MODEL_ID: str = os.getenv("WATSONX_MODEL_ID", "ibm/granite-4-h-small")

# Critic/guardrail model (Task 6).
# Default: meta-llama/llama-3-3-70b-instruct — a capable instruction-following
# model available in this deployment that reliably returns structured JSON verdicts.
# If ibm/granite-guardian-3-8b becomes available in your deployment you can set:
#   WATSONX_GUARDIAN_MODEL_ID=ibm/granite-guardian-3-8b
GUARDIAN_MODEL_ID: str = os.getenv(
    "WATSONX_GUARDIAN_MODEL_ID", "meta-llama/llama-3-3-70b-instruct"
)

# Backend selector — change to "ollama" or "openai" to swap implementations.
LLM_BACKEND: str = os.getenv("LLM_BACKEND", "watsonx")

# ---------------------------------------------------------------------------
# Autonomous agent settings (agent_runner.py)
# ---------------------------------------------------------------------------

# Default country processed by the autonomous agent when --country is not given.
AGENT_DEFAULT_COUNTRY: str = os.getenv("AGENT_DEFAULT_COUNTRY", "Angola")

# How many hours between autonomous loop cycles (--loop mode).
AGENT_LOOP_HOURS: float = float(os.getenv("AGENT_LOOP_HOURS", "3"))

# Minimum FRP (MW) used by the agent; defaults to same as the interactive app.
AGENT_MIN_FRP: float = float(os.getenv("AGENT_MIN_FRP", str(float(os.getenv("DEFAULT_FRP_THRESHOLD", "10.0")))))

# ---------------------------------------------------------------------------
# Debug / testing flags
# ---------------------------------------------------------------------------

# When True, _generate_summary() replaces the real spread_index in its output
# with the historic fabricated value (1,953,840 km²) so the full
# generator→critic→correction pipeline can be exercised against the live API
# without needing to manufacture bad data in the database.
# Never set this in production.
FORCE_FABRICATED_TEST: bool = os.getenv("FORCE_FABRICATED_TEST", "0").strip() == "1"

# ---------------------------------------------------------------------------
# Cache settings
# ---------------------------------------------------------------------------

# SQLite file used for all local caches (fire data + weather data).
DB_PATH: str = os.getenv("DB_PATH", "wildfire_cache.db")

# TTL for NASA FIRMS fire data cache (minutes).
CACHE_TTL_MINUTES: int = int(os.getenv("CACHE_TTL_MINUTES", "30"))

# TTL for Open-Meteo weather data cache (minutes).
WEATHER_CACHE_TTL_MINUTES: int = int(os.getenv("WEATHER_CACHE_TTL_MINUTES", "60"))

# ---------------------------------------------------------------------------
# Risk / forecast settings
# ---------------------------------------------------------------------------

# Minimum FRP (MW) shown by default; user can adjust via sidebar slider.
DEFAULT_FRP_THRESHOLD: float = float(os.getenv("DEFAULT_FRP_THRESHOLD", "10.0"))

# Size of each forecast grid cell in degrees (0.25° ≈ 28 km at the equator).
FORECAST_GRID_DEG: float = float(os.getenv("FORECAST_GRID_DEG", "0.25"))

# Number of days of FIRMS history used to build forecast features.
FORECAST_HISTORY_DAYS: int = int(os.getenv("FORECAST_HISTORY_DAYS", "7"))

# ---------------------------------------------------------------------------
# Country bounding boxes — imported from the shared, dependency-free module.
# Anything doing `from config import COUNTRY_BBOX` continues to work unchanged.
# ---------------------------------------------------------------------------
from country_bboxes import COUNTRY_BBOX  # noqa: E402 (after env-var block)
