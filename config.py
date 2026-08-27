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
WATSONX_API_KEY: str = _clean(os.getenv("WATSONX_API_KEY"))
WATSONX_PROJECT_ID: str = _clean(os.getenv("WATSONX_PROJECT_ID"))
WATSONX_URL: str = _clean(os.getenv("WATSONX_URL")) or "https://us-south.ml.cloud.ibm.com"

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
# Country bounding boxes  — W, S, E, N  (longitude_min, lat_min, lon_max, lat_max)
#
# Designed as a plain dict so a future geo_lookup.py can populate the same
# shape (e.g. from a world GeoJSON) with zero changes to callers.
# ---------------------------------------------------------------------------

COUNTRY_BBOX: dict = {
    "Brazil":                        "-73.99,-33.75,-28.85,5.27",
    "Australia":                     "113.34,-43.64,153.57,-10.68",
    "United States":                 "-124.74,24.52,-66.95,49.38",
    "Canada":                        "-141.00,41.68,-52.62,83.11",
    "Indonesia":                     "95.01,-11.01,141.02,5.91",
    "Russia":                        "19.64,41.19,180.00,81.86",
    "Democratic Republic of Congo":  "12.18,-13.46,31.30,5.39",
    "Angola":                        "11.67,-18.04,24.08,-4.39",
    "Mozambique":                     "30.21,-26.87,40.84,-10.47",
    "Mexico":                        "-117.13,14.53,-86.70,32.72",
    "Bolivia":                       "-69.65,-22.90,-57.47,-9.67",
    "Venezuela":                     "-73.35,0.65,-59.80,12.20",
    "Argentina":                     "-73.56,-55.06,-53.64,-21.78",
    "India":                         "68.18,8.07,97.40,35.51",
    "China":                         "73.50,18.16,134.77,53.56",
    "Nigeria":                       "2.69,4.27,14.68,13.89",
    "South Africa":                  "16.46,-34.83,32.89,-22.13",
    "Portugal":                      "-9.53,36.96,-6.19,42.15",
    "Greece":                        "19.37,34.80,29.64,41.75",
    "Chile":                         "-75.64,-55.90,-66.96,-17.51",
}
