"""
app.py — Wildfire Dashboard AI.

Entry point: streamlit run app.py

Wires together all modules into a six-tab Streamlit application:
  🗺️  Map          — Active fire points + forecast risk grid (Folium)
  📊  Risk Summary — Metric cards + AI risk analysis
  🔮  Forecast     — Top-10 risk cells table + AI forecast interpretation
  💬  Chat         — Context-aware chat with RiskContext + ForecastResult injected
  🤖  Agent Status — Autonomous agent run history + health indicator
  🌿  Land Cover   — On-demand Sentinel-2 fetch, NDVI map, land cover classification
"""

import datetime

import streamlit as st
import folium
from folium.plugins import MarkerCluster
from streamlit_folium import st_folium
import pandas as pd

import re

import config
import agent_store
import agent_runner
import artifacts as _artifacts
from ingestor import get_fire_data
from risk_engine import compute_risk
from forecast_engine import run_forecast
from llm_gateway import get_gateway, UNVERIFIED_MARKER
import sentinel_fetch as _sentinel_fetch
import landcover_classifier as _lc


# ---------------------------------------------------------------------------
# LLM response renderer — separates Evidence block from narrative
# ---------------------------------------------------------------------------

def _render_llm_response(text: str, level: str = "info") -> None:
    """Render an LLM response with the Evidence block visually separated.

    Looks for an ``Evidence:`` section (and optional ``Confidence:`` line)
    before the narrative text and displays them in a styled container so they
    are immediately scannable rather than buried in prose.

    Parameters
    ----------
    text:
        Raw LLM output string.
    level:
        Streamlit alert level for the narrative portion — "error", "warning",
        or "info".  Matches risk level conventions used by the caller.
    """
    # ---- Split into evidence block, confidence line, and narrative ----
    # Pattern: optional leading whitespace, then "Evidence:" heading, then
    # bullet lines, then optional "Confidence: X%" line, then narrative.
    evidence_lines: list[str] = []
    confidence_line: str = ""
    narrative: str = text

    ev_match = re.search(
        r"(?im)^Evidence:\s*\n((?:^\s*[-•]\s*.+\n?)+)",
        text,
    )
    if ev_match:
        raw_bullets = ev_match.group(1)
        evidence_lines = [
            re.sub(r"^\s*[-•]\s*", "", ln).strip()
            for ln in raw_bullets.splitlines()
            if ln.strip()
        ]
        # Everything after the evidence block is potential confidence + narrative
        after_evidence = text[ev_match.end():]
    else:
        after_evidence = text

    conf_match = re.search(r"(?im)^Confidence:\s*(\d+%?)\s*$", after_evidence)
    if conf_match:
        confidence_line = conf_match.group(1).strip()
        # Narrative = everything after the Confidence line
        narrative = after_evidence[conf_match.end():].strip()
    else:
        narrative = after_evidence.strip()

    # ---- Render evidence block (only when the model actually produced one) ----
    if evidence_lines:
        bullets_html = "".join(
            f"<li style='margin:2px 0;'>{line}</li>"
            for line in evidence_lines
        )
        conf_html = (
            f"<p style='margin:8px 0 0;font-weight:600;'>"
            f"Confidence: {confidence_line}</p>"
            if confidence_line
            else ""
        )
        st.markdown(
            f"""
<div style="
    background:#f0f4ff;
    border-left:4px solid #3b82d4;
    border-radius:4px;
    padding:10px 14px;
    margin-bottom:10px;
    font-size:0.92em;
">
  <p style="margin:0 0 6px;font-weight:700;letter-spacing:.03em;color:#1f2328;">
    📋 Evidence
  </p>
  <ul style="margin:0;padding-left:18px;color:#1f2328;">
    {bullets_html}
  </ul>
  {conf_html}
</div>
""",
            unsafe_allow_html=True,
        )

    # ---- Render narrative with appropriate alert colour ----
    if narrative:
        if level == "error":
            st.error(narrative)
        elif level == "warning":
            st.warning(narrative)
        else:
            st.info(narrative)

# ---------------------------------------------------------------------------
# Page config — must be the first Streamlit call
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="Wildfire Dashboard AI",
    page_icon="🔥",
    layout="wide",
)

# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

with st.sidebar:
    st.title("🔥 Wildfire Dashboard AI")

    country = st.selectbox("Country", sorted(config.COUNTRY_BBOX.keys()))

    days_map = {"Last 48 hours": 2, "Last 7 days": 7}
    days_label = st.radio("Time Range", list(days_map.keys()), index=0)
    days = days_map[days_label]

    horizon_map = {"24 hours": 1, "7 days": 7}
    horizon_label = st.radio(
        "Forecast Horizon",
        list(horizon_map.keys()),
        index=0,
        help=(
            "24 hours — next-day fire probability from current weather conditions.\n\n"
            "7 days — weekly outlook using 7-day weather forecast averages. "
            "Carries significantly more uncertainty than the 24-hour estimate."
        ),
    )
    horizon_days = horizon_map[horizon_label]

    min_frp = st.slider(
        "Min FRP threshold (MW)",
        0.0,
        500.0,
        config.DEFAULT_FRP_THRESHOLD,
        step=5.0,
    )

    refresh = st.button("🔄 Refresh Data")
    if refresh:
        st.session_state["refresh_counter"] = (
            st.session_state.get("refresh_counter", 0) + 1
        )

    # ── Developer debug toggle — Guardrails fabrication test ─────────────
    # Only shown when FORCE_FABRICATED_TEST=1 is already set in the environment
    # (so it never appears in a normal production session).
    _env_force = config.FORCE_FABRICATED_TEST
    if _env_force or st.query_params.get("debug") == "guardrails":
        st.divider()
        st.caption("🛠 Developer Mode")
        _force_fab = st.checkbox(
            "Inject fabricated spread_index",
            value=_env_force,
            help=(
                "Replaces the spread_index in the generated summary with the "
                "historically-bugged 1,953,840 km² value so the full "
                "critic→correction loop can be exercised against the live API."
            ),
            key="force_fabricated",
        )
        # Propagate the checkbox state into config so llm_gateway picks it up.
        config.FORCE_FABRICATED_TEST = _force_fab
        if _force_fab:
            st.warning(
                "⚠️ **Fabrication-injection active.** Summaries will contain a "
                "known-wrong spread value to trigger the guardrail pipeline."
            )

# ---------------------------------------------------------------------------
# Configuration guard — fail fast with a clear UI message, not a traceback
# ---------------------------------------------------------------------------

if not config.FIRMS_MAP_KEY:
    st.error(
        "**FIRMS_MAP_KEY is not configured.** "
        "Register for a free key at https://firms.modaps.eosdis.nasa.gov/api/area/ "
        "and add `FIRMS_MAP_KEY=<your_key>` to your `.env` file, then restart the app."
    )
    st.stop()

# ---------------------------------------------------------------------------
# Data loading — fire detections
# ---------------------------------------------------------------------------

@st.cache_data(ttl=config.CACHE_TTL_MINUTES * 60, show_spinner=False)
def load_fire_data(country, days, min_frp, _refresh_flag):
    return get_fire_data(country, days, min_frp, force_refresh=bool(_refresh_flag))


try:
    with st.spinner("Fetching fire data..."):
        fire_df, _ingest_seconds = load_fire_data(
            country, days, min_frp, st.session_state.get("refresh_counter", 0)
        )
    if _ingest_seconds is not None:
        st.sidebar.caption(
            f"⏱ Cold-cache ingest: {_ingest_seconds:.1f}s "
            f"({days}d window, {len(fire_df):,} detections)"
        )
except RuntimeError as e:
    st.error(f"**Configuration error:** {e}")
    st.stop()
except Exception as e:
    st.error(f"**Failed to load fire data:** {e}")
    st.stop()

risk_ctx = compute_risk(fire_df, country, days)

# ---------------------------------------------------------------------------
# Forecast loading — store in session_state to avoid redundant recomputes
# ---------------------------------------------------------------------------

forecast_key = (country, days, min_frp, horizon_days)
if (
    "forecast_key" not in st.session_state
    or st.session_state.forecast_key != forecast_key
    or refresh
):
    horizon_label_short = "24-hour" if horizon_days == 1 else "7-day"
    with st.spinner(f"Running {horizon_label_short} forecast model..."):
        try:
            st.session_state.forecast_result = run_forecast(
                fire_df, country, min_frp, horizon_days
            )
        except Exception as e:
            st.error(f"**Forecast error:** {e}")
            st.stop()
        st.session_state.forecast_key = forecast_key

forecast_result = st.session_state.forecast_result

# ---------------------------------------------------------------------------
# LLM Gateway initialisation
# ---------------------------------------------------------------------------

try:
    gateway = get_gateway()
except RuntimeError as e:
    st.error(f"⚠️ LLM not configured: {e}")
    gateway = None

# ---------------------------------------------------------------------------
# LLM session-state cache invalidation
# ---------------------------------------------------------------------------

# Include config.FORCE_FABRICATED_TEST in the cache key so that toggling the
# debug injection checkbox immediately invalidates any previously cached
# summary — preventing a stale fabricated text from persisting on-screen.
summary_cache_key = (country, days, min_frp, config.FORCE_FABRICATED_TEST)
if (
    "summary_key" not in st.session_state
    or st.session_state.summary_key != summary_cache_key
):
    st.session_state.summary_text = None
    st.session_state.summary_key = summary_cache_key

forecast_interp_key = (country, days, min_frp)
if (
    "forecast_interp_key" not in st.session_state
    or st.session_state.forecast_interp_key != forecast_interp_key
):
    st.session_state.forecast_interp_text = None
    st.session_state.forecast_interp_key = forecast_interp_key

# ---------------------------------------------------------------------------
# Tabs
# ---------------------------------------------------------------------------

# Ensure agent_runs table exists (idempotent — safe to call every rerun).
agent_store.init_schema()

# ── Land cover model — loaded once per process, not per rerun ───────────────
@st.cache_resource(show_spinner=False)
def _get_landcover_model():
    return _lc.load_landcover_model()

tab_map, tab_summary, tab_forecast, tab_chat, tab_agent, tab_landcover = st.tabs(
    ["🗺️ Map", "📊 Risk Summary", "🔮 Forecast", "💬 Chat", "🤖 Agent Status", "🌿 Land Cover"]
)

# ── Tab 1: Map ───────────────────────────────────────────────────────────────

with tab_map:
    try:
        st.subheader(f"Active Fires & Forecast Risk — {country}")

        bbox = config.COUNTRY_BBOX[country]
        w, s, e, n = [float(x) for x in bbox.split(",")]
        center_lat, center_lon = (s + n) / 2, (w + e) / 2

        m = folium.Map(
            location=[center_lat, center_lon], zoom_start=5, tiles="CartoDB Voyager"
        )

        # Layer 1: Active fire points (clustered, coloured by FRP quartile)
        # Hard cap at MAP_RENDER_LIMIT rows to prevent the Folium→srcdoc payload
        # from exceeding browser/Streamlit limits (~116k markers silently blank).
        # The cap is applied to the map only; the CSV export uses the full dataset.
        MAP_RENDER_LIMIT = 5_000
        fire_layer = folium.FeatureGroup(name="🔴 Active Fires", show=True)
        if not fire_df.empty:
            map_df = (
                fire_df.nlargest(MAP_RENDER_LIMIT, "frp")
                if len(fire_df) > MAP_RENDER_LIMIT
                else fire_df
            )
            if len(fire_df) > MAP_RENDER_LIMIT:
                st.caption(
                    f"⚠️ Map shows top {MAP_RENDER_LIMIT:,} detections by FRP "
                    f"({len(fire_df):,} total). Use **Download 30-day dataset (CSV)** "
                    "for the full dataset."
                )
            frp_75 = map_df["frp"].quantile(0.75)
            frp_50 = map_df["frp"].quantile(0.50)
            cluster = MarkerCluster().add_to(fire_layer)
            for row in map_df.itertuples(index=False):
                color = (
                    "red"
                    if row.frp >= frp_75
                    else ("orange" if row.frp >= frp_50 else "yellow")
                )
                folium.CircleMarker(
                    location=[row.latitude, row.longitude],
                    radius=5,
                    color=color,
                    fill=True,
                    fill_opacity=0.7,
                    popup=(
                        f"FRP: {row.frp:.1f} MW<br>"
                        f"Date: {row.acq_date}<br>"
                        f"Confidence: {row.confidence}"
                    ),
                ).add_to(cluster)
        fire_layer.add_to(m)

        # Layer 2: Forecast grid cells coloured by risk_band
        forecast_layer = folium.FeatureGroup(name="🟠 24h Forecast Risk", show=True)
        band_colors = {
            "LOW": "#2ecc71",
            "MEDIUM": "#f39c12",
            "HIGH": "#e67e22",
            "EXTREME": "#e74c3c",
        }
        half = config.FORECAST_GRID_DEG / 2
        for cell in forecast_result.cells:
            color = band_colors.get(cell.risk_band, "#95a5a6")
            folium.Rectangle(
                bounds=[
                    [cell.lat_center - half, cell.lon_center - half],
                    [cell.lat_center + half, cell.lon_center + half],
                ],
                color=color,
                fill=True,
                fill_opacity=0.15,
                weight=0.5,
                popup=(
                    f"Risk: {cell.risk_band}<br>"
                    f"Prob: {cell.fire_prob:.1%}<br>"
                    f"Fires(7d): {cell.historical_fire_count}"
                ),
            ).add_to(forecast_layer)
        forecast_layer.add_to(m)

        # Log tile-load failures to the browser console so they are visible in
        # DevTools instead of silently disappearing.
        tile_error_js = folium.Element(
            "<script>"
            "document.addEventListener('DOMContentLoaded', function() {"
            "  var maps = Object.values(window).filter(function(v) {"
            "    return v && v._container && v._container.classList.contains('leaflet-container');"
            "  });"
            "  maps.forEach(function(lmap) {"
            "    lmap.eachLayer(function(layer) {"
            "      if (layer.on) {"
            "        layer.on('tileerror', function(e) {"
            "          console.error('[Wildfire] Tile load failed:', e.tile ? e.tile.src : e);"
            "        });"
            "      }"
            "    });"
            "  });"
            "});"
            "</script>"
        )
        m.get_root().html.add_child(tile_error_js)

        folium.LayerControl().add_to(m)

        # ── Map Legend ────────────────────────────────────────────────────────
        legend_html = """
        <div style="
            position: fixed;
            bottom: 30px; right: 10px;
            background: white;
            border: 1px solid #ccc;
            border-radius: 6px;
            padding: 10px 14px;
            font-size: 12px;
            line-height: 1.7;
            z-index: 9999;
            min-width: 220px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.2);
        ">
          <b>Map Legend</b><br>
          <hr style="margin:4px 0">
          <b>🔴 Individual fire detections</b><br>
          Color = FRP relative to current dataset:<br>
          &nbsp; <span style="color:red">&#9679;</span> Red &nbsp;— high FRP (≥ 75th percentile)<br>
          &nbsp; <span style="color:orange">&#9679;</span> Orange — moderate FRP (≥ 50th percentile)<br>
          &nbsp; <span style="color:#c8c800">&#9679;</span> Yellow — low FRP (&lt; 50th percentile)<br>
          <i style="color:#555">(FRP = Fire Radiative Power in MW)</i><br>
          <hr style="margin:4px 0">
          <b>Clustered circles</b> (zoom out)<br>
          &nbsp; Number = count of fire detections<br>
          &nbsp; Color = cluster size (Leaflet default):<br>
          &nbsp; <span style="color:green">&#9679;</span> Green — small cluster (&lt; 10)<br>
          &nbsp; <span style="color:#c8a800">&#9679;</span> Yellow — medium cluster (10–99)<br>
          &nbsp; <span style="color:#c0392b">&#9679;</span> Red &nbsp;— large cluster (≥ 100)<br>
          &nbsp; Size also scales with cluster count<br>
          <hr style="margin:4px 0">
          <b>Forecast grid cells</b><br>
          &nbsp; <span style="color:#2ecc71">&#9632;</span> Green — LOW risk<br>
          &nbsp; <span style="color:#f39c12">&#9632;</span> Orange — MEDIUM risk<br>
          &nbsp; <span style="color:#e67e22">&#9632;</span> Dark orange — HIGH risk<br>
          &nbsp; <span style="color:#e74c3c">&#9632;</span> Red — EXTREME risk<br>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))

        st_folium(m, width="100%", height=550)
    except Exception as _tab_err:
        st.error(f"**Map tab error:** {_tab_err}")

# ---------------------------------------------------------------------------
# Export helpers
# ---------------------------------------------------------------------------


@st.cache_data(ttl=config.CACHE_TTL_MINUTES * 60, show_spinner=False)
def _load_30d_fire_data(country: str, min_frp: float, _refresh_flag: int) -> pd.DataFrame:
    """Fetch a 30-day FIRMS window purely for CSV export.

    This path is completely isolated from the sidebar time-range selection
    and MUST NOT feed into Map, Risk Summary, or Forecast rendering.
    """
    from ingestor import get_fire_data  # already imported at module level; repeated for clarity

    df, _ = get_fire_data(country, days=30, min_frp=min_frp)
    return df


def _build_export_csv(
    fire_df: pd.DataFrame, forecast_result, country: str, days: int
) -> bytes:
    """
    Return plain (uncompressed) UTF-8 CSV bytes for use in st.download_button.

    Streamlit's SelectiveGZipMiddleware re-compresses all HTTP responses that
    aren't in its narrow exclusion list.  Serving gzip bytes through
    st.download_button causes double-compression: the browser strips the outer
    HTTP gzip layer and delivers the still-compressed inner bytes as the
    downloaded file, which macOS/Windows unarchive tools reject.

    The gzip layer exists only in artifacts.build_export_csv (for on-disk
    storage efficiency in agent_artifacts/).  This function decompresses
    immediately so the download button always delivers a plain .csv.
    """
    import gzip as _gzip
    return _gzip.decompress(
        _artifacts.build_export_csv(fire_df, forecast_result, country, days)
    )


def _build_model_script() -> bytes:
    """Read wildfire_model_export.py from disk and return its bytes."""
    with open("wildfire_model_export.py", "rb") as fh:
        return fh.read()


# ── Tab 2: Risk Summary ──────────────────────────────────────────────────────

with tab_summary:
    try:
        st.subheader(f"Current Fire Risk — {country} ({days_label})")

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Fire Detections", risk_ctx.fire_count)
        col2.metric("Total FRP", f"{risk_ctx.total_frp:.0f} MW")
        col3.metric("Max FRP", f"{risk_ctx.max_frp:.0f} MW")
        col4.metric("Risk Level", risk_ctx.risk_level)

        st.divider()

        # ── Export buttons ────────────────────────────────────────────────────
        st.markdown("#### ⬇️ Export")
        exp_col1, exp_col2 = st.columns(2)

        with exp_col1:
            csv_filename = (
                f"{country.lower().replace(' ', '_')}_wildfire_data_"
                f"{datetime.date.today().isoformat()}.csv"
            )
            csv_bytes = _build_export_csv(fire_df, forecast_result, country, days)
            st.download_button(
                label="📥 Download data (CSV)",
                data=csv_bytes,
                file_name=csv_filename,
                mime="text/csv",
                help=(
                    "Downloads the current filtered fire detections enriched with "
                    "forecast probability for the nearest grid cell."
                ),
            )

        with exp_col2:
            try:
                model_script_bytes = _build_model_script()
                st.download_button(
                    label="📥 Download model code (.py)",
                    data=model_script_bytes,
                    file_name="wildfire_model_export.py",
                    mime="text/x-python",
                    help=(
                        "Standalone Python script reproducing the feature engineering, "
                        "pseudo-labelling, and XGBoost training/inference. "
                        "No credentials are included — set FIRMS_MAP_KEY via env variable."
                    ),
                )
            except FileNotFoundError:
                st.error("wildfire_model_export.py not found — cannot offer model download.")

        # ── 30-day CSV export (independent of sidebar time range) ─────────────
        st.markdown("---")
        exp_col3, _ = st.columns([1, 1])
        with exp_col3:
            csv30_filename = (
                f"{country.lower().replace(' ', '_')}_wildfire_30d_"
                f"{datetime.date.today().isoformat()}.csv"
            )
            if st.button(
                "📦 Download 30-day dataset (CSV)",
                help=(
                    "Fetches the last 30 days of FIRMS detections for this country "
                    "(independent of the Time Range selector above) and generates a "
                    "downloadable CSV. This data is never fed into Map, Risk Summary, "
                    "or Forecast rendering."
                ),
                key="btn_30d_csv",
            ):
                with st.spinner("Fetching 30-day FIRMS window…"):
                    try:
                        df_30d = _load_30d_fire_data(
                            country, min_frp,
                            st.session_state.get("refresh_counter", 0),
                        )
                        csv30_bytes = _build_export_csv(df_30d, None, country, 30)
                        st.download_button(
                            label=f"⬇️ Save {len(df_30d):,} detections as CSV",
                            data=csv30_bytes,
                            file_name=csv30_filename,
                            mime="text/csv",
                            key="dl_30d_csv",
                        )
                        st.caption(
                            f"30-day window loaded: **{len(df_30d):,} detections** "
                            f"({country}). Click the button above to save."
                        )
                    except Exception as _e30:
                        st.error(f"30-day fetch failed: {_e30}")

        st.divider()

        st.markdown("#### 🤖 AI Risk Analysis")

        # ── Agent-run toggle ──────────────────────────────────────────────
        agent_run_summary = agent_store.get_latest_run(country)
        use_agent_summary = st.toggle(
            "Use latest agent run",
            value=bool(agent_run_summary),
            key="use_agent_summary",
            disabled=not bool(agent_run_summary),
            help=(
                "When ON, displays the most recent successful automated-agent "
                "summary for this country instead of calling the live pipeline."
            ),
        )

        if use_agent_summary and agent_run_summary:
            _ar = agent_run_summary
            _age_s = (
                datetime.datetime.now(datetime.timezone.utc)
                - datetime.datetime.fromisoformat(_ar["started_at"])
            ).total_seconds()
            _age_min = int(_age_s // 60)
            _age_str = (
                f"{_age_min} min ago" if _age_min < 120
                else f"{_age_min // 60}h ago"
            )
            _gv = _ar.get("guardrail_verdict", "n/a") or "n/a"
            st.caption(
                f"🤖 Last updated by automated agent: {_age_str} "
                f"(guardrail: **{_gv}**)"
            )
            summary_raw = _ar.get("summary_text") or ""
            if summary_raw:
                level = risk_ctx.risk_level
                render_level = (
                    "error" if level in ("HIGH", "EXTREME")
                    else "warning" if level == "MEDIUM"
                    else "info"
                )
                if UNVERIFIED_MARKER in summary_raw:
                    st.markdown(
                        "<div style='background:#fff3cd;border-left:4px solid #e6a817;"
                        "border-radius:4px;padding:8px 14px;margin-bottom:8px;"
                        "font-size:0.93em;'>"
                        "⚠️ <strong>UNVERIFIED</strong> — Agent summary could not be "
                        "validated against source data. Review with caution."
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    summary_raw = summary_raw.replace(UNVERIFIED_MARKER, "")
                _render_llm_response(summary_raw, level=render_level)
            else:
                st.info("Agent run completed but produced no summary text.")

        else:
            # Live pipeline path
            regen_summary = st.button(
                "🔄 Generate / Regenerate Summary", key="regen_summary"
            )
            if gateway is None:
                st.info("Configure watsonx credentials to enable AI analysis.")
            elif regen_summary:
                with st.spinner("Generating AI summary..."):
                    st.session_state.summary_text = gateway.summarize(risk_ctx)

            if st.session_state.get("summary_text"):
                level = risk_ctx.risk_level
                render_level = (
                    "error" if level in ("HIGH", "EXTREME")
                    else "warning" if level == "MEDIUM"
                    else "info"
                )
                summary_raw = st.session_state.summary_text
                if UNVERIFIED_MARKER in summary_raw:
                    st.markdown(
                        "<div style='background:#fff3cd;border-left:4px solid #e6a817;"
                        "border-radius:4px;padding:8px 14px;margin-bottom:8px;"
                        "font-size:0.93em;'>"
                        "⚠️ <strong>UNVERIFIED</strong> — This summary could not be "
                        "validated against source data after two attempts. Review with caution."
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    summary_raw = summary_raw.replace(UNVERIFIED_MARKER, "")
                _render_llm_response(summary_raw, level=render_level)
            elif gateway is not None and not regen_summary:
                st.caption("Click the button above to generate an AI risk summary.")
    except Exception as _tab_err:
        st.error(f"**Risk Summary tab error:** {_tab_err}")

# ── Tab 3: Forecast ──────────────────────────────────────────────────────────

with tab_forecast:
    try:
        _horizon_label = "24-Hour" if forecast_result.forecast_horizon_hours == 24 else "7-Day"
        st.subheader(f"{_horizon_label} Fire Risk Forecast — {country}")

        _uncertainty_note = (
            " The 7-day outlook uses a week-long weather forecast average and carries "
            "substantially more uncertainty than the 24-hour estimate."
            if forecast_result.forecast_horizon_hours != 24
            else ""
        )
        st.warning(
            "⚠️ **Probabilistic estimate** — This forecast is based on historical fire "
            f"activity and weather conditions. It is not a certainty.{_uncertainty_note} "
            "Always follow official guidance."
        )

        if forecast_result.model_used == "XGBoost":
            model_label = "🤖 ML Model (XGBoost)"
        else:
            model_label = (
                "📐 Deterministic Fallback "
                "(insufficient pseudo-labelled samples — need ≥10 cells with both "
                "fire-active and fire-free history)"
            )
        st.caption(
            f"Model used: {model_label} | "
            f"Generated: {forecast_result.generated_at} UTC | "
            f"Horizon: {forecast_result.forecast_horizon_hours}h"
        )

        st.markdown("#### Top 10 Highest-Risk Cells")
        if forecast_result.cells:
            top10 = forecast_result.cells[:10]
            table_data = [
                {
                    "Lat": round(c.lat_center, 3),
                    "Lon": round(c.lon_center, 3),
                    "Fire Prob (%)": round(c.fire_prob * 100, 1),
                    "Risk Band": c.risk_band,
                    "Fires (7d)": c.historical_fire_count,
                    "Temp 24h (°C)": round(
                        c.feature_snapshot.get("temp_24h_mean", float("nan")), 1
                    ),
                    "Humidity 24h (%)": round(
                        c.feature_snapshot.get("humidity_24h_mean", float("nan")), 1
                    ),
                }
                for c in top10
            ]
            st.dataframe(pd.DataFrame(table_data), use_container_width=True)

            # ── SHAP "Why this risk?" section ────────────────────────────────
            # Re-enable per-cell drill-down by setting SHOW_PER_CELL_SHAP = True
            # once the max_depth / probability-saturation investigation is resolved.
            SHOW_PER_CELL_SHAP = False

            if forecast_result.model_used == "XGBoost":
                st.markdown(
                    "##### 🔍 Why this risk? — Weather Feature Contributions"
                )
                st.caption(
                    "Model trained on weather conditions only — fire history is shown "
                    "separately as context, not as a model input. "
                    "Bar width = share of total prediction push for that cell."
                )

                # ── Overall summary: mean |SHAP| per feature across top-10 ──
                cells_with_shap = [c for c in top10 if c.shap_contribs]
                if cells_with_shap:
                    import collections as _col
                    # Accumulate mean absolute SHAP per feature across all
                    # cells that have SHAP data (always top-10 on XGBoost path).
                    feature_abs_sum = _col.defaultdict(float)
                    feature_labels_map = {}
                    n_cells = len(cells_with_shap)
                    for cell in cells_with_shap:
                        for contrib in cell.shap_contribs:
                            feature_abs_sum[contrib["feature"]] += abs(contrib["shap"])
                            feature_labels_map[contrib["feature"]] = contrib["label"]

                    # Sort by mean absolute SHAP descending
                    mean_abs = sorted(
                        [
                            (feat, feature_abs_sum[feat] / n_cells, feature_labels_map[feat])
                            for feat in feature_abs_sum
                        ],
                        key=lambda x: x[1],
                        reverse=True,
                    )
                    # Normalise to percentages of total absolute push
                    total_mean_abs = sum(v for _, v, _ in mean_abs) or 1.0
                    summary_rows = [
                        (lbl, val / total_mean_abs * 100)
                        for _, val, lbl in mean_abs
                    ]

                    # Build the entire card as one HTML block so the wrapping
                    # <div> is self-contained in a single Streamlit element.
                    bar_rows_html = ""
                    for feat_lbl, pct_val in summary_rows:
                        bar_pct = min(pct_val, 100.0)
                        bar_rows_html += (
                            f'<div style="margin:4px 0;">'
                            f'<span style="font-size:0.88em;width:130px;display:inline-block;'
                            f'color:#1f2328;">{feat_lbl}</span>'
                            f'<span style="font-size:0.88em;color:#3b5998;'
                            f'font-weight:600;">{pct_val:.1f}% of push</span>'
                            f'<div style="background:#e5e7eb;border-radius:3px;height:8px;'
                            f'margin-top:2px;max-width:340px;">'
                            f'<div style="background:#3b5998;border-radius:3px;height:8px;'
                            f'width:{bar_pct:.1f}%;"></div>'
                            f'</div></div>'
                        )
                    st.markdown(
                        "<div style='background:#f7f8fa;border:1px solid #e5e7eb;"
                        "border-radius:6px;padding:10px 14px;margin-bottom:12px;'>"
                        "<p style='margin:0 0 8px;font-weight:700;font-size:0.93em;"
                        "color:#1f2328;'>📊 Overall — What's driving risk across "
                        f"the top {n_cells} cells right now</p>"
                        f"{bar_rows_html}"
                        "</div>",
                        unsafe_allow_html=True,
                    )

                # ── Per-cell expanders (drill-down) ───────────────────────────
                # Gated behind SHOW_PER_CELL_SHAP — disabled until the XGBoost
                # max_depth / probability-saturation issue is resolved, which
                # causes many top cells to share identical SHAP breakdowns.
                # Set SHOW_PER_CELL_SHAP = True above to re-enable.
                if SHOW_PER_CELL_SHAP:
                    for rank, cell in enumerate(top10, start=1):
                        if not cell.shap_contribs:
                            continue
                        label = (
                            f"Cell {rank} · "
                            f"Lat {cell.lat_center:.2f}, Lon {cell.lon_center:.2f} · "
                            f"{cell.risk_band} · {cell.fire_prob*100:.1f}%"
                        )
                        with st.expander(label, expanded=(rank <= 3)):
                            for contrib in cell.shap_contribs:
                                pct  = contrib["pct"]
                                shap = contrib["shap"]
                                feat_label = contrib["label"]
                                direction = "▲" if shap >= 0 else "▼"
                                bar_color = "#c0392b" if shap >= 0 else "#2980b9"
                                bar_pct = min(abs(pct), 100.0)
                                sign_str = f"{pct:+.1f}%"
                                st.markdown(
                                    f"""
<div style="margin:3px 0;">
  <span style="font-size:0.88em;width:130px;display:inline-block;
               color:#1f2328;">{feat_label}</span>
  <span style="font-size:0.88em;color:{bar_color};
               font-weight:600;">{direction} {sign_str}</span>
  <div style="background:#e5e7eb;border-radius:3px;height:7px;
              margin-top:2px;max-width:340px;">
    <div style="background:{bar_color};border-radius:3px;height:7px;
                width:{bar_pct:.1f}%;"></div>
  </div>
</div>""",
                                    unsafe_allow_html=True,
                                )
            else:
                st.caption(
                    "ℹ️ SHAP breakdown unavailable — model fell back to the "
                    "deterministic scorer (insufficient training samples)."
                )
        else:
            st.info("No forecast data available for the selected filters.")

        st.divider()

        st.markdown("#### 🤖 AI Forecast Interpretation")

        # ── Agent-run toggle (forecast) ───────────────────────────────────
        agent_run_forecast = agent_store.get_latest_run(country)
        use_agent_forecast = st.toggle(
            "Use latest agent run",
            value=bool(agent_run_forecast),
            key="use_agent_forecast",
            disabled=not bool(agent_run_forecast),
            help=(
                "When ON, displays the most recent successful automated-agent "
                "forecast interpretation for this country."
            ),
        )

        if use_agent_forecast and agent_run_forecast:
            _ar = agent_run_forecast
            _age_s = (
                datetime.datetime.now(datetime.timezone.utc)
                - datetime.datetime.fromisoformat(_ar["started_at"])
            ).total_seconds()
            _age_min = int(_age_s // 60)
            _age_str = (
                f"{_age_min} min ago" if _age_min < 120
                else f"{_age_min // 60}h ago"
            )
            _gv = _ar.get("guardrail_verdict", "n/a") or "n/a"
            st.caption(
                f"🤖 Last updated by automated agent: {_age_str} "
                f"(guardrail: **{_gv}**)"
            )
            interp_raw = _ar.get("forecast_text") or ""
            if interp_raw:
                if UNVERIFIED_MARKER in interp_raw:
                    st.markdown(
                        "<div style='background:#fff3cd;border-left:4px solid #e6a817;"
                        "border-radius:4px;padding:8px 14px;margin-bottom:8px;"
                        "font-size:0.93em;'>"
                        "⚠️ <strong>UNVERIFIED</strong> — Agent forecast could not be "
                        "validated against source data. Review with caution."
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    interp_raw = interp_raw.replace(UNVERIFIED_MARKER, "")
                _render_llm_response(interp_raw, level="warning")
            else:
                st.info("Agent run completed but produced no forecast interpretation.")

        else:
            # Live pipeline path
            regen_forecast = st.button(
                "🔄 Generate / Regenerate Forecast Analysis", key="regen_forecast"
            )
            if gateway is None:
                st.info("Configure watsonx credentials to enable AI forecast analysis.")
            elif regen_forecast:
                with st.spinner("Generating AI forecast interpretation..."):
                    st.session_state.forecast_interp_text = gateway.interpret_forecast(
                        forecast_result
                    )

            if st.session_state.get("forecast_interp_text"):
                interp_raw = st.session_state.forecast_interp_text
                if UNVERIFIED_MARKER in interp_raw:
                    st.markdown(
                        "<div style='background:#fff3cd;border-left:4px solid #e6a817;"
                        "border-radius:4px;padding:8px 14px;margin-bottom:8px;"
                        "font-size:0.93em;'>"
                        "⚠️ <strong>UNVERIFIED</strong> — This forecast interpretation could not "
                        "be validated against source data after two attempts. Review with caution."
                        "</div>",
                        unsafe_allow_html=True,
                    )
                    interp_raw = interp_raw.replace(UNVERIFIED_MARKER, "")
                _render_llm_response(interp_raw, level="warning")
            elif gateway is not None and not regen_forecast:
                st.caption("Click the button above to generate an AI forecast interpretation.")
    except Exception as _tab_err:
        st.error(f"**Forecast tab error:** {_tab_err}")

# ── Tab 4: Chat ──────────────────────────────────────────────────────────────

with tab_chat:
    try:
        st.subheader("💬 Ask the Wildfire Analyst")

        if "messages" not in st.session_state:
            st.session_state.messages = []

        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])

        if gateway is None:
            st.info("Configure watsonx credentials to enable the chat assistant.")
        else:
            if prompt := st.chat_input("Ask about fire risk, forecast, or recommendations..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)
                with st.chat_message("assistant"):
                    with st.spinner("Thinking..."):
                        reply = gateway.chat(
                            st.session_state.messages[:-1], risk_ctx, forecast_result
                        )
                    st.markdown(reply)
                st.session_state.messages.append({"role": "assistant", "content": reply})
    except Exception as _tab_err:
        st.error(f"**Chat tab error:** {_tab_err}")

# ── Tab 5: Agent Status ───────────────────────────────────────────────────────

with tab_agent:
    try:
        st.subheader("🤖 Autonomous Agent Status")

        # ── Health indicator ─────────────────────────────────────────────
        _all_runs = agent_store.get_recent_runs(20)
        _loop_hours = config.AGENT_LOOP_HOURS

        if _all_runs:
            _last = _all_runs[0]
            _last_age_s = (
                datetime.datetime.now(datetime.timezone.utc)
                - datetime.datetime.fromisoformat(_last["started_at"])
            ).total_seconds()
            _last_age_h = _last_age_s / 3600

            if _last["status"] == "failed":
                _health_dot, _health_label, _health_color = "●", "Failed", "#e74c3c"
            elif _last_age_h > _loop_hours * 2:
                _health_dot, _health_label, _health_color = "●", "Stale", "#e6a817"
            else:
                _health_dot, _health_label, _health_color = "●", "Running", "#2ecc71"

            st.markdown(
                f"<p style='font-size:1.1em;margin-bottom:4px;'>"
                f"<span style='color:{_health_color};font-size:1.3em;'>{_health_dot}</span>"
                f"&nbsp;<strong>Agent: {_health_label}</strong>&nbsp;"
                f"<span style='color:#57606a;font-size:0.9em;'>"
                f"— last run {int(_last_age_h * 60)} min ago "
                f"({_last['country']}, status: {_last['status']})"
                f"</span></p>",
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                "<p style='font-size:1.1em;'>"
                "<span style='color:#57606a;font-size:1.3em;'>●</span>"
                "&nbsp;<strong>Agent: No runs yet</strong>"
                "</p>",
                unsafe_allow_html=True,
            )

        # ── Manual trigger button ────────────────────────────────────────
        st.markdown("---")
        _agent_col1, _agent_col2 = st.columns([2, 5])
        with _agent_col1:
            _run_now = st.button(
                "▶ Run agent now",
                key="run_agent_now",
                help=(
                    f"Triggers one synchronous agent cycle for the currently "
                    f"selected country ({country}) with min FRP {min_frp:.0f} MW."
                ),
            )
        with _agent_col2:
            if _run_now:
                with st.spinner(
                    f"Running agent for {country}… this takes 30–90 s."
                ):
                    try:
                        _gw_agent = get_gateway() if gateway is None else gateway
                    except RuntimeError:
                        _gw_agent = None
                    _result = agent_runner.run_once(
                        country=country,
                        min_frp=min_frp,
                        horizon_days=1,
                        force_refresh=False,
                        gateway=_gw_agent,
                    )
                if _result["status"] == "success":
                    st.success(
                        f"✓ Agent run complete in {_result['latency_seconds']:.1f}s "
                        f"— guardrail: **{_result['guardrail_verdict']}**"
                    )
                elif _result["status"] == "partial":
                    st.warning(
                        f"⚠ Partial run ({_result['latency_seconds']:.1f}s) — "
                        f"metrics saved, AI text unavailable. "
                        f"{_result.get('error_message', '')}"
                    )
                else:
                    st.error(
                        f"✗ Agent run failed: {_result.get('error_message', 'unknown error')}"
                    )
                # Force table refresh
                _all_runs = agent_store.get_recent_runs(20)

        # ── Run history table ────────────────────────────────────────────
        st.markdown("#### Last 20 agent runs")
        if _all_runs:
            # Summary table (metadata only — no blobs)
            _table_rows = []
            for _r in _all_runs:
                _ts = _r["started_at"][:19].replace("T", " ")
                _lat = (
                    f"{_r['latency_seconds']:.1f}s"
                    if _r.get("latency_seconds") is not None
                    else "—"
                )
                _err = (_r.get("error_message") or "")[:60]
                _table_rows.append(
                    {
                        "Started (UTC)": _ts,
                        "Country": _r["country"],
                        "Status": _r["status"],
                        "Latency": _lat,
                        "Guardrail": _r.get("guardrail_verdict") or "n/a",
                        "Error": _err,
                    }
                )

            _df_runs = pd.DataFrame(_table_rows)

            # Colour-code status column with background highlighting via
            # Streamlit's built-in dataframe styling.
            def _status_style(val: str) -> str:
                return {
                    "success": "background-color:#d4edda;color:#155724",
                    "partial":  "background-color:#fff3cd;color:#856404",
                    "failed":   "background-color:#f8d7da;color:#721c24",
                    "running":  "background-color:#cce5ff;color:#004085",
                }.get(val, "")

            _styled = _df_runs.style.applymap(_status_style, subset=["Status"])
            st.dataframe(_styled, use_container_width=True, hide_index=True)

            # ── Downloads column — one expander per run ──────────────────
            st.markdown("#### Downloads")
            st.caption(
                "Artifacts are read from disk at render time — "
                "only runs that have an artifacts_dir stored will show buttons."
            )
            for _r in _all_runs:
                _adir = _r.get("artifacts_dir")
                _run_ts = _r["started_at"][:19].replace("T", " ")
                _run_label = (
                    f"{_run_ts}  ·  {_r['country']}  ·  {_r['status']}"
                )
                if not _adir:
                    # Older run or failed run without artifacts — skip silently.
                    continue
                with st.expander(_run_label, expanded=False):
                    _dl_c1, _dl_c2, _dl_c3, _dl_c4, _dl_c5 = st.columns(5)
                    _rid = _r["run_id"]

                    # Col 1 — 48h risk-metrics dataset (decompressed for download)
                    _csv_bytes = _artifacts.read_artifact_csv(_adir, "dataset.csv.gz")
                    with _dl_c1:
                        if _csv_bytes:
                            st.download_button(
                                label="📊 Dataset (48h)",
                                data=_csv_bytes,
                                file_name=f"dataset_{_rid[:8]}.csv",
                                mime="text/csv",
                                key=f"dl_csv_{_rid}",
                                help="48h FIRMS detections — risk-metrics window (plain CSV).",
                            )
                        else:
                            st.caption("dataset —")

                    # Col 2 — 7-day XGBoost training window (decompressed; XGBoost runs only)
                    _fw_bytes = _artifacts.read_artifact_csv(
                        _adir, "dataset_forecast_window.csv.gz"
                    )
                    with _dl_c2:
                        if _fw_bytes:
                            st.download_button(
                                label="🗄️ Training data (7d)",
                                data=_fw_bytes,
                                file_name=f"dataset_forecast_window_{_rid[:8]}.csv",
                                mime="text/csv",
                                key=f"dl_fw_{_rid}",
                                help=(
                                    "7-day FIRMS window — actual XGBoost training input "
                                    "(plain CSV). Pair with model.json to reproduce this booster."
                                ),
                            )
                        else:
                            st.caption("training data —\n_(deterministic run)_")

                    # Col 3 — model (booster or sentinel)
                    _model_bytes = _artifacts.read_artifact(_adir, "model.json")
                    with _dl_c3:
                        if _model_bytes:
                            st.download_button(
                                label="🤖 Model",
                                data=_model_bytes,
                                file_name=f"model_{_rid[:8]}.json",
                                mime="application/json",
                                key=f"dl_model_{_rid}",
                                help="XGBoost booster (JSON), or sentinel note for deterministic runs.",
                            )
                        else:
                            st.caption("model —")

                    # Col 4 — model script
                    _script_bytes = _artifacts.read_artifact(_adir, "model_script.py")
                    with _dl_c4:
                        if _script_bytes:
                            st.download_button(
                                label="📄 Script",
                                data=_script_bytes,
                                file_name="wildfire_model_export.py",
                                mime="text/x-python",
                                key=f"dl_script_{_rid}",
                                help="wildfire_model_export.py — standalone reproducible training script.",
                            )
                        else:
                            st.caption("script —")

                    # Col 5 — markdown report
                    _report_bytes = _artifacts.read_artifact(_adir, "report.md")
                    with _dl_c5:
                        if _report_bytes:
                            st.download_button(
                                label="📝 Report",
                                data=_report_bytes,
                                file_name=f"report_{_rid[:8]}.md",
                                mime="text/markdown",
                                key=f"dl_report_{_rid}",
                                help="Markdown report: metadata + artifact inventory + AI text.",
                            )
                        else:
                            st.caption("report —")

                    # Folder size summary
                    _sz = _artifacts.artifact_dir_size(_adir)
                    st.caption(
                        f"Artifacts folder: `{_adir}` — total {_sz / 1024:.1f} KB"
                    )
        else:
            st.info(
                "No agent runs yet. Click **▶ Run agent now** above or start the "
                "agent from the terminal:\n\n"
                "```bash\npython agent_runner.py --country Angola\n```"
            )

        # ── Configuration summary ─────────────────────────────────────────
        with st.expander("⚙ Agent configuration"):
            st.json(
                {
                    "default_country": config.AGENT_DEFAULT_COUNTRY,
                    "loop_interval_hours": config.AGENT_LOOP_HOURS,
                    "min_frp_mw": config.AGENT_MIN_FRP,
                    "db_path": config.DB_PATH,
                    "generator_model": config.GRANITE_MODEL_ID,
                    "critic_model": config.GUARDIAN_MODEL_ID,
                }
            )

    except Exception as _tab_err:
        st.error(f"**Agent Status tab error:** {_tab_err}")

# ── Tab 6: Land Cover ────────────────────────────────────────────────────────

with tab_landcover:
    try:
        st.subheader(f"🌿 Land Cover & Vegetation — {country}")

        # ── Info banner ──────────────────────────────────────────────────────
        st.caption(
            "Fetches a Sentinel-2 HLS tile (30 m) via NASA Earthdata for the "
            "selected country, computes NDVI, and classifies the scene with a "
            "MobileNetV2 model trained on EuroSAT (93.4% test accuracy). "
            "**Each fetch downloads ~110 MB and takes 30–60 s** — click once and wait."
        )

        # ── Date range selector ──────────────────────────────────────────────
        lc_col1, lc_col2 = st.columns(2)
        with lc_col1:
            lc_start = st.date_input(
                "Start date",
                value=datetime.date.today() - datetime.timedelta(days=90),
                key="lc_start",
            )
        with lc_col2:
            lc_end = st.date_input(
                "End date",
                value=datetime.date.today(),
                key="lc_end",
            )

        lc_fetch_btn = st.button(
            "🛰️ Fetch Sentinel-2 tile & analyse",
            key="lc_fetch",
            help=(
                "Downloads the least-cloudy HLS S30 granule for the selected "
                "country and date range, then computes NDVI and runs the land "
                "cover classifier."
            ),
        )

        # ── Fetch + analyse on button press ─────────────────────────────────
        if lc_fetch_btn:
            bbox = config.COUNTRY_BBOX[country]
            date_range = (lc_start.isoformat(), lc_end.isoformat())

            with st.spinner(
                f"Fetching Sentinel-2 tile for {country} "
                f"({date_range[0]} → {date_range[1]})… "
                "this takes 30–60 s."
            ):
                try:
                    lc_result = _sentinel_fetch.fetch_sentinel2_tile(
                        bbox, date_range
                    )
                    st.session_state["lc_result"] = lc_result
                    st.session_state["lc_country"] = country
                except Exception as _lc_fetch_err:
                    st.error(f"Sentinel-2 fetch failed: {_lc_fetch_err}")
                    st.session_state.pop("lc_result", None)

        # ── Display cached result (persists across reruns) ───────────────────
        lc_data = st.session_state.get("lc_result")
        if lc_data is not None:
            import numpy as np
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            import io

            lc_cached_country = st.session_state.get("lc_country", country)

            # ── Metadata row ─────────────────────────────────────────────────
            st.divider()
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Country (cached)", lc_cached_country)
            m2.metric("Acquisition date", lc_data["acquisition_date"])
            m3.metric("Cloud cover", f"{lc_data['cloud_cover']:.1f}%")
            m4.metric("Fetch time", f"{lc_data['elapsed_seconds']:.0f} s")
            st.caption(f"Source: {lc_data['source']}  ·  Granule: `{lc_data['granule_id']}`")

            # ── NDVI computation ─────────────────────────────────────────────
            B4 = lc_data.get("B4")
            B8 = lc_data.get("B8")
            B3 = lc_data.get("B3")
            B2 = lc_data.get("B2")

            bands_ok = all(v is not None for v in (B4, B8, B3, B2))

            if bands_ok:
                ndvi = (B8.astype(float) - B4.astype(float)) / (
                    B8.astype(float) + B4.astype(float) + 1e-9
                )
                ndvi = np.clip(ndvi, -1.0, 1.0)
                ndvi_mean = float(np.nanmean(ndvi))
                ndvi_veg_pct = float(100 * np.nanmean(ndvi > 0.3))

                # ── NDVI + RGB plots ──────────────────────────────────────────
                step = 4  # downsample 3660→915 for display
                ndvi_ds = ndvi[::step, ::step]
                r_ch = np.clip(B4 * 3.5, 0, 1)[::step, ::step]
                g_ch = np.clip(B3 * 3.5, 0, 1)[::step, ::step]
                b_ch = np.clip(B2 * 3.5, 0, 1)[::step, ::step]
                rgb = np.stack([r_ch, g_ch, b_ch], axis=-1)
                # Replace NaN with 0 for display
                rgb = np.nan_to_num(rgb, nan=0.0)

                fig, axes = plt.subplots(1, 2, figsize=(13, 5))

                im = axes[0].imshow(ndvi_ds, cmap="RdYlGn", vmin=-0.2, vmax=0.8)
                plt.colorbar(im, ax=axes[0], label="NDVI", fraction=0.046, pad=0.04)
                axes[0].set_title(
                    f"NDVI  —  {lc_cached_country}\n"
                    f"mean={ndvi_mean:.3f}   vegetation (>0.3): {ndvi_veg_pct:.1f}%",
                    fontsize=10,
                )
                axes[0].axis("off")

                axes[1].imshow(rgb)
                axes[1].set_title(
                    f"True-colour RGB (B4/B3/B2 ×3.5)\n"
                    f"{lc_data['acquisition_date']}  cloud={lc_data['cloud_cover']:.1f}%",
                    fontsize=10,
                )
                axes[1].axis("off")

                fig.tight_layout()
                buf = io.BytesIO()
                fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
                plt.close(fig)
                buf.seek(0)
                st.image(buf, use_container_width=True)

                # ── NDVI stats ────────────────────────────────────────────────
                _valid_pct = float(100 * (~np.isnan(B8) & ~np.isnan(B4)).mean())
                n1, n2, n3, n4 = st.columns(4)
                n1.metric("Mean NDVI (valid px)", f"{ndvi_mean:.3f}")
                n2.metric("Dense vegetation (NDVI>0.3)", f"{ndvi_veg_pct:.1f}%",
                          help="% of valid pixels with NDVI>0.3 — excludes nodata")
                n3.metric(
                    "Bare / water (NDVI<0.1)",
                    f"{100*float(np.nanmean(ndvi < 0.1)):.1f}%",
                    help="% of valid pixels — excludes nodata",
                )
                n4.metric("Scene valid data", f"{_valid_pct:.0f}%",
                          help="Fraction of the full granule covered by real data (vs nodata/edge padding)")

                # ── Land cover classification ─────────────────────────────────
                st.divider()
                st.markdown("#### 🏷️ Land Cover Classification")

                lc_model = _get_landcover_model()
                if lc_model is None:
                    st.warning(
                        "Land cover model not loaded — "
                        "`models/global6_classifier.pt` missing."
                    )
                else:
                    # ── Geographic confidence badge ────────────────────────────
                    # Coverage tiers (4 levels):
                    #   ✅ Validated      — EuroSAT in-domain (Greece, Portugal)
                    #   🔵 Improved       — Phase B fine-tune on real patches (Angola)
                    #   🟡 Lightly checked — Task 4: 5 real tiles reviewed per country,
                    #                        NDVI consistency check only, no fine-tuning
                    #   ⚠️ Experimental   — completely untested
                    _EUROSAT_VALIDATED  = {"Greece", "Portugal"}
                    _ANGOLA_IMPROVED    = {"Angola"}
                    # Task 4 lightly-checked countries with per-country bucket.
                    # Each tuple: (bucket, tile_count, one-line qualitative note)
                    _LIGHTLY_CHECKED = {
                        "Brazil":    ("Plausible",    5,
                                      "4/5 tiles NDVI-consistent; 1 tile Built_up on NDVI=0.84 "
                                      "(Amazon domain-gap; same failure mode as original Angola bug)."),
                        "India":     ("Mixed",        5,
                                      "1 plausible, 3 borderline, 1 implausible (Forest_Veg at NDVI=−0.07). "
                                      "Seasonal bare soil + monsoon moisture confuses model."),
                        "Australia": ("Mixed",        5,
                                      "3 plausible (vegetation/cropland consistent); 1 implausible "
                                      "(Forest_Veg on NDVI=−0.99 likely cloud/shadow artifact)."),
                        "Mexico":    ("Unreliable",   4,
                                      "3/4 tiles Forest_Veg predicted on NDVI < 0.14 (semi-arid scenes). "
                                      "High risk of domain mismatch for arid/dry-season Mexican land cover."),
                    }

                    _geo_validated  = lc_cached_country in _EUROSAT_VALIDATED
                    _geo_angola     = lc_cached_country in _ANGOLA_IMPROVED
                    _geo_lc_info    = _LIGHTLY_CHECKED.get(lc_cached_country)

                    if _geo_validated:
                        _geo_badge_color = "#2ecc71"
                        _geo_badge_icon  = "✅"
                        _geo_badge_text  = (
                            "Validated — EuroSAT training domain covers this region. "
                            "Global-6 model: all 4 active classes ≥ 94.6% accuracy."
                        )
                    elif _geo_angola:
                        _geo_badge_color = "#3b82d4"
                        _geo_badge_icon  = "🔵"
                        _geo_badge_text  = (
                            "Improved experimental (Phase B) — Global-6 model retrained "
                            "with Angola Sentinel-2 data. "
                            "Forest_Vegetation and Built_up classification substantially "
                            "improved (70% and 63%); Bare_Sparse and Wetland reliable (100%/97%). "
                            "Cropland (45%) and Water (40%) are weaker than the original model "
                            "due to spectral overlap — use the NDVI metric above as a "
                            "complementary vegetation signal."
                        )
                    elif _geo_lc_info is not None:
                        _lc_bucket, _lc_tiles, _lc_note = _geo_lc_info
                        _geo_badge_color = "#c9a000"
                        _geo_badge_icon  = "🟡"
                        _geo_badge_text  = (
                            f"Lightly checked — {_lc_tiles} sample tiles reviewed, "
                            f"NDVI-consistency check only, no fine-tuning (Task 4). "
                            f"Overall: {_lc_bucket}. {_lc_note} "
                            "This is NOT equivalent to Phase B validation. "
                            "Use NDVI above as the primary signal."
                        )
                    else:
                        _geo_badge_color = "#e6a817"
                        _geo_badge_icon  = "⚠️"
                        _geo_badge_text  = (
                            "Experimental — no training or validation data for this region. "
                            "Results can be confidently wrong even at high confidence scores. "
                            "Use the NDVI metric above as a more reliable vegetation signal."
                        )
                    st.markdown(
                        f"<div style='display:inline-block;padding:5px 12px;"
                        f"background:{_geo_badge_color}22;border:1px solid {_geo_badge_color};"
                        f"border-radius:5px;font-size:0.9em;margin-bottom:8px;'>"
                        f"{_geo_badge_icon} <strong>Geographic coverage:</strong> "
                        f"{_geo_badge_text}"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                    # ── Find best valid-data crop window ───────────────────────
                    # HLS tiles can have large nodata margins (~60% for Angola).
                    # The geometric centre often falls in the nodata region, which
                    # collapses all reflectance to zero and forces a SeaLake
                    # classification.  Strategy: scan a 5×5 grid of candidate
                    # 1024×1024 windows and pick the one with the least nodata.
                    CROP_HALF = 512   # 1024×1024 px crop at 30m ≈ 31 km × 31 km
                    NODATA_REJECT = 0.20  # reject windows with >20% nodata
                    # Minimum B4 (red) reflectance to count a pixel as "land"
                    # for crop selection purposes.  HLS water pixels have B4≈0.0
                    # (real zero — not NaN), so they pass np.isnan() and would
                    # cause the crop selector to pick water-dominant windows.
                    # Using a positive threshold excludes open water and dark
                    # shadows while accepting all real land reflectances (soil,
                    # vegetation, urban all have B4 > 0.002 in HLS).
                    _LAND_MIN_REFL = 0.001

                    # ── Cloud-filter via HLS Fmask QA band ─────────────────────
                    # Fmask bits (HLSS30 v2.0, HLS User Guide Table 7):
                    #   bit 1 = Cloud, bit 2 = Adjacent-to-cloud, bit 3 = Cloud shadow
                    # A pixel is "valid land" only if:
                    #   (a) B4 > _LAND_MIN_REFL  (positive reflectance — excludes nodata/water)
                    #   (b) NOT cloud-contaminated per Fmask bits 1/2/3
                    # When Fmask is unavailable (old fetch, network failure, etc.) we
                    # fall back to the B4-only criterion so nothing breaks.
                    import sentinel_fetch as _sf
                    _fmask_arr = lc_data.get("Fmask")
                    if _fmask_arr is not None:
                        _cloud_mask = _sf.is_cloud_contaminated(_fmask_arr)  # True = contaminated
                    else:
                        _cloud_mask = None  # no Fmask available — use B4 only

                    def _is_valid_pixel_win(b4_win, y0, x0, y1, x1):
                        """True where pixel passes both reflectance AND cloud filter."""
                        refl_ok = b4_win > _LAND_MIN_REFL
                        if _cloud_mask is not None:
                            cloud_ok = ~_cloud_mask[y0:y1, x0:x1]
                            return refl_ok & cloud_ok
                        return refl_ok

                    h, w = B4.shape
                    best_y0, best_x0 = None, None
                    best_valid_frac = 0.0
                    for _gy in range(1, 6):        # 5 rows
                        for _gx in range(1, 6):    # 5 columns
                            cy_c = int(h * _gy / 6)
                            cx_c = int(w * _gx / 6)
                            _y0 = max(0, cy_c - CROP_HALF)
                            _y1 = min(h, cy_c + CROP_HALF)
                            _x0 = max(0, cx_c - CROP_HALF)
                            _x1 = min(w, cx_c + CROP_HALF)
                            if (_y1 - _y0) < 64 or (_x1 - _x0) < 64:
                                continue
                            # Count pixels that are valid land AND cloud-free.
                            _win = B4[_y0:_y1, _x0:_x1]
                            _valid_win = _is_valid_pixel_win(_win, _y0, _x0, _y1, _x1)
                            valid_frac = float(np.nanmean(_valid_win))
                            if valid_frac > best_valid_frac:
                                best_valid_frac = valid_frac
                                best_y0, best_x0 = _y0, _x0

                    if best_y0 is None or best_valid_frac < (1.0 - NODATA_REJECT):
                        # Fallback: use the row/col with most valid cloud-free land pixels
                        if _cloud_mask is not None:
                            _land_clear = (B4 > _LAND_MIN_REFL) & (~_cloud_mask)
                        else:
                            _land_clear = B4 > _LAND_MIN_REFL
                        row_valid = _land_clear.sum(axis=1)
                        col_valid = _land_clear.sum(axis=0)
                        if row_valid.max() == 0:
                            # No cloud-free land pixels anywhere in the scene —
                            # report honestly rather than falling back to a
                            # cloud-contaminated crop silently.
                            st.warning(
                                "⚠️ No cloud-free land pixels found in this granule. "
                                f"Cloud filter removed all valid pixels "
                                f"({'Fmask active' if _cloud_mask is not None else 'B4 threshold only'}). "
                                "Try a different date range or a clear-sky scene."
                            )
                            st.stop()
                        best_row = int(row_valid.argmax())
                        best_col = int(col_valid.argmax())
                        best_y0 = max(0, best_row - CROP_HALF)
                        best_x0 = max(0, best_col - CROP_HALF)
                        _fb_win_b4 = B4[best_y0:min(h, best_y0+2*CROP_HALF),
                                        best_x0:min(w, best_x0+2*CROP_HALF)]
                        _fb_valid = _is_valid_pixel_win(
                            _fb_win_b4,
                            best_y0, best_x0,
                            min(h, best_y0+2*CROP_HALF), min(w, best_x0+2*CROP_HALF)
                        )
                        best_valid_frac = float(np.nanmean(_fb_valid))

                    y0_c = best_y0
                    y1_c = min(h, y0_c + 2 * CROP_HALF)
                    x0_c = best_x0
                    x1_c = min(w, x0_c + 2 * CROP_HALF)

                    tile_r = np.nan_to_num(B4[y0_c:y1_c, x0_c:x1_c], nan=0.0)
                    tile_g = np.nan_to_num(B3[y0_c:y1_c, x0_c:x1_c], nan=0.0)
                    tile_b = np.nan_to_num(B2[y0_c:y1_c, x0_c:x1_c], nan=0.0)
                    tile_rgb = np.stack([tile_r, tile_g, tile_b], axis=-1)

                    # Clip sub-zero reflectance to 0 (noise artefacts below
                    # zero are non-physical and drive spurious SeaLake scores)
                    tile_r = np.clip(tile_r, 0, None)
                    tile_g = np.clip(tile_g, 0, None)
                    tile_b = np.clip(tile_b, 0, None)
                    tile_rgb = np.stack([tile_r, tile_g, tile_b], axis=-1)

                    # Crop NDVI (used for consistency check shown to user)
                    crop_b8 = np.clip(
                        np.nan_to_num(B8[y0_c:y1_c, x0_c:x1_c], nan=0.0), 0, None
                    )
                    crop_ndvi_arr = np.clip(
                        (crop_b8.astype(float) - tile_r.astype(float))
                        / (crop_b8.astype(float) + tile_r.astype(float) + 1e-9),
                        -1, 1,
                    )
                    _crop_ndvi_mean = float(np.mean(crop_ndvi_arr))
                    _crop_veg_pct   = float(100 * np.mean(crop_ndvi_arr > 0.3))

                    # ── App-side debug log (matches classify_tile log) ─────────
                    import time as _lc_time
                    _lc_ts = _lc_time.strftime("%H:%M:%S")
                    try:
                        with open("/tmp/classify_tile_debug.log", "a") as _lf:
                            _lf.write(f"\n[{_lc_ts}] APP SIDE: about to call classify_tile\n")
                            _lf.write(f"  tile_rgb shape={tile_rgb.shape} dtype={tile_rgb.dtype}\n")
                            _lf.write(f"  tile_rgb id={id(tile_rgb)}\n")
                            for _ci, _cn in enumerate(["ch0(B4/R)", "ch1(B3/G)", "ch2(B2/B)"]):
                                _ch = tile_rgb[:, :, _ci]
                                _lf.write(
                                    f"  {_cn}: min={float(_ch.min()):.6f}  max={float(_ch.max()):.6f}"
                                    f"  mean={float(_ch.mean()):.6f}  nonzero={100*float((_ch!=0).mean()):.1f}%\n"
                                )
                            _lf.write(f"  crop_ndvi_mean={_crop_ndvi_mean:.3f}  veg>0.3={_crop_veg_pct:.1f}%\n")
                            _lf.write(f"  crop window: y=[{y0_c},{y1_c}] x=[{x0_c},{x1_c}]\n")
                    except Exception:
                        pass

                    # Show crop diagnostics
                    st.caption(
                        f"Classifier input: best valid-data crop  "
                        f"y=[{y0_c},{y1_c}] x=[{x0_c},{x1_c}]  "
                        f"valid pixels: {best_valid_frac*100:.0f}%  "
                        f"crop NDVI mean={_crop_ndvi_mean:.3f}  "
                        f"veg>0.3: {_crop_veg_pct:.1f}%"
                    )

                    lc_pred = _lc.classify_tile(lc_model, tile_rgb)

                    # ── Model confidence badge ─────────────────────────────────
                    conf     = lc_pred["confidence"]
                    _T_used  = lc_pred.get("temperature", 1.0)
                    badge_color = (
                        "#2ecc71" if conf >= 0.75
                        else "#e6a817" if conf >= 0.50
                        else "#e74c3c"
                    )
                    badge_label = (
                        "High" if conf >= 0.75
                        else "Medium" if conf >= 0.50
                        else "Low"
                    )
                    st.markdown(
                        f"<div style='display:inline-block;padding:6px 14px;"
                        f"background:{badge_color};border-radius:6px;"
                        f"color:white;font-weight:bold;font-size:1.1em;'>"
                        f"🏷️ {lc_pred['class']}  "
                        f"<span style='font-size:0.85em;font-weight:normal;'>"
                        f"({conf*100:.1f}% — {badge_label} confidence)"
                        f"</span></div>",
                        unsafe_allow_html=True,
                    )
                    # Calibration honesty note.
                    # Temperature scaling (T={_T_used:.4f}) is applied — but T≈1.03
                    # is near-unity, so its practical effect on any individual prediction
                    # is 0.1–1 pp.  High-confidence wrong predictions (e.g. 92% Built_up
                    # on a dense-vegetation scene) are caused by a training-data domain
                    # gap, not by miscalibration, and temperature scaling does not fix them.
                    # The NDVI metric above is the reliable cross-check for this.
                    st.caption(
                        f"Temperature scaling applied (T={_T_used:.4f}) — "
                        "confidence is calibrated but had minimal effect on known "
                        "high-confidence-wrong cases (0.1–1 pp reduction). "
                        "High confidence does not mean correct prediction for "
                        "out-of-domain regions — use NDVI above as the primary check."
                    )
                    st.write("")

                    # ── All-class probability bar chart ───────────────────────
                    all_probs = lc_pred["all_probs"]
                    prob_df = pd.DataFrame(
                        sorted(all_probs.items(), key=lambda x: x[1], reverse=True),
                        columns=["Class", "Probability"],
                    )
                    prob_df["Probability %"] = (prob_df["Probability"] * 100).round(1)

                    fig2, ax2 = plt.subplots(figsize=(7, 4))
                    colors = [
                        badge_color if c == lc_pred["class"] else "#adb5bd"
                        for c in prob_df["Class"]
                    ]
                    ax2.barh(
                        prob_df["Class"], prob_df["Probability %"],
                        color=colors, edgecolor="none",
                    )
                    ax2.set_xlabel("Probability (%)")
                    ax2.set_title(
                        f"All-class probabilities — best valid-data crop ({lc_cached_country})",
                        fontsize=10,
                    )
                    ax2.invert_yaxis()
                    fig2.tight_layout()
                    buf2 = io.BytesIO()
                    fig2.savefig(buf2, format="png", dpi=100, bbox_inches="tight")
                    plt.close(fig2)
                    buf2.seek(0)
                    st.image(buf2, use_container_width=False, width=520)

            else:
                missing = [k for k, v in [("B4", B4), ("B8", B8), ("B3", B3), ("B2", B2)] if v is None]
                st.warning(
                    f"Bands {missing} were not loaded from this granule — "
                    "NDVI and classification unavailable. "
                    "Try a different date range."
                )

        elif not lc_fetch_btn:
            st.info(
                f"Select a date range and click **🛰️ Fetch Sentinel-2 tile & analyse** "
                f"to load a Sentinel-2 scene for **{country}**."
            )

    except Exception as _lc_tab_err:
        st.error(f"**Land Cover tab error:** {_lc_tab_err}")
        import traceback
        st.code(traceback.format_exc())
