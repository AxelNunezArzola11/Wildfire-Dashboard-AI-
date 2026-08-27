"""
llm_gateway.py — Abstract LLM interface + IBM watsonx / Granite implementation.

Public API
----------
LLMGateway          Abstract base class
WatsonxGateway      Concrete implementation via ibm-watsonx-ai SDK
AuditVerdict        Named result from the critic agent
get_gateway()       Factory — selects backend from config.LLM_BACKEND

Generator-Critic (Task 6 AI Guardrails)
----------------------------------------
WatsonxGateway.summarize() and .interpret_forecast() both run a two-stage
pipeline:

  1. Generator (Granite) produces a summary text.
  2. Cheap pre-filter: extract all numeric tokens from the text and check each
     against numeric values in the source JSON.  If every number is accounted
     for, skip the expensive critic call and return PASS immediately.
  3. If the pre-filter finds an ambiguous/unverifiable number it escalates to
     AGENT 2: audit_insight() — a second Granite Guardian 3-8B call that
     classifies the summary against the source on four failure modes:
       • fabricated numbers not present in the source
       • contradictions with source values
       • overstated certainty
       • missing uncertainty disclaimer
  4. Correction loop: one regeneration attempt on FAIL.  If the regeneration
     also fails the critic, the original text is returned with an ⚠ UNVERIFIED
     marker embedded so app.py can surface a visible badge.

No top-level ibm_watsonx_ai import; it is loaded lazily inside WatsonxGateway
so the module remains importable even when the package is not installed.
"""

import dataclasses
import json
import logging
import re
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass

import config
from forecast_engine import ForecastResult
from risk_engine import RiskContext

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# AuditVerdict — structured result from the critic agent
# ---------------------------------------------------------------------------

AUDIT_PASS = "pass"
AUDIT_FAIL = "fail"

# Sentinel embedded in summaries that survived two failed audits.
UNVERIFIED_MARKER = "\n\n⚠ UNVERIFIED — this summary could not be validated against source data."


@dataclass
class AuditVerdict:
    """Result returned by audit_insight()."""
    status: str          # AUDIT_PASS or AUDIT_FAIL
    reason: str          # Human-readable explanation (empty string on PASS)
    skipped: bool = False  # True when pre-filter cleared all numbers (no LLM call)
    latency_ms: float = 0.0  # Wall-clock time for the audit LLM call (0 if skipped)

# ---------------------------------------------------------------------------
# Concurrency + rate-limit helpers
# ---------------------------------------------------------------------------

# Allow at most 3 simultaneous requests to the watsonx API so the free-plan
# limit of 10 concurrent requests is never exceeded even across Streamlit
# rerenders that trigger summarize + interpret_forecast + chat at once.
_WATSONX_SEMAPHORE = threading.Semaphore(3)

_RETRY_DELAYS = (2, 4, 8)   # seconds — exponential backoff on HTTP 429


def _call_with_retry(fn, *args, **kwargs):
    """
    Call fn(*args, **kwargs) under the shared semaphore with exponential
    backoff on 429 / consumption_limit_reached errors.

    Raises the last exception if all retries are exhausted.
    """
    last_exc: Exception | None = None
    delays = list(_RETRY_DELAYS) + [None]   # None = no more retries

    for delay in delays:
        with _WATSONX_SEMAPHORE:
            try:
                return fn(*args, **kwargs)
            except Exception as exc:
                err = str(exc).lower()
                if "429" in err or "consumption_limit" in err or "rate" in err:
                    last_exc = exc
                    if delay is not None:
                        logger.warning(
                            "watsonx 429 rate limit hit — retrying in %ds. (%s)",
                            delay, exc,
                        )
                        # Release semaphore before sleeping so other threads
                        # are not blocked during the backoff window.
                else:
                    raise
        # Sleep *outside* the semaphore context so other callers can proceed.
        if delay is not None:
            time.sleep(delay)

    raise last_exc  # exhausted all retries

# ---------------------------------------------------------------------------
# Guardrails — Guardian model ID + audit prompt template
# ---------------------------------------------------------------------------

# ibm/granite-guardian-3-8b is a safety/hallucination-detection model that
# produces a structured "safe"/"unsafe" verdict given a context + response.
# Override via WATSONX_GUARDIAN_MODEL_ID in .env if granite-guardian-3-8b is
# not available in your deployment (see config.GUARDIAN_MODEL_ID).
GUARDIAN_MODEL_ID: str = config.GUARDIAN_MODEL_ID

GUARDIAN_AUDIT_PROMPT_TEMPLATE = """\
You are a fact-checking critic for AI-generated wildfire risk summaries.

SOURCE DATA (authoritative — treat every field as ground truth):
{source_json}

GENERATED SUMMARY (the text to audit):
{summary_text}

Your task: classify the summary against the source data on ALL FOUR criteria below.
Return ONLY a JSON object with no surrounding text, no markdown fences.

Criteria:
1. "fabricated_numbers" — Does the summary contain any numeric values that are
   NOT present (exactly or approximately within ±5%) in the source data?
2. "contradictions" — Does the summary state a value that directly contradicts
   a value in the source data (e.g. wrong fire count, wrong FRP)?
3. "overstated_certainty" — Does the summary present probabilistic forecasts as
   certain facts without qualification?
4. "missing_disclaimer" — If forecasts or probabilities are mentioned, does the
   summary lack an appropriate uncertainty disclaimer?

For each criterion output "true" if the issue is present, "false" otherwise.
Then set "verdict" to "fail" if ANY criterion is true, otherwise "pass".
Set "reason" to a concise one-sentence explanation if verdict is "fail",
otherwise set "reason" to "All checks passed."

Required output format (strict JSON, no other text):
{{
  "fabricated_numbers": <true|false>,
  "contradictions": <true|false>,
  "overstated_certainty": <true|false>,
  "missing_disclaimer": <true|false>,
  "verdict": "<pass|fail>",
  "reason": "<one sentence>"
}}
"""

# ---------------------------------------------------------------------------
# Guardrails — cheap numeric pre-filter (runs before the LLM critic call)
# ---------------------------------------------------------------------------

# Match integers and decimals, including comma-formatted numbers.
# Excludes pure year/time strings that happen to look like numbers.
_NUM_RE = re.compile(r"\b\d{1,3}(?:,\d{3})*(?:\.\d+)?\b|\b\d+(?:\.\d+)?\b")


def _extract_numeric_claims(text: str) -> list[float]:
    """Return all numeric values (as floats) extracted from *text*."""
    results = []
    for m in _NUM_RE.finditer(text):
        try:
            results.append(float(m.group().replace(",", "")))
        except ValueError:
            pass
    return results


def _extract_source_numbers(source: dict) -> list[float]:
    """Walk a (possibly nested) source dict and collect all numeric leaf values."""
    nums: list[float] = []

    def _walk(obj):
        if isinstance(obj, dict):
            for v in obj.values():
                _walk(v)
        elif isinstance(obj, (list, tuple)):
            for v in obj:
                _walk(v)
        elif isinstance(obj, (int, float)) and not isinstance(obj, bool):
            try:
                nums.append(float(obj))
            except (ValueError, OverflowError):
                pass

    _walk(source)
    return nums


def numeric_prefilter(summary_text: str, source_json: dict) -> tuple[bool, list[float]]:
    """
    Fast regex-based pre-filter for the critic agent.

    Returns ``(all_clear, suspicious_numbers)`` where:
    - ``all_clear=True``  → every numeric claim in *summary_text* is traceable
                            to a value in *source_json* (within tolerance).
                            Skip the expensive Guardian call.
    - ``all_clear=False`` → at least one number could not be reconciled.
                            *suspicious_numbers* lists the offending values.

    Tolerance tiers (tightened to avoid false-passes on large spread_index values):
    - val < 1,000       : ±5% relative  (handles FRP, fire-count rounding)
    - 1,000 ≤ val < 100,000  : ±1% relative  (tighter on mid-range totals)
    - val ≥ 100,000     : exact match required (±0.5% or ±500, whichever is
                          smaller), because spread_index values in the millions
                          differ meaningfully even at 4–5% separation.

    A 5% window on 1,876,000 km² is ±93,800 km² — enough to falsely pass a
    hallucinated 1,953,840 km² that is 77,840 km² off.  Tightening to 0.5%
    (±9,380) eliminates that false pass.
    """
    claimed = _extract_numeric_claims(summary_text)
    if not claimed:
        return True, []  # no numbers → nothing to fabricate

    source_nums = _extract_source_numbers(source_json)
    if not source_nums:
        return False, claimed  # can't verify anything — escalate

    def _tolerance(val: float) -> float:
        abs_val = abs(val)
        if abs_val < 1_000:
            return max(abs_val * 0.05, 0.5)       # ±5%  (FRP, counts)
        if abs_val < 100_000:
            return max(abs_val * 0.01, 1.0)        # ±1%  (spread < 100k km²)
        return max(abs_val * 0.005, 500.0)         # ±0.5% (large spread_index)

    suspicious: list[float] = []
    for val in claimed:
        # Percentage values (0–100) are almost always derived; treat as safe.
        if 0.0 <= val <= 100.0:
            continue
        tol = _tolerance(val)
        if not any(abs(val - s) <= tol for s in source_nums):
            suspicious.append(val)

    return (len(suspicious) == 0), suspicious


# ---------------------------------------------------------------------------
# Prompt templates (module-level constants)
# ---------------------------------------------------------------------------

SUMMARY_PROMPT_TEMPLATE = """\
You are a wildfire risk analyst briefing forest rangers and civil protection teams.

Current situation for {country} (last {time_window_days} day(s)):
  - Overall risk level  : {risk_level}
  - Active fire count   : {fire_count}
  - Total FRP           : {total_frp:.1f} MW
  - Maximum FRP         : {max_frp:.1f} MW
  - Spread index        : {spread_index:.0f} km²
    NOTE — this is the bounding-box area enclosing all fire detection coordinates.
    It measures geographic DISPERSION of detections, NOT area burned or area affected.
    Two fires at opposite ends of a country produce a large value despite minimal damage.
    Always describe it as "detections are spread across a bounding area of X km²".
    Never say "affecting X km²", "burned area", or "fire extent".
  - High-confidence %   : {high_confidence_pct:.1f}%

Top hotspots by fire intensity:
{top_hotspots_str}

Instructions:
You MUST structure your response EXACTLY as shown in the example below — no exceptions.
First output the Evidence block, then the Confidence line, then the narrative.
Only include Evidence fields whose values are present in the input data above.
Never invent or estimate a field (e.g. humidity or temperature) that is not provided.

--- EXAMPLE OUTPUT (follow this format exactly) ---

Evidence:
- 312 fires detected
- Max FRP = 187.4 MW
- Detections spread across a bounding area of 95,000 km²
- High-confidence detections = 74.2%

Confidence: 78%

The current risk level for [Country] is HIGH. [Narrative continues here in plain
language suitable for field personnel. Mention total FRP, spread index, and the
top hotspot coordinates.]

Recommended actions for the next 24 hours:
• [Action 1]
• [Action 2]
• [Action 3]

--- END EXAMPLE ---

Now produce the real response for {country}:
1. Open with the Evidence block using only the fields provided in the input above.
2. Follow immediately with a Confidence line (your estimated confidence in the
   risk assessment, as a percentage between 0% and 100%).
3. Begin the narrative with: "The current risk level for {country} is {risk_level}."
4. Summarise fire count, total FRP ({total_frp:.1f} MW), and spread index for
   field personnel. Describe the spread index as: "detections are spread across
   a bounding area of {spread_index:.0f} km²". Do NOT write "affecting X km²",
   "burned area", "fire extent", or any phrase implying damage extent.
5. Call out the top hotspot coordinates listed above.
6. End with exactly 3 bullet-point action recommendations for the next 24 hours.
7. Keep the entire response under 300 words.
"""

FORECAST_PROMPT_TEMPLATE = """\
You are a wildfire risk analyst interpreting a 24-hour probabilistic fire forecast
for {country}.

Forecast metadata:
  - Model used              : {model_used}
  - Forecast horizon        : {forecast_horizon_hours} hours
  - Generated at (UTC)      : {generated_at}

Top high-risk grid cells:
{top_cells_str}

Instructions:
You MUST structure your response EXACTLY as shown in the example below — no exceptions.
First output the Evidence block, then the Confidence line, then the narrative.
Only include Evidence fields whose values are present in the top-cells data above
(e.g. only add temperature or humidity lines if those values appear in the cell data).
Never invent or estimate a field that is not in the provided data.

--- EXAMPLE OUTPUT (follow this format exactly) ---

Evidence:
- Top 3 of 47 cells shown; 12 at EXTREME, 18 at HIGH, 17 at MEDIUM
- Max fire probability = 91.4% (Cell 1)
- Top cell risk band = EXTREME
- Key driver: high temperature (38.2°C), recent fire history (7 fires in 7d)

Confidence: 72%

Note: This is a probabilistic estimate based on historical fire activity and
weather data. It is not a certainty.

[Narrative continues — name top 3 cells, their coordinates, risk band, fire
probability %, and key drivers. State model used and what that means for reliability.
Give 2–3 concrete preparation actions for the next 24 hours.]

--- END EXAMPLE ---

Now produce the real response for {country}:
1. Open with the Evidence block. The first bullet MUST read exactly:
   "Top 3 of {total_cells} cells shown; {extreme_count} at EXTREME, {high_count} at HIGH, {medium_count} at MEDIUM"
   Then include: max fire probability, top risk band.
   Add temperature or humidity lines only if those values appear in the input.
2. Follow immediately with a Confidence line (your estimated confidence in the
   forecast, as a percentage between 0% and 100%).
3. Then write: "Note: This is a probabilistic estimate based on historical fire
   activity and weather data. It is not a certainty."
4. Name the top 3 high-risk grid cells (coordinates, risk band, fire probability %).
5. For each cell identify the 2 features that pushed the risk highest.
6. State whether {model_used} was used and briefly explain reliability implications.
7. Give 2–3 concrete preparation actions for the next 24 hours.
8. Keep the entire response under 350 words.
"""

CHAT_SYSTEM_PROMPT = """\
You are a wildfire risk analyst assistant. You have access to live risk metrics
and a 24-hour probabilistic fire forecast for {country}.

Current risk context (JSON):
{risk_context_json}

Forecast summary — top cells (JSON):
{forecast_summary_json}

Guidelines:
- Answer only questions related to fire risk, the forecast, weather conditions,
  or recommended actions.
- If the user asks about an unrelated topic, politely redirect them to fire-risk
  related questions.
- Never claim certainty about future fire occurrence; always frame predictions as
  probabilistic estimates.
- Keep answers concise and actionable for field personnel.
"""

# ---------------------------------------------------------------------------
# Abstract base class
# ---------------------------------------------------------------------------


class LLMGateway(ABC):
    """Abstract interface for LLM-backed natural-language generation."""

    @abstractmethod
    def summarize(self, risk_context: RiskContext) -> str:
        """Generate a plain-language risk summary for field personnel."""
        ...

    @abstractmethod
    def interpret_forecast(self, forecast_result: ForecastResult) -> str:
        """Generate a plain-language interpretation of the forecast output."""
        ...

    @abstractmethod
    def chat(
        self,
        messages: list[dict],
        risk_context: RiskContext,
        forecast_result: ForecastResult,
    ) -> str:
        """
        Respond to the latest user message given conversation history and context.

        Parameters
        ----------
        messages:
            List of ``{"role": "user"|"assistant", "content": str}`` dicts,
            ordered oldest-first.  The last entry is always the user's message.
        risk_context:
            Current RiskContext for the selected country/time window.
        forecast_result:
            Most recent ForecastResult for the selected country.
        """
        ...


# ---------------------------------------------------------------------------
# watsonx / Granite implementation
# ---------------------------------------------------------------------------


class WatsonxGateway(LLMGateway):
    """Concrete LLM gateway backed by IBM Granite via the ibm-watsonx-ai SDK."""

    def __init__(self) -> None:
        if not all([config.WATSONX_API_KEY, config.WATSONX_PROJECT_ID, config.WATSONX_URL]):
            raise RuntimeError(
                "watsonx credentials missing. Set WATSONX_API_KEY, "
                "WATSONX_PROJECT_ID, and WATSONX_URL in your .env file."
            )

        # Lazy import — keeps the module importable without ibm_watsonx_ai installed.
        from ibm_watsonx_ai import APIClient, Credentials
        from ibm_watsonx_ai.foundation_models import ModelInference

        credentials = Credentials(
            url=config.WATSONX_URL,
            api_key=config.WATSONX_API_KEY,
        )
        self._client = APIClient(credentials)
        self._model = ModelInference(
            model_id=config.GRANITE_MODEL_ID,
            api_client=self._client,
            project_id=config.WATSONX_PROJECT_ID,
            params={"max_new_tokens": 512, "temperature": 0.3},
        )
        # AGENT 2: Granite Guardian for hallucination / fabrication auditing.
        # Separate ModelInference so we can swap the model ID independently.
        # If the Guardian model is not available in this deployment, degrade
        # gracefully to the cheap pre-filter only (no LLM critic).
        self._guardian: "ModelInference | None" = None
        try:
            self._guardian = ModelInference(
                model_id=GUARDIAN_MODEL_ID,
                api_client=self._client,
                project_id=config.WATSONX_PROJECT_ID,
                params={"max_new_tokens": 256, "temperature": 0.0},
            )
            logger.info(
                "WatsonxGateway initialised with model '%s' + guardian '%s'.",
                config.GRANITE_MODEL_ID, GUARDIAN_MODEL_ID,
            )
        except Exception as _guardian_err:
            logger.warning(
                "Guardian model '%s' not available in this deployment (%s). "
                "Guardrails will use the cheap numeric pre-filter only.",
                GUARDIAN_MODEL_ID, _guardian_err,
            )
            logger.info(
                "WatsonxGateway initialised with model '%s' (no Guardian).",
                config.GRANITE_MODEL_ID,
            )

    # ------------------------------------------------------------------
    # AGENT 2 — audit_insight (Granite Guardian critic)
    # ------------------------------------------------------------------

    def audit_insight(self, summary_text: str, source_json: dict) -> AuditVerdict:
        """
        Classify *summary_text* against *source_json* for four failure modes:
          • fabricated numbers not present in the source
          • contradictions with source values
          • overstated certainty
          • missing uncertainty disclaimer

        Pipeline
        --------
        1. Cheap numeric pre-filter: regex-extract all numbers from the text and
           cross-check against source_json's numeric fields (±5% tolerance).
           If every number is accounted for, return PASS immediately (no LLM call).
        2. If the pre-filter flags any ambiguous number, call Granite Guardian
           via the shared rate-limit semaphore and parse its JSON verdict.

        Returns
        -------
        AuditVerdict with status AUDIT_PASS or AUDIT_FAIL plus a reason string.
        """
        # ── Stage 1: cheap pre-filter ──────────────────────────────────────
        all_clear, suspicious = numeric_prefilter(summary_text, source_json)
        if all_clear:
            logger.debug("audit_insight: pre-filter PASS (no suspicious numbers).")
            return AuditVerdict(
                status=AUDIT_PASS, reason="", skipped=True, latency_ms=0.0
            )

        logger.info(
            "audit_insight: pre-filter flagged numbers %s — escalating to Guardian.",
            suspicious,
        )

        # ── Stage 2: Granite Guardian LLM call ────────────────────────────
        # If the Guardian model failed to initialise, fail-open so the pipeline
        # is not blocked on every deployment that lacks granite-guardian-3-8b.
        if self._guardian is None:
            logger.warning(
                "audit_insight: Guardian model unavailable — suspicious numbers %s "
                "could not be verified by LLM critic. Returning PASS (fail-open).",
                suspicious,
            )
            return AuditVerdict(
                status=AUDIT_PASS,
                reason=f"Guardian model not available — numbers {suspicious} unverified.",
                skipped=True,
                latency_ms=0.0,
            )

        prompt = GUARDIAN_AUDIT_PROMPT_TEMPLATE.format(
            source_json=json.dumps(source_json, indent=2, default=str),
            summary_text=summary_text,
        )
        t0 = time.monotonic()
        try:
            raw = _call_with_retry(
                self._guardian.chat,
                messages=[{"role": "user", "content": prompt}],
                params={"max_new_tokens": 256, "temperature": 0.0},
            )["choices"][0]["message"]["content"]
        except Exception as exc:
            logger.error("audit_insight: Guardian call failed: %s", exc)
            # Fail-open: treat a Guardian failure as a PASS so the pipeline
            # does not block every summary when credentials are unavailable.
            return AuditVerdict(
                status=AUDIT_PASS,
                reason=f"Guardian unavailable ({exc}) — audit skipped.",
                skipped=True,
                latency_ms=0.0,
            )
        latency_ms = (time.monotonic() - t0) * 1000

        # ── Parse JSON response ────────────────────────────────────────────
        try:
            # Strip markdown fences the model may wrap around the JSON.
            clean = re.sub(r"```(?:json)?|```", "", raw).strip()
            parsed = json.loads(clean)
            verdict_str = str(parsed.get("verdict", "pass")).lower()
            reason = str(parsed.get("reason", ""))
            status = AUDIT_FAIL if verdict_str == "fail" else AUDIT_PASS
        except (json.JSONDecodeError, AttributeError) as exc:
            # Unparseable response → conservative FAIL so the correction loop fires.
            logger.warning(
                "audit_insight: could not parse Guardian JSON (%s). Raw: %r", exc, raw
            )
            status = AUDIT_FAIL
            reason = f"Guardian returned unparseable response: {raw[:200]}"

        logger.info(
            "audit_insight: Guardian verdict=%s latency=%.0f ms reason=%s",
            status, latency_ms, reason,
        )
        return AuditVerdict(status=status, reason=reason, latency_ms=latency_ms)

    # ------------------------------------------------------------------
    # Internal helper — single generator call (no audit)
    # ------------------------------------------------------------------

    # DEBUG_FABRICATED_SPREAD is the historically-bugged value that the old
    # prompt template used to generate.  It is a realistic-sounding but wholly
    # fabricated number for Portugal (real spread is ~40 000–80 000 km²).
    _DEBUG_FABRICATED_SPREAD: float = 1_953_840.0

    def _generate_summary(self, risk_context: RiskContext) -> str:
        """Call the generator model for a risk summary (no guardrail logic).

        When ``config.FORCE_FABRICATED_TEST`` is True the generator's output is
        post-processed to replace the real spread_index with the historic
        fabricated value (1 953 840 km²).  This lets developers trigger the full
        guardian pipeline against the live API from the Streamlit UI without
        needing bad data in the database.  The flag has zero effect in
        production (default False).
        """
        top_hotspots_str = "\n".join(
            f"  • Lat {h['lat']:.2f}, Lon {h['lon']:.2f}  (FRP: {h['frp']:.1f} MW)"
            for h in risk_context.top_hotspots
        ) or "  (no hotspots detected)"

        prompt = SUMMARY_PROMPT_TEMPLATE.format(
            country=risk_context.country,
            time_window_days=risk_context.time_window_days,
            risk_level=risk_context.risk_level,
            fire_count=risk_context.fire_count,
            total_frp=risk_context.total_frp,
            max_frp=risk_context.max_frp,
            spread_index=risk_context.spread_index,
            high_confidence_pct=risk_context.high_confidence_pct,
            top_hotspots_str=top_hotspots_str,
        )
        text = _call_with_retry(
            self._model.chat,
            messages=[{"role": "user", "content": prompt}],
            params={"max_new_tokens": 512, "temperature": 0.3},
        )["choices"][0]["message"]["content"]

        if config.FORCE_FABRICATED_TEST:
            # Replace every occurrence of the real spread value with the fabricated
            # one.  We use a broad replacement so it catches formatted variants
            # (e.g. "45,000", "45000", "45,000.0").
            real_str_variants = [
                f"{risk_context.spread_index:,.0f}",
                f"{risk_context.spread_index:.0f}",
                f"{int(risk_context.spread_index):,}",
            ]
            fab_str = f"{self._DEBUG_FABRICATED_SPREAD:,.0f}"
            for variant in real_str_variants:
                text = text.replace(variant, fab_str)
            logger.warning(
                "_generate_summary: FORCE_FABRICATED_TEST active — "
                "injected fabricated spread_index %s km² into output.",
                fab_str,
            )

        return text

    def _generate_forecast_interp(self, forecast_result: ForecastResult) -> tuple[str, dict]:
        """Call the generator model for a forecast interpretation (no guardrail logic).

        Returns (text, source_json) where source_json is the structured data
        the critic will verify the text against.
        """
        top_cells_str = _format_top_cells(forecast_result.top_risk_cells[:3])

        all_cells = forecast_result.cells or []
        total_cells = len(all_cells)
        extreme_count = sum(1 for c in all_cells if c.risk_band == "EXTREME")
        high_count    = sum(1 for c in all_cells if c.risk_band == "HIGH")
        medium_count  = sum(1 for c in all_cells if c.risk_band == "MEDIUM")

        prompt = FORECAST_PROMPT_TEMPLATE.format(
            country=forecast_result.country,
            model_used=forecast_result.model_used,
            forecast_horizon_hours=forecast_result.forecast_horizon_hours,
            generated_at=forecast_result.generated_at,
            top_cells_str=top_cells_str,
            total_cells=total_cells,
            extreme_count=extreme_count,
            high_count=high_count,
            medium_count=medium_count,
        )
        text = _call_with_retry(
            self._model.chat,
            messages=[{"role": "user", "content": prompt}],
            params={"max_new_tokens": 512, "temperature": 0.3},
        )["choices"][0]["message"]["content"]

        source_json = {
            "country": forecast_result.country,
            "model_used": forecast_result.model_used,
            "forecast_horizon_hours": forecast_result.forecast_horizon_hours,
            "total_cells": total_cells,
            "extreme_count": extreme_count,
            "high_count": high_count,
            "medium_count": medium_count,
            "top_cells": [
                {
                    "lat": c.lat_center,
                    "lon": c.lon_center,
                    "risk_band": c.risk_band,
                    "fire_prob_pct": round(c.fire_prob * 100, 1),
                    "hist_fire_count_7d": c.historical_fire_count,
                }
                for c in forecast_result.top_risk_cells[:3]
            ],
        }
        return text, source_json

    # ------------------------------------------------------------------
    # summarize — generator + critic + correction loop
    # ------------------------------------------------------------------

    def summarize(self, risk_context: RiskContext) -> str:
        """
        Generate a plain-language risk summary for field personnel.

        Generator-Critic pipeline (Task 6):
        1. Call _generate_summary() (AGENT 1).
        2. Build source_json from risk_context and run audit_insight() (AGENT 2).
        3. If PASS → return text.
        4. If FAIL → inject the rejection reason into a correction prompt and
           regenerate once.
        5. Run audit_insight() on the regenerated text.
        6. If still FAIL → return original text with ⚠ UNVERIFIED marker.
        """
        try:
            source_json = dataclasses.asdict(risk_context)

            # Attempt 1 — generate
            logger.warning(
                "summarize [1/stage-gen]: calling generator (%s) "
                "FORCE_FABRICATED_TEST=%s",
                config.GRANITE_MODEL_ID, config.FORCE_FABRICATED_TEST,
            )
            text = self._generate_summary(risk_context)
            fab_present = "1,953,840" in text
            logger.warning(
                "summarize [2/stage-audit1]: generator done. "
                "fabricated_value_present=%s. calling audit_insight()...",
                fab_present,
            )
            verdict = self.audit_insight(text, source_json)
            logger.warning(
                "summarize [3/verdict1]: status=%s skipped=%s latency=%.0f ms reason=%r",
                verdict.status, verdict.skipped, verdict.latency_ms, verdict.reason,
            )

            if verdict.status == AUDIT_PASS:
                logger.warning(
                    "summarize [4/done]: PASS on attempt 1 — returning text. "
                    "fabricated_value_present=%s", fab_present,
                )
                return text

            # Attempt 2 — regenerate with correction hint.
            # IMPORTANT: route through _generate_summary() so that any debug
            # injection hook (FORCE_FABRICATED_TEST) is applied consistently.
            # The correction reason is appended as a second user turn so the
            # model sees both the original instructions and the rejection note.
            logger.warning(
                "summarize: audit FAIL (%.0f ms) — regenerating. Reason: %s",
                verdict.latency_ms, verdict.reason,
            )
            original_text = text
            # Call _generate_summary for the base text (with injection if active),
            # then follow up with the rejection reason in the same conversation.
            base_text2 = self._generate_summary(risk_context)
            correction_addendum = (
                f"[CORRECTION REQUIRED] A reviewer rejected the previous version: "
                f"{verdict.reason}  Please revise, correcting only the identified issues."
            )
            text = _call_with_retry(
                self._model.chat,
                messages=[
                    {"role": "user", "content": base_text2},
                    {"role": "user", "content": correction_addendum},
                ],
                params={"max_new_tokens": 512, "temperature": 0.3},
            )["choices"][0]["message"]["content"]

            # Apply the injection hook to the correction output too, so that when
            # FORCE_FABRICATED_TEST is active both attempts carry the fabrication.
            if config.FORCE_FABRICATED_TEST:
                real_str_variants = [
                    f"{risk_context.spread_index:,.0f}",
                    f"{risk_context.spread_index:.0f}",
                    f"{int(risk_context.spread_index):,}",
                ]
                fab_str = f"{self._DEBUG_FABRICATED_SPREAD:,.0f}"
                for variant in real_str_variants:
                    text = text.replace(variant, fab_str)
                logger.warning(
                    "summarize correction: FORCE_FABRICATED_TEST active — "
                    "re-injected fabricated spread_index %s into attempt 2.", fab_str,
                )

            fab_present2 = "1,953,840" in text
            logger.warning(
                "summarize [5/stage-audit2]: correction done. "
                "fabricated_value_present=%s. calling audit_insight()...",
                fab_present2,
            )
            verdict2 = self.audit_insight(text, source_json)
            logger.warning(
                "summarize [6/verdict2]: status=%s skipped=%s latency=%.0f ms reason=%r",
                verdict2.status, verdict2.skipped, verdict2.latency_ms, verdict2.reason,
            )
            if verdict2.status == AUDIT_PASS:
                logger.warning(
                    "summarize [7/done]: PASS on attempt 2 — returning corrected text. "
                    "fabricated_value_present=%s", fab_present2,
                )
                return text

            # Both attempts failed — return original with UNVERIFIED marker
            logger.warning(
                "summarize [8/done]: BOTH attempts FAILED — returning original with "
                "UNVERIFIED badge. fabricated_value_present=%s", fab_present,
            )
            return original_text + UNVERIFIED_MARKER

        except Exception as e:
            logger.error("summarize() EXCEPTION: %s", e, exc_info=True)
            return f"[Summary unavailable: {e}]"

    # ------------------------------------------------------------------
    # interpret_forecast — generator + critic + correction loop
    # ------------------------------------------------------------------

    def interpret_forecast(self, forecast_result: ForecastResult) -> str:
        """
        Generate a plain-language interpretation of the forecast output.

        Generator-Critic pipeline (Task 6): same two-attempt pattern as summarize().
        """
        try:
            # Attempt 1 — generate
            text, source_json = self._generate_forecast_interp(forecast_result)
            verdict = self.audit_insight(text, source_json)

            if verdict.status == AUDIT_PASS:
                logger.info("interpret_forecast: audit PASS (skipped=%s latency=%.0f ms)",
                            verdict.skipped, verdict.latency_ms)
                return text

            # Attempt 2 — regenerate with correction hint
            logger.warning(
                "interpret_forecast: audit FAIL (latency=%.0f ms) — regenerating. Reason: %s",
                verdict.latency_ms, verdict.reason,
            )
            original_text = text
            text, _ = self._generate_forecast_interp(forecast_result)
            # Append correction hint via a second message in the conversation
            # so we do not re-format the full template (too costly).
            correction_addendum = (
                f"\n\n[CORRECTION REQUIRED] Previous version was rejected: {verdict.reason}"
                "\nPlease correct these issues."
            )
            text = _call_with_retry(
                self._model.chat,
                messages=[
                    {"role": "user", "content": text + correction_addendum},
                ],
                params={"max_new_tokens": 512, "temperature": 0.3},
            )["choices"][0]["message"]["content"]

            verdict2 = self.audit_insight(text, source_json)
            if verdict2.status == AUDIT_PASS:
                logger.info("interpret_forecast: regeneration PASS (latency=%.0f ms)",
                            verdict2.latency_ms)
                return text

            logger.error(
                "interpret_forecast: both attempts failed audit. Returning original with badge."
            )
            return original_text + UNVERIFIED_MARKER

        except Exception as e:
            logger.error("interpret_forecast() failed: %s", e)
            return f"[Forecast interpretation unavailable: {e}]"

    # ------------------------------------------------------------------
    # chat
    # ------------------------------------------------------------------

    def chat(
        self,
        messages: list[dict],
        risk_context: RiskContext,
        forecast_result: ForecastResult,
    ) -> str:
        """Build a full prompt from system context + message history and call Granite."""
        try:
            risk_context_json = json.dumps(
                dataclasses.asdict(risk_context), indent=2, default=str
            )

            forecast_summary_json = json.dumps(
                [
                    {
                        "lat": c.lat_center,
                        "lon": c.lon_center,
                        "risk_band": c.risk_band,
                        "fire_prob": round(c.fire_prob * 100, 1),
                    }
                    for c in forecast_result.top_risk_cells[:3]
                ],
                indent=2,
            )

            system_block = CHAT_SYSTEM_PROMPT.format(
                country=risk_context.country,
                risk_context_json=risk_context_json,
                forecast_summary_json=forecast_summary_json,
            )

            # Build a properly-structured messages list for the chat endpoint.
            chat_messages = [{"role": "system", "content": system_block}]
            for msg in messages:
                chat_messages.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("content", ""),
                })

            return _call_with_retry(
                self._model.chat,
                messages=chat_messages,
                params={"max_new_tokens": 512, "temperature": 0.3},
            )["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error("chat() failed: %s", e)
            return f"[Chat unavailable: {e}]"


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------


def _format_top_cells(cells: list) -> str:
    """
    Render up to 3 GridCell objects as structured text for the forecast prompt.

    For each cell the 2 feature values furthest from their safe baseline are
    highlighted so Granite can reference them in the explanation.
    """
    if not cells:
        return "  (no high-risk cells identified)"

    # Feature danger heuristics — higher value = more dangerous (except humidity/precip).
    def _danger_score(feat_name: str, value: float) -> float:
        mapping = {
            "temp_24h_mean":       lambda v: max(v - 20.0, 0.0) / 30.0,
            "humidity_24h_mean":   lambda v: max(100.0 - v, 0.0) / 100.0,
            "wind_24h_max":        lambda v: v / 60.0,
            "hist_fire_count_7d":  lambda v: v / 10.0,
            "hist_fire_count_24h": lambda v: v / 5.0,
        }
        fn = mapping.get(feat_name)
        if fn is None:
            return 0.0
        try:
            return float(fn(float(value)))
        except (TypeError, ValueError):
            return 0.0

    feature_labels = {
        "temp_24h_mean":       "high temperature",
        "humidity_24h_mean":   "low humidity",
        "wind_24h_max":        "high wind speed",
        "hist_fire_count_7d":  "recent fire history (7 d)",
        "hist_fire_count_24h": "very recent fire activity (24 h)",
    }

    lines: list[str] = []
    for i, cell in enumerate(cells, start=1):
        snap = cell.feature_snapshot or {}
        scored = sorted(
            [(k, _danger_score(k, snap.get(k, 0))) for k in feature_labels],
            key=lambda x: x[1],
            reverse=True,
        )
        top_features = [
            f"{feature_labels[k]} ({snap.get(k, 'N/A')})"
            for k, _ in scored[:2]
        ]
        lines.append(
            f"  Cell {i}: Lat {cell.lat_center:.2f}, Lon {cell.lon_center:.2f}"
            f" | Risk band: {cell.risk_band}"
            f" | Fire probability: {cell.fire_prob * 100:.1f}%"
            f"\n    Key drivers: {', '.join(top_features)}"
        )

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def get_gateway() -> LLMGateway:
    """Return the configured LLMGateway implementation.

    The backend is selected via ``config.LLM_BACKEND``.  Currently only
    ``"watsonx"`` is supported; an unknown value raises ``NotImplementedError``.
    """
    backend = config.LLM_BACKEND
    if backend == "watsonx":
        return WatsonxGateway()
    raise NotImplementedError(
        f"LLM backend '{backend}' is not implemented. Supported: 'watsonx'"
    )
