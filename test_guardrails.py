"""
test_guardrails.py — Task 6 AI Guardrails test suite.

Tests
-----
A. Normal correct summary  → passes both pre-filter and critic (no badge).
B. Fabricated-number summary (1953840 km² bug) → pre-filter flags it,
   critic issues FAIL verdict, correction loop fires, and if both attempts
   fail the UNVERIFIED marker is visible in the final output.

All LLM calls are mocked so the tests run without real watsonx credentials.

Run with:
    python -m pytest test_guardrails.py -v
"""

import dataclasses
import json
import time
import unittest
from unittest.mock import MagicMock, patch, call

from llm_gateway import (
    AUDIT_FAIL,
    AUDIT_PASS,
    UNVERIFIED_MARKER,
    AuditVerdict,
    WatsonxGateway,
    numeric_prefilter,
    _extract_numeric_claims,
    _extract_source_numbers,
)
from risk_engine import RiskContext


# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

def _make_risk_context(spread_index: float = 45_000.0) -> RiskContext:
    """Return a minimal RiskContext with known values."""
    return RiskContext(
        country="Portugal",
        time_window_days=2,
        fire_count=312,
        total_frp=1874.5,
        max_frp=187.4,
        mean_frp=6.0,
        hotspot_density=3.2,
        high_confidence_pct=74.2,
        spread_index=spread_index,
        risk_level="HIGH",
        top_hotspots=[
            {"lat": 38.50, "lon": -8.12, "frp": 187.4, "acq_date": "2025-07-01"},
        ],
    )


def _make_correct_summary(ctx: RiskContext) -> str:
    """Build a factually-correct summary that reflects the source exactly."""
    return (
        f"Evidence:\n"
        f"- {ctx.fire_count} fires detected\n"
        f"- Max FRP = {ctx.max_frp:.1f} MW\n"
        f"- Detections spread across a bounding area of {ctx.spread_index:,.0f} km²\n"
        f"- High-confidence detections = {ctx.high_confidence_pct:.1f}%\n\n"
        f"Confidence: 78%\n\n"
        f"The current risk level for {ctx.country} is {ctx.risk_level}. "
        f"There are {ctx.fire_count} active fire detections with a total FRP of "
        f"{ctx.total_frp:.1f} MW. Detections are spread across a bounding area of "
        f"{ctx.spread_index:,.0f} km², indicating a dispersed pattern.\n\n"
        f"Note: This is a probabilistic estimate based on historical fire activity "
        f"and weather data. It is not a certainty.\n\n"
        f"Recommended actions for the next 24 hours:\n"
        f"• Pre-position aerial resources near Lat 38.50, Lon -8.12\n"
        f"• Issue advisory for all HIGH-risk zones\n"
        f"• Monitor wind conditions closely\n"
    )


def _make_fabricated_summary(ctx: RiskContext) -> str:
    """Build a summary that injects the old 1953840 km² fabricated bug."""
    # The source spread_index is 45_000 km²; this text claims 1,953,840 km².
    return (
        f"Evidence:\n"
        f"- {ctx.fire_count} fires detected\n"
        f"- Max FRP = {ctx.max_frp:.1f} MW\n"
        f"- Detections spread across a bounding area of 1,953,840 km²\n"  # ← fabricated
        f"- High-confidence detections = {ctx.high_confidence_pct:.1f}%\n\n"
        f"Confidence: 78%\n\n"
        f"The current risk level for {ctx.country} is {ctx.risk_level}. "
        f"Fires are affecting an area of 1,953,840 km² across the country.\n\n"
        f"Recommended actions for the next 24 hours:\n"
        f"• Evacuate all regions\n"
        f"• Declare national emergency\n"
        f"• Request international assistance\n"
    )


def _guardian_fail_response(reason: str) -> dict:
    """Return a mock watsonx API response that Guardian would give for a FAIL verdict."""
    payload = json.dumps({
        "fabricated_numbers": True,
        "contradictions": True,
        "overstated_certainty": False,
        "missing_disclaimer": False,
        "verdict": "fail",
        "reason": reason,
    })
    return {"choices": [{"message": {"content": payload}}]}


def _guardian_pass_response() -> dict:
    """Return a mock watsonx API response that Guardian would give for a PASS verdict."""
    payload = json.dumps({
        "fabricated_numbers": False,
        "contradictions": False,
        "overstated_certainty": False,
        "missing_disclaimer": False,
        "verdict": "pass",
        "reason": "All checks passed.",
    })
    return {"choices": [{"message": {"content": payload}}]}


def _generator_response(text: str) -> dict:
    """Return a mock watsonx API response shaped like the generator model returns."""
    return {"choices": [{"message": {"content": text}}]}


# ---------------------------------------------------------------------------
# Helper — build a WatsonxGateway with mocked models
# ---------------------------------------------------------------------------

def _make_gateway() -> WatsonxGateway:
    """
    Instantiate WatsonxGateway without real credentials by patching the
    ibm_watsonx_ai imports and config values.
    """
    mock_model = MagicMock()
    mock_guardian = MagicMock()
    mock_client = MagicMock()

    with (
        patch("llm_gateway.config.WATSONX_API_KEY", "fake-key"),
        patch("llm_gateway.config.WATSONX_PROJECT_ID", "fake-proj"),
        patch("llm_gateway.config.WATSONX_URL", "https://us-south.ml.cloud.ibm.com"),
        patch("llm_gateway.config.GRANITE_MODEL_ID", "ibm/granite-4-h-small"),
        patch.dict("sys.modules", {
            "ibm_watsonx_ai": MagicMock(),
            "ibm_watsonx_ai.foundation_models": MagicMock(),
        }),
    ):
        gw = WatsonxGateway.__new__(WatsonxGateway)
        gw._model = mock_model
        gw._guardian = mock_guardian
        gw._client = mock_client

    return gw


# ===========================================================================
# Unit tests — numeric_prefilter (pure logic, no mocks needed)
# ===========================================================================

class TestNumericPrefilter(unittest.TestCase):
    """Test the fast regex-based numeric pre-filter in isolation."""

    def test_no_numbers_in_text_is_clear(self):
        text = "Fire risk is HIGH. All units should remain on standby."
        source = {"fire_count": 312, "spread_index": 45000.0}
        all_clear, suspicious = numeric_prefilter(text, source)
        self.assertTrue(all_clear)
        self.assertEqual(suspicious, [])

    def test_matching_numbers_are_clear(self):
        # spread_index is 45000 in source; 45,000 appears in text (comma formatted)
        ctx = _make_risk_context()
        text = _make_correct_summary(ctx)
        source = dataclasses.asdict(ctx)
        all_clear, suspicious = numeric_prefilter(text, source)
        self.assertTrue(all_clear, f"Expected clear but got suspicious: {suspicious}")

    def test_fabricated_number_is_flagged(self):
        ctx = _make_risk_context()
        text = _make_fabricated_summary(ctx)  # contains 1,953,840 not in source
        source = dataclasses.asdict(ctx)
        all_clear, suspicious = numeric_prefilter(text, source)
        self.assertFalse(all_clear)
        # 1953840 should appear in suspicious list
        self.assertTrue(
            any(abs(v - 1_953_840) < 1 for v in suspicious),
            f"Expected 1953840 in suspicious list but got: {suspicious}",
        )

    def test_percentages_below_100_are_safe(self):
        # 74.2% confidence, 78% confidence → should NOT be flagged
        text = "High-confidence detections = 74.2%. Confidence: 78%."
        source = {"high_confidence_pct": 74.2}
        all_clear, suspicious = numeric_prefilter(text, source)
        self.assertTrue(all_clear, f"Percentages ≤100 should not be suspicious: {suspicious}")

    def test_extract_numeric_claims(self):
        text = "1,953,840 km² across 312 fires with FRP of 187.4 MW"
        nums = _extract_numeric_claims(text)
        self.assertIn(1953840.0, nums)
        self.assertIn(312.0, nums)
        self.assertIn(187.4, nums)

    def test_extract_source_numbers_nested(self):
        source = {
            "fire_count": 312,
            "total_frp": 1874.5,
            "top_hotspots": [{"lat": 38.5, "lon": -8.12, "frp": 187.4}],
        }
        nums = _extract_source_numbers(source)
        self.assertIn(312.0, nums)
        self.assertIn(1874.5, nums)
        self.assertIn(187.4, nums)


# ===========================================================================
# Integration tests — full audit_insight() pipeline
# ===========================================================================

class TestAuditInsight(unittest.TestCase):
    """Test audit_insight() end-to-end with mocked Guardian."""

    def setUp(self):
        self.gw = _make_gateway()
        self.ctx = _make_risk_context()
        self.source = dataclasses.asdict(self.ctx)

    def test_correct_summary_passes_pre_filter_no_guardian_call(self):
        """
        Test (a): a factually-correct summary whose numbers all match source
        should PASS the pre-filter, skipping the expensive Guardian LLM call.
        """
        t_start = time.monotonic()
        summary = _make_correct_summary(self.ctx)
        verdict = self.gw.audit_insight(summary, self.source)
        elapsed_ms = (time.monotonic() - t_start) * 1000

        self.assertEqual(verdict.status, AUDIT_PASS)
        self.assertTrue(verdict.skipped, "Pre-filter should have cleared it (skipped=True)")
        self.assertEqual(verdict.latency_ms, 0.0)
        # Guardian model should NOT have been called at all
        self.gw._guardian.chat.assert_not_called()

        print(f"\n[TEST A] Correct summary → PASS (pre-filter, no LLM). "
              f"Pre-filter elapsed: {elapsed_ms:.1f} ms")

    def test_fabricated_number_triggers_guardian_and_fails(self):
        """
        Test (b): a summary containing 1,953,840 km² (not in source which has
        45,000 km²) should be flagged by the pre-filter, escalated to Guardian,
        and receive a FAIL verdict.
        """
        summary = _make_fabricated_summary(self.ctx)

        # Mock Guardian to return a FAIL verdict
        fail_response = _guardian_fail_response(
            "Summary contains fabricated spread index 1,953,840 km² not present in source (45,000 km²)."
        )
        self.gw._guardian.chat.return_value = fail_response

        t_start = time.monotonic()
        verdict = self.gw.audit_insight(summary, self.source)
        elapsed_ms = (time.monotonic() - t_start) * 1000

        self.assertEqual(verdict.status, AUDIT_FAIL)
        self.assertFalse(verdict.skipped)
        self.assertIn("fabricated", verdict.reason.lower())
        self.gw._guardian.chat.assert_called_once()

        print(f"\n[TEST B] Fabricated summary → FAIL (Guardian). "
              f"Guardian latency: {verdict.latency_ms:.1f} ms  "
              f"Total elapsed: {elapsed_ms:.1f} ms  "
              f"Reason: {verdict.reason}")

    def test_guardian_pass_verdict_accepted(self):
        """Guardian returns 'pass' → AuditVerdict.status == AUDIT_PASS."""
        # Use a text with a number that pre-filter can't verify — force escalation
        text = "Fires are active across 1234567 km²."
        source = {"spread_index": 45000.0}

        self.gw._guardian.chat.return_value = _guardian_pass_response()
        verdict = self.gw.audit_insight(text, source)

        self.assertEqual(verdict.status, AUDIT_PASS)
        self.gw._guardian.chat.assert_called_once()

    def test_guardian_unavailable_fails_open(self):
        """If Guardian raises an exception, audit_insight() fails open (returns PASS)."""
        text = "Fires detected across 9999999 km²."
        source = {"spread_index": 45000.0}

        self.gw._guardian.chat.side_effect = RuntimeError("Connection error")
        verdict = self.gw.audit_insight(text, source)

        self.assertEqual(verdict.status, AUDIT_PASS)
        self.assertTrue(verdict.skipped)


# ===========================================================================
# Integration tests — full summarize() correction loop
# ===========================================================================

class TestSummarizeGuardrailLoop(unittest.TestCase):
    """Test the full generator-critic-correction loop in summarize()."""

    def setUp(self):
        self.gw = _make_gateway()
        self.ctx = _make_risk_context()

    def test_correct_summary_passes_on_first_attempt(self):
        """
        Scenario: generator returns a correct summary → pre-filter clears it →
        no Guardian call, no regeneration. summarize() returns the text as-is.
        """
        correct = _make_correct_summary(self.ctx)
        self.gw._model.chat.return_value = _generator_response(correct)

        t_start = time.monotonic()
        result = self.gw.summarize(self.ctx)
        elapsed_ms = (time.monotonic() - t_start) * 1000

        self.assertEqual(result, correct)
        self.assertNotIn(UNVERIFIED_MARKER, result)
        self.gw._model.chat.assert_called_once()   # only one generator call
        self.gw._guardian.chat.assert_not_called() # no Guardian call needed

        print(f"\n[SUMMARY TEST A] Correct → PASS first attempt. "
              f"Total elapsed: {elapsed_ms:.1f} ms")

    def test_fabricated_summary_triggers_regeneration(self):
        """
        Scenario: generator returns fabricated summary (attempt 1) → pre-filter
        flags 1,953,840 km² → Guardian FAIL → correction loop fires.

        Correction path (fixed): _generate_summary() called again for base_text2,
        then a follow-up refinement call is made.  That is 3 model calls total:
          call 1: attempt 1 generation          → fabricated
          call 2: attempt 2 base (_generate_summary) → fabricated (no injection in mock)
          call 3: attempt 2 refinement          → corrected

        The corrected text does NOT contain 1,953,840 so the pre-filter clears it
        on the second audit → PASS → no UNVERIFIED badge.
        """
        fabricated = _make_fabricated_summary(self.ctx)
        corrected = _make_correct_summary(self.ctx)

        # 3 generator calls now: gen1, gen2-base, gen2-refinement
        self.gw._model.chat.side_effect = [
            _generator_response(fabricated),   # attempt 1 generation
            _generator_response(fabricated),   # attempt 2 _generate_summary base
            _generator_response(corrected),    # attempt 2 refinement → corrected
        ]
        # Guardian called once for failing attempt 1; attempt 2 corrected text
        # passes the pre-filter so Guardian is not called again.
        self.gw._guardian.chat.side_effect = [
            _guardian_fail_response(
                "Fabricated spread index 1,953,840 km² not in source."
            ),
        ]

        t_start = time.monotonic()
        result = self.gw.summarize(self.ctx)
        elapsed_ms = (time.monotonic() - t_start) * 1000

        self.assertEqual(result, corrected)
        self.assertNotIn(UNVERIFIED_MARKER, result)
        self.assertEqual(self.gw._model.chat.call_count, 3)
        # Guardian called only once — corrected text cleared the pre-filter.
        self.assertEqual(self.gw._guardian.chat.call_count, 1)

        print(f"\n[SUMMARY TEST B] Fabricated → regeneration → PASS (pre-filter on attempt 2). "
              f"Guardian called 1 time. Total elapsed: {elapsed_ms:.1f} ms")

    def test_both_attempts_fail_produces_unverified_badge(self):
        """
        Scenario: both attempts fail audit → summarize() returns original text
        with UNVERIFIED_MARKER appended.

        Correction path (fixed): 4 model calls total:
          call 1: attempt 1 generation
          call 2: attempt 2 base (_generate_summary)
          call 3: attempt 2 refinement
          (both attempt 1 and attempt 2 outputs contain 1,953,840 → both fail Guardian)
        """
        fabricated = _make_fabricated_summary(self.ctx)

        # 3 generator calls: gen1, gen2-base, gen2-refinement (all return fabricated)
        self.gw._model.chat.side_effect = [
            _generator_response(fabricated),   # attempt 1
            _generator_response(fabricated),   # attempt 2 _generate_summary base
            _generator_response(fabricated),   # attempt 2 refinement
        ]
        # Guardian called twice — once per attempt, both fail
        self.gw._guardian.chat.side_effect = [
            _guardian_fail_response("Fabricated spread index."),
            _guardian_fail_response("Still fabricated after correction."),
        ]

        t_start = time.monotonic()
        result = self.gw.summarize(self.ctx)
        elapsed_ms = (time.monotonic() - t_start) * 1000

        self.assertIn(UNVERIFIED_MARKER, result)
        # The original fabricated text should still be present
        self.assertIn("1,953,840", result)
        self.assertEqual(self.gw._model.chat.call_count, 3)
        self.assertEqual(self.gw._guardian.chat.call_count, 2)

        print(f"\n[SUMMARY TEST C] Both attempts fail → UNVERIFIED badge. "
              f"Guardian called {self.gw._guardian.chat.call_count} time(s). "
              f"Total elapsed: {elapsed_ms:.1f} ms")


# ===========================================================================
# Run tests and print timing summary
# ===========================================================================

if __name__ == "__main__":
    unittest.main(verbosity=2)
