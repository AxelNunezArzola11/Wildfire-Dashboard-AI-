"""
test_email_alerts.py — Self-contained test for email_alerts.py

Tests:
  1. Fail-open: ALERT_EMAIL_ENABLED unset → no crash, returns "skipped-not-configured"
  2. Idempotency: simulate HIGH→EXTREME→EXTREME→MEDIUM→EXTREME transition sequence
     and assert exactly 2 sends would have fired.
  3. Mock SMTP server: spin up a local DebuggingServer on localhost:10025 using
     Python's smtpd module (or aiosmtpd if available), confirm the constructed
     MIMEMultipart message is structurally valid and that smtplib can connect.
"""

import os
import sys
import sqlite3
import tempfile
import threading
import smtpd
import asyncore
import time
import textwrap

# ── Isolate test from any real .env by setting env vars explicitly before import ──
os.environ.setdefault("FIRMS_MAP_KEY", "test_firms_key")
os.environ.setdefault("WATSONX_API_KEY", "")
os.environ.setdefault("WATSONX_PROJECT_ID", "")

# ---------------------------------------------------------------------------
# Test helpers
# ---------------------------------------------------------------------------

def _make_risk_ctx(level: str, country: str = "Angola"):
    """Return a minimal RiskContext with the given risk_level."""
    from risk_engine import RiskContext
    return RiskContext(
        country=country,
        time_window_days=2,
        fire_count=0 if level == "NO DATA" else 300,
        total_frp=0.0 if level == "NO DATA" else 4000.0,
        max_frp=0.0 if level == "NO DATA" else 500.0,
        mean_frp=0.0 if level == "NO DATA" else 13.3,
        hotspot_density=0.0,
        high_confidence_pct=80.0,
        spread_index=0.0 if level == "NO DATA" else 200_000.0,
        risk_level=level,
        top_hotspots=[],
    )


def _override_db(tmp_db_path: str):
    """Monkey-patch config.DB_PATH so tests write to a temp DB, not the real one."""
    import config
    config.DB_PATH = tmp_db_path


PASS = "✓"
FAIL = "✗"
results = []

def check(label: str, condition: bool, detail: str = ""):
    mark = PASS if condition else FAIL
    msg = f"  {mark}  {label}"
    if detail:
        msg += f"\n       {detail}"
    print(msg)
    results.append((label, condition))

# ============================================================================
# Test 1 — Fail-open: ALERT_EMAIL_ENABLED unset or false
# ============================================================================

print("\n" + "=" * 60)
print("TEST 1: Fail-open (no credentials configured)")
print("=" * 60)

with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as _f:
    tmp_db1 = _f.name

_override_db(tmp_db1)

# Ensure env vars are absent / false
for _k in ["ALERT_EMAIL_ENABLED", "ALERT_SMTP_USER", "ALERT_SMTP_APP_PASSWORD", "ALERT_EMAIL_TO"]:
    os.environ.pop(_k, None)

# Re-import config fresh so ALERT_EMAIL_ENABLED picks up the cleared env
import importlib
import config as _config_module
importlib.reload(_config_module)
# Re-import email_alerts after config reload
import email_alerts as _ea
importlib.reload(_ea)

try:
    outcome = _ea.check_and_send_alert("Angola", _make_risk_ctx("EXTREME"))
    check(
        "ALERT_EMAIL_ENABLED unset → no exception, returns skipped-not-configured",
        outcome == "skipped-not-configured",
        f"got: {outcome!r}",
    )
    print(f"       → outcome = {outcome!r}")
except Exception as exc:
    check("ALERT_EMAIL_ENABLED unset → no exception", False, f"raised: {exc}")

# Verify direct send also fails-open
os.environ["ALERT_EMAIL_ENABLED"] = "false"
importlib.reload(_config_module)
importlib.reload(_ea)

try:
    sent = _ea.send_extreme_risk_alert("Angola", _make_risk_ctx("EXTREME"))
    check(
        "ALERT_EMAIL_ENABLED=false → send_extreme_risk_alert returns False without crash",
        sent is False,
        f"got: {sent!r}",
    )
    print(f"       → returned {sent!r}")
except Exception as exc:
    check("ALERT_EMAIL_ENABLED=false → no exception", False, f"raised: {exc}")

os.unlink(tmp_db1)

# ============================================================================
# Test 2 — Idempotency: HIGH → EXTREME → EXTREME → MEDIUM → EXTREME
# Expected sends: 2 (first EXTREME transition + second EXTREME after MEDIUM)
# ============================================================================

print("\n" + "=" * 60)
print("TEST 2: Idempotency sequence HIGH→EXTREME→EXTREME→MEDIUM→EXTREME")
print("        (expect exactly 2 sends, not 3)")
print("=" * 60)

with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as _f:
    tmp_db2 = _f.name

_override_db(tmp_db2)

# Enable alerts in config but use a spy send function that never actually connects
send_count = 0
send_log = []

def _spy_send(country, risk_ctx):
    """Replacement for send_extreme_risk_alert that records calls without SMTP."""
    global send_count
    send_count += 1
    send_log.append((country, risk_ctx.risk_level))
    return True  # simulate success

import email_alerts
importlib.reload(email_alerts)

# Patch _is_alert_enabled to return True, and the actual send to use the spy
email_alerts._is_alert_enabled = lambda: True
email_alerts.send_extreme_risk_alert = _spy_send

sequence = ["HIGH", "EXTREME", "EXTREME", "MEDIUM", "EXTREME"]
outcomes = []
for step_level in sequence:
    ctx = _make_risk_ctx(step_level)
    outcome = email_alerts.check_and_send_alert("Angola", ctx)
    outcomes.append((step_level, outcome))
    print(f"       level={step_level:8s}  outcome={outcome}")

expected_outcomes = [
    ("HIGH",    "skipped-not-extreme"),
    ("EXTREME", "sent"),
    ("EXTREME", "skipped-already-alerted"),
    ("MEDIUM",  "skipped-not-extreme"),
    ("EXTREME", "sent"),
]
check(
    "Sequence outcomes match expected exactly",
    outcomes == expected_outcomes,
    f"got: {outcomes}",
)
check(
    "Exactly 2 sends fired (not 3)",
    send_count == 2,
    f"send_count={send_count}, send_log={send_log}",
)
print(f"       → send_count={send_count}, send_log={send_log}")

os.unlink(tmp_db2)

# ============================================================================
# Test 3 — Message construction + local SMTP server
# Spin up Python's smtpd.DebuggingServer on localhost:10025, send a test
# message through it, capture the raw DATA to prove the message is valid.
# ============================================================================

print("\n" + "=" * 60)
print("TEST 3: Message construction + local mock SMTP server")
print("=" * 60)

# First, verify _build_message produces a structurally valid MIME object
import email_alerts as _ea_msg
importlib.reload(_ea_msg)

# Temporarily configure SMTP_USER / EMAIL_TO for message construction
_config_module.ALERT_SMTP_USER = "sender@example.com"
_config_module.ALERT_EMAIL_TO  = "recipient@example.com"
_config_module.ALERT_SMTP_HOST = "127.0.0.1"
_config_module.ALERT_SMTP_PORT = "10025"
_config_module.ALERT_EMAIL_ENABLED = "true"
_config_module.ALERT_SMTP_APP_PASSWORD = "testpassword"

# Reload email_alerts so it picks up the config changes
importlib.reload(_ea_msg)

ctx_extreme = _make_risk_ctx("EXTREME")
msg_obj = _ea_msg._build_message("Angola", ctx_extreme)

check(
    "MIMEMultipart subject contains 'EXTREME' and country name",
    "EXTREME" in msg_obj["Subject"] and "Angola" in msg_obj["Subject"],
    f"Subject: {msg_obj['Subject']!r}",
)
check(
    "From header set",
    msg_obj["From"] == "sender@example.com",
    f"From: {msg_obj['From']!r}",
)
check(
    "To header set",
    msg_obj["To"] == "recipient@example.com",
    f"To: {msg_obj['To']!r}",
)

# get_payload(decode=True) decodes base64/quoted-printable automatically.
_raw_bytes = msg_obj.get_payload(0).get_payload(decode=True)
body_payload = _raw_bytes.decode("utf-8") if _raw_bytes else ""
check(
    "Body contains country name",
    "Angola" in body_payload,
    f"Body snippet: {body_payload[:80]!r}",
)
check(
    "Body contains dashboard URL",
    "streamlit.app" in body_payload,
    f"URL found: {'streamlit.app' in body_payload}",
)
check(
    "Body contains fire count",
    "300" in body_payload,
    f"fire count snippet: {body_payload[:200]!r}",
)

print(f"\n       Message preview (first 400 chars of body):")
print(textwrap.indent(body_payload[:400], "       | "))

# ── Spin up a local mock SMTP server ─────────────────────────────────────────
captured_messages = []

class _CaptureSMTP(smtpd.SMTPServer):
    def process_message(self, peer, mailfrom, rcpttos, data, **kwargs):
        captured_messages.append({
            "mailfrom": mailfrom,
            "rcpttos": rcpttos,
            "data": data.decode("utf-8", errors="replace") if isinstance(data, bytes) else data,
        })

MOCK_PORT = 10025

def _run_server():
    server = _CaptureSMTP(("127.0.0.1", MOCK_PORT), None)
    # Run asyncore loop briefly to accept connections
    asyncore.loop(timeout=0.1, count=100)

server_thread = threading.Thread(target=_run_server, daemon=True)
server_thread.start()
time.sleep(0.3)  # let server bind

# Patch smtplib.SMTP to skip STARTTLS (localhost has no TLS cert)
import smtplib
_orig_smtp = smtplib.SMTP

class _PlainSMTP(smtplib.SMTP):
    def starttls(self, *args, **kwargs):
        pass  # skip TLS on localhost
    def login(self, *args, **kwargs):
        pass  # skip auth on localhost

smtplib.SMTP = _PlainSMTP

try:
    sent_ok = _ea_msg.send_extreme_risk_alert("Angola", ctx_extreme)
    time.sleep(0.3)  # give asyncore loop time to process

    check(
        "send_extreme_risk_alert returns True with mock server",
        sent_ok is True,
        f"got: {sent_ok!r}",
    )
    check(
        "Mock SMTP server captured exactly 1 message",
        len(captured_messages) == 1,
        f"captured_messages count: {len(captured_messages)}",
    )
    if captured_messages:
        cap = captured_messages[0]
        check(
            "Captured message From matches ALERT_SMTP_USER",
            cap["mailfrom"] == "sender@example.com",
            f"mailfrom: {cap['mailfrom']!r}",
        )
        check(
            "Captured message To matches ALERT_EMAIL_TO",
            "recipient@example.com" in cap["rcpttos"],
            f"rcpttos: {cap['rcpttos']!r}",
        )
        check(
            "Captured DATA contains EXTREME subject",
            "EXTREME" in cap["data"],
            f"data snippet: {cap['data'][:200]!r}",
        )
        print(f"\n       Captured SMTP DATA (first 500 chars):")
        print(textwrap.indent(cap["data"][:500], "       | "))
except Exception as exc:
    check("Mock SMTP send completes without exception", False, f"raised: {exc}")
finally:
    smtplib.SMTP = _orig_smtp

# ============================================================================
# Test 4 — Failed send returns "failed", NOT "skipped-not-configured",
#           and alert_state is NOT recorded (so next cycle retries)
# ============================================================================

print("\n" + "=" * 60)
print("TEST 4: SMTP failure returns 'failed' and leaves state un-recorded")
print("=" * 60)

with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as _f:
    tmp_db4 = _f.name

_override_db(tmp_db4)

import email_alerts as _ea4
importlib.reload(_ea4)

# _is_alert_enabled → True, but send → False (simulates wrong password, etc.)
_ea4._is_alert_enabled = lambda: True
_ea4.send_extreme_risk_alert = lambda country, risk_ctx: False  # always fails

outcome_failed = _ea4.check_and_send_alert("Angola", _make_risk_ctx("EXTREME"))
check(
    "SMTP failure → outcome is 'failed' (not 'skipped-not-configured')",
    outcome_failed == "failed",
    f"got: {outcome_failed!r}",
)

# Confirm state was NOT written to alert_state — so next cycle retries
conn4 = sqlite3.connect(tmp_db4)
row4 = conn4.execute(
    "SELECT last_alerted_level FROM alert_state WHERE country = ?", ("Angola",)
).fetchone()
conn4.close()
check(
    "alert_state NOT recorded after failed send (retry will fire next cycle)",
    row4 is None,
    f"row in alert_state: {dict(row4) if row4 else None!r}",
)
print(f"       → outcome={outcome_failed!r}, alert_state row={row4}")

os.unlink(tmp_db4)

# ============================================================================
# Summary
# ============================================================================

print("\n" + "=" * 60)
total = len(results)
passed = sum(1 for _, ok in results if ok)
failed = total - passed
print(f"RESULTS: {passed}/{total} passed  ({failed} failed)")
print("=" * 60)
if failed:
    print("\nFailed checks:")
    for label, ok in results:
        if not ok:
            print(f"  {FAIL} {label}")
sys.exit(0 if failed == 0 else 1)
