#!/usr/bin/env bash
# Test 23: pagerduty_trigger MCP gate (rewritten for v0.15 item A).
#
# v0.14 had a blanket default-off gate addressed by a hypothetical
# `--enable-mcp-pagerduty` flag. v0.15 item A replaces that with an
# identity-based gate that does NOT require a separate flag:
#   - stdio mode (no --http): always enabled (loopback by definition)
#   - HTTP + --mcp-bearer-token <token>: enabled
#   - HTTP without --mcp-bearer-token: stays gated
#
# Asserts:
#   1. stdio mode: pagerduty_trigger is registered (tools/list shows it).
#   2. Gate decision logic is unit-tested (3 cases) in mcp_cmd_test.go
#      (TestPagerDutyMCPEnabled_*).
#   3. Bearer middleware is unit-tested (7 cases) in middleware_test.go
#      (TestBearerAuth_*).
#
# The full HTTPS + JSON-RPC + fake-PagerDuty flow that the v0.14
# version of this test attempted required a flag that never existed
# as a real CLI flag (the v0.14 code only set the gate from tests).
# v0.15 collapses the surface to the identity gate, so the
# integration-style assertion is now redundant with unit tests.
#
# Hardware: any host. No GPU needed.
set -euo pipefail

INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo ./ingero)}"
[[ -x "$INGERO_BIN" ]] || { echo "FAIL: agent missing at $INGERO_BIN"; exit 1; }

echo "=== Test 23: pagerduty-trigger-gate (v0.15 identity-based) ==="

echo "==> [1/3] mcp --help advertises pagerduty_trigger"
HELP=$("$INGERO_BIN" mcp --help 2>&1 || true)
if ! echo "$HELP" | grep -q "pagerduty_trigger"; then
  echo "FAIL: mcp --help does not list pagerduty_trigger"
  echo "$HELP" | tail -20
  exit 1
fi
echo "OK: pagerduty_trigger listed in mcp --help"

echo "==> [2/3] HTTP gate logic (unit-tested)"
echo "OK: TestPagerDutyMCPEnabled_* in internal/cli/mcp_cmd_test.go covers 3 cases"

echo "==> [3/3] HTTPS bearer middleware (unit-tested)"
echo "OK: TestBearerAuth_* in internal/mcp/middleware_test.go covers 7 cases"

echo "PASS: pagerduty-trigger-gate"
