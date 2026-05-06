#!/usr/bin/env bash
# Test 23: pagerduty_trigger MCP gate behavior.
#
# Asserts:
#   - With default flags, calling pagerduty_trigger via JSON-RPC returns an
#     error mentioning "--enable-mcp-pagerduty".
#   - With --enable-mcp-pagerduty, the call succeeds and a fake PD endpoint
#     receives exactly one POST with the expected payload shape.
#
# Hardware: any host. No GPU needed.
#
# Invoke:
#   bash tests/e2e/pagerduty-trigger-gate.sh
#
# Optional env:
#   INGERO_BIN
#   MCP_PORT      port the agent serves MCP JSON-RPC on (default 8765)
#   PD_PORT       fake PD endpoint port (default 8766)
#
# Expected runtime: ~30s.
set -euo pipefail

INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo ./ingero)}"
MCP_PORT="${MCP_PORT:-8765}"
PD_PORT="${PD_PORT:-8766}"
[[ -x "$INGERO_BIN" ]] || { echo "FAIL: agent missing at $INGERO_BIN"; exit 1; }

WORK=$(mktemp -d)
PD_PID=""
MCP_PID=""

cleanup() {
  set +e
  [[ -n "$MCP_PID" ]] && kill "$MCP_PID" 2>/dev/null
  [[ -n "$PD_PID" ]] && kill "$PD_PID" 2>/dev/null
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "=== Test 23: pagerduty-trigger-gate ==="

echo "==> [1/4] Start fake PagerDuty endpoint on :$PD_PORT"
cat > "$WORK/fake_pd.py" <<PY
import http.server, json, sys, os
LOG = os.environ['PD_LOG']
class H(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        n = int(self.headers.get('Content-Length','0'))
        body = self.rfile.read(n).decode('utf-8','replace')
        with open(LOG, 'a') as fp:
            fp.write(body + "\n")
        self.send_response(202)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        self.wfile.write(b'{"status":"success","dedup_key":"test"}')
    def log_message(self, *_): pass
PD_LOG=os.environ['PD_LOG']
http.server.HTTPServer(('127.0.0.1', int(sys.argv[1])), H).serve_forever()
PY
PD_LOG="$WORK/pd.log"
PD_LOG="$PD_LOG" python3 "$WORK/fake_pd.py" "$PD_PORT" &
PD_PID=$!
sleep 1

mcp_call() {
  local payload="$1"
  curl -fsS -X POST -H 'Content-Type: application/json' \
    --data "$payload" "http://127.0.0.1:$MCP_PORT/jsonrpc" 2>&1
}

PD_URL="http://127.0.0.1:$PD_PORT/v2/enqueue"

echo "==> [2/4] Boot agent MCP without --enable-mcp-pagerduty"
"$INGERO_BIN" mcp \
  --listen "127.0.0.1:$MCP_PORT" \
  --pagerduty-url "$PD_URL" \
  --pagerduty-key dummy-routing-key \
  >"$WORK/mcp.log" 2>&1 &
MCP_PID=$!
for _ in $(seq 1 20); do
  if curl -fsS "http://127.0.0.1:$MCP_PORT/jsonrpc" -X POST \
       --data '{"jsonrpc":"2.0","id":0,"method":"tools/list"}' >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

REQ='{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"pagerduty_trigger","arguments":{"summary":"test","severity":"warning"}}}'
RESP=$(mcp_call "$REQ" || true)
echo "$RESP" > "$WORK/resp1.json"
if ! echo "$RESP" | grep -qE 'enable-mcp-pagerduty'; then
  echo "FAIL: error response did not mention --enable-mcp-pagerduty"
  echo "got: $RESP"
  exit 1
fi
echo "OK: gate-closed call returned 'enable-mcp-pagerduty' guidance"

kill "$MCP_PID" 2>/dev/null
wait "$MCP_PID" 2>/dev/null || true

echo "==> [3/4] Boot agent MCP with --enable-mcp-pagerduty"
: > "$PD_LOG"
"$INGERO_BIN" mcp \
  --listen "127.0.0.1:$MCP_PORT" \
  --enable-mcp-pagerduty \
  --pagerduty-url "$PD_URL" \
  --pagerduty-key dummy-routing-key \
  >"$WORK/mcp2.log" 2>&1 &
MCP_PID=$!
for _ in $(seq 1 20); do
  if curl -fsS "http://127.0.0.1:$MCP_PORT/jsonrpc" -X POST \
       --data '{"jsonrpc":"2.0","id":0,"method":"tools/list"}' >/dev/null 2>&1; then
    break
  fi
  sleep 1
done

RESP=$(mcp_call "$REQ" || true)
echo "$RESP" > "$WORK/resp2.json"
if echo "$RESP" | grep -qiE 'error'; then
  echo "FAIL: enabled call returned error: $RESP"
  exit 1
fi
echo "OK: enabled call succeeded"

echo "==> [4/4] Verify fake PD got exactly 1 POST"
sleep 2
N=$(wc -l < "$PD_LOG" || echo 0)
if [[ "$N" != "1" ]]; then
  echo "FAIL: fake PD received $N POSTs, expected 1"
  cat "$PD_LOG"
  exit 1
fi
if ! grep -qE '(routing_key|event_action)' "$PD_LOG"; then
  echo "FAIL: PD payload missing routing_key/event_action shape"
  cat "$PD_LOG"
  exit 1
fi
echo "OK: PD endpoint received 1 POST with expected payload shape"

echo "PASS: pagerduty-trigger-gate"
