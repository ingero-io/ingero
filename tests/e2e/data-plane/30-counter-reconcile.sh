#!/usr/bin/env bash
# Test 30: Counter reconciliation under sustained load (data-plane gate 2).
#
# At the end of a 30-min deterministic workload, captures FOUR-stage counts
# plus an external mirror, then asserts:
#   - |agent - fleet| / agent < 5%       (covers adaptive sampling)
#   - |fleet - echo|  / fleet < 1%        (processor stochastic drops)
#   - |echo  - mcp|   / echo  == 0        (HARD: any difference is a typo
#                                          or window-edge bug)
#   - |fleet - external| / fleet < 5%     (parallel exporter mirror)
#
# Hardware: 1xA10 + 1 multi-GPU node. Single-A10 acceptable for a smoke
# pass (the 2D/Peer rows are silent, not asserted here).
#
# Invoke:
#   sudo bash tests/e2e/data-plane/30-counter-reconcile.sh
#
# Optional env:
#   INGERO_BIN
#   ECHO_DB           default /var/lib/ingero/echo.db
#   ECHO_MCP_URL      default http://127.0.0.1:8080
#   FLEET_DBG_LOG     fleet OTLP debug exporter log (or omit to skip stage 2)
#   EXT_OTLP_PORT     default 4319
#   DURATION_S        default 1800 (30 min). Set 300 for smoke pass.
#
# Expected runtime: DURATION_S + ~120s.
set -euo pipefail
. "$(dirname "$0")/../_lib.sh"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo ./ingero)}"
ECHO_DB="${ECHO_DB:-/var/lib/ingero/echo.db}"
ECHO_MCP_URL="${ECHO_MCP_URL:-http://127.0.0.1:8080}"
FLEET_DBG_LOG="${FLEET_DBG_LOG:-}"
EXT_OTLP_PORT="${EXT_OTLP_PORT:-4319}"
DURATION_S="${DURATION_S:-1800}"

[[ -x "$INGERO_BIN" ]] || { echo "FAIL: agent missing at $INGERO_BIN"; exit 1; }
command -v sqlite3 >/dev/null || { echo "FAIL: sqlite3 missing"; exit 1; }
command -v docker >/dev/null || { echo "FAIL: docker missing"; exit 1; }

WORK=$(mktemp -d)
AGENT_PID=""
WL_PID=""
EXT_COLLECTOR="ingero-ext-test30-$$"

cleanup() {
  set +e
  kill_agent
  [[ -n "$WL_PID" ]] && kill "$WL_PID" 2>/dev/null
  docker rm -f "$EXT_COLLECTOR" >/dev/null 2>&1
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "=== Test 30: counter-reconcile (DURATION_S=$DURATION_S) ==="

T0=$(date -u +%s)

echo "==> [setup] Boot external mirror"
cat > "$WORK/ext.yaml" <<YAML
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:$EXT_OTLP_PORT
exporters:
  debug:
    verbosity: detailed
service:
  pipelines:
    metrics:
      receivers: [otlp]
      exporters: [debug]
YAML
docker run -d --rm --name "$EXT_COLLECTOR" --network host \
  -v "$WORK/ext.yaml:/etc/otelcol-contrib/config.yaml:ro" \
  otel/opentelemetry-collector-contrib:latest \
  --config=/etc/otelcol-contrib/config.yaml >/dev/null
sleep 4

echo "==> [setup] Start deterministic workload (memcpy_stress)"
python3 "$REPO_ROOT/tests/workloads/synthetic/memcpy_stress.py" >"$WORK/wl.log" 2>&1 &
WL_PID=$!

echo "==> [setup] Boot agent with debug + dual export"
sudo "$INGERO_BIN" trace --record --db "$WORK/trace.db" \
  --duration "${DURATION_S}s" \
  --debug \
  --otlp localhost:4318 \
  --otlp-mirror "localhost:$EXT_OTLP_PORT" \
  >"$WORK/agent.log" 2>&1 &
AGENT_PID=$!

# Wait for the agent's --duration window.
wait "$AGENT_PID" || true
T1=$(date -u +%s)
sleep 5
docker logs "$EXT_COLLECTOR" >"$WORK/ext.log" 2>&1

echo "==> [count] Capture stage counts"
# Agent emit count (debug line "OTLP: pushing N").
AGENT=$(awk '/OTLP: pushing/ {n+=$NF} END{print n+0}' "$WORK/agent.log")
# Fleet receive (optional; if not provided, treat as agent count).
if [[ -n "$FLEET_DBG_LOG" && -f "$FLEET_DBG_LOG" ]]; then
  FLEET=$(awk '/Metrics #datapoints/ {n+=$NF} END{print n+0}' "$FLEET_DBG_LOG")
else
  echo "WARN: FLEET_DBG_LOG not provided; treating fleet=agent"
  FLEET="$AGENT"
fi
# Echo store rows in window.
if [[ -f "$ECHO_DB" ]]; then
  ECHO=$(sqlite3 "$ECHO_DB" \
    "SELECT count(*) FROM events WHERE timestamp BETWEEN $T0 AND $T1" 2>/dev/null || echo 0)
else
  echo "FAIL: ECHO_DB missing at $ECHO_DB"
  exit 1
fi
# MCP query for the same SELECT via run_analysis.
MCP_RESP=$(curl -fsS -X POST -H 'Content-Type: application/json' \
  --data "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"run_analysis\",\"arguments\":{\"sql\":\"SELECT count(*) FROM events WHERE timestamp BETWEEN $T0 AND $T1\"}}}" \
  "$ECHO_MCP_URL/jsonrpc" 2>/dev/null || true)
MCP=$(echo "$MCP_RESP" | grep -oE '[0-9]+' | head -1 || echo 0)
# External mirror datapoint count from debug exporter.
EXT=$(awk '/Metrics #datapoints/ {n+=$NF} END{print n+0}' "$WORK/ext.log")

echo "agent=$AGENT  fleet=$FLEET  echo=$ECHO  mcp=$MCP  external=$EXT"

assert_ratio() {
  local label="$1" a="$2" b="$3" tol="$4" hard="$5"
  if [[ -z "$a" || "$a" == "0" ]]; then
    echo "FAIL: $label denominator is zero (a=$a)"
    return 1
  fi
  local diff=$(awk -v x="$a" -v y="$b" 'BEGIN { d=(x>y?x-y:y-x)/x; printf "%.5f", d }')
  local ok=$(awk -v d="$diff" -v t="$tol" 'BEGIN { print (d <= t) ? 1 : 0 }')
  if [[ "$ok" == "1" ]]; then
    echo "OK: $label diff=$diff <= $tol"
  else
    echo "FAIL: $label diff=$diff > $tol (hard=$hard)"
    return 1
  fi
}

FAILED=0
assert_ratio "|agent-fleet|/agent"     "$AGENT" "$FLEET" 0.05 soft || FAILED=$((FAILED+1))
assert_ratio "|fleet-echo|/fleet"      "$FLEET" "$ECHO"  0.01 soft || FAILED=$((FAILED+1))
# echo == mcp is HARD (zero tolerance).
if [[ "$ECHO" != "$MCP" ]]; then
  echo "FAIL: |echo-mcp| HARD: echo=$ECHO mcp=$MCP must be equal"
  FAILED=$((FAILED+1))
else
  echo "OK: echo == mcp (HARD)"
fi
assert_ratio "|fleet-external|/fleet"  "$FLEET" "$EXT"   0.05 soft || FAILED=$((FAILED+1))

if (( FAILED > 0 )); then
  echo "FAIL: $FAILED reconcile assertion(s)"
  exit 1
fi
echo "PASS: counter-reconcile"
