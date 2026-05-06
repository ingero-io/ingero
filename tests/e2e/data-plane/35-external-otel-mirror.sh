#!/usr/bin/env bash
# Test 35: Parallel external OTEL pipeline (data-plane gate 7).
#
# - Configures fleet with TWO exporters: Echo (existing path) + sidecar
#   otelcol-contrib running `debug` exporter.
# - Runs a 5-minute workload.
#
# Asserts:
#   - echo_store.count == otelcol_debug.count within 1% tolerance
#   - No Echo-side filter silently masks data not seen externally.
#
# Hardware: any A10.
#
# Invoke:
#   sudo bash tests/e2e/data-plane/35-external-otel-mirror.sh
#
# Optional env:
#   INGERO_BIN
#   ECHO_DB           default /var/lib/ingero/echo.db
#   EXT_OTLP_PORT     default 4319
#   DURATION_S        default 300
#
# Expected runtime: DURATION_S + ~30s.
set -euo pipefail
. "$(dirname "$0")/../_lib.sh"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo ./ingero)}"
ECHO_DB="${ECHO_DB:-/var/lib/ingero/echo.db}"
EXT_OTLP_PORT="${EXT_OTLP_PORT:-4319}"
DURATION_S="${DURATION_S:-300}"

[[ -x "$INGERO_BIN" ]] || { echo "FAIL: agent missing at $INGERO_BIN"; exit 1; }
command -v sqlite3 >/dev/null || { echo "FAIL: sqlite3 missing"; exit 1; }
command -v docker >/dev/null || { echo "FAIL: docker missing"; exit 1; }

WORK=$(mktemp -d)
AGENT_PID=""
WL_PID=""
EXT_COLLECTOR="ingero-ext-test35-$$"

cleanup() {
  set +e
  kill_agent
  [[ -n "$WL_PID" ]] && kill "$WL_PID" 2>/dev/null
  docker rm -f "$EXT_COLLECTOR" >/dev/null 2>&1
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "=== Test 35: external-otel-mirror (DURATION_S=$DURATION_S) ==="

echo "==> [1/3] Boot external mirror with debug exporter"
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

echo "==> [2/3] Boot agent + workload, agent dual-exports to Echo + mirror"
T0=$(date -u +%s)
python3 "$REPO_ROOT/tests/workloads/synthetic/memcpy_stress.py" >"$WORK/wl.log" 2>&1 &
WL_PID=$!
sudo "$INGERO_BIN" trace --record --db "$WORK/trace.db" \
  --duration "${DURATION_S}s" \
  --otlp localhost:4318 \
  --otlp-mirror "localhost:$EXT_OTLP_PORT" \
  >"$WORK/agent.log" 2>&1 &
AGENT_PID=$!
wait "$AGENT_PID" || true
T1=$(date -u +%s)
sleep 5
docker logs "$EXT_COLLECTOR" >"$WORK/ext.log" 2>&1

echo "==> [3/3] Compare echo store vs external"
if [[ ! -f "$ECHO_DB" ]]; then
  echo "FAIL: ECHO_DB missing at $ECHO_DB"
  exit 1
fi
ECHO=$(sqlite3 "$ECHO_DB" \
  "SELECT count(*) FROM events WHERE timestamp BETWEEN $T0 AND $T1" 2>/dev/null || echo 0)
EXT=$(awk '/Metrics #datapoints/ {n+=$NF} END{print n+0}' "$WORK/ext.log")

echo "echo=$ECHO ext=$EXT"
if [[ -z "$ECHO" || "$ECHO" == "0" ]]; then
  echo "FAIL: echo count is zero (no data flowed via Echo path)"
  exit 1
fi
if [[ -z "$EXT" || "$EXT" == "0" ]]; then
  echo "FAIL: ext count is zero (mirror saw nothing)"
  exit 1
fi
DIFF=$(awk -v a="$ECHO" -v b="$EXT" 'BEGIN { d=(a>b?a-b:b-a)/a; printf "%.4f", d }')
OK=$(awk -v d="$DIFF" 'BEGIN { print (d <= 0.01) ? 1 : 0 }')
if [[ "$OK" != "1" ]]; then
  echo "FAIL: |echo-ext|/echo = $DIFF > 0.01 (potential silent Echo-side filter)"
  exit 1
fi
echo "OK: echo == ext within 1% (diff=$DIFF)"

echo "PASS: external-otel-mirror"
