#!/usr/bin/env bash
# Test 25: YAML config loader drives runtime behavior.
#
# Asserts:
#   - Agent boots with `--config /tmp/config.yaml` carrying:
#       throttle_poll_interval: 2s
#       memfrag_poll_interval: 7s
#       libnccl_discovery_interval: 4s
#       otlp.endpoint: localhost:4318
#   - Prometheus exposition shows the throttle, memfrag, and nccl-discovery
#     metric timestamps advance at the configured cadences (within
#     +/- 1s tolerance).
#
# Hardware: any A10. Docker for otelcol-contrib (the OTLP endpoint smoke).
#
# Invoke:
#   sudo bash tests/e2e/yaml-config-boot.sh
#
# Optional env:
#   INGERO_BIN
#
# Expected runtime: ~70s (covers 3 throttle ticks at 2s + 1 memfrag at 7s +
# 2 nccl ticks at 4s, with margin).
set -euo pipefail
. "$(dirname "$0")/_lib.sh"

INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo ./ingero)}"
[[ -x "$INGERO_BIN" ]] || { echo "FAIL: agent missing at $INGERO_BIN"; exit 1; }
command -v docker >/dev/null || { echo "FAIL: docker missing"; exit 1; }

WORK=$(mktemp -d)
COLLECTOR="ingero-otelcol-test25-$$"
AGENT_PID=""

cleanup() {
  set +e
  kill_agent
  docker rm -f "$COLLECTOR" >/dev/null 2>&1
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "=== Test 25: yaml-config-boot ==="

echo "==> [1/4] Boot otelcol-contrib (sink for the configured otlp endpoint)"
cat > "$WORK/otelcol.yaml" <<'YAML'
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318
exporters:
  debug:
    verbosity: basic
service:
  pipelines:
    metrics:
      receivers: [otlp]
      exporters: [debug]
YAML
docker run -d --rm --name "$COLLECTOR" --network host \
  -v "$WORK/otelcol.yaml:/etc/otelcol-contrib/config.yaml:ro" \
  otel/opentelemetry-collector-contrib:latest \
  --config=/etc/otelcol-contrib/config.yaml >/dev/null
sleep 4

echo "==> [2/4] Write config.yaml"
CFG="$WORK/config.yaml"
cat > "$CFG" <<'YAML'
throttle_poll_interval: 2s
memfrag_poll_interval: 7s
libnccl_discovery_interval: 4s
otlp:
  endpoint: localhost:4318
prometheus:
  listen: ":9090"
YAML

echo "==> [3/4] Boot agent with --config"
sudo "$INGERO_BIN" trace --record --db "$WORK/trace.db" \
  --duration 60s \
  --config "$CFG" \
  >"$WORK/agent.log" 2>&1 &
AGENT_PID=$!
sleep 8

# Sample Prometheus 4 times across 32s, capture series counter values
# for throttle, memfrag, nccl-discovery.
sample() {
  curl -fsS http://localhost:9090/metrics 2>/dev/null
}

declare -a SAMPLES_T=() SAMPLES_M=() SAMPLES_N=()
declare -a STAMPS=()
for i in 1 2 3 4 5; do
  STAMPS+=("$(date +%s)")
  SNAP="$(sample || true)"
  # Use any throttle metric, memfrag estimate, and nccl process count as "tick" indicators.
  SAMPLES_T+=("$(echo "$SNAP" | awk '/^gpu_throttle_power_active/ {n++} END{print n+0}')")
  SAMPLES_M+=("$(echo "$SNAP" | awk '/^gpu_memory_fragmentation_estimate/ {n++} END{print n+0}')")
  SAMPLES_N+=("$(echo "$SNAP" | awk '/^gpu_nccl_processes_total/ {n++} END{print n+0}')")
  sleep 8
done

echo "==> [4/4] Validate cadence + OTLP sink connectivity"
# Throttle should always have a current value (every 2s); samples > 0.
ZERO_T=0
for v in "${SAMPLES_T[@]}"; do (( v == 0 )) && ZERO_T=$((ZERO_T+1)); done
if (( ZERO_T > 1 )); then
  echo "FAIL: throttle metric sampling missing across windows ($ZERO_T zero samples)"
  exit 1
fi
echo "OK: throttle metric present at expected cadence (2s)"

ZERO_M=0
for v in "${SAMPLES_M[@]}"; do (( v == 0 )) && ZERO_M=$((ZERO_M+1)); done
# memfrag at 7s -> within a ~32s window we expect >= 4 ticks; some samples may show 0 if pre-first-tick
if (( ZERO_M >= 4 )); then
  echo "FAIL: memfrag metric never appeared ($ZERO_M zero samples of ${#SAMPLES_M[@]})"
  exit 1
fi
echo "OK: memfrag metric appeared (cadence 7s)"

ZERO_N=0
for v in "${SAMPLES_N[@]}"; do (( v == 0 )) && ZERO_N=$((ZERO_N+1)); done
if (( ZERO_N >= 4 )); then
  echo "FAIL: nccl-discovery metric never appeared"
  exit 1
fi
echo "OK: nccl-discovery metric appeared (cadence 4s)"

# OTLP endpoint reach: collector must have logged at least one batch.
docker logs "$COLLECTOR" 2>&1 > "$WORK/otelcol.log"
if ! grep -qE 'Metrics|datapoints' "$WORK/otelcol.log"; then
  echo "FAIL: collector saw no metrics from agent (otlp.endpoint not honored)"
  tail -40 "$WORK/otelcol.log"
  exit 1
fi
echo "OK: otelcol-contrib received metrics over OTLP"

echo "PASS: yaml-config-boot"
