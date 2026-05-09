#!/usr/bin/env bash
# Tests 8 + 9: OTLP push round-trip + Prometheus pull endpoint with v0.14
# metric names.
#
# Asserts:
#   - Test 8: All 14 v0.14 metric names appear at least once in the
#             otelcol-contrib debug exporter log when the agent pushes
#             over OTLP HTTP for 60s under a real CUDA workload.
#   - Test 9: Same metric names also appear in the Prometheus text-format
#             exposition served by `--prometheus :9090`.
#
# Hardware: any A10. Docker required (otelcol-contrib).
#
# Invoke:
#   sudo bash tests/e2e/otlp-receiver-roundtrip.sh
#
# Optional env:
#   INGERO_BIN, WORKLOAD_PY (same defaults as trace-v014-flags.sh)
#
# Expected runtime: ~110s.
set -euo pipefail
. "$(dirname "$0")/_lib.sh"

INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo ./ingero)}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORKLOAD_PY="${WORKLOAD_PY:-$REPO_ROOT/tests/workloads/synthetic/memcpy_stress.py}"

[[ -x "$INGERO_BIN" ]] || { echo "FAIL: agent missing at $INGERO_BIN"; exit 1; }
command -v docker >/dev/null || { echo "FAIL: docker missing"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "FAIL: nvidia-smi missing"; exit 1; }

WORK=$(mktemp -d)
COLLECTOR="ingero-otelcol-test89-$$"
WORKLOAD_PID=""
AGENT_PID=""

cleanup() {
  set +e
  kill_agent
  [[ -n "$WORKLOAD_PID" ]] && kill "$WORKLOAD_PID" 2>/dev/null
  docker rm -f "$COLLECTOR" >/dev/null 2>&1
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "=== Tests 8 + 9: otlp-receiver-roundtrip ==="

echo "==> [1/5] Boot otelcol-contrib"
cat > "$WORK/otelcol-config.yaml" <<'YAML'
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318
exporters:
  debug:
    verbosity: detailed
service:
  pipelines:
    metrics:
      receivers: [otlp]
      exporters: [debug]
YAML
docker run -d --rm --name "$COLLECTOR" --network host \
  -v "$WORK/otelcol-config.yaml:/etc/otelcol-contrib/config.yaml:ro" \
  otel/opentelemetry-collector-contrib:latest \
  --config=/etc/otelcol-contrib/config.yaml >/dev/null
sleep 5

echo "==> [2/5] Start CUDA workload"
if [[ -f "$WORKLOAD_PY" ]]; then
  python3 "$WORKLOAD_PY" >"$WORK/workload.log" 2>&1 &
  WORKLOAD_PID=$!
fi

echo "==> [3/5] Run agent for 60s with OTLP push + Prometheus pull"
sudo "$INGERO_BIN" trace --record --db "$WORK/trace.db" \
  --duration 60s \
  --throttle-poll-interval 5s \
  --memfrag-poll-interval 5s \
  --libnccl-discovery-interval 5s \
  --otlp localhost:4318 \
  --prometheus :9090 \
  >"$WORK/agent.log" 2>&1 &
AGENT_PID=$!

# Mid-run, scrape Prometheus.
sleep 45
PROM="$WORK/prom.txt"
curl -fsS http://localhost:9090/metrics > "$PROM" || {
  echo "FAIL: could not scrape Prometheus on :9090"
  cat "$WORK/agent.log" | tail -40
  exit 1
}

wait "$AGENT_PID" || true
sleep 3

echo "==> [4/5] Inspect collector log"
COLL_LOG="$WORK/otelcol.log"
docker logs "$COLLECTOR" >"$COLL_LOG" 2>&1

REQUIRED=(
  "gpu.cuda.operation.duration"
  "gpu.cuda.operation.count"
  "gpu.throttle.power_active"
  "gpu.throttle.thermal_active"
  "gpu.throttle.sw_active"
  "gpu.throttle.hw_active"
  "gpu.nccl.process_loaded"
  "gpu.nccl.processes_total"
  "gpu.memory.used"
  "gpu.memory.free"
  "gpu.memory.total"
  "gpu.memory.fragmentation_estimate"
  "gpu.memory.process.allocated_bytes"
  "gpu.memcpy.bytes_total"
  "gpu.memcpy.duration_ms"
)

OTLP_MISS=0
for m in "${REQUIRED[@]}"; do
  if grep -q "$m" "$COLL_LOG"; then
    echo "OK: OTLP saw $m"
  else
    echo "FAIL: OTLP missing $m"
    OTLP_MISS=$((OTLP_MISS+1))
  fi
done

echo "==> [5/5] Inspect Prometheus scrape"
PROM_MISS=0
for m in "${REQUIRED[@]}"; do
  # Prometheus text format converts dots to underscores.
  pname="$(echo "$m" | tr '.' '_')"
  if grep -q "^${pname}" "$PROM" || grep -q "${pname}{" "$PROM"; then
    echo "OK: Prometheus saw $pname"
  else
    echo "FAIL: Prometheus missing $pname"
    PROM_MISS=$((PROM_MISS+1))
  fi
done

if (( OTLP_MISS > 0 || PROM_MISS > 0 )); then
  echo "FAIL: otlp_missing=$OTLP_MISS prom_missing=$PROM_MISS"
  exit 1
fi

echo "PASS: otlp-receiver-roundtrip (tests 8 + 9)"
