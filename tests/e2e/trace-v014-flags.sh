#!/usr/bin/env bash
# Test 2: First trace with v0.14 polling + discovery flags.
#
# Asserts:
#   - Agent runs 30s with all v0.14 flags wired, no panic / fatal.
#   - otelcol-contrib debug exporter logs >= 1 datapoint each for:
#       gpu.cuda.operation.duration
#       gpu.throttle.power_active
#       gpu.memory.fragmentation_estimate
#   - The agent's local trace DB has rows in the run window.
#
# Hardware: any A10 with NVIDIA driver. Docker required (for otelcol-contrib).
#
# Invoke:
#   sudo bash tests/e2e/trace-v014-flags.sh
#
# Optional env:
#   INGERO_BIN       path to the agent binary (default: ./ingero on PATH)
#   WORKLOAD_PY      path to a CUDA workload that runs >= 30s
#                    (default: tests/workloads/training/gpt2_stress.py)
#
# Expected runtime: ~75s (60s warm + 30s trace + teardown).
set -euo pipefail

INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo ./ingero)}"
REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
WORKLOAD_PY="${WORKLOAD_PY:-$REPO_ROOT/tests/workloads/training/gpt2_stress.py}"

if [[ ! -x "$INGERO_BIN" ]]; then
  echo "FAIL: agent binary not found at $INGERO_BIN"
  exit 1
fi
if ! command -v docker >/dev/null; then
  echo "FAIL: docker not installed"
  exit 1
fi
if ! command -v nvidia-smi >/dev/null; then
  echo "FAIL: nvidia-smi missing"
  exit 1
fi

WORK=$(mktemp -d)
COLLECTOR="ingero-otelcol-test2-$$"
WORKLOAD_PID=""

cleanup() {
  set +e
  if [[ -n "$WORKLOAD_PID" ]] && kill -0 "$WORKLOAD_PID" 2>/dev/null; then
    kill "$WORKLOAD_PID" 2>/dev/null
    wait "$WORKLOAD_PID" 2>/dev/null
  fi
  docker rm -f "$COLLECTOR" >/dev/null 2>&1
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "=== Test 2: trace-v014-flags ==="

echo "==> [1/5] Write otelcol-contrib config"
cat > "$WORK/otelcol-config.yaml" <<'YAML'
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318
      grpc:
        endpoint: 0.0.0.0:4317
exporters:
  debug:
    verbosity: detailed
service:
  pipelines:
    metrics:
      receivers: [otlp]
      exporters: [debug]
    traces:
      receivers: [otlp]
      exporters: [debug]
    logs:
      receivers: [otlp]
      exporters: [debug]
YAML

echo "==> [2/5] Boot otelcol-contrib"
docker run -d --rm \
  --name "$COLLECTOR" \
  --network host \
  -v "$WORK/otelcol-config.yaml:/etc/otelcol-contrib/config.yaml:ro" \
  otel/opentelemetry-collector-contrib:latest \
  --config=/etc/otelcol-contrib/config.yaml >/dev/null

# Wait for the OTLP HTTP listener to come up.
for _ in $(seq 1 20); do
  if docker logs "$COLLECTOR" 2>&1 | grep -q "Everything is ready"; then
    break
  fi
  sleep 1
done

echo "==> [3/5] Start CUDA workload"
if [[ -f "$WORKLOAD_PY" ]]; then
  python3 "$WORKLOAD_PY" >"$WORK/workload.log" 2>&1 &
  WORKLOAD_PID=$!
  sleep 5
else
  echo "WARN: workload script $WORKLOAD_PY missing; agent will run against idle GPU"
fi

echo "==> [4/5] Run agent for 30s with v0.14 flags"
DB="$WORK/trace.db"
sudo "$INGERO_BIN" trace --record --db "$DB" \
  --duration 30s \
  --throttle-poll-interval 5s \
  --memfrag-poll-interval 5s \
  --libnccl-discovery-interval 5s \
  --otlp localhost:4318 >"$WORK/agent.log" 2>&1 || true

if grep -qiE 'panic|fatal error|runtime error' "$WORK/agent.log"; then
  echo "FAIL: panic / fatal in agent log"
  tail -50 "$WORK/agent.log"
  exit 1
fi
echo "OK: agent ran 30s with no panic"

echo "==> [5/5] Assert metric datapoints in collector"
sleep 3
COLL_LOG="$WORK/otelcol.log"
docker logs "$COLLECTOR" >"$COLL_LOG" 2>&1

declare -a REQUIRED=(
  "gpu.cuda.operation.duration"
  "gpu.throttle.power_active"
  "gpu.memory.fragmentation_estimate"
)
MISSING=0
for m in "${REQUIRED[@]}"; do
  if grep -q "$m" "$COLL_LOG"; then
    echo "OK: collector saw $m"
  else
    echo "FAIL: collector did not see $m"
    MISSING=$((MISSING+1))
  fi
done
if (( MISSING > 0 )); then
  echo "==> tail of collector log:"
  tail -120 "$COLL_LOG"
  exit 1
fi

# Trace DB should have rows.
if ! command -v sqlite3 >/dev/null; then
  echo "WARN: sqlite3 missing; skipping trace DB row assertion"
else
  ROWS=$(sqlite3 "$DB" "SELECT count(*) FROM events" 2>/dev/null || echo 0)
  if (( ROWS == 0 )); then
    echo "FAIL: trace DB has zero events"
    exit 1
  fi
  echo "OK: trace DB has $ROWS events"
fi

echo "PASS: trace-v014-flags"
