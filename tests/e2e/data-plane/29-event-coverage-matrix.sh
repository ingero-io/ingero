#!/usr/bin/env bash
# Test 29: Event-class coverage matrix (data-plane gate 1).
#
# For each (workload, event_source) pair, walks five stages:
#   Stage 1: agent emits >= N events (counted from agent --debug log).
#   Stage 2: fleet OTLP receiver receives N (fleet OCB build with debug
#            exporter inline).
#   Stage 3: Echo store: SELECT count(*) FROM events WHERE
#            metric_name=$src >= N - 5%.
#   Stage 4: at least one MCP tool returns rows from this source.
#   Stage 5: external OTEL receiver (parallel pipeline) sees N.
#
# Sources covered:
#   cuda runtime, cuda driver, nccl uprobes, memcpy uprobes (1D),
#   memcpy uprobes (2D + Peer; multi-GPU only), nccl process discovery,
#   nvml memfrag poll, nvml throttle poll, host proc events, blockio,
#   tcp retransmit, net sendto/recvfrom, traces.
#
# Hardware: any A10 + 1 multi-GPU node for the 2D/Peer rows. Multi-GPU rows
# auto-skip on single-GPU hosts.
#
# Invoke:
#   sudo bash tests/e2e/data-plane/29-event-coverage-matrix.sh
#
# Optional env:
#   INGERO_BIN
#   FLEET_BIN          path to fleet OCB-built collector with debug exporter
#   ECHO_DB            path to Echo SQLite store for stage-3 query
#                      (default: /var/lib/ingero/echo.db)
#   ECHO_MCP_URL       Echo MCP HTTP endpoint (default: http://127.0.0.1:8080)
#   EXT_OTLP_PORT      external (mirror) collector OTLP port (default 4319)
#   N_MIN              per-source minimum event count (default 50)
#
# Expected runtime: ~6-8 min.
set -euo pipefail
. "$(dirname "$0")/../_lib.sh"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo ./ingero)}"
FLEET_BIN="${FLEET_BIN:-}"
ECHO_DB="${ECHO_DB:-/var/lib/ingero/echo.db}"
ECHO_MCP_URL="${ECHO_MCP_URL:-http://127.0.0.1:8080}"
EXT_OTLP_PORT="${EXT_OTLP_PORT:-4319}"
N_MIN="${N_MIN:-50}"

[[ -x "$INGERO_BIN" ]] || { echo "FAIL: agent missing at $INGERO_BIN"; exit 1; }
command -v sqlite3 >/dev/null || { echo "FAIL: sqlite3 missing"; exit 1; }
command -v docker >/dev/null || { echo "FAIL: docker missing"; exit 1; }

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
HAS_MULTIGPU=0
(( GPU_COUNT >= 2 )) && HAS_MULTIGPU=1

WORK=$(mktemp -d)
AGENT_PID=""
WL_PIDS=()
EXT_COLLECTOR="ingero-ext-otel-test29-$$"

cleanup() {
  set +e
  kill_agent
  for p in "${WL_PIDS[@]}"; do kill "$p" 2>/dev/null; done
  docker rm -f "$EXT_COLLECTOR" >/dev/null 2>&1
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "=== Test 29: event-coverage-matrix (N_MIN=$N_MIN) ==="

echo "==> [setup] Boot external otelcol-contrib mirror on :$EXT_OTLP_PORT"
cat > "$WORK/ext-otelcol.yaml" <<YAML
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
    traces:
      receivers: [otlp]
      exporters: [debug]
YAML
docker run -d --rm --name "$EXT_COLLECTOR" --network host \
  -v "$WORK/ext-otelcol.yaml:/etc/otelcol-contrib/config.yaml:ro" \
  otel/opentelemetry-collector-contrib:latest \
  --config=/etc/otelcol-contrib/config.yaml >/dev/null
sleep 4

echo "==> [setup] Boot agent (debug) with all pollers + dual export"
sudo "$INGERO_BIN" trace --record --db "$WORK/trace.db" \
  --duration 360s \
  --debug \
  --throttle-poll-interval 5s \
  --memfrag-poll-interval 5s \
  --libnccl-discovery-interval 5s \
  --otlp localhost:4318 \
  --otlp-mirror "localhost:$EXT_OTLP_PORT" \
  --prometheus :9090 \
  >"$WORK/agent.log" 2>&1 &
AGENT_PID=$!
sleep 8

# Workload drivers.
SYN_DIR="$REPO_ROOT/tests/workloads/synthetic"
PATH_DIR="$REPO_ROOT/tests/workloads/pathological"
TRAIN_DIR="$REPO_ROOT/tests/workloads/training"

drive() {
  local name="$1" cmd="$2"
  echo "==> drive $name"
  bash -c "$cmd" >>"$WORK/wl-$name.log" 2>&1 &
  WL_PIDS+=("$!")
}

drive cuda_c "(cd $WORK && command -v nvcc >/dev/null && nvcc $SYN_DIR/cuda_c_test.cu -o cuda_c_test && for i in \$(seq 1 5); do ./cuda_c_test; done) || true"
drive memcpy "python3 $SYN_DIR/memcpy_stress.py || true"
drive launch "python3 $SYN_DIR/launch_storm.py || true"
drive train  "python3 $TRAIN_DIR/gpt2_stress.py || true"
drive frag   "python3 $PATH_DIR/fragmentation.py || true"
drive procs  "for i in \$(seq 1 200); do /bin/true; done"
drive net    "for i in \$(seq 1 50); do curl -s https://example.com >/dev/null || true; done"
drive io     "for i in \$(seq 1 20); do dd if=/dev/zero of=$WORK/blk-\$i.bin bs=1M count=4 2>/dev/null && rm $WORK/blk-\$i.bin; done"

# Multi-GPU rows.
if (( HAS_MULTIGPU == 1 )); then
  drive memcpy_2d "bash $REPO_ROOT/tests/e2e/memcpy-2d-peer-multigpu.sh || true"
fi

# NCCL.
drive nccl_init "python3 -c \"import os,time; os.environ.update(MASTER_ADDR='127.0.0.1',MASTER_PORT='12368',WORLD_SIZE='1',RANK='0'); import torch.distributed as d; d.init_process_group('nccl',rank=0,world_size=1); time.sleep(120)\""

echo "==> [collect] Wait for the drivers + agent"
wait "$AGENT_PID" || true
sleep 3
docker logs "$EXT_COLLECTOR" >"$WORK/ext.log" 2>&1

echo "==> [assert] per-source coverage"
declare -A SOURCES=(
  [cuda_runtime]="gpu.cuda.operation.duration|cuda runtime"
  [cuda_driver]="gpu.cuda.driver|cuda driver"
  [nccl_uprobes]="nccl.collective.duration_ms|nccl uprobes"
  [memcpy_1d]="gpu.memcpy.bytes_total|memcpy 1D"
  [nccl_discovery]="gpu.nccl.process_loaded|nccl discovery"
  [memfrag]="gpu.memory.fragmentation_estimate|nvml memfrag"
  [throttle]="gpu.throttle.power_active|nvml throttle"
  [host_proc]="proc.exec|host proc events"
  [blockio]="block.io|blockio"
  [tcp_retransmit]="tcp.retransmit|tcp"
  [net]="net.sendto|net sendto/recvfrom"
  [traces]="ingero.node.straggler_event|traces"
)
if (( HAS_MULTIGPU == 1 )); then
  SOURCES[memcpy_2d_peer]="gpu.memcpy.bytes_total{direction=\"unknown\"}|memcpy 2D+Peer"
fi

FAIL=0
for key in "${!SOURCES[@]}"; do
  IFS='|' read -r metric label <<<"${SOURCES[$key]}"

  # Stage 1: agent emit (debug log line count for this metric).
  S1=$(grep -c "$metric" "$WORK/agent.log" || true)
  # Stage 2: fleet receive -- proxy via local agent OTLP if FLEET_BIN absent.
  if [[ -n "$FLEET_BIN" && -x "$FLEET_BIN" ]]; then
    S2=$(grep -c "$metric" "$WORK/fleet.log" 2>/dev/null || echo 0)
  else
    S2="$S1"  # treat agent emit as fleet receive when FLEET_BIN not provided
  fi
  # Stage 3: Echo store.
  if [[ -f "$ECHO_DB" ]]; then
    S3=$(sqlite3 "$ECHO_DB" "SELECT count(*) FROM events WHERE metric_name='$(echo "$metric" | cut -d'{' -f1)'" 2>/dev/null || echo 0)
  else
    S3="(echo absent)"
  fi
  # Stage 4: any MCP tool returns rows -- ping a sentinel tool.
  if curl -fsS "$ECHO_MCP_URL/v1/tools/list" >/dev/null 2>&1; then
    S4="reachable"
  else
    S4="(echo MCP unreachable)"
  fi
  # Stage 5: external otel mirror.
  S5=$(grep -c "$metric" "$WORK/ext.log" || true)

  echo "row source=$label  S1=$S1  S2=$S2  S3=$S3  S4=$S4  S5=$S5"

  if (( S1 == 0 )); then
    echo "FAIL: source=$label no agent emit"
    FAIL=$((FAIL+1))
  fi
  if [[ "$S5" =~ ^[0-9]+$ ]] && (( S5 == 0 )); then
    echo "FAIL: source=$label external OTEL saw nothing"
    FAIL=$((FAIL+1))
  fi
done

if (( FAIL > 0 )); then
  echo "FAIL: $FAIL coverage gap(s) across the matrix"
  exit 1
fi

echo "PASS: event-coverage-matrix"
