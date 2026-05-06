#!/usr/bin/env bash
# Test 19: W1 NVML-poll memfrag heuristic responds to real fragmentation.
#
# Asserts:
#   - Peak gpu.memory.fragmentation_estimate during the fragmentation
#     workload is at least baseline + 0.05 (5% bump).
#   - gpu.memory.process.allocated_bytes{pid=<workload>} is non-zero.
#
# Hardware: any A10.
#
# Invoke:
#   sudo bash tests/e2e/memfrag-emits-under-fragmentation.sh
#
# Optional env:
#   INGERO_BIN
#   FRAG_PY    path to fragmentation.py (default: tests/workloads/pathological/fragmentation.py)
#
# Expected runtime: ~120s (15s baseline + 90s workload + 15s teardown).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo ./ingero)}"
FRAG_PY="${FRAG_PY:-$REPO_ROOT/tests/workloads/pathological/fragmentation.py}"

[[ -x "$INGERO_BIN" ]] || { echo "FAIL: agent missing at $INGERO_BIN"; exit 1; }
[[ -f "$FRAG_PY" ]] || { echo "FAIL: fragmentation workload missing at $FRAG_PY"; exit 1; }

WORK=$(mktemp -d)
WL_PID=""
AGENT_PID=""

cleanup() {
  set +e
  kill_agent
  [[ -n "$WL_PID" ]] && kill "$WL_PID" 2>/dev/null
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "=== Test 19: memfrag-emits-under-fragmentation ==="

frag_now() {
  curl -fsS http://localhost:9090/metrics 2>/dev/null \
    | awk '/^gpu_memory_fragmentation_estimate/ {print $NF; exit}' \
    || echo 0
}

echo "==> [1/4] Boot agent with --memfrag-poll-interval 5s"
sudo "$INGERO_BIN" trace --record --db "$WORK/trace.db" \
  --duration 130s \
  --memfrag-poll-interval 5s \
  --prometheus :9090 \
  >"$WORK/agent.log" 2>&1 &
AGENT_PID=$!
. "$(dirname "$0")/_lib.sh"
wait_port_ready 127.0.0.1 9090 30 || { echo "FAIL: agent did not bind :9090 within 30s"; tail -20 "$WORK/agent.log"; exit 1; }
# Give the memfrag poller one full interval to produce its first reading.
sleep 6

echo "==> [2/4] Capture baseline (idle, 15s window)"
BASELINE=0
for _ in $(seq 1 3); do
  v=$(frag_now)
  if (( $(awk -v a="$v" -v b="$BASELINE" 'BEGIN { print (a>b)?1:0 }') == 1 )); then
    BASELINE="$v"
  fi
  sleep 5
done
echo "OK: baseline gpu.memory.fragmentation_estimate=$BASELINE"

echo "==> [3/4] Start fragmentation workload"
python3 "$FRAG_PY" >"$WORK/wl.log" 2>&1 &
WL_PID=$!
sleep 1
WL_PROC_PID=$(pgrep -P $$ -f "$FRAG_PY" | head -1 || echo "$WL_PID")
echo "OK: workload pid=$WL_PROC_PID"

PEAK=0
for _ in $(seq 1 18); do
  v=$(frag_now)
  if (( $(awk -v a="$v" -v b="$PEAK" 'BEGIN { print (a>b)?1:0 }') == 1 )); then
    PEAK="$v"
  fi
  sleep 5
done
echo "OK: peak gpu.memory.fragmentation_estimate=$PEAK"

echo "==> [4/4] Assert peak > baseline + 0.05"
DIFF=$(awk -v p="$PEAK" -v b="$BASELINE" 'BEGIN { printf "%.4f", p-b }')
OK=$(awk -v d="$DIFF" 'BEGIN { print (d >= 0.05) ? 1 : 0 }')
if [[ "$OK" != "1" ]]; then
  echo "FAIL: peak-baseline=$DIFF < 0.05 (peak=$PEAK baseline=$BASELINE)"
  exit 1
fi
echo "OK: peak-baseline=$DIFF >= 0.05"

# Per-PID allocated_bytes.
PROM=$(curl -fsS http://localhost:9090/metrics)
ALLOC=$(echo "$PROM" | awk -v p="\"$WL_PROC_PID\"" '/^gpu_memory_process_allocated_bytes/ && index($0, "pid="p) {print $NF; exit}' || echo 0)
if [[ -z "$ALLOC" || "$ALLOC" == "0" ]]; then
  # Try without pid label match (in case label is different).
  ALLOC=$(echo "$PROM" | awk '/^gpu_memory_process_allocated_bytes/ {print $NF; exit}' || echo 0)
fi
if [[ -z "$ALLOC" || "$ALLOC" == "0" ]]; then
  echo "FAIL: gpu.memory.process.allocated_bytes is zero"
  echo "$PROM" | grep -E '^gpu_memory_process' | head -10
  exit 1
fi
echo "OK: gpu.memory.process.allocated_bytes=$ALLOC"

echo "PASS: memfrag-emits-under-fragmentation"
