#!/usr/bin/env bash
# Test 16: libnccl process discovery emits gpu.nccl.process_loaded with
# correct labels.
#
# Asserts:
#   - At least one Prometheus line for `gpu.nccl.process_loaded` with the
#     expected `pid`, `libnccl_path`, and `libnccl_version` labels.
#   - `gpu.nccl.processes_total` >= 1 while the workload runs.
#
# Hardware: any A10. Python with PyTorch installed (Lambda images include it).
#
# Invoke:
#   sudo bash tests/e2e/nccl-discovery-asserts.sh
#
# Optional env:
#   INGERO_BIN
#
# Expected runtime: ~70s.
set -euo pipefail
. "$(dirname "$0")/_lib.sh"

INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo ./ingero)}"
[[ -x "$INGERO_BIN" ]] || { echo "FAIL: agent missing at $INGERO_BIN"; exit 1; }

WORK=$(mktemp -d)
WORKLOAD_PID=""
AGENT_PID=""

cleanup() {
  set +e
  kill_agent
  [[ -n "$WORKLOAD_PID" ]] && kill "$WORKLOAD_PID" 2>/dev/null
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "=== Test 16: nccl-discovery-asserts ==="

cat > "$WORK/torch_nccl.py" <<'PY'
import os, sys, time
os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', '12355')
os.environ.setdefault('WORLD_SIZE', '1')
os.environ.setdefault('RANK', '0')
import torch
import torch.distributed as dist
dist.init_process_group(backend='nccl', rank=0, world_size=1)
print(f"NCCL initialized; pid={os.getpid()}", flush=True)
time.sleep(60)
PY

echo "==> [1/3] Start torch+NCCL workload"
python3 "$WORK/torch_nccl.py" >"$WORK/workload.log" 2>&1 &
WORKLOAD_PID=$!
# Wait for NCCL to actually load.
for _ in $(seq 1 20); do
  if grep -q "NCCL initialized" "$WORK/workload.log"; then break; fi
  sleep 1
done
if ! grep -q "NCCL initialized" "$WORK/workload.log"; then
  echo "FAIL: torch did not initialize NCCL"
  cat "$WORK/workload.log"
  exit 1
fi
WORKLOAD_PROC_PID=$(awk -F'pid=' '/pid=/ {print $2; exit}' "$WORK/workload.log")
echo "OK: workload pid=$WORKLOAD_PROC_PID has loaded libnccl"

echo "==> [2/3] Boot agent with --libnccl-discovery-interval 3s"
sudo "$INGERO_BIN" trace --record --db "$WORK/trace.db" \
  --duration 30s \
  --libnccl-discovery-interval 3s \
  --prometheus :9090 \
  >"$WORK/agent.log" 2>&1 &
AGENT_PID=$!
sleep 12

echo "==> [3/3] Scrape Prometheus + assert"
PROM="$WORK/prom.txt"
curl -fsS http://localhost:9090/metrics > "$PROM" || {
  echo "FAIL: could not scrape :9090"
  cat "$WORK/agent.log" | tail -30
  exit 1
}

if ! grep -E '^gpu_nccl_process_loaded' "$PROM" >/dev/null; then
  echo "FAIL: no gpu.nccl.process_loaded line in Prometheus output"
  grep -E 'nccl|gpu_nccl' "$PROM" | head -20
  exit 1
fi

LINE=$(grep -E '^gpu_nccl_process_loaded' "$PROM" | grep -E "pid=\"$WORKLOAD_PROC_PID\"" || true)
if [[ -z "$LINE" ]]; then
  echo "FAIL: no process_loaded line for pid=$WORKLOAD_PROC_PID"
  grep -E '^gpu_nccl_process_loaded' "$PROM" | head -10
  exit 1
fi
echo "OK: $LINE"

# Check labels.
for lbl in libnccl_path libnccl_version; do
  if ! echo "$LINE" | grep -qE "${lbl}=\"[^\"]+\""; then
    echo "FAIL: label $lbl missing or empty"
    exit 1
  fi
done
echo "OK: process_loaded carries libnccl_path + libnccl_version"

# processes_total >= 1. The exporter writes this as an integer (count of
# discovered processes), so plain bash arithmetic suffices; bc is not in
# the minimal toolchain on Lambda VMs.
TOTAL=$(awk '/^gpu_nccl_processes_total/ {print $2; exit}' "$PROM" || echo 0)
TOTAL_INT=${TOTAL%%.*}
if (( TOTAL_INT < 1 )); then
  echo "FAIL: gpu_nccl_processes_total=$TOTAL < 1"
  exit 1
fi
echo "OK: gpu_nccl_processes_total=$TOTAL"

echo "PASS: nccl-discovery-asserts"
