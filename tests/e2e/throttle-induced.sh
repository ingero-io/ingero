#!/usr/bin/env bash
# Test 20: W2-poller throttle reasons (induced by power-cap).
#
# Asserts:
#   - gpu.throttle.power_active{gpu_uuid=<uuid>} flips to 1 within 2x poll
#     interval after `nvidia-smi -pl <min>` is applied to a hot GPU.
#   - Flips back to 0 within 2x poll interval after the power-limit is
#     restored.
#
# Hardware: any A10 with sustained workload + sudo permission for nvidia-smi.
#
# Invoke:
#   sudo bash tests/e2e/throttle-induced.sh
#
# Optional env:
#   INGERO_BIN
#   WORKLOAD_PY    sustained CUDA workload
#                  (default: tests/workloads/training/gpt2_stress.py)
#
# Expected runtime: ~90s.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo ./ingero)}"
WORKLOAD_PY="${WORKLOAD_PY:-$REPO_ROOT/tests/workloads/training/gpt2_stress.py}"
POLL_S=5

[[ -x "$INGERO_BIN" ]] || { echo "FAIL: agent missing at $INGERO_BIN"; exit 1; }
command -v nvidia-smi >/dev/null || { echo "FAIL: nvidia-smi missing"; exit 1; }

# Use awk NR==1 instead of `| head -1`: head closes the pipe after
# one line, which causes nvidia-smi to receive SIGPIPE on its next
# write. With `set -o pipefail` the whole pipeline returns 141 and
# `set -e` exits the script silently. awk reads all input first,
# then prints only the first row, so the producer never sees a
# closed pipe. Origin: v0.14.1 e2e harness debugging.
GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | awk 'NR==1')
ORIG_LIMIT=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | awk 'NR==1')
MIN_LIMIT=$(nvidia-smi --query-gpu=power.min_limit --format=csv,noheader,nounits | awk 'NR==1')
echo "GPU $GPU_UUID power.limit=$ORIG_LIMIT W power.min=$MIN_LIMIT W"

WORK=$(mktemp -d)
WL_PID=""
AGENT_PID=""
RESTORED=0

. "$(dirname "$0")/_lib.sh"

cleanup() {
  set +e
  kill_agent
  [[ -n "$WL_PID" ]] && kill "$WL_PID" 2>/dev/null
  if [[ "$RESTORED" == "0" ]]; then
    sudo nvidia-smi -pl "$ORIG_LIMIT" >/dev/null 2>&1
  fi
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "=== Test 20: throttle-induced ==="

power_active() {
  curl -fsS http://localhost:9090/metrics 2>/dev/null \
    | awk -v u="\"$GPU_UUID\"" '/^gpu_throttle_power_active/ && index($0, "gpu_uuid="u) {print $NF; exit}' \
    || echo 0
}

echo "==> [1/5] Start sustained CUDA workload"
WL_BIN=""
if [[ -f "$WORKLOAD_PY" ]]; then
  python3 "$WORKLOAD_PY" >"$WORK/wl.log" 2>&1 &
  WL_PID=$!
elif WL_BIN=$(ensure_cuda_busy 2>/tmp/cuda_busy.err); then
  echo "OK: using bundled cuda_busy stresser at $WL_BIN"
  "$WL_BIN" --duration 0 >"$WORK/wl.log" 2>&1 &
  WL_PID=$!
else
  echo "SKIP: no workload available (PyTorch missing AND nvcc-build of cuda_busy failed)"
  cat /tmp/cuda_busy.err 2>/dev/null
  exit 0
fi
sleep 10

echo "==> [2/5] Boot agent with --throttle-poll-interval ${POLL_S}s"
sudo "$INGERO_BIN" trace --record --db "$WORK/trace.db" \
  --duration 80s \
  --throttle-poll-interval ${POLL_S}s \
  --prometheus :9090 \
  >"$WORK/agent.log" 2>&1 &
AGENT_PID=$!
. "$(dirname "$0")/_lib.sh"
wait_port_ready 127.0.0.1 9090 30 || { echo "FAIL: agent did not bind :9090 within 30s"; tail -20 "$WORK/agent.log"; exit 1; }
# Give the throttle poller a full interval to produce its baseline reading.
sleep "$(( POLL_S + 2 ))"

echo "==> [3/5] Apply power-cap to $MIN_LIMIT W"
sudo nvidia-smi -pl "$MIN_LIMIT" >/dev/null

# Wait up to 2 * poll for power_active to flip to 1.
DEADLINE=$(( SECONDS + 2 * POLL_S + 5 ))
FOUND=0
while (( SECONDS < DEADLINE )); do
  v=$(power_active)
  if [[ "$v" == "1" ]]; then FOUND=1; break; fi
  sleep 1
done
if [[ "$FOUND" != "1" ]]; then
  echo "FAIL: gpu_throttle_power_active did not flip to 1 within $((2*POLL_S+5))s"
  curl -fsS http://localhost:9090/metrics | grep -E 'throttle' | head -10
  exit 1
fi
echo "OK: power_active = 1 after power-cap"

echo "==> [4/5] Stop workload + restore power limit ($ORIG_LIMIT W)"
# Stopping the workload first so GPU power draw genuinely drops; NVML
# keeps the power-throttle reason "active" while the chip is still
# pulling near the cap, so leaving the workload running and just
# raising the cap does not reliably clear the flag within a useful
# window. With the workload gone the chip drops to idle and the
# agent observes the clear within 1-2 poll intervals.
[[ -n "$WL_PID" ]] && kill "$WL_PID" 2>/dev/null
WL_PID=""
sleep 2
sudo nvidia-smi -pl "$ORIG_LIMIT" >/dev/null
RESTORED=1

DEADLINE=$(( SECONDS + 3 * POLL_S + 5 ))
CLEARED=0
while (( SECONDS < DEADLINE )); do
  v=$(power_active)
  if [[ "$v" == "0" ]]; then CLEARED=1; break; fi
  sleep 1
done
if [[ "$CLEARED" != "1" ]]; then
  echo "FAIL: gpu_throttle_power_active did not clear to 0 within $((3*POLL_S+5))s after workload stop"
  exit 1
fi
echo "OK: power_active = 0 after workload stop + limit restore"

echo "==> [5/5] Done"
echo "PASS: throttle-induced"
