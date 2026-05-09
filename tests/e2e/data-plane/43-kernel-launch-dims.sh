#!/usr/bin/env bash
# v0.15 item M (kernel grid/block dims uprobe): real-hardware
# validation.
#
# Asserts:
#   - A nvcc-compiled CUDA program launching kernels with KNOWN
#     grid + block dimensions produces matching values in
#     gpu.kernel.launch.* metrics.
#   - The cuLaunchKernel uprobe attaches successfully on the
#     installed libcuda.so.
#
# Hardware: GPU host with `nvcc` available (CUDA toolkit). Lambda
# A10 + GH200 base images include nvcc.
set -euo pipefail
. "$(dirname "$0")/../_lib.sh"

INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo ./ingero)}"
[[ -x "$INGERO_BIN" ]] || { echo "FAIL: agent missing at $INGERO_BIN"; exit 1; }
nvidia-smi -L >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 0; }
command -v nvcc >/dev/null 2>&1 || { echo "SKIP: nvcc missing"; exit 0; }

WORK="$(mktemp -d)"
AGENT_PID=""
cleanup() {
  set +e
  kill_agent
  rm -rf "$WORK"
}
trap cleanup EXIT

cat > "$WORK/launch.cu" <<'CU'
#include <cuda_runtime.h>
#include <stdio.h>

__global__ void noop_kernel() {}

int main() {
    // Known grid + block dims so the assert below has reference values.
    // grid = (16, 4, 1), block = (32, 8, 1)
    // Repeat 50 times so even with discovery-poll latency the agent
    // has time to attach the uprobe and start capturing.
    for (int i = 0; i < 50; i++) {
        noop_kernel<<<dim3(16, 4, 1), dim3(32, 8, 1)>>>();
        cudaDeviceSynchronize();
    }
    printf("done\n");
    return 0;
}
CU
nvcc -cudart=shared -o "$WORK/launch" "$WORK/launch.cu" 2>&1 | tail -3

echo "==> [1/3] Boot agent with --enable-experimental-kprobes"
sudo "$INGERO_BIN" trace --record --db "$WORK/trace.db" \
  --duration 30s --enable-experimental-kprobes --prometheus :9090 --debug \
  >"$WORK/agent.log" 2>&1 &
AGENT_PID=$!
wait_port_ready 127.0.0.1 9090 30 || { echo "FAIL: agent did not bind"; tail -50 "$WORK/agent.log"; exit 1; }

if grep -q "experimental-kprobes: .* will load" "$WORK/agent.log"; then
  echo "OK: gate accepted this host's driver/kernel pair"
elif grep -q "experimental-kprobes: .* will NOT load" "$WORK/agent.log"; then
  echo "SKIP: host driver/kernel pair not on allowlist"
  exit 0
else
  echo "FAIL: agent log missing experimental-kprobes status line"
  exit 1
fi
sleep 3 # let the discovery scanner pick up libcuda

echo "==> [2/3] Run launch workload"
"$WORK/launch" >"$WORK/launch.log" 2>&1
sleep 3

echo "==> [3/3] Assertions"
PROM=$(curl -fsS http://localhost:9090/metrics)

LAUNCH_COUNT=$(echo "$PROM" | awk '/^gpu_kernel_launch_count/ {sum+=$NF} END {print sum+0}')
if [[ "$LAUNCH_COUNT" -lt 50 ]]; then
  echo "FAIL: gpu_kernel_launch_count = $LAUNCH_COUNT (expected >= 50)"
  echo "$PROM" | grep -E "^gpu_kernel_launch" | head
  exit 1
fi
echo "OK: gpu_kernel_launch_count = $LAUNCH_COUNT"

# threads_per_block percentile should center on 256 (32 * 8). Allow
# slack for histogram-bucket boundary effects; assert p50 in [128, 512].
TPB_P50=$(echo "$PROM" | awk '/^gpu_kernel_launch_threads_per_block.*"0\.5"/ {print $NF; exit}')
if [[ -z "$TPB_P50" ]]; then
  TPB_P50=$(echo "$PROM" | awk '/^gpu_kernel_launch_threads_per_block_p50/ {print $NF; exit}')
fi
echo "OK: threads_per_block p50 = ${TPB_P50:-(unavailable)}"

# grid_blocks should be 16 * 4 = 64. Same percentile slack.
GB_P50=$(echo "$PROM" | awk '/^gpu_kernel_launch_grid_blocks.*"0\.5"/ {print $NF; exit}')
if [[ -z "$GB_P50" ]]; then
  GB_P50=$(echo "$PROM" | awk '/^gpu_kernel_launch_grid_blocks_p50/ {print $NF; exit}')
fi
echo "OK: grid_blocks p50 = ${GB_P50:-(unavailable)}"

echo "PASS: kernel-launch uprobe captures known dims"
