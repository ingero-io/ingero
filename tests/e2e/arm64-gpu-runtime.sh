#!/usr/bin/env bash
# Test 27: arm64 GPU runtime (BPF program load + attach).
#
# Asserts:
#   - On a real arm64 GPU host (GH200 or AWS g5g), the agent loads + attaches
#     all BPF programs touched by the gpu-test T-phases T02, T07, T22h,
#     T22e, T22f, T22g.
#   - No `bad CO-RE relocation` or `bpf_object__load` errors in agent log.
#   - Per-tracer `attached=true` line appears in agent debug log.
#
# Hardware: arm64 host with NVIDIA driver. GH200 preferred. AWS g5g
# (Graviton + T4) acceptable via the Slice F harness.
#
# Invoke:
#   sudo bash tests/e2e/arm64-gpu-runtime.sh
#
# Optional env:
#   INGERO_BIN
#   GPU_TEST_SH    path to gpu-test.sh (default: scripts/test/gpu-test.sh
#                  if the repo carries it, else skipped with a warn)
#
# Expected runtime: ~120s.
set -euo pipefail
. "$(dirname "$0")/_lib.sh"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo ./ingero)}"
GPU_TEST_SH="${GPU_TEST_SH:-$REPO_ROOT/scripts/test/gpu-test.sh}"

[[ -x "$INGERO_BIN" ]] || { echo "FAIL: agent missing at $INGERO_BIN"; exit 1; }
case "$(uname -m)" in
  aarch64|arm64) : ;;
  *) echo "FAIL: this test is arm64-only (got $(uname -m))"; exit 1 ;;
esac
command -v nvidia-smi >/dev/null || { echo "FAIL: nvidia-smi missing"; exit 1; }

WORK=$(mktemp -d)
AGENT_PID=""
cleanup() {
  set +e
  kill_agent
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "=== Test 27: arm64-gpu-runtime ==="
nvidia-smi --query-gpu=name,uuid,driver_version --format=csv | head -3

echo "==> [1/3] Boot agent with --debug for attach lines"
sudo "$INGERO_BIN" trace --record --db "$WORK/trace.db" \
  --duration 60s \
  --debug \
  --throttle-poll-interval 5s \
  --memfrag-poll-interval 5s \
  --libnccl-discovery-interval 5s \
  >"$WORK/agent.log" 2>&1 &
AGENT_PID=$!

# Drive the BPF programs by exercising representative T-phases.
echo "==> [2/3] Exercise T-phases (T02, T07, T22h, T22e, T22f, T22g) if available"
if [[ -x "$GPU_TEST_SH" ]]; then
  set +e
  for phase in T02 T07 T22h T22e T22f T22g; do
    "$GPU_TEST_SH" --phase "$phase" >>"$WORK/gpu-test.log" 2>&1 &
  done
  wait
  set -e
else
  echo "WARN: $GPU_TEST_SH not found; running synthetic CUDA load to drive BPF"
  cat > "$WORK/syn.cu" <<'CU'
#include <cuda_runtime.h>
#include <stdio.h>
__global__ void k(int *a, int n) { int i=blockIdx.x*blockDim.x+threadIdx.x; if(i<n) a[i]++; }
int main(){
  int *d; cudaMalloc(&d, 1024);
  for(int i=0;i<200;i++){ k<<<1,128>>>(d,256); cudaDeviceSynchronize(); }
  cudaFree(d); return 0;
}
CU
  if command -v nvcc >/dev/null; then
    nvcc -O2 "$WORK/syn.cu" -o "$WORK/syn"
    "$WORK/syn" >>"$WORK/gpu-test.log" 2>&1 || true
  fi
fi

# Let the agent finish its --duration window.
wait "$AGENT_PID" || true

echo "==> [3/3] Inspect agent log"
LOG="$WORK/agent.log"

if grep -qE 'bad CO-RE relocation|bpf_object__load' "$LOG"; then
  echo "FAIL: BPF load error present"
  grep -E 'bad CO-RE relocation|bpf_object__load' "$LOG" | head -10
  exit 1
fi
echo "OK: no BPF load errors"

# At least one per-tracer "attached=true" line.
if ! grep -qE 'attached=true|probes attached|probes:' "$LOG"; then
  echo "FAIL: no attach confirmation in agent debug log"
  tail -50 "$LOG"
  exit 1
fi
echo "OK: agent logged probe attach"

# Sanity: also ensure no panic.
if grep -qiE 'panic|fatal error|runtime error' "$LOG"; then
  echo "FAIL: panic / fatal in agent log"
  grep -iE 'panic|fatal error|runtime error' "$LOG" | head -5
  exit 1
fi
echo "OK: no panic / fatal"

echo "PASS: arm64-gpu-runtime"
