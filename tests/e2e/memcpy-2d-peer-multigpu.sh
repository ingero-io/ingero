#!/usr/bin/env bash
# Test 18: Memcpy uprobes 2D + Peer (multi-GPU).
#
# Asserts:
#   - gpu.memcpy.bytes_total{direction="unknown"} is non-zero (2D variants
#     encode direction=5/unknown because the kind argument lives where libbpf's PT_REGS_PARMn macros cannot read it).
#   - gpu.memcpy.bytes_total{direction="d2d"} grows by Peer's expected byte
#     count.
#   - NO 2D events leak into direction="h2h" (regression guard:
#     unreadable kind must produce direction="unknown", never the
#     default-when-zero "h2h" value).
#
# Hardware: 2x H100 SXM5 (Lambda multi-GPU). Tolerates any 2-GPU box.
#
# Invoke:
#   sudo bash tests/e2e/memcpy-2d-peer-multigpu.sh
#
# Optional env:
#   INGERO_BIN, NVCC
#
# Expected runtime: ~50s.
set -euo pipefail

INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo ./ingero)}"
NVCC="${NVCC:-$(command -v nvcc || true)}"
[[ -x "$INGERO_BIN" ]] || { echo "FAIL: agent missing at $INGERO_BIN"; exit 1; }
[[ -n "$NVCC" && -x "$NVCC" ]] || { echo "FAIL: nvcc missing"; exit 1; }

GPU_COUNT=$(nvidia-smi -L | wc -l)
if (( GPU_COUNT < 2 )); then
  echo "FAIL: this test needs >= 2 GPUs (have $GPU_COUNT)"
  exit 1
fi

WORK=$(mktemp -d)
AGENT_PID=""
cleanup() {
  set +e
  kill_agent
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "=== Test 18: memcpy-2d-peer-multigpu ==="

echo "==> [1/4] Compile CUDA program"
cat > "$WORK/memcpy_2d_peer.cu" <<'CU'
// 100 cudaMemcpy2D + 100 cudaMemcpyPeer (GPU0 -> GPU1).
//   2D copy:  width=1 MiB rows, height=1, pitch=1 MiB, count=100  -> 100 MiB
//   Peer:     buffer = 4 MiB, count=100                            -> 400 MiB
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#define CHK(call) do { \
  cudaError_t e = (call); \
  if (e != cudaSuccess) { \
    fprintf(stderr, "CUDA err %s:%d %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
    exit(1); \
  } \
} while (0)

int main(void) {
    int n = 0;
    CHK(cudaGetDeviceCount(&n));
    if (n < 2) { fprintf(stderr, "need >=2 GPUs\n"); return 1; }

    const size_t MB = 1024UL * 1024UL;
    const size_t W = 1 * MB;
    const size_t H = 1;
    const size_t PEER_SZ = 4 * MB;
    const int N = 100;

    CHK(cudaSetDevice(0));
    void *src2d, *dst2d;
    CHK(cudaMalloc(&src2d, W * H));
    CHK(cudaMalloc(&dst2d, W * H));
    void *peer_src;
    CHK(cudaMalloc(&peer_src, PEER_SZ));
    CHK(cudaSetDevice(1));
    void *peer_dst;
    CHK(cudaMalloc(&peer_dst, PEER_SZ));
    CHK(cudaSetDevice(0));
    cudaDeviceEnablePeerAccess(1, 0);
    sleep(2);

    fprintf(stderr, "phase: cudaMemcpy2D\n");
    for (int i = 0; i < N; i++) {
        CHK(cudaMemcpy2D(dst2d, W, src2d, W, W, H, cudaMemcpyDeviceToDevice));
    }
    fprintf(stderr, "phase: cudaMemcpyPeer\n");
    for (int i = 0; i < N; i++) {
        CHK(cudaMemcpyPeer(peer_dst, 1, peer_src, 0, PEER_SZ));
    }
    CHK(cudaDeviceSynchronize());

    cudaFree(src2d); cudaFree(dst2d); cudaFree(peer_src);
    cudaSetDevice(1); cudaFree(peer_dst);
    return 0;
}
CU
"$NVCC" -O2 -cudart=shared "$WORK/memcpy_2d_peer.cu" -o "$WORK/memcpy_2d_peer" 2>"$WORK/nvcc.log" || {
  echo "FAIL: nvcc compile failed"
  cat "$WORK/nvcc.log"
  exit 1
}
echo "OK: compiled"

echo "==> [2/4] Boot agent"
sudo "$INGERO_BIN" trace --record --db "$WORK/trace.db" \
  --duration 50s \
  --prometheus :9090 \
  >"$WORK/agent.log" 2>&1 &
AGENT_PID=$!
. "$(dirname "$0")/_lib.sh"
wait_port_ready 127.0.0.1 9090 30 || { echo "FAIL: agent did not bind :9090 within 30s"; tail -20 "$WORK/agent.log"; exit 1; }

# Capture pre-workload baselines.
PROM_BEFORE="$WORK/prom_before.txt"
curl -fsS http://localhost:9090/metrics > "$PROM_BEFORE" || true

bytes_for() {
  local dir="$1" file="$2"
  awk -v d="\"$dir\"" '/^gpu_memcpy_bytes_total/ && index($0, "direction="d) {print $NF; exit}' "$file" 2>/dev/null || echo 0
}

D2D_BEFORE=$(bytes_for d2d "$PROM_BEFORE")
UNKNOWN_BEFORE=$(bytes_for unknown "$PROM_BEFORE")

echo "==> [3/4] Run workload"
"$WORK/memcpy_2d_peer" >"$WORK/workload.log" 2>&1
sleep 2

echo "==> [4/4] Assertions"
PROM_AFTER="$WORK/prom_after.txt"
curl -fsS http://localhost:9090/metrics > "$PROM_AFTER"

D2D_AFTER=$(bytes_for d2d "$PROM_AFTER")
UNKNOWN_AFTER=$(bytes_for unknown "$PROM_AFTER")
H2H_AFTER=$(bytes_for h2h "$PROM_AFTER")

D2D_DELTA=$(awk -v a="$D2D_AFTER" -v b="${D2D_BEFORE:-0}" 'BEGIN { print a-b }')
UNK_DELTA=$(awk -v a="$UNKNOWN_AFTER" -v b="${UNKNOWN_BEFORE:-0}" 'BEGIN { print a-b }')
EXPECT_PEER=$(( 100 * 4 * 1024 * 1024 ))

# Assertion 1: unknown is non-zero.
if (( $(awk -v u="$UNK_DELTA" 'BEGIN { print (u > 0) ? 1 : 0 }') == 0 )); then
  echo "FAIL: direction=unknown delta=$UNK_DELTA (expected non-zero from 100 cudaMemcpy2D)"
  exit 1
fi
echo "OK: direction=unknown delta=$UNK_DELTA bytes"

# Assertion 2: d2d delta within +/- 10% of expected Peer bytes.
DIFF=$(awk -v a="$D2D_DELTA" -v e="$EXPECT_PEER" 'BEGIN { d=(a>e?a-e:e-a)/e; printf "%.4f", d }')
OK=$(awk -v d="$DIFF" 'BEGIN { print (d <= 0.10) ? 1 : 0 }')
if [[ "$OK" != "1" ]]; then
  echo "FAIL: d2d delta=$D2D_DELTA expected ~$EXPECT_PEER (diff=$DIFF > 0.10)"
  exit 1
fi
echo "OK: d2d delta=$D2D_DELTA matches Peer bytes within 10%"

# Assertion 3: no h2h leak. h2h_after must be the same as before (treat empty as 0).
H2H_BEFORE=$(bytes_for h2h "$PROM_BEFORE")
LEAK=$(awk -v a="${H2H_AFTER:-0}" -v b="${H2H_BEFORE:-0}" 'BEGIN { print a-b }')
if (( $(awk -v l="$LEAK" 'BEGIN { print (l > 0) ? 1 : 0 }') == 1 )); then
  echo "FAIL: 2D leak into direction=h2h (delta=$LEAK)"
  exit 1
fi
echo "OK: no leak into direction=h2h"

echo "PASS: memcpy-2d-peer-multigpu"
