#!/usr/bin/env bash
# Test 17: Memcpy uprobes 1D direction matrix.
#
# Asserts (within 5% tolerance):
#   - gpu.memcpy.bytes_total{direction="h2d"}     ~ 100 MiB
#   - gpu.memcpy.bytes_total{direction="d2h"}     ~ 200 MiB
#   - gpu.memcpy.bytes_total{direction="d2d"}     ~ 400 MiB
#   - gpu.memcpy.bytes_total{direction="default"} ~ 800 MiB
#
# Hardware: any A10 with NVIDIA driver + nvcc available (CUDA Toolkit).
#
# Invoke:
#   sudo bash tests/e2e/memcpy-1d-direction-matrix.sh
#
# Optional env:
#   INGERO_BIN
#   NVCC          path to nvcc (default: nvcc on PATH)
#
# Expected runtime: ~60s.
#
# Why a custom CUDA program (not memcpy_stress.py):
#   memcpy_stress.py drives H2D/D2H/D2D via PyTorch but does not exercise the
#   `cudaMemcpyDefault` direction explicitly, and it transfers variable-size
#   buffers, which makes the byte-total assertion hard to bound. This test
#   compiles a tiny C program with deterministic copy counts + sizes for a
#   tight tolerance check.
set -euo pipefail

INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo ./ingero)}"
NVCC="${NVCC:-$(command -v nvcc || true)}"
[[ -x "$INGERO_BIN" ]] || { echo "FAIL: agent missing at $INGERO_BIN"; exit 1; }
[[ -n "$NVCC" && -x "$NVCC" ]] || { echo "FAIL: nvcc missing"; exit 1; }

WORK=$(mktemp -d)
AGENT_PID=""
cleanup() {
  set +e
  kill_agent
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "=== Test 17: memcpy-1d-direction-matrix ==="

echo "==> [1/4] Compile CUDA program"
cat > "$WORK/memcpy_matrix.cu" <<'CU'
// Deterministic 1D memcpy matrix:
//   100 H2D     of 1 MiB  = 100 MiB
//   100 D2H     of 2 MiB  = 200 MiB
//   100 D2D     of 4 MiB  = 400 MiB
//   100 default of 8 MiB  = 800 MiB
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
    const size_t MB = 1024UL * 1024UL;
    const size_t H2D_SZ = 1 * MB;
    const size_t D2H_SZ = 2 * MB;
    const size_t D2D_SZ = 4 * MB;
    const size_t DEF_SZ = 8 * MB;
    const int N = 100;

    void *h1 = malloc(H2D_SZ);
    void *h2 = malloc(D2H_SZ);
    void *h3 = malloc(DEF_SZ);
    void *d1, *d2, *d3a, *d3b, *d4;
    CHK(cudaMalloc(&d1, H2D_SZ));
    CHK(cudaMalloc(&d2, D2H_SZ));
    CHK(cudaMalloc(&d3a, D2D_SZ));
    CHK(cudaMalloc(&d3b, D2D_SZ));
    CHK(cudaMalloc(&d4, DEF_SZ));
    sleep(2); // let the agent attach uprobes before any copies fire

    fprintf(stderr, "phase: H2D\n");
    for (int i = 0; i < N; i++) CHK(cudaMemcpy(d1, h1, H2D_SZ, cudaMemcpyHostToDevice));
    fprintf(stderr, "phase: D2H\n");
    for (int i = 0; i < N; i++) CHK(cudaMemcpy(h2, d2, D2H_SZ, cudaMemcpyDeviceToHost));
    fprintf(stderr, "phase: D2D\n");
    for (int i = 0; i < N; i++) CHK(cudaMemcpy(d3b, d3a, D2D_SZ, cudaMemcpyDeviceToDevice));
    fprintf(stderr, "phase: default\n");
    for (int i = 0; i < N; i++) CHK(cudaMemcpy(d4, h3, DEF_SZ, cudaMemcpyDefault));
    CHK(cudaDeviceSynchronize());

    free(h1); free(h2); free(h3);
    cudaFree(d1); cudaFree(d2); cudaFree(d3a); cudaFree(d3b); cudaFree(d4);
    return 0;
}
CU
"$NVCC" -O2 "$WORK/memcpy_matrix.cu" -o "$WORK/memcpy_matrix" 2>"$WORK/nvcc.log" || {
  echo "FAIL: nvcc compile failed"
  cat "$WORK/nvcc.log"
  exit 1
}
echo "OK: compiled $WORK/memcpy_matrix"

echo "==> [2/4] Boot agent with Prometheus pull"
sudo "$INGERO_BIN" trace --record --db "$WORK/trace.db" \
  --duration 60s \
  --prometheus :9090 \
  >"$WORK/agent.log" 2>&1 &
AGENT_PID=$!
. "$(dirname "$0")/_lib.sh"
wait_port_ready 127.0.0.1 9090 30 || { echo "FAIL: agent did not bind :9090 within 30s"; tail -20 "$WORK/agent.log"; exit 1; }

echo "==> [3/4] Run workload"
"$WORK/memcpy_matrix" >"$WORK/workload.log" 2>&1
sleep 2

echo "==> [4/4] Assert per-direction totals"
PROM="$WORK/prom.txt"
curl -fsS http://localhost:9090/metrics > "$PROM"

declare -A EXPECT_MIB=(
  [h2d]=100
  [d2h]=200
  [d2d]=400
  [default]=800
)
TOLERANCE=0.05
FAILED=0
for dir in h2d d2h d2d default; do
  expect=${EXPECT_MIB[$dir]}
  actual=$(awk -v d="\"$dir\"" '/^gpu_memcpy_bytes_total/ && index($0, "direction="d) {print $NF; exit}' "$PROM" || echo 0)
  if [[ -z "$actual" || "$actual" == "0" ]]; then
    echo "FAIL: no value for direction=$dir"
    FAILED=$((FAILED+1))
    continue
  fi
  actual_mib=$(awk -v a="$actual" 'BEGIN { printf "%.3f", a/(1024*1024) }')
  diff=$(awk -v a="$actual_mib" -v e="$expect" 'BEGIN { d = (a > e ? a-e : e-a)/e; printf "%.4f", d }')
  ok=$(awk -v d="$diff" -v t="$TOLERANCE" 'BEGIN { print (d <= t) ? 1 : 0 }')
  if [[ "$ok" == "1" ]]; then
    echo "OK: direction=$dir actual=${actual_mib}MiB expect=${expect}MiB diff=$diff"
  else
    echo "FAIL: direction=$dir actual=${actual_mib}MiB expect=${expect}MiB diff=$diff > $TOLERANCE"
    FAILED=$((FAILED+1))
  fi
done

(( FAILED == 0 )) || { echo "FAIL: $FAILED direction(s) out of tolerance"; exit 1; }
echo "PASS: memcpy-1d-direction-matrix"
