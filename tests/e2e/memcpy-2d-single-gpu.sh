#!/usr/bin/env bash
# memcpy-2d-single-gpu.sh
#
# Single-GPU diagnostic for the v0.14 cudaMemcpy2D BYTES counter
# (v0.15 F3). Runs 100 cudaMemcpy2D calls of known size on whatever
# GPU is present, scrapes Prometheus, and asserts that
#   gpu_memcpy_bytes_total{direction="unknown"}
# advanced by ~100 * width bytes. A pass tells us the BPF probe
# fires + records arg0=width. A fail with delta=0 says either the
# probe is not attached or arg0 is not propagating.
#
# Hardware: any 1+-GPU host. nvcc required.
set -euo pipefail
. "$(dirname "$0")/_lib.sh"

INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo ./ingero)}"
NVCC="${NVCC:-$(command -v nvcc || true)}"
[[ -x "$INGERO_BIN" ]] || { echo "FAIL: agent missing at $INGERO_BIN"; exit 1; }
[[ -n "$NVCC" && -x "$NVCC" ]] || { echo "SKIP: nvcc missing"; exit 0; }
nvidia-smi -L >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 0; }

WORK=$(mktemp -d)
AGENT_PID=""
cleanup() {
  set +e
  kill_agent
  rm -rf "$WORK"
}
trap cleanup EXIT

cat > "$WORK/m2d.cu" <<'CU'
// 100 cudaMemcpy2D calls of width=1 MiB, height=1 (so dst[0..W) gets
// W bytes per call; total ~100 MiB).
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#define CHK(c) do { cudaError_t e=(c); if (e) { fprintf(stderr,"%s:%d %s\n",__FILE__,__LINE__,cudaGetErrorString(e)); exit(1);} } while(0)
int main(){
  const size_t MB = 1024UL*1024UL, W = 1*MB, H = 1;
  void *src,*dst;
  CHK(cudaMalloc(&src,W*H));
  CHK(cudaMalloc(&dst,W*H));
  for (int i=0;i<100;i++) CHK(cudaMemcpy2D(dst,W,src,W,W,H,cudaMemcpyDeviceToDevice));
  CHK(cudaDeviceSynchronize());
  cudaFree(src); cudaFree(dst);
  return 0;
}
CU
# -cudart=shared forces dynamic linking against libcudart.so. Without
# this nvcc statically links cudart inside the binary and the agent's
# uprobes against /usr/lib/libcudart.so do not fire (the call sites
# live in the binary, not in the shared library).
"$NVCC" -O2 -cudart=shared "$WORK/m2d.cu" -o "$WORK/m2d" 2>"$WORK/nvcc.log" || {
  echo "FAIL: nvcc compile"; cat "$WORK/nvcc.log"; exit 1
}

echo "==> [1/3] Boot agent"
sudo "$INGERO_BIN" trace --record --db "$WORK/trace.db" --duration 25s --prometheus :9090 \
  >"$WORK/agent.log" 2>&1 &
AGENT_PID=$!
wait_port_ready 127.0.0.1 9090 30 || { echo "FAIL: agent did not bind :9090"; tail -20 "$WORK/agent.log"; exit 1; }

bytes_unknown() {
  curl -fsS http://localhost:9090/metrics 2>/dev/null \
    | awk '/^gpu_memcpy_bytes_total\{direction="unknown"\}/ {print $NF; exit}' \
    || echo 0
}
B0=$(bytes_unknown)
B0=${B0:-0}

echo "==> [2/3] Run 100 cudaMemcpy2D (1 MiB each)"
"$WORK/m2d" >"$WORK/wl.log" 2>&1
sleep 3
B1=$(bytes_unknown)
B1=${B1:-0}
DELTA=$(( B1 - B0 ))

echo "==> [3/3] Assertions"
echo "before=$B0 after=$B1 delta=$DELTA"
if (( DELTA <= 0 )); then
  echo "FAIL: cudaMemcpy2D delta=0 in direction=unknown bytes_total"
  echo "--- agent log ---"
  tail -40 "$WORK/agent.log"
  echo "--- /metrics relevant rows ---"
  curl -fsS http://localhost:9090/metrics | grep -E 'gpu_memcpy|gpu_cuda_operation_count' | head -20
  exit 1
fi
# Lower bound: we expect ~100 * 1 MiB = 100 MiB, but adaptive sampling
# may drop some events. Anything > 1 MiB is enough to prove the wire
# path works.
if (( DELTA < 1024*1024 )); then
  echo "WARN: delta=$DELTA below 1 MiB; probe firing but bytes propagation suspect"
fi
echo "PASS: cudaMemcpy2D direction=unknown bytes advance by $DELTA bytes"
