#!/bin/bash
# Shared helpers for the e2e harness scripts. Source from the test:
#   . "$(dirname "$0")/_lib.sh"

# wait_port_ready <host> <port> [timeout_seconds]
# Polls TCP connect to the given endpoint, returns 0 when bound, 1 on
# timeout. Default timeout 30s. Used to bridge the variable startup
# delay between agent launch and Prometheus listener bind on Lambda
# VMs (4s-static sleeps were not enough on cold-start sudo runs).
wait_port_ready() {
  local host="${1:-127.0.0.1}"
  local port="${2:?port required}"
  local timeout="${3:-30}"
  local start=$SECONDS
  while (( SECONDS - start < timeout )); do
    if (echo > /dev/tcp/"$host"/"$port") >/dev/null 2>&1; then
      return 0
    fi
    sleep 0.5
  done
  return 1
}

# kill_agent
# Cleans up any lingering `ingero trace` process. Use from cleanup()
# traps in scripts that boot the agent. The naive
#   sudo "$INGERO_BIN" trace ... &
#   AGENT_PID=$!
# pattern captures the SUDO PID, not the actual ingero PID, so a
# subsequent `sudo kill "$AGENT_PID"` only kills sudo and leaves the
# real agent alive. Subsequent boots then fail with
# "another ingero trace is running". pkill across the actual
# process name dodges that.
kill_agent() {
  sudo pkill -f "ingero-bin trace" 2>/dev/null || true
  sudo pkill -f "ingero trace" 2>/dev/null || true
  sleep 1
}

# ensure_cuda_busy
# Compiles and caches tests/workloads/cuda_busy.cu, prints the path
# to the cached binary. Returns 1 (and prints to stderr) if nvcc is
# missing or the build fails, so callers can decide how to fall
# back. The cache lives in /tmp so it survives across test runs in
# the same VM.
#
# Usage:
#   WL=$(ensure_cuda_busy) || { echo "SKIP: nvcc unavailable"; exit 0; }
#   "$WL" --duration 30 &
#   WL_PID=$!
ensure_cuda_busy() {
  local repo_root="${REPO_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}"
  local src="$repo_root/tests/workloads/cuda_busy.cu"
  local bin="/tmp/ingero-cuda-busy"
  if [[ ! -f "$src" ]]; then
    echo "ensure_cuda_busy: source missing at $src" >&2
    return 1
  fi
  if [[ -x "$bin" && "$bin" -nt "$src" ]]; then
    echo "$bin"
    return 0
  fi
  if ! command -v nvcc >/dev/null 2>&1; then
    echo "ensure_cuda_busy: nvcc missing" >&2
    return 1
  fi
  if ! nvcc -O2 "$src" -o "$bin" 2>/tmp/cuda_busy_build.log; then
    echo "ensure_cuda_busy: nvcc build failed; see /tmp/cuda_busy_build.log" >&2
    return 1
  fi
  echo "$bin"
}

# set_strict_with_traps
# Wraps `set -euo pipefail` with an ERR trap that logs the failing
# line + command before the script exits, so silent
# pipefail+SIGPIPE exits become loud. Source this near the top of
# any script that sets strict mode; it replaces a bare
# `set -euo pipefail`.
set_strict_with_traps() {
  set -euo pipefail
  trap 'rc=$?; echo "FAIL [line $LINENO, exit $rc]: ${BASH_COMMAND}" >&2' ERR
}
