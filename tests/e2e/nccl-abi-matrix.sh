#!/usr/bin/env bash
# Test 28: NCCL ABI matrix.
#
# Asserts:
#   - For each of three NCCL versions (2.18.x, 2.20.x, 2.21.x):
#     - PyTorch + NCCL initializes with that version.
#     - Running ncclAllReduce 100 times in a 2-rank job produces
#       >= 100 samples in the nccl.collective.duration_ms histogram with
#       op_type=ncclAllReduce.
#
# Hardware: any multi-GPU node (>= 2 GPUs) so torch.distributed can run a
# 2-rank job locally. Single-GPU is acceptable if the env tolerates
# WORLD_SIZE=2 across CUDA_VISIBLE_DEVICES sharing.
#
# Invoke:
#   sudo bash tests/e2e/nccl-abi-matrix.sh
#
# Optional env:
#   INGERO_BIN
#   NCCL_VERSIONS    space-separated list. Default sweeps three ABI eras
#                    on currently-available PyPI nvidia-nccl-cu12 wheels:
#                    2.18.3 (pre-2.20 era), 2.22.3 (mid era),
#                    2.26.2 (current era as of v0.14). Pre-v0.14.1 the
#                    default was "2.18.6 2.20.5 2.21.5" but those tags
#                    were never published to PyPI; verify via:
#                      curl -s https://pypi.org/pypi/nvidia-nccl-cu12/json \
#                        | jq -r '.releases | keys | .[]'
#
# Expected runtime: ~300s (3 venvs x install + run).
set -euo pipefail
. "$(dirname "$0")/_lib.sh"

INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo ./ingero)}"
NCCL_VERSIONS="${NCCL_VERSIONS:-2.18.3 2.22.3 2.26.2}"
[[ -x "$INGERO_BIN" ]] || { echo "FAIL: agent missing at $INGERO_BIN"; exit 1; }
command -v python3 >/dev/null || { echo "FAIL: python3 missing"; exit 1; }

GPU_COUNT=$(nvidia-smi -L 2>/dev/null | wc -l)
if (( GPU_COUNT < 2 )); then
  echo "WARN: only $GPU_COUNT GPU visible; the 2-rank job will share devices"
fi

# The agent's libnccl-discovery scanner allowlists only canonical
# library install roots (/usr/lib*, /usr/local, /opt/{nccl,cuda,
# conda,pytorch,python}); /tmp and /home are denied. A venv that
# lands in /tmp causes the scanner to refuse the libnccl.so it
# finds, so no uprobes attach and the assertion gets 0 samples.
# Put the per-version venvs under /opt/python/ so they fall inside
# the allowlist. Falls back to /tmp with a warning when /opt is
# not writable (CI runners without sudo).
VENV_PARENT=/opt/python/v0141-nccl-test
if sudo -n mkdir -p "$VENV_PARENT" 2>/dev/null && sudo -n chown "$USER" "$VENV_PARENT" 2>/dev/null; then
  WORK="$VENV_PARENT"
else
  echo "WARN: /opt/python not writable; falling back to /tmp (agent allowlist will REJECT venv libnccl paths)"
  WORK=$(mktemp -d)
fi
mkdir -p "$WORK"
AGENT_PID=""
cleanup() {
  set +e
  kill_agent
  if [[ "$WORK" == "$VENV_PARENT" ]]; then
    sudo rm -rf "$WORK" 2>/dev/null
  else
    rm -rf "$WORK"
  fi
}
trap cleanup EXIT

echo "=== Test 28: nccl-abi-matrix (versions: $NCCL_VERSIONS) ==="

cat > "$WORK/allreduce.py" <<'PY'
import os, sys, time
import torch
import torch.distributed as dist

rank = int(os.environ['RANK'])
world = int(os.environ['WORLD_SIZE'])
os.environ.setdefault('MASTER_ADDR', '127.0.0.1')
os.environ.setdefault('MASTER_PORT', '12399')
torch.cuda.set_device(rank % torch.cuda.device_count())
dist.init_process_group(backend='nccl', rank=rank, world_size=world)
print(f"rank={rank} nccl-version={torch.cuda.nccl.version()}", flush=True)
t = torch.ones(1024 * 1024, device='cuda')
for i in range(100):
    dist.all_reduce(t)
torch.cuda.synchronize()
dist.destroy_process_group()
print("done", flush=True)
PY

run_for_version() {
  local ver="$1"
  local venv="$WORK/venv-$ver"
  echo "==> [version $ver] make venv + install"
  python3 -m venv "$venv"
  # shellcheck disable=SC1091
  source "$venv/bin/activate"
  pip install --quiet --upgrade pip
  # Pin torch + nccl wheel.
  if ! pip install --quiet "torch" "nvidia-nccl-cu12==$ver" 2>"$WORK/pip-$ver.log"; then
    echo "FAIL: pip install nvidia-nccl-cu12==$ver failed"
    cat "$WORK/pip-$ver.log"
    deactivate
    return 1
  fi

  echo "==> [version $ver] boot agent with NCCL uprobes"
  sudo "$INGERO_BIN" trace --record --db "$WORK/trace-$ver.db" \
    --duration 90s \
    --libnccl-discovery-interval 3s \
    --prometheus :9090 \
    >"$WORK/agent-$ver.log" 2>&1 &
  AGENT_PID=$!
  sleep 5

  echo "==> [version $ver] run 2-rank ncclAllReduce x 100"
  RANK=0 WORLD_SIZE=2 python "$WORK/allreduce.py" >"$WORK/r0-$ver.log" 2>&1 &
  R0=$!
  RANK=1 WORLD_SIZE=2 python "$WORK/allreduce.py" >"$WORK/r1-$ver.log" 2>&1 &
  R1=$!
  wait $R0 $R1

  sleep 3
  PROM="$WORK/prom-$ver.txt"
  curl -fsS http://localhost:9090/metrics > "$PROM" 2>/dev/null || true

  # Stop agent before next iteration.
  kill_agent
  AGENT_PID=""

  # Assertion: nccl.collective.duration_ms histogram count >= 100 for
  # op_type=ncclAllReduce.
  COUNT=$(awk '/^nccl_collective_duration_ms_count/ && index($0, "op_type=\"ncclAllReduce\"") {print $NF; exit}' "$PROM" || echo 0)
  if [[ -z "$COUNT" ]]; then COUNT=0; fi
  if (( COUNT < 100 )); then
    echo "FAIL: version $ver got only $COUNT samples (expected >= 100)"
    grep -E '^nccl_collective_duration_ms' "$PROM" | head -10
    deactivate
    return 1
  fi
  echo "OK: version $ver got $COUNT samples"

  deactivate
  return 0
}

OVERALL=0
for v in $NCCL_VERSIONS; do
  if ! run_for_version "$v"; then
    OVERALL=$((OVERALL+1))
  fi
done

if (( OVERALL > 0 )); then
  echo "FAIL: $OVERALL NCCL version(s) did not pass"
  exit 1
fi

echo "PASS: nccl-abi-matrix ($NCCL_VERSIONS)"
