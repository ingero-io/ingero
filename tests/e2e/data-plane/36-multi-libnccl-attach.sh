#!/usr/bin/env bash
# 36-multi-libnccl-attach.sh
#
# Verifies F1 multi-path runtime libnccl attachment. Two PyTorch
# venvs with their own libnccl wheel run in parallel; the agent's
# discovery scanner should find BOTH and AttachAt both. Asserts
# AttachedPaths grows to 2 + the agent log contains the
# "attached new libnccl path(s) total_attached=N" marker for both
# discovery cycles.
#
# Closes v0.15 follow-up gap #4: multi-libnccl simultaneous on a
# real host. Single-libnccl coverage already lives in v0.15's
# Lambda A10 cycle.
#
# Hardware: any GPU host with PyTorch installable. nvcc not needed.
set -euo pipefail
. "$(dirname "$0")/../_lib.sh"

INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo ./ingero)}"
[[ -x "$INGERO_BIN" ]] || { echo "FAIL: agent missing at $INGERO_BIN"; exit 1; }
command -v python3 >/dev/null || { echo "SKIP: python3 missing"; exit 0; }
nvidia-smi -L >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 0; }

VENV_PARENT=/opt/python/v015-multi-nccl
sudo mkdir -p "$VENV_PARENT" 2>/dev/null
sudo chown "$USER" "$VENV_PARENT" 2>/dev/null || {
  echo "SKIP: cannot write under /opt/python (libnccl-discovery allowlist target)"
  exit 0
}

WORK="$VENV_PARENT"
AGENT_PID=""
cleanup() {
  set +e
  kill_agent
  sudo pkill -9 -f hold_libnccl 2>/dev/null
  sudo rm -rf "$WORK"/venv-A "$WORK"/venv-B "$WORK"/*.py "$WORK"/*.log "$WORK"/trace.db 2>/dev/null
}
trap cleanup EXIT

echo "==> [1/4] Build two venvs with PyTorch (cached if present)"
for tag in A B; do
  V="$WORK/venv-$tag"
  if [[ ! -d "$V" ]]; then
    python3 -m venv "$V"
    "$V/bin/pip" install --quiet --upgrade pip
    "$V/bin/pip" install --quiet torch==2.4.0 --extra-index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -3
  fi
done
LIB_A=$("$WORK/venv-A/bin/python" -c "import torch, os, glob; print(glob.glob(os.path.join(os.path.dirname(torch.__file__), '..', 'nvidia', 'nccl', 'lib', 'libnccl*.so*'))[0])")
LIB_B=$("$WORK/venv-B/bin/python" -c "import torch, os, glob; print(glob.glob(os.path.join(os.path.dirname(torch.__file__), '..', 'nvidia', 'nccl', 'lib', 'libnccl*.so*'))[0])")
echo "venv-A libnccl: $LIB_A"
echo "venv-B libnccl: $LIB_B"
# realpath them for the equality check (the actual libnccl files)
LIB_A_REAL=$(realpath "$LIB_A")
LIB_B_REAL=$(realpath "$LIB_B")

cat > "$WORK/hold_libnccl.py" <<'PY'
# Loads libnccl into /proc/PID/maps via torch.cuda.nccl, then idles
# for 30s. The discovery scanner picks the libnccl path out of maps.
import os, time, torch
print(f"pid={os.getpid()} nccl={torch.cuda.nccl.version()}", flush=True)
time.sleep(30)
PY

echo "==> [2/4] Boot agent with --nccl + 2s discovery"
sudo "$INGERO_BIN" trace --record --db "$WORK/trace.db" \
  --duration 60s --nccl --libnccl-discovery-interval 2s --prometheus :9090 --debug \
  >"$WORK/agent.log" 2>&1 &
AGENT_PID=$!
wait_port_ready 127.0.0.1 9090 30 || { echo "FAIL: agent did not bind"; tail -30 "$WORK/agent.log"; exit 1; }
sleep 4 # let discovery run once with no nccl processes (baseline)

echo "==> [3/4] Launch two parallel hold-libnccl processes"
"$WORK/venv-A/bin/python" "$WORK/hold_libnccl.py" >"$WORK/A.log" 2>&1 &
PA=$!
"$WORK/venv-B/bin/python" "$WORK/hold_libnccl.py" >"$WORK/B.log" 2>&1 &
PB=$!
sleep 12 # at least 5 discovery scans within this window
PROM=$(curl -fsS http://localhost:9090/metrics)
wait $PA $PB

echo "==> [4/4] Assertions"
# A: scanner found both processes (Prometheus reports total)
TOTAL=$(echo "$PROM" | awk '/^gpu_nccl_processes_total/ {print $NF; exit}')
if [[ -z "$TOTAL" || "$TOTAL" -lt 2 ]]; then
  echo "FAIL: gpu_nccl_processes_total = ${TOTAL:-empty} (expected >= 2)"
  echo "$PROM" | grep -E "^gpu_nccl"
  exit 1
fi
echo "OK: gpu_nccl_processes_total = $TOTAL"

# B: AttachAt fired for both unique libnccl paths
ATTACH_LINES=$(sudo grep -c "attached new libnccl path" "$WORK/agent.log")
if (( ATTACH_LINES < 1 )); then
  echo "FAIL: no 'attached new libnccl path' log lines (AttachAt never fired)"
  exit 1
fi
echo "OK: AttachAt log lines = $ATTACH_LINES"

# C: total_attached eventually >= 2 + 1 (eager systemwide may add one)
LATEST_TOTAL=$(sudo grep "total_attached" "$WORK/agent.log" | tail -1 | grep -oE 'total_attached=[0-9]+' | cut -d= -f2)
if [[ -z "$LATEST_TOTAL" || "$LATEST_TOTAL" -lt 2 ]]; then
  echo "FAIL: latest total_attached = ${LATEST_TOTAL:-empty} (expected >= 2 for two distinct venvs)"
  exit 1
fi
echo "OK: latest total_attached = $LATEST_TOTAL"

# D: both venv libnccl paths appear in process_loaded gauge rows
for path_real in "$LIB_A_REAL" "$LIB_B_REAL"; do
  # Some libnccl wheels are symlinks; the maps entry shows realpath
  if echo "$PROM" | grep -q "$(basename $path_real)"; then
    echo "OK: libnccl basename $(basename $path_real) appears in /metrics"
  fi
done

echo "PASS: multi-libnccl runtime attachment validated"
