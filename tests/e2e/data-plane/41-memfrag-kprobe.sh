#!/usr/bin/env bash
# v0.15 item K (W1 memfrag IOCTL kprobe): real-hardware validation.
#
# Asserts:
#   - Agent boots with --enable-experimental-kprobes on a tested
#     driver+kernel pair (Lambda A10 image baseline by default).
#   - The agent log carries the "experimental-kprobes: ... will load"
#     line indicating the gate accepted the host.
#   - A short PyTorch fragmentation workload triggers a non-zero
#     count on `gpu_memfrag_ioctl_event_total` in /metrics.
#
# Hardware: any GPU host with NVIDIA driver loaded + PyTorch
# installable. The cmd-number distribution is host-specific and
# not asserted; only the existence of events is asserted.
set -euo pipefail
. "$(dirname "$0")/../_lib.sh"

INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo ./ingero)}"
[[ -x "$INGERO_BIN" ]] || { echo "FAIL: agent missing at $INGERO_BIN"; exit 1; }
nvidia-smi -L >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 0; }

WORK="$(mktemp -d)"
AGENT_PID=""
cleanup() {
  set +e
  kill_agent
  rm -rf "$WORK"
}
trap cleanup EXIT

cat > "$WORK/frag_workload.py" <<'PY'
# Generates fragmentation-heavy IOCTL traffic: many small allocs +
# frees, interspersed with a few large allocs. Real workloads do
# this during DataLoader churn, layer-wise allocators, etc.
import torch
import time
torch.cuda.init()
xs = []
for i in range(200):
    # 1024 elements + linear growth; bounded so we don't OOM the GPU.
    xs.append(torch.empty((1 << 10) + i, device='cuda'))
    if i % 5 == 0 and len(xs) > 5:
        del xs[0]
        torch.cuda.empty_cache()
time.sleep(2)
print("done")
PY

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
  echo "  detail: $(grep experimental-kprobes "$WORK/agent.log" | head -1)"
  exit 0
else
  echo "FAIL: agent log missing experimental-kprobes status line"
  tail -30 "$WORK/agent.log"
  exit 1
fi

echo "==> [2/3] Run PyTorch fragmentation workload"
python3 "$WORK/frag_workload.py" >"$WORK/workload.log" 2>&1 || {
  echo "WARN: workload exited with non-zero; continuing to assertions"
  tail -10 "$WORK/workload.log"
}
sleep 3

echo "==> [3/3] Assertions"
PROM=$(curl -fsS http://localhost:9090/metrics)
TOTAL_EVENTS=$(echo "$PROM" | awk '/^gpu_memfrag_ioctl_event_total/ {sum+=$NF} END {print sum+0}')
if [[ "$TOTAL_EVENTS" -lt 1 ]]; then
  echo "FAIL: gpu_memfrag_ioctl_event_total = $TOTAL_EVENTS (expected >= 1)"
  echo "$PROM" | grep -E "^gpu_memfrag" | head
  exit 1
fi
echo "OK: gpu_memfrag_ioctl_event_total = $TOTAL_EVENTS"

# Distinct cmd labels: real workloads always use multiple ioctl
# command codes (alloc/free/sync). 1 is suspicious.
DISTINCT_CMDS=$(echo "$PROM" | awk '/^gpu_memfrag_ioctl_event_total\{/ {print $1}' | sort -u | wc -l)
if [[ "$DISTINCT_CMDS" -lt 2 ]]; then
  echo "WARN: only $DISTINCT_CMDS distinct cmd codes (real workload usually has >= 2)"
fi
echo "OK: distinct cmd codes = $DISTINCT_CMDS"

echo "PASS: memfrag kprobe fires under PyTorch fragmentation workload"
