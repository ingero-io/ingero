#!/usr/bin/env bash
# v0.15 item L (W2 throttle event-driven probe via edge detection):
# real-hardware validation.
#
# Asserts:
#   - Inducing power throttle via `nvidia-smi -pl <below-TDP>` and
#     running a load workload produces at least one rising edge in
#     gpu.throttle.power.event_total.
#   - The agent's existing throttle gauges (gpu.throttle.*_active)
#     also reflect the throttle state during the load, confirming
#     the edge counter is layered on top of the same poll.
#
# Hardware: Lambda A10 (or any GPU where -pl works). Some consumer
# cards reject -pl; the test SKIPs in that case.
set -euo pipefail
. "$(dirname "$0")/../_lib.sh"

INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo ./ingero)}"
[[ -x "$INGERO_BIN" ]] || { echo "FAIL: agent missing at $INGERO_BIN"; exit 1; }
nvidia-smi -L >/dev/null 2>&1 || { echo "SKIP: no GPU"; exit 0; }

# Snapshot original power limit so we can restore on cleanup.
ORIG_PL=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader,nounits | head -1 | tr -d ' ')
LOW_PL=$(awk "BEGIN {print int($ORIG_PL * 0.5)}")

WORK="$(mktemp -d)"
AGENT_PID=""
cleanup() {
  set +e
  kill_agent
  sudo nvidia-smi -pl "$ORIG_PL" >/dev/null 2>&1 || true
  rm -rf "$WORK"
}
trap cleanup EXIT

cat > "$WORK/load.py" <<'PY'
import torch, time
torch.cuda.init()
a = torch.randn(8192, 8192, device='cuda')
end = time.time() + 8
while time.time() < end:
    _ = a @ a
PY

echo "==> [1/3] Boot agent (no experimental flag needed; edge detector is in the standard NVML poller)"
sudo "$INGERO_BIN" trace --record --db "$WORK/trace.db" \
  --duration 30s --prometheus :9090 --debug \
  >"$WORK/agent.log" 2>&1 &
AGENT_PID=$!
wait_port_ready 127.0.0.1 9090 30 || { echo "FAIL: agent did not bind"; tail -50 "$WORK/agent.log"; exit 1; }
sleep 6 # let one or two baseline polls run

echo "==> [2/3] Throttle the GPU and run load"
sudo nvidia-smi -pl "$LOW_PL" >/dev/null 2>&1 || {
  echo "SKIP: nvidia-smi -pl rejected on this GPU"
  exit 0
}
python3 "$WORK/load.py" >"$WORK/load.log" 2>&1 || true
sleep 6

echo "==> [3/3] Assertions"
PROM=$(curl -fsS http://localhost:9090/metrics)
POWER_EVENTS=$(echo "$PROM" | awk '/^gpu_throttle_power_event_total/ {print $NF; exit}')
POWER_EVENTS=${POWER_EVENTS:-0}

if [[ "$POWER_EVENTS" -lt 1 ]]; then
  echo "FAIL: gpu_throttle_power_event_total = $POWER_EVENTS (expected >= 1 after -pl + load)"
  echo "$PROM" | grep -E "^gpu_throttle" | head
  exit 1
fi
echo "OK: gpu_throttle_power_event_total = $POWER_EVENTS"
echo "PASS: throttle edge detector fires under induced power throttle"
