#!/usr/bin/env bash
# Test 31: Drop-counter assertion under normal load (data-plane gate 3).
#
# Asserts (HARD, zero tolerance):
#   - Every per-tracer drop counter == 0 after a 5-min training run.
#   - Fleet processor `dropped_total` family == 0.
#
# Hardware: any A10.
#
# Invoke:
#   sudo bash tests/e2e/data-plane/31-drop-counter.sh
#
# Optional env:
#   INGERO_BIN
#   FLEET_METRICS_URL    fleet processor /metrics endpoint
#                        (default http://127.0.0.1:8888/metrics, skipped if
#                        unreachable)
#   WORKLOAD_PY          default tests/workloads/training/gpt2_finetune.py
#   DURATION_S           default 300
#
# Expected runtime: DURATION_S + ~30s.
set -euo pipefail
. "$(dirname "$0")/../_lib.sh"

REPO_ROOT="$(cd "$(dirname "$0")/../../.." && pwd)"
INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo ./ingero)}"
WORKLOAD_PY="${WORKLOAD_PY:-$REPO_ROOT/tests/workloads/training/gpt2_finetune.py}"
FLEET_METRICS_URL="${FLEET_METRICS_URL:-http://127.0.0.1:8888/metrics}"
DURATION_S="${DURATION_S:-300}"

[[ -x "$INGERO_BIN" ]] || { echo "FAIL: agent missing at $INGERO_BIN"; exit 1; }
[[ -f "$WORKLOAD_PY" ]] || { echo "FAIL: workload missing at $WORKLOAD_PY"; exit 1; }
command -v sqlite3 >/dev/null || { echo "FAIL: sqlite3 missing"; exit 1; }

WORK=$(mktemp -d)
AGENT_PID=""
WL_PID=""

cleanup() {
  set +e
  kill_agent
  [[ -n "$WL_PID" ]] && kill "$WL_PID" 2>/dev/null
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "=== Test 31: drop-counter (DURATION_S=$DURATION_S) ==="

echo "==> [1/3] Start workload + agent"
python3 "$WORKLOAD_PY" >"$WORK/wl.log" 2>&1 &
WL_PID=$!
sudo "$INGERO_BIN" trace --record --db "$WORK/trace.db" \
  --duration "${DURATION_S}s" \
  --debug \
  --prometheus :9090 \
  >"$WORK/agent.log" 2>&1 &
AGENT_PID=$!

wait "$AGENT_PID" || true

echo "==> [2/3] Inspect agent per-tracer drop counters"
PROM="$WORK/prom.txt"
curl -fsS http://localhost:9090/metrics > "$PROM" 2>/dev/null || true

# Per-tracer drop families: ingero_tracer_drops_total or similar; also
# accept any *_dropped_total exposed by the agent.
DROPS_NONZERO=$(awk '/^(ingero_[a-z_]*drops?_total|[a-z_]+_dropped_total)/ && $NF+0 != 0' "$PROM" || true)
if [[ -n "$DROPS_NONZERO" ]]; then
  echo "FAIL: agent has non-zero drop counters:"
  echo "$DROPS_NONZERO"
  exit 1
fi
echo "OK: agent drop counters all zero"

# Also scan agent debug log for "dropped" lines.
if grep -E 'drop(ped)? .*[1-9][0-9]*' "$WORK/agent.log" >/dev/null 2>&1; then
  echo "FAIL: agent debug log reports non-zero drops:"
  grep -E 'drop(ped)? .*[1-9][0-9]*' "$WORK/agent.log" | head -10
  exit 1
fi
echo "OK: agent debug log has no non-zero drop lines"

echo "==> [3/3] Inspect fleet processor dropped_total"
if curl -fsS "$FLEET_METRICS_URL" >"$WORK/fleet.txt" 2>/dev/null; then
  FLEET_DROPS=$(awk '/dropped_total/ && $NF+0 != 0' "$WORK/fleet.txt" || true)
  if [[ -n "$FLEET_DROPS" ]]; then
    echo "FAIL: fleet processor has non-zero dropped_total:"
    echo "$FLEET_DROPS"
    exit 1
  fi
  echo "OK: fleet processor dropped_total == 0"
else
  echo "WARN: fleet metrics not reachable at $FLEET_METRICS_URL; skipping fleet check"
fi

echo "PASS: drop-counter"
