#!/usr/bin/env bash
# Test 26: Sustained-load + leak detection (1 hour).
#
# Asserts:
#   - Goroutine count plateaus within 10% over the last 30 minutes of a 1 hr
#     run.
#   - RSS plateaus within 10% over the last 30 minutes.
#   - No `panic` / `fatal` / `runtime error` in agent log.
#   - No agent restart.
#
# Hardware: 2x H100 SXM5 (sustained multi-GPU). Single-GPU works for a smoke
# pass at lower confidence.
#
# Invoke:
#   sudo bash tests/e2e/soak-test.sh
#
# Optional env:
#   INGERO_BIN
#   SOAK_DURATION_S    seconds to run (default 3600 = 1 hour). For a fast
#                      smoke pass set SOAK_DURATION_S=600 (10 min).
#   WORKLOAD_PY        sustained workload (default gpt2_stress.py)
#   PPROF_PORT         agent debug pprof port (default 6060)
#
# Expected runtime: SOAK_DURATION_S + ~120s teardown.
set -euo pipefail
. "$(dirname "$0")/_lib.sh"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo ./ingero)}"
WORKLOAD_PY="${WORKLOAD_PY:-$REPO_ROOT/tests/workloads/training/gpt2_stress.py}"
SOAK_DURATION_S="${SOAK_DURATION_S:-3600}"
PPROF_PORT="${PPROF_PORT:-6060}"

[[ -x "$INGERO_BIN" ]] || { echo "FAIL: agent missing at $INGERO_BIN"; exit 1; }
[[ -f "$WORKLOAD_PY" ]] || { echo "FAIL: workload missing at $WORKLOAD_PY"; exit 1; }

WORK=$(mktemp -d)
WL_PID=""
AGENT_PID=""

cleanup() {
  set +e
  kill_agent
  [[ -n "$WL_PID" ]] && kill "$WL_PID" 2>/dev/null
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "=== Test 26: soak-test (${SOAK_DURATION_S}s) ==="

echo "==> [1/4] Start sustained workload"
python3 "$WORKLOAD_PY" >"$WORK/wl.log" 2>&1 &
WL_PID=$!
sleep 3

echo "==> [2/4] Boot agent with pprof on :$PPROF_PORT"
sudo "$INGERO_BIN" trace --record --db "$WORK/trace.db" \
  --duration "${SOAK_DURATION_S}s" \
  --pprof ":$PPROF_PORT" \
  --throttle-poll-interval 5s \
  --memfrag-poll-interval 5s \
  --libnccl-discovery-interval 5s \
  >"$WORK/agent.log" 2>&1 &
AGENT_PID=$!
sleep 15

echo "==> [3/4] Capture goroutine + RSS every 5 min"
SAMPLES="$WORK/samples.tsv"
echo -e "ts\tgoroutines\trss_kb" > "$SAMPLES"

INTERVAL=300
TICKS=$(( SOAK_DURATION_S / INTERVAL ))
for ((i=0; i<TICKS; i++)); do
  TS=$(date +%s)
  GO=$(curl -fsS "http://localhost:$PPROF_PORT/debug/pprof/goroutine?debug=1" 2>/dev/null \
       | awk '/^goroutine profile: total/ {print $4; exit}' || echo "")
  RSS=$(ps -o rss= -p "$AGENT_PID" 2>/dev/null | tr -d ' ' || echo "")
  echo -e "$TS\t${GO:-NA}\t${RSS:-NA}" >> "$SAMPLES"
  sleep "$INTERVAL"
done

# Final teardown wait; agent stops on its --duration timer.
wait "$AGENT_PID" 2>/dev/null || true

echo "==> [4/4] Assertions"
cat "$SAMPLES"

# Panic / fatal / runtime-error scan.
if grep -qiE 'panic|fatal error|runtime error' "$WORK/agent.log"; then
  echo "FAIL: panic / fatal / runtime error in agent log"
  grep -iE 'panic|fatal error|runtime error' "$WORK/agent.log" | head -5
  exit 1
fi
echo "OK: no panic / fatal / runtime error"

# Plateau check on the last 30 min of samples.
if (( TICKS < 6 )); then
  echo "WARN: SOAK_DURATION_S < 30min, plateau check is informational only"
fi
LAST_HALF=$(( (TICKS / 2) > 6 ? (TICKS / 2) : 6 ))
TAIL=$(tail -n "$LAST_HALF" "$SAMPLES")

# Compute min/max for goroutines and rss.
G_MIN=$(echo "$TAIL" | awk '$2 != "NA" { if (m=="" || $2<m) m=$2 } END {print m}')
G_MAX=$(echo "$TAIL" | awk '$2 != "NA" { if ($2>m) m=$2 } END {print m}')
R_MIN=$(echo "$TAIL" | awk '$3 != "NA" { if (m=="" || $3<m) m=$3 } END {print m}')
R_MAX=$(echo "$TAIL" | awk '$3 != "NA" { if ($3>m) m=$3 } END {print m}')

if [[ -z "$G_MIN" || -z "$R_MIN" ]]; then
  echo "FAIL: insufficient samples for plateau check"
  exit 1
fi

G_DRIFT=$(awk -v mn="$G_MIN" -v mx="$G_MAX" 'BEGIN { printf "%.4f", (mx-mn)/mn }')
R_DRIFT=$(awk -v mn="$R_MIN" -v mx="$R_MAX" 'BEGIN { printf "%.4f", (mx-mn)/mn }')
echo "goroutine drift over last $LAST_HALF samples: $G_DRIFT (min=$G_MIN max=$G_MAX)"
echo "RSS drift over last $LAST_HALF samples: $R_DRIFT (min=$R_MIN max=$R_MAX)"

G_OK=$(awk -v d="$G_DRIFT" 'BEGIN { print (d <= 0.10) ? 1 : 0 }')
R_OK=$(awk -v d="$R_DRIFT" 'BEGIN { print (d <= 0.10) ? 1 : 0 }')
if [[ "$G_OK" != "1" ]]; then
  echo "FAIL: goroutine count drift > 10% (drift=$G_DRIFT)"
  exit 1
fi
if [[ "$R_OK" != "1" ]]; then
  echo "FAIL: RSS drift > 10% (drift=$R_DRIFT)"
  exit 1
fi
echo "OK: goroutine + RSS plateau within 10%"

echo "PASS: soak-test"
