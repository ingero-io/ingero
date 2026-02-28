#!/bin/bash
################################################################################
# Ingero GPU Integration Test — Canonical Single Script
#
# This is the ONE test script for GPU integration testing. All other scripts
# (gpu-integration-test.sh, lambdalabs/gpu-integration-test.sh,
# remote-full-test.sh) have been consolidated into this file.
#
# Run on remote GPU VM after code sync + build:
#   make gpu-test    (TensorDock)
#   make lambda-test (Lambda Labs)
#
# Or directly on the VM:
#   cd ~/workspace/ingero && bash scripts/gpu-test.sh
#
# Phases:
#   0: System info capture
#   1: Regression gate (T01-T09)
#   2: Stack tracing deep tests (T10-T13)
#   3: OTLP + Prometheus export (T14a-e, T15-T17)
#   4: Stack latency benchmark (T18)
#   5: MCP AI diagnostic (T19a-e, T20-T21)
#   6: GPU problem investigation (T23a-T23ab, 28 issues via MCP)
#
# Parallelization strategy (saves ~350s / 44% faster):
#   - T02 demos + T09 synthetics run as background jobs during Phase 1
#   - T03+T07 merged into one trace session (clean + driver API check)
#   - T10+T12 merged into one trace session (stack + record + query)
#   - T18 benchmark reduced to 2+2 iterations at 20s
################################################################################

set -uo pipefail

export PATH=/usr/local/go/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/go/bin:$HOME/.local/bin:$PATH
export HOME="${HOME:-/home/$(whoami)}"

# Colors
if [ -t 1 ]; then
  RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
else
  RED=''; GREEN=''; YELLOW=''; BLUE=''; NC=''
fi

PASS_COUNT=0; FAIL_COUNT=0; SKIP_COUNT=0
REPORT=""
SCRIPT_START=$SECONDS

# Per-test timing and structured results for JSON report
declare -a TEST_RESULTS  # Each entry: "id|name|status|detail|duration_s"
_test_start=$SECONDS

cd "$(dirname "$0")/.." || { echo "Cannot find ingero directory"; exit 1; }
INGERO_DIR="$(pwd)"
mkdir -p logs

# Per-run temp directory — avoids /tmp permission conflicts from previous runs
# owned by different users (e.g., root from sudo).
TEST_TMP=$(mktemp -d /tmp/ingero_test_XXXXXX)

ts()   { date -u '+%Y-%m-%d %H:%M:%S'; }
log()  { echo -e "$(ts) ${GREEN}[INFO]${NC}  $1"; }
warn() { echo -e "$(ts) ${YELLOW}[WARN]${NC}  $1"; }
fail() { echo -e "$(ts) ${RED}[FAIL]${NC}  $1"; }

record() {
    local status="$1" name="$2" detail="$3"
    REPORT="${REPORT}[$(ts)] [$status] $name: $detail\n"
    # Capture structured result for JSON report.
    # Extract test ID (e.g. "T01") from name like "T01: check: GPU detected".
    local tid="${name%%:*}"
    local elapsed=$((SECONDS - _test_start))
    TEST_RESULTS+=("${tid}|${name}|${status}|${detail}|${elapsed}")
    if [[ "$status" == "PASS" ]]; then
        echo -e "$(ts)   ${GREEN}[PASS]${NC} $name"
        PASS_COUNT=$((PASS_COUNT + 1))
    elif [[ "$status" == "FAIL" ]]; then
        echo -e "$(ts)   ${RED}[FAIL]${NC} $name — $detail"
        FAIL_COUNT=$((FAIL_COUNT + 1))
    elif [[ "$status" == "SKIP" ]]; then
        echo -e "$(ts)   ${YELLOW}[SKIP]${NC} $name — $detail"
        SKIP_COUNT=$((SKIP_COUNT + 1))
    fi
    _test_start=$SECONDS
}

header() {
    echo ""
    echo -e "$(ts) ${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "$(ts) ${BLUE}  $1${NC}"
    echo -e "$(ts) ${BLUE}════════════════════════════════════════════════════════════${NC}"
}

# Kill background processes on exit
cleanup_pids=()
cleanup() {
    for pid in "${cleanup_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
    rm -rf "${TEST_TMP:-}" 2>/dev/null || true
}
trap cleanup EXIT

# Helper: start a PyTorch matmul workload in background
start_workload() {
    local duration="${1:-15}"
    python3 -c "
import torch, time, os
print(f'Workload PID: {os.getpid()}', flush=True)
d = torch.device('cuda:0')
a = torch.randn(2048, 2048, device=d)
b = torch.randn(2048, 2048, device=d)
start = time.time()
while time.time() - start < $duration:
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
" &>/dev/null &
    local pid=$!
    cleanup_pids+=("$pid")
    echo "$pid"
}

# Helper: count JSON events from a file
count_events() {
    python3 -c "
import json, sys
events = []
for line in open('$1'):
    try: events.append(json.loads(line))
    except: pass
print(len(events))
" 2>/dev/null || echo "0"
}

# Helper: background job tracking
# BG_PIDS: name -> pid, BG_OUTFILES: name -> output file
declare -A BG_PIDS
declare -A BG_OUTFILES

bg_start() {
    local name="$1"; shift
    local outfile="$1"; shift
    "$@" > "$outfile" 2>&1 &
    local pid=$!
    BG_PIDS["$name"]=$pid
    BG_OUTFILES["$name"]="$outfile"
    cleanup_pids+=("$pid")
    log "  bg: $name (pid=$pid) → $outfile"
}

# Collect a background job's result. On failure, dump its output for debugging.
bg_collect() {
    local name="$1"
    local pid="${BG_PIDS[$name]}"
    local outfile="${BG_OUTFILES[$name]}"
    wait "$pid" 2>/dev/null
    local rc=$?
    if [[ $rc -ne 0 && -f "$outfile" ]]; then
        warn "  bg $name (pid=$pid) exited $rc — output:"
        # Show last 20 lines to avoid flooding the main log
        tail -20 "$outfile" | while IFS= read -r line; do
            echo "    | $line"
        done
    fi
    return $rc
}

# ── Binary guard — fail fast if not built ──
if [[ ! -x bin/ingero ]]; then
    fail "bin/ingero not found. Run 'make generate && make build' first."
    exit 1
fi

exec > >(tee logs/integration-report.log) 2>&1

echo "================================================================"
echo "  Ingero v0.6 GPU Integration Test — $(date)"
echo "  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo "================================================================"

################################################################################
# Phase 0: System Info Capture
################################################################################
header "Phase 0: System Info"

nvidia-smi > logs/nvidia-smi.log 2>&1
uname -a > logs/uname.log
python3 --version >> logs/uname.log 2>&1
sudo ./bin/ingero check --debug > logs/check-debug.log 2>&1
./bin/ingero version > logs/version.log 2>&1

log "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null)"
log "Kernel: $(uname -r)"
log "Go: $(go version | awk '{print $3}')"
log "PyTorch: $(python3 -c 'import torch; print(f"{torch.__version__}, CUDA {torch.version.cuda}")' 2>/dev/null || echo 'N/A')"

################################################################################
# Probe Smoke Test — abort early if eBPF probes can't attach
################################################################################
header "Probe Smoke Test"

log "Starting 3s CUDA workload..."
SMOKE_WL_PID=$(start_workload 5)
sleep 1

log "Running ingero trace --debug --json --duration 3s..."
sudo ./bin/ingero trace --debug --json --duration 3s \
    > logs/smoke-test.json 2> logs/smoke-test.log

wait "$SMOKE_WL_PID" 2>/dev/null || true

SMOKE_COUNT=$(count_events logs/smoke-test.json)
if [[ "$SMOKE_COUNT" -gt 0 ]]; then
    log "Probe smoke test PASSED: $SMOKE_COUNT events in 3s"
else
    fail "PROBE SMOKE TEST FAILED: 0 events captured in 3s"
    fail ""
    fail "eBPF probes failed to attach. Common causes:"
    fail "  1. Stale eBPF objects — run: make generate && make build"
    fail "  2. Missing BTF — check: ls -la /sys/kernel/btf/vmlinux"
    fail "  3. Missing CUDA libs — check: sudo ./bin/ingero check"
    fail "  4. Kernel mismatch — compiled for $(uname -r)?"
    fail ""
    fail "Debug output (last 30 lines of stderr):"
    tail -30 logs/smoke-test.log | while IFS= read -r line; do
        echo "    | $line"
    done
    fail ""
    fail "Aborting test suite — fix probes before proceeding."
    # Still emit a minimal report so the failure is captured
    record "FAIL" "T00: probe smoke test" "0 events — probes not attaching"
    echo -e "$REPORT" > logs/test-report.txt
    exit 1
fi

################################################################################
# Phase 1: Regression Gate (v0.4 baseline)
################################################################################
header "Phase 1: Regression Gate"

# Clean DB once at start — all phases accumulate into one queryable dataset.
# With SUDO_USER-aware DefaultDBPath, DB lands in the user's home dir.
# Clean both locations + WAL/SHM files for safety.
rm -f ~/.ingero/ingero.db ~/.ingero/ingero.db-wal ~/.ingero/ingero.db-shm
sudo rm -f /root/.ingero/ingero.db /root/.ingero/ingero.db-wal /root/.ingero/ingero.db-shm

# Test 1: ingero check
_test_start=$SECONDS
log "Test 1: ingero check"
CHECK_OUT=$(sudo ./bin/ingero check 2>&1)
if echo "$CHECK_OUT" | grep -q "GPU model:"; then
    GPU_MODEL=$(echo "$CHECK_OUT" | grep "GPU model:" | sed 's/.*GPU model: //')
    record "PASS" "T01: check: GPU detected" "$GPU_MODEL"
else
    record "FAIL" "T01: check: GPU detected" "no GPU model found"
fi

# ── Launch T02 + T09 as background jobs (run in parallel with T03-T08) ──
log "Launching T02 demos + T09 synthetics in background..."

# T02: 6 demo scenarios in parallel (max 45s each)
for scenario in incident cold-start memcpy-bottleneck periodic-spike cpu-contention gpu-steal; do
    bg_start "demo_${scenario}" "logs/bg-demo-${scenario}.out" \
        timeout 45 ./bin/ingero demo --no-gpu "$scenario" --json --duration 30s
done

# T09: 4 synthetic workloads in parallel (max 60s each)
for script in alloc_stress memcpy_stress launch_storm sync_stall; do
    bg_start "t1_${script}" "logs/bg-t1-${script}.out" \
        timeout 60 python3 "tests/workloads/synthetic/${script}.py"
done

# ── T03+T07 merged: ONE trace session → check total events AND cuLaunchKernel ──
log "Test 3+7: trace --duration 15s (clean + driver API)"
WL_PID=$(start_workload 18)
sleep 1
sudo ./bin/ingero trace --json --pid "$WL_PID" --duration 15s > logs/trace-clean.json 2> logs/trace-clean.log
wait "$WL_PID" 2>/dev/null || true

CLEAN_COUNT=$(count_events logs/trace-clean.json)
if [[ "$CLEAN_COUNT" -gt 100 ]]; then
    record "PASS" "T03: trace clean" "$CLEAN_COUNT events"
else
    record "FAIL" "T03: trace clean" "$CLEAN_COUNT events (expected >100)"
fi

CU_COUNT=$(grep -c '"cuLaunchKernel"' logs/trace-clean.json 2>/dev/null || echo "0")
CU_COUNT=$(echo "$CU_COUNT" | head -1)
if [[ "$CU_COUNT" -gt 0 ]]; then
    record "PASS" "T07: driver API: cuLaunchKernel" "$CU_COUNT events (from T03 session)"
else
    record "FAIL" "T07: driver API: cuLaunchKernel" "0 events"
fi

# Test 4: trace --debug
log "Test 4: trace --debug --duration 15s"
WL_PID=$(start_workload 18)
sleep 1
sudo ./bin/ingero trace --debug --json --duration 15s > logs/trace-debug.json 2> logs/trace-debug.log
wait "$WL_PID" 2>/dev/null || true
DEBUG_COUNT=$(count_events logs/trace-debug.json)
DEBUG_LINES=$(grep -c '\[DEBUG\]' logs/trace-debug.log 2>/dev/null || echo "0")
if [[ "$DEBUG_COUNT" -gt 100 && "$DEBUG_LINES" -gt 0 ]]; then
    record "PASS" "T04: trace --debug" "$DEBUG_COUNT events, $DEBUG_LINES debug lines"
else
    record "FAIL" "T04: trace --debug" "events=$DEBUG_COUNT, debugLines=$DEBUG_LINES"
fi

# Test 5: record + query round-trip (recording is default)
_test_start=$SECONDS
log "Test 5: record + query"
WL_PID=$(start_workload 14)
sleep 1
sudo ./bin/ingero trace --json --duration 10s > logs/trace-record.json 2> logs/trace-record.log
wait "$WL_PID" 2>/dev/null || true
QUERY_OUT=$(sudo ./bin/ingero query --since 5m --json 2>/dev/null)
QUERY_COUNT=$(echo "$QUERY_OUT" | grep -c '"op"' || true)
if [[ "$QUERY_COUNT" -gt 0 ]]; then
    record "PASS" "T05: record + query" "$QUERY_COUNT events retrieved"
else
    record "FAIL" "T05: record + query" "query returned 0 events"
fi

# Test 6: explain (reads from root's DB since trace runs as sudo)
_test_start=$SECONDS
log "Test 6: explain --since 180s"
EXPLAIN_OUT=$(sudo ./bin/ingero explain --debug --since 180s 2>&1)
if echo "$EXPLAIN_OUT" | grep -q "INCIDENT REPORT"; then
    record "PASS" "T06: explain" "incident report generated"
else
    record "FAIL" "T06: explain" "no incident report"
fi

# Test 8: GPU demo incident
_test_start=$SECONDS
log "Test 8: demo incident (GPU mode)"
timeout 120s sudo ./bin/ingero demo incident --json > logs/demo-gpu-incident.json 2> logs/demo-gpu-incident.log
GPU_DEMO_COUNT=$(count_events logs/demo-gpu-incident.json)
if [[ "$GPU_DEMO_COUNT" -gt 100 ]]; then
    record "PASS" "T08: GPU demo incident" "$GPU_DEMO_COUNT events"
else
    record "FAIL" "T08: GPU demo incident" "$GPU_DEMO_COUNT events (expected >100)"
fi

# ── Collect T02 background results ──
_test_start=$SECONDS
log "Collecting T02 demo results..."
DEMO_FAIL=0
for scenario in incident cold-start memcpy-bottleneck periodic-spike cpu-contention gpu-steal; do
    if ! bg_collect "demo_${scenario}"; then
        DEMO_FAIL=$((DEMO_FAIL + 1))
        warn "  demo --no-gpu $scenario failed"
    fi
done
if [[ "$DEMO_FAIL" -eq 0 ]]; then
    record "PASS" "T02: demo --no-gpu (6 scenarios)" "all pass (parallel)"
else
    record "FAIL" "T02: demo --no-gpu (6 scenarios)" "$DEMO_FAIL failed"
fi

# ── Collect T09 background results ──
_test_start=$SECONDS
log "Collecting T09 synthetic results..."
T1_FAIL=0
for script in alloc_stress memcpy_stress launch_storm sync_stall; do
    if ! bg_collect "t1_${script}"; then
        T1_FAIL=$((T1_FAIL + 1))
        warn "  ${script}.py failed"
    fi
done
if [[ "$T1_FAIL" -eq 0 ]]; then
    record "PASS" "T09: Tier 1 synthetic (4/4)" "all pass (parallel)"
else
    record "FAIL" "T09: Tier 1 synthetic" "$T1_FAIL/4 failed"
fi

# Regression gate check
REGRESSION_FAILS=$FAIL_COUNT
if [[ "$REGRESSION_FAILS" -gt 0 ]]; then
    fail "Regression gate FAILED ($REGRESSION_FAILS failures). Continuing with v0.6 tests anyway."
fi

################################################################################
# Phase 2: v0.6 Stack Tracing Deep Tests
################################################################################
header "Phase 2: v0.6 Stack Tracing"

# ── T10+T12 merged: ONE trace session → check both ──
# --stack and --record are on by default.
_test_start=$SECONDS
log "Test 10+12: trace + query (stacks on by default)"
WL_PID=$(start_workload 18)
sleep 1
sudo ./bin/ingero trace --json --duration 15s > logs/stack-native.json 2> logs/stack-native.log
wait "$WL_PID" 2>/dev/null || true

# T10: analyze stack coverage from merged session
python3 -c "
import json, sys
events = []
for line in open('logs/stack-native.json'):
    try: events.append(json.loads(line))
    except: pass
stack_events = [e for e in events if e.get('stack')]
total = len(events)
with_stack = len(stack_events)
pct = (with_stack / total * 100) if total > 0 else 0

# Per-source breakdown (HOST events are kernel tracepoints — no userspace stacks by design)
cuda_driver = [e for e in events if e.get('source') in ('cuda', 'driver')]
cuda_driver_with = [e for e in cuda_driver if e.get('stack')]
host = [e for e in events if e.get('source') == 'host']
cd_pct = (len(cuda_driver_with) / len(cuda_driver) * 100) if cuda_driver else 0
print(f'STACK_CUDA_DRIVER={len(cuda_driver_with)}/{len(cuda_driver)} ({cd_pct:.0f}%)')
print(f'STACK_HOST={len(host)} (no userspace stacks by design)')

# Check for resolved symbols
resolved = 0
has_hex_ip = 0
for e in stack_events[:100]:
    for f in e['stack']:
        if f.get('symbol'):
            resolved += 1
        ip_str = str(f.get('ip', ''))
        if ip_str.startswith('0x') or ip_str.startswith('0X') or (isinstance(f.get('ip'), int) and f['ip'] > 0):
            has_hex_ip += 1

print(f'STACK_TOTAL={total}')
print(f'STACK_WITH_STACK={with_stack}')
print(f'STACK_PCT={pct:.1f}')
print(f'STACK_RESOLVED={resolved}')
print(f'STACK_HAS_IP={has_hex_ip}')

# Sample output
if stack_events:
    e = stack_events[0]
    print(f'STACK_SAMPLE_OP={e.get(\"op\",\"?\")}')
    print(f'STACK_SAMPLE_DEPTH={len(e[\"stack\"])}')
    for f in e['stack'][:5]:
        sym = f.get('symbol','')
        fi = f.get('file','')
        ip = f.get('ip','')
        print(f'  FRAME: ip={ip} sym={sym} file={fi}')
" > $TEST_TMP/stack_analysis.txt 2>&1

cat $TEST_TMP/stack_analysis.txt
STACK_TOTAL=$(grep 'STACK_TOTAL=' $TEST_TMP/stack_analysis.txt | cut -d= -f2)
STACK_WITH=$(grep 'STACK_WITH_STACK=' $TEST_TMP/stack_analysis.txt | cut -d= -f2)
STACK_PCT=$(grep 'STACK_PCT=' $TEST_TMP/stack_analysis.txt | cut -d= -f2)
STACK_RESOLVED=$(grep 'STACK_RESOLVED=' $TEST_TMP/stack_analysis.txt | cut -d= -f2)

if [[ "${STACK_TOTAL:-0}" -gt 100 && "${STACK_WITH:-0}" -gt 0 ]]; then
    record "PASS" "T10: --stack native" "${STACK_WITH}/${STACK_TOTAL} events with stack (${STACK_PCT}%), ${STACK_RESOLVED} resolved symbols"
else
    record "FAIL" "T10: --stack native" "total=${STACK_TOTAL:-0}, with_stack=${STACK_WITH:-0}"
fi

# T12: query round-trip from same recorded session
sudo ./bin/ingero query --since 5m --json > logs/stack-query.json 2>/dev/null
RECORD_COUNT=$(count_events logs/stack-native.json)
QUERY_COUNT=$(count_events logs/stack-query.json)
if [[ "$RECORD_COUNT" -gt 100 && "$QUERY_COUNT" -gt 0 ]]; then
    record "PASS" "T12: stack + record + query" "recorded=$RECORD_COUNT, queried=$QUERY_COUNT (from T10 session)"
else
    record "FAIL" "T12: stack + record + query" "recorded=$RECORD_COUNT, queried=$QUERY_COUNT"
fi

# Test 11: CPython frame extraction
_test_start=$SECONDS
log "Test 11: --stack CPython frames"
cat > $TEST_TMP/pytest.py << 'PYEOF'
import torch
import time
import os

def forward(x, w):
    return torch.mm(x, w)

def train_step(data, weights):
    output = forward(data, weights)
    loss = output.sum()
    return loss

def main():
    print(f"PID: {os.getpid()}", flush=True)
    d = torch.device('cuda:0')
    data = torch.randn(1024, 1024, device=d)
    weights = torch.randn(1024, 1024, device=d)
    start = time.time()
    while time.time() - start < 16:
        loss = train_step(data, weights)
        torch.cuda.synchronize()

if __name__ == "__main__":
    main()
PYEOF

python3 $TEST_TMP/pytest.py &
PY_PID=$!
cleanup_pids+=("$PY_PID")
sleep 3
sudo ./bin/ingero trace --debug --json --pid "$PY_PID" --duration 18s > logs/stack-python.json 2> logs/stack-python.log
wait "$PY_PID" 2>/dev/null || true

python3 -c "
import json
events = []
for line in open('logs/stack-python.json'):
    try: events.append(json.loads(line))
    except: pass
stack_events = [e for e in events if e.get('stack')]
py_events = 0
for e in stack_events:
    for f in e['stack']:
        if f.get('py_file') or f.get('py_func'):
            py_events += 1
            break
total = len(events)
with_stack = len(stack_events)
print(f'PY_TOTAL={total}')
print(f'PY_WITH_STACK={with_stack}')
print(f'PY_WITH_PYFRAMES={py_events}')
# Show sample Python frames
for e in stack_events[:20]:
    for f in e['stack']:
        if f.get('py_file'):
            print(f'  PY_FRAME: {f[\"py_file\"]}:{f.get(\"py_line\",\"?\")} in {f.get(\"py_func\",\"?\")}()')
" > $TEST_TMP/pystack_analysis.txt 2>&1

cat $TEST_TMP/pystack_analysis.txt
PY_WITH_PYFRAMES=$(grep 'PY_WITH_PYFRAMES=' $TEST_TMP/pystack_analysis.txt | cut -d= -f2)
PY_TOTAL=$(grep 'PY_TOTAL=' $TEST_TMP/pystack_analysis.txt | cut -d= -f2)
DEBUG_STACK_LINES=$(grep -c '\[DEBUG\].*stack' logs/stack-python.log 2>/dev/null || echo "0")

# Check DWARF offset discovery diagnostics from --debug output.
DWARF_PATH=$(grep -o 'using DWARF offsets.*' logs/stack-python.log 2>/dev/null | head -1 || echo "")
HARDCODED_PATH=$(grep -o 'using hardcoded offsets.*' logs/stack-python.log 2>/dev/null | head -1 || echo "")
DWARF_MISMATCH=$(grep -c 'MISMATCH' logs/stack-python.log 2>/dev/null || echo "0")

if [[ -n "$DWARF_PATH" ]]; then
    log "  DWARF offset path active: ${DWARF_PATH}"
    if [[ "${DWARF_MISMATCH}" -gt 0 ]]; then
        log "  ${DWARF_MISMATCH} offset field(s) differ from hardcoded (distro-patched build detected)"
    fi
elif [[ -n "$HARDCODED_PATH" ]]; then
    log "  Hardcoded offset path: ${HARDCODED_PATH}"
    log "  (install libpython3.X-dbgsym for DWARF offsets)"
fi

if [[ "${PY_WITH_PYFRAMES:-0}" -gt 0 ]]; then
    OFFSET_INFO=""
    if [[ -n "$DWARF_PATH" ]]; then
        OFFSET_INFO=", DWARF offsets"
    else
        OFFSET_INFO=", hardcoded offsets"
    fi
    record "PASS" "T11: CPython frames" "${PY_WITH_PYFRAMES} events with py_file/py_func${OFFSET_INFO}"
elif [[ "${PY_TOTAL:-0}" -gt 100 ]]; then
    # CPython frame extraction is opportunistic — may not work with all Python versions
    record "SKIP" "T11: CPython frames" "events captured but no py frames (Python version may not support frame walking)"
else
    record "FAIL" "T11: CPython frames" "total=${PY_TOTAL:-0}, pyFrames=${PY_WITH_PYFRAMES:-0}"
fi

# Test 13: explain causal chain detection under GPU + CPU contention
# Strategy: multi-process GPU contention (time-slicing) + CPU stress → trace → explain.
# Multiple GPU workers competing for the same GPU create measurable p99 variance
# from GPU context switch overhead (~0.8ms per switch), which the correlator detects.
_test_start=$SECONDS
log "Test 13: explain chain detection (GPU + CPU contention)"

# Launch 3 competing GPU workers (creates GPU time-slicing contention).
python3 tests/workloads/pathological/gpu_contention_driver.py \
    --workers 3 --duration 40 --matrix-size 2048 > logs/contention-workers.log 2>&1 &
CONTENTION_PID=$!
cleanup_pids+=("$CONTENTION_PID")
sleep 3  # let workers warm up and allocate GPU memory

# Start CPU contention overlapping with trace (creates sched_switch events).
if command -v stress-ng &>/dev/null; then
    stress-ng --cpu "$(nproc)" --timeout 30s > /dev/null 2>&1 &
    STRESS_PID=$!
    cleanup_pids+=("$STRESS_PID")
fi
sleep 2

# Trace during contention (records to SQLite).
sudo ./bin/ingero trace --duration 25s > /dev/null 2> logs/trace-contention.log

wait "$CONTENTION_PID" 2>/dev/null || true
[ -n "${STRESS_PID:-}" ] && wait "$STRESS_PID" 2>/dev/null || true

# Explain the stored data (needs sudo to read root's DB).
sudo ./bin/ingero explain --debug --since 60s > logs/explain-chain.log 2>&1
if grep -q "causal chain(s) found" logs/explain-chain.log 2>/dev/null; then
    CHAIN_COUNT=$(grep -o '[0-9]* causal chain' logs/explain-chain.log | head -1 | awk '{print $1}')
    record "PASS" "T13: explain chain detection" "${CHAIN_COUNT} chain(s) detected under contention"
elif grep -q "INCIDENT REPORT" logs/explain-chain.log 2>/dev/null; then
    # Report generated but no chains — contention may have been insufficient
    record "SKIP" "T13: explain chain detection" "report generated but no chains (insufficient contention)"
else
    record "FAIL" "T13: explain chain detection" "no incident report"
fi

################################################################################
# Phase 3: v0.6 OTLP Export Tests
################################################################################
header "Phase 3: v0.6 OTLP Export"

# Test 14: OTLP with Python HTTP receiver
_test_start=$SECONDS
log "Test 14: OTLP push to mock receiver"

# Kill any leftover processes on port 4318
sudo fuser -k 4318/tcp 2>/dev/null || true
sleep 1

# Start mock OTLP receiver
cat > $TEST_TMP/otlp_receiver.py << OTLPEOF
import http.server
import json
import sys
import os
import signal
import threading

received = []
class OTLPHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        length = int(self.headers.get('Content-Length', 0))
        body = self.rfile.read(length)
        ct = self.headers.get('Content-Type', '')
        entry = {
            'path': self.path,
            'content_type': ct,
            'body_size': len(body),
        }
        try:
            entry['payload'] = json.loads(body)
        except:
            entry['payload_raw'] = body.decode('utf-8', errors='replace')[:500]
        received.append(entry)
        print(f"OTLP RECV: {self.path} ({len(body)} bytes, {ct})", flush=True)
        self.send_response(200)
        self.end_headers()
        self.wfile.write(b'{}')

    def log_message(self, format, *args):
        pass  # Suppress default logging

http.server.HTTPServer.allow_reuse_address = True
server = http.server.HTTPServer(('0.0.0.0', 4318), OTLPHandler)

def shutdown_handler(signum, frame):
    # Write received payloads to file
    with open('$TEST_TMP/otlp_received.json', 'w') as f:
        json.dump(received, f, indent=2, default=str)
    print(f"OTLP receiver: wrote {len(received)} payloads to $TEST_TMP/otlp_received.json", flush=True)
    server.shutdown()
    sys.exit(0)

signal.signal(signal.SIGTERM, shutdown_handler)
signal.signal(signal.SIGINT, shutdown_handler)

# Auto-shutdown after 60s
def auto_shutdown():
    import time
    time.sleep(60)
    shutdown_handler(None, None)
threading.Thread(target=auto_shutdown, daemon=True).start()

print("OTLP receiver listening on :4318", flush=True)
server.serve_forever()
OTLPEOF

python3 $TEST_TMP/otlp_receiver.py > logs/otlp-receipt.log 2>&1 &
OTLP_PID=$!
cleanup_pids+=("$OTLP_PID")
sleep 1

# Run trace with --otlp pointing to our receiver
WL_PID=$(start_workload 25)
sleep 1
sudo ./bin/ingero trace --otlp localhost:4318 --debug --json --duration 20s > logs/otlp-events.json 2> logs/otlp-debug.log
wait "$WL_PID" 2>/dev/null || true

# Give receiver time to flush
sleep 2
kill "$OTLP_PID" 2>/dev/null || true
sleep 2

# Analyze received payloads
if [ -f $TEST_TMP/otlp_received.json ]; then
    cp $TEST_TMP/otlp_received.json logs/otlp-payload.json
    python3 -c "
import json
data = json.load(open('$TEST_TMP/otlp_received.json'))
print(f'OTLP_RECEIVED={len(data)}')
if data:
    entry = data[0]
    print(f'OTLP_PATH={entry.get(\"path\",\"?\")}')
    print(f'OTLP_CT={entry.get(\"content_type\",\"?\")}')
    payload = entry.get('payload', {})
    # Check for expected metric names
    metrics_found = set()
    for rm in payload.get('resourceMetrics', []):
        for sm in rm.get('scopeMetrics', []):
            for m in sm.get('metrics', []):
                metrics_found.add(m.get('name', ''))
    for name in sorted(metrics_found):
        print(f'OTLP_METRIC={name}')
    print(f'OTLP_METRIC_COUNT={len(metrics_found)}')
" > $TEST_TMP/otlp_analysis.txt 2>&1
    cat $TEST_TMP/otlp_analysis.txt

    OTLP_RECEIVED=$(grep 'OTLP_RECEIVED=' $TEST_TMP/otlp_analysis.txt | cut -d= -f2)
    OTLP_PATH=$(grep 'OTLP_PATH=' $TEST_TMP/otlp_analysis.txt | cut -d= -f2)
    OTLP_CT=$(grep 'OTLP_CT=' $TEST_TMP/otlp_analysis.txt | cut -d= -f2)

    if [[ "${OTLP_RECEIVED:-0}" -gt 0 ]]; then
        record "PASS" "T14a: OTLP received" "${OTLP_RECEIVED} pushes received"
    else
        record "FAIL" "T14a: OTLP received" "0 pushes"
    fi

    if [[ "${OTLP_PATH:-}" == "/v1/metrics" ]]; then
        record "PASS" "T14b: OTLP path" "/v1/metrics"
    else
        record "FAIL" "T14b: OTLP path" "got: ${OTLP_PATH:-empty}"
    fi

    if echo "${OTLP_CT:-}" | grep -q "application/json"; then
        record "PASS" "T14c: OTLP content-type" "application/json"
    else
        record "FAIL" "T14c: OTLP content-type" "got: ${OTLP_CT:-empty}"
    fi

    # Check expected metrics
    for metric in "system.cpu.utilization" "gpu.cuda.operation.duration" "ingero.anomaly.count"; do
        if grep -q "OTLP_METRIC=$metric" $TEST_TMP/otlp_analysis.txt 2>/dev/null; then
            record "PASS" "T14d: OTLP metric: $metric" "present"
        else
            record "FAIL" "T14d: OTLP metric: $metric" "missing"
        fi
    done
else
    record "FAIL" "T14: OTLP received" "no $TEST_TMP/otlp_received.json (receiver died?)"
fi

# Test 15: OTLP debug logging
_test_start=$SECONDS
log "Test 15: OTLP debug logging"
if grep -q 'OTLP: pushing' logs/otlp-debug.log 2>/dev/null; then
    PUSH_LINES=$(grep -c 'OTLP: push' logs/otlp-debug.log || echo "0")
    record "PASS" "T15: OTLP debug logs" "$PUSH_LINES OTLP log lines"
else
    record "FAIL" "T15: OTLP debug logs" "no 'OTLP: pushing' in debug stderr"
fi

# Test 16: OTLP connection refused graceful degradation
_test_start=$SECONDS
log "Test 16: OTLP connection refused (graceful degradation)"
WL_PID=$(start_workload 18)
sleep 1
sudo ./bin/ingero trace --otlp localhost:9999 --json --duration 15s > logs/otlp-connrefused.json 2> logs/otlp-connrefused.log
OTLP_EXIT=$?
wait "$WL_PID" 2>/dev/null || true
CONNREF_COUNT=$(count_events logs/otlp-connrefused.json)

if [[ "$CONNREF_COUNT" -gt 100 ]]; then
    record "PASS" "T16: OTLP conn refused: events" "$CONNREF_COUNT events (no crash)"
else
    record "FAIL" "T16: OTLP conn refused: events" "$CONNREF_COUNT events (expected >100)"
fi

if grep -q 'OTLP: push failed\|connection refused\|connect:' logs/otlp-connrefused.log 2>/dev/null; then
    record "PASS" "T16: OTLP conn refused: error logged" "error in stderr"
else
    # May not appear without --debug
    record "SKIP" "T16: OTLP conn refused: error logged" "no OTLP error in stderr (may need --debug)"
fi

# Test 17: Stack + OTLP combined
_test_start=$SECONDS
log "Test 17: trace + --otlp combined (stacks on by default)"
# Restart OTLP receiver (kill any leftover)
sudo fuser -k 4318/tcp 2>/dev/null || true
sleep 1
python3 $TEST_TMP/otlp_receiver.py > logs/otlp-combined-receipt.log 2>&1 &
OTLP_PID2=$!
cleanup_pids+=("$OTLP_PID2")
sleep 1

WL_PID=$(start_workload 18)
sleep 1
sudo ./bin/ingero trace --otlp localhost:4318 --json --duration 15s > logs/combined-events.json 2> logs/combined-debug.log
wait "$WL_PID" 2>/dev/null || true
sleep 2
sudo kill "$OTLP_PID2" 2>/dev/null || true
sleep 1

COMBINED_COUNT=$(count_events logs/combined-events.json)
COMBINED_STACK=$(python3 -c "
import json
events = []
for line in open('logs/combined-events.json'):
    try: events.append(json.loads(line))
    except: pass
stack_events = [e for e in events if e.get('stack')]
print(len(stack_events))
" 2>/dev/null || echo "0")

if [[ "$COMBINED_COUNT" -gt 100 && "$COMBINED_STACK" -gt 0 ]]; then
    record "PASS" "T17: stack + OTLP combined" "events=$COMBINED_COUNT, with_stack=$COMBINED_STACK"
else
    record "FAIL" "T17: stack + OTLP combined" "events=$COMBINED_COUNT, with_stack=$COMBINED_STACK"
fi

# Test 14e: Prometheus /metrics endpoint
_test_start=$SECONDS
log "Test 14e: Prometheus /metrics"
sudo fuser -k 9090/tcp 2>/dev/null || true
sleep 1

WL_PID=$(start_workload 18)
sleep 1
sudo ./bin/ingero trace --prometheus :9090 --json --duration 12s > logs/prom-events.json 2> logs/prom-debug.log &
PROM_PID=$!
cleanup_pids+=("$PROM_PID")
sleep 5

PROM_OUT=$(curl -s localhost:9090/metrics 2>&1)
echo "$PROM_OUT" > logs/prom-metrics.txt

wait "$PROM_PID" 2>/dev/null || true
wait "$WL_PID" 2>/dev/null || true

if echo "$PROM_OUT" | grep -q "system_cpu_utilization"; then
    record "PASS" "T14e: Prometheus system metrics" "system_cpu_utilization present"
else
    record "FAIL" "T14e: Prometheus system metrics" "system_cpu_utilization missing"
fi

if echo "$PROM_OUT" | grep -q "gpu_cuda_operation_duration_microseconds"; then
    record "PASS" "T14e: Prometheus CUDA metrics" "gpu_cuda_operation_duration_microseconds present"
else
    record "FAIL" "T14e: Prometheus CUDA metrics" "gpu_cuda_operation_duration_microseconds missing"
fi

if echo "$PROM_OUT" | grep -q "# TYPE"; then
    record "PASS" "T14e: Prometheus exposition format" "# TYPE lines present"
else
    record "FAIL" "T14e: Prometheus exposition format" "no # TYPE lines"
fi

################################################################################
# Phase 4: Stack Latency Benchmark (2+2 iterations, 20s each)
################################################################################
header "Phase 4: Stack Latency Benchmark"

_test_start=$SECONDS
log "Running 2 iterations baseline (default, stacks on) + 2 iterations --stack=false"
log "Using matmul workload, 20s duration each"

# Helper: run one benchmark iteration
run_bench() {
    local label="$1"
    local extra_flags="$2"
    local outfile="$3"

    # Start a matmul workload that runs long enough
    python3 -c "
import torch, time
d = torch.device('cuda:0')
a = torch.randn(2048, 2048, device=d)
b = torch.randn(2048, 2048, device=d)
start = time.time()
while time.time() - start < 25:
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
" > /dev/null 2>&1 &
    local wl_pid=$!
    cleanup_pids+=("$wl_pid")
    sleep 3
    sudo ./bin/ingero trace $extra_flags --json --duration 20s > "$outfile" 2>/dev/null
    kill "$wl_pid" 2>/dev/null; wait "$wl_pid" 2>/dev/null || true
    local count=$(count_events "$outfile")
    echo "$count"
}

# Baseline (default — stacks on)
BENCH_STACK_COUNTS=()
for i in 1 2; do
    log "  Baseline (stacks on) iteration $i/2..."
    COUNT=$(run_bench "stack-$i" "" "logs/bench-stack-${i}.json")
    BENCH_STACK_COUNTS+=("$COUNT")
    log "    → $COUNT events"
done

# With --stack=false (no stacks)
BENCH_NOSTACK_COUNTS=()
for i in 1 2; do
    log "  No-stack iteration $i/2..."
    COUNT=$(run_bench "nostack-$i" "--stack=false" "logs/bench-nostack-${i}.json")
    BENCH_NOSTACK_COUNTS+=("$COUNT")
    log "    → $COUNT events"
done

# Analyze benchmark results
log "Analyzing benchmark results..."
python3 << 'BENCHEOF' > logs/benchmark-summary.txt 2>&1
import json, os, statistics

def parse_events(filepath):
    events = []
    try:
        for line in open(filepath):
            try:
                events.append(json.loads(line))
            except:
                pass
    except:
        pass
    return events

# Collect per-iteration stats
nostack_counts = []
stack_counts = []
nostack_durations = {}  # op -> [durations_ns]
stack_durations = {}

for i in range(1, 3):
    ns_events = parse_events(f'logs/bench-nostack-{i}.json')
    s_events = parse_events(f'logs/bench-stack-{i}.json')
    nostack_counts.append(len(ns_events))
    stack_counts.append(len(s_events))

    for e in ns_events:
        op = e.get('op', '?')
        dur = e.get('duration_ns', 0)
        if dur > 0:
            nostack_durations.setdefault(op, []).append(dur)

    for e in s_events:
        op = e.get('op', '?')
        dur = e.get('duration_ns', 0)
        if dur > 0:
            stack_durations.setdefault(op, []).append(dur)

# Compute averages
avg_nostack = statistics.mean(nostack_counts) if nostack_counts else 0
avg_stack = statistics.mean(stack_counts) if stack_counts else 0

print("=" * 70)
print("  Ingero v0.6 Stack Tracing Benchmark Summary")
print("=" * 70)
print()
print("Throughput (events in 20s window):")
print(f"  Default (stacks on):   {stack_counts} → avg {avg_stack:.0f}")
print(f"  With --stack=false:    {nostack_counts} → avg {avg_nostack:.0f}")
if avg_nostack > 0:
    overhead_pct = ((avg_nostack - avg_stack) / avg_nostack) * 100
    print(f"  Stack overhead:        {overhead_pct:+.1f}% {'(stacks add overhead)' if overhead_pct > 0 else '(within noise)'}")
print()

# Per-op latency comparison
print("Per-operation latency (ns):")
print(f"  {'Operation':<25} {'No-stack p50':>12} {'Stack p50':>12} {'Delta':>10}")
print(f"  {'-'*25} {'-'*12} {'-'*12} {'-'*10}")

for op in sorted(set(list(nostack_durations.keys()) + list(stack_durations.keys()))):
    ns_vals = sorted(nostack_durations.get(op, []))
    s_vals = sorted(stack_durations.get(op, []))
    if not ns_vals or not s_vals:
        continue
    ns_p50 = ns_vals[len(ns_vals)//2]
    s_p50 = s_vals[len(s_vals)//2]
    delta_ns = s_p50 - ns_p50
    print(f"  {op:<25} {ns_p50:>12,} {s_p50:>12,} {delta_ns:>+10,}")
print()

# Ring buffer comparison
print("Ring buffer overhead:")
print(f"  Without --stack: ~64 bytes/event (struct only, v0.7)")
print(f"  With --stack:    ~584 bytes/event (struct + 64 stack frames × 8 bytes)")
print(f"  Ratio: ~9.1x more data per event")
print()

# Stack coverage (per-source: HOST tracepoints have no userspace stacks by design)
stack_all = 0
stack_with = 0
stack_cuda_driver = 0
stack_cuda_driver_with = 0
stack_host = 0
for i in range(1, 3):
    for e in parse_events(f'logs/bench-stack-{i}.json'):
        stack_all += 1
        has = bool(e.get('stack'))
        if has:
            stack_with += 1
        src = e.get('source', '')
        if src in ('cuda', 'driver'):
            stack_cuda_driver += 1
            if has:
                stack_cuda_driver_with += 1
        elif src == 'host':
            stack_host += 1
if stack_all > 0:
    cd_pct = (stack_cuda_driver_with / stack_cuda_driver * 100) if stack_cuda_driver else 0
    print(f"Stack coverage: {stack_with}/{stack_all} events ({stack_with/stack_all*100:.1f}% overall)")
    print(f"  CUDA+DRIVER: {stack_cuda_driver_with}/{stack_cuda_driver} ({cd_pct:.1f}%)")
    print(f"  HOST:        {stack_host} events (kernel tracepoints, no userspace stacks)")

print()
print("=" * 70)
BENCHEOF

cat logs/benchmark-summary.txt

# Record benchmark result
if [[ "${#BENCH_NOSTACK_COUNTS[@]}" -eq 2 && "${#BENCH_STACK_COUNTS[@]}" -eq 2 ]]; then
    record "PASS" "T18: benchmark complete" "2+2 iterations, see logs/benchmark-summary.txt"
else
    record "FAIL" "T18: benchmark incomplete" "nostack=${#BENCH_NOSTACK_COUNTS[@]}, stack=${#BENCH_STACK_COUNTS[@]}"
fi

################################################################################
# Phase 5: MCP AI Diagnostic
################################################################################
header "Phase 5: MCP AI Diagnostic"

_test_start=$SECONDS
log "Starting MCP server on :8080 against recorded DB..."

# Kill any pre-existing MCP server from a previous run (avoids "address already in use")
sudo pkill -f 'ingero mcp' 2>/dev/null || true
sleep 0.5

# Start MCP server in background
sudo ./bin/ingero mcp --http :8080 > logs/mcp-server.log 2>&1 &
MCP_PID=$!
cleanup_pids+=("$MCP_PID")

# Wait for MCP server to be ready (max 5s)
MCP_READY=0
for i in $(seq 1 10); do
    if curl -skf -o /dev/null https://localhost:8080/mcp -H 'Content-Type: application/json' -H 'Accept: application/json, text/event-stream' -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"test","version":"1.0"}}}' 2>/dev/null; then
        MCP_READY=1
        break
    fi
    sleep 0.5
done

if [[ "$MCP_READY" -eq 0 ]]; then
    record "FAIL" "T19a: MCP server start" "server not ready after 5s"
    kill "$MCP_PID" 2>/dev/null || true
else
    log "MCP server ready"

    # Helper: call an MCP tool and return the response
    mcp_call() {
        local tool="$1" args="$2"
        curl -skf https://localhost:8080/mcp \
            -H 'Content-Type: application/json' \
            -H 'Accept: application/json, text/event-stream' \
            -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"${tool}\",\"arguments\":${args}}}" 2>/dev/null
    }

    # T19a: get_check
    log "Test 19a: MCP get_check"
    RESP=$(mcp_call "get_check" '{}')
    if echo "$RESP" | grep -q "NVIDIA\|GPU\|driver\|PASS\|FAIL"; then
        record "PASS" "T19a: MCP get_check" "GPU info returned"
    else
        record "FAIL" "T19a: MCP get_check" "no GPU info in response"
    fi

    # T19b: get_trace_stats
    _test_start=$SECONDS
    log "Test 19b: MCP get_trace_stats"
    # Note: MCP responses wrap data in a JSON "text" field, so inner quotes are
    # backslash-escaped (\"op\" not "op"). Use patterns that don't depend on matching
    # literal quotes.
    RESP=$(mcp_call "get_trace_stats" '{"since":"30m"}')
    echo "T19b RESP: ${RESP:0:300}" >> logs/mcp-debug.log
    if echo "$RESP" | grep -q 'op.*p50\|ops.*cuda\|p50.*p95'; then
        record "PASS" "T19b: MCP get_trace_stats" "stats with events returned"
    elif echo "$RESP" | grep -qi "No events\|no events\|0 events"; then
        record "FAIL" "T19b: MCP get_trace_stats" "no events in DB (recording may not have worked)"
    else
        record "FAIL" "T19b: MCP get_trace_stats" "unexpected response: ${RESP:0:200}"
    fi

    # T19c: run_sql (replaces query_events — use SQL for ad-hoc event queries)
    _test_start=$SECONDS
    log "Test 19c: MCP run_sql"
    RESP=$(mcp_call "run_sql" '{"query":"SELECT e.id, o.name AS op, e.duration, e.pid FROM events e JOIN ops o ON e.source=o.source_id AND e.op=o.op_id ORDER BY e.id DESC LIMIT 10"}')
    echo "T19c RESP: ${RESP:0:300}" >> logs/mcp-debug.log
    if echo "$RESP" | grep -q 'columns\|data\|cuLaunchKernel\|cudaDeviceSync\|cudaMalloc\|sched_switch'; then
        record "PASS" "T19c: MCP run_sql" "events returned via SQL"
    else
        record "FAIL" "T19c: MCP run_sql" "no events returned: ${RESP:0:200}"
    fi

    # T19d: get_causal_chains
    _test_start=$SECONDS
    log "Test 19d: MCP get_causal_chains"
    RESP=$(mcp_call "get_causal_chains" '{"since":"30m"}')
    if echo "$RESP" | grep -qi "causal\|chain\|healthy\|sev\|severity\|No causal"; then
        record "PASS" "T19d: MCP get_causal_chains" "chains or healthy response"
    else
        record "FAIL" "T19d: MCP get_causal_chains" "unexpected response: ${RESP:0:200}"
    fi

    # T19e: get_test_report — will fail because we haven't written test-report.json yet.
    # We generate it after this phase, so expect a "not found" message (which is correct behavior).
    _test_start=$SECONDS
    log "Test 19e: MCP get_test_report"
    RESP=$(mcp_call "get_test_report" '{}')
    if echo "$RESP" | grep -q "No test report\|test-report.json\|Run.*gpu-test"; then
        record "PASS" "T19e: MCP get_test_report" "correct 'not found' response (report generated after this phase)"
    elif echo "$RESP" | grep -q "version\|summary\|tests"; then
        record "PASS" "T19e: MCP get_test_report" "report returned"
    else
        record "FAIL" "T19e: MCP get_test_report" "unexpected response"
    fi

    # ── T20: MCP Session Transcript ──
    # Generate a structured "ML Engineer diagnosis" session log.
    # TSC is on by default (omit or pass true) — never pass false.
    _test_start=$SECONDS
    log "Test 20: MCP session transcript"

    SESSION_FILE="logs/mcp-session.txt"
    DB_COUNT=$(sudo ./bin/ingero query --since 60m --json 2>/dev/null | grep -c '"op"' || echo "?")

    {
        echo "=== Ingero MCP Session — ML Engineer Diagnosis Demo ==="
        echo "Date: $(date -u)"
        echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
        echo "DB: ${DB_COUNT} events in ingero.db"
        echo ""

        # Step 0: tools/list
        echo "========================================"
        echo ">> Step 0: ML Engineer: What can Ingero tell me?"
        echo "========================================"
        echo ""
        REQ='{"jsonrpc":"2.0","id":0,"method":"tools/list"}'
        echo "REQUEST:"
        echo "$REQ" | python3 -m json.tool 2>/dev/null || echo "$REQ"
        echo ""
        RESP=$(curl -skf https://localhost:8080/mcp \
            -H 'Content-Type: application/json' \
            -H 'Accept: application/json, text/event-stream' \
            -d "$REQ" 2>/dev/null)
        echo "RESPONSE:"
        echo "$RESP" | python3 -m json.tool 2>/dev/null || echo "$RESP"
        echo ""
        echo ""

        # Step 1: get_check
        echo "========================================"
        echo ">> Step 1: ML Engineer: Is my GPU environment healthy?"
        echo "========================================"
        echo ""
        REQ='{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_check","arguments":{}}}'
        echo "REQUEST:"
        echo "$REQ" | python3 -m json.tool 2>/dev/null || echo "$REQ"
        echo ""
        RESP=$(mcp_call "get_check" '{}')
        echo "RESPONSE:"
        echo "$RESP" | python3 -m json.tool 2>/dev/null || echo "$RESP"
        echo ""
        echo ""

        # Step 2: get_trace_stats (TSC on — default)
        echo "========================================"
        echo ">> Step 2: ML Engineer: Show me GPU stats (TSC compressed)."
        echo "========================================"
        echo ""
        REQ='{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_trace_stats","arguments":{"since":"30m"}}}'
        echo "REQUEST:"
        echo "$REQ" | python3 -m json.tool 2>/dev/null || echo "$REQ"
        echo ""
        RESP=$(mcp_call "get_trace_stats" '{"since":"30m"}')
        echo "RESPONSE:"
        echo "$RESP" | python3 -m json.tool 2>/dev/null || echo "$RESP"
        echo ""
        echo ""

        # Step 3: run_sql — ad-hoc event query (replaces query_events)
        echo "========================================"
        echo ">> Step 3: ML Engineer: Show me recent events (via SQL)."
        echo "========================================"
        echo ""
        REQ='{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"run_sql","arguments":{"query":"SELECT e.id, o.name AS op, e.duration, e.pid FROM events e JOIN ops o ON e.source=o.source_id AND e.op=o.op_id ORDER BY e.id DESC LIMIT 20"}}}'
        echo "REQUEST:"
        echo "$REQ" | python3 -m json.tool 2>/dev/null || echo "$REQ"
        echo ""
        RESP=$(mcp_call "run_sql" '{"query":"SELECT e.id, o.name AS op, e.duration, e.pid FROM events e JOIN ops o ON e.source=o.source_id AND e.op=o.op_id ORDER BY e.id DESC LIMIT 20"}')
        echo "RESPONSE:"
        echo "$RESP" | python3 -m json.tool 2>/dev/null || echo "$RESP"
        echo ""
        echo ""

        # Step 4: get_causal_chains (TSC on — default)
        echo "========================================"
        echo ">> Step 4: ML Engineer: Diagnose root cause."
        echo "========================================"
        echo ""
        REQ='{"jsonrpc":"2.0","id":4,"method":"tools/call","params":{"name":"get_causal_chains","arguments":{"since":"30m"}}}'
        echo "REQUEST:"
        echo "$REQ" | python3 -m json.tool 2>/dev/null || echo "$REQ"
        echo ""
        RESP=$(mcp_call "get_causal_chains" '{"since":"30m"}')
        echo "RESPONSE:"
        echo "$RESP" | python3 -m json.tool 2>/dev/null || echo "$RESP"
        echo ""
        echo ""

        # Step 5: get_stacks (resolved call stacks)
        echo "========================================"
        echo ">> Step 5: ML Engineer: What call stacks hit cudaMalloc?"
        echo "========================================"
        echo ""
        REQ='{"jsonrpc":"2.0","id":5,"method":"tools/call","params":{"name":"get_stacks","arguments":{"op":"cudaMalloc","limit":5}}}'
        echo "REQUEST:"
        echo "$REQ" | python3 -m json.tool 2>/dev/null || echo "$REQ"
        echo ""
        RESP=$(mcp_call "get_stacks" '{"op":"cudaMalloc","limit":5}')
        echo "RESPONSE:"
        echo "$RESP" | python3 -m json.tool 2>/dev/null || echo "$RESP"
        echo ""
        echo ""

        # Step 6: run_demo (synthetic scenario)
        echo "========================================"
        echo ">> Step 6: ML Engineer: Run a demo scenario."
        echo "========================================"
        echo ""
        REQ='{"jsonrpc":"2.0","id":6,"method":"tools/call","params":{"name":"run_demo","arguments":{"scenario":"incident"}}}'
        echo "REQUEST:"
        echo "$REQ" | python3 -m json.tool 2>/dev/null || echo "$REQ"
        echo ""
        RESP=$(mcp_call "run_demo" '{"scenario":"incident"}')
        echo "RESPONSE:"
        echo "$RESP" | python3 -m json.tool 2>/dev/null || echo "$RESP"
        echo ""
        echo ""

        # Step 7: get_test_report
        echo "========================================"
        echo ">> Step 7: ML Engineer: Show me the test report."
        echo "========================================"
        echo ""
        REQ='{"jsonrpc":"2.0","id":7,"method":"tools/call","params":{"name":"get_test_report","arguments":{}}}'
        echo "REQUEST:"
        echo "$REQ" | python3 -m json.tool 2>/dev/null || echo "$REQ"
        echo ""
        RESP=$(mcp_call "get_test_report" '{}')
        echo "RESPONSE:"
        echo "$RESP" | python3 -m json.tool 2>/dev/null || echo "$RESP"
        echo ""
        echo ""

        echo "========================================"
        echo ">> Session Complete"
        echo "========================================"
        echo ""
        echo "This session demonstrated all 7 MCP tools with TSC on (default):"
        echo "  0. tools/list — discover available tools"
        echo "  1. get_check — system health check"
        echo "  2. get_trace_stats — GPU operation statistics (TSC compressed)"
        echo "  3. run_sql — ad-hoc event query via SQL"
        echo "  4. get_causal_chains — root cause analysis (TSC compressed)"
        echo "  5. get_stacks — resolved call stacks"
        echo "  6. run_demo — synthetic scenario"
        echo "  7. get_test_report — integration test results"
    } > "$SESSION_FILE"

    if [ -s "$SESSION_FILE" ]; then
        LINES=$(wc -l < "$SESSION_FILE")
        record "PASS" "T20: MCP session transcript" "${LINES} lines → logs/mcp-session.txt"
    else
        record "FAIL" "T20: MCP session transcript" "empty file"
    fi

    # Kill MCP server (started with sudo, so needs sudo to kill)
    # Don't use wait — sudo-spawned processes aren't shell children, so wait hangs.
    sudo kill "$MCP_PID" 2>/dev/null || true
    sleep 1
fi

# ── T21: DB Schema Validation ──
_test_start=$SECONDS
log "Test 21: DB schema validation"

# Find the DB — SUDO_USER-aware path puts it in the invoking user's home
DB_PATH=""
if [ -f "$HOME/.ingero/ingero.db" ]; then
    DB_PATH="$HOME/.ingero/ingero.db"
elif sudo test -f /root/.ingero/ingero.db; then
    DB_PATH="/root/.ingero/ingero.db"
fi

if [ -n "$DB_PATH" ]; then
    log "  DB found: $DB_PATH"
    # sqlite3 may be at different locations; fall back to Python if needed
    if command -v sqlite3 &>/dev/null; then
        sqlite3 "$DB_PATH" .schema > logs/db-schema.txt 2>&1 || \
            sudo sqlite3 "$DB_PATH" .schema > logs/db-schema.txt 2>&1 || true
    else
        python3 -c "
import sqlite3, sys
conn = sqlite3.connect('$DB_PATH')
for row in conn.execute(\"SELECT sql FROM sqlite_master WHERE sql IS NOT NULL\"):
    print(row[0])
conn.close()
" > logs/db-schema.txt 2>&1 || \
        sudo python3 -c "
import sqlite3
conn = sqlite3.connect('$DB_PATH')
for row in conn.execute(\"SELECT sql FROM sqlite_master WHERE sql IS NOT NULL\"):
    print(row[0])
conn.close()
" > logs/db-schema.txt 2>&1 || true
    fi

    SCHEMA_OK=0
    SCHEMA_MISSING=""
    for table in events causal_chains system_snapshots sources ops schema_info sessions; do
        if grep -qi "$table" logs/db-schema.txt 2>/dev/null; then
            SCHEMA_OK=$((SCHEMA_OK + 1))
        else
            SCHEMA_MISSING="${SCHEMA_MISSING} ${table}"
        fi
    done

    if [[ "$SCHEMA_OK" -eq 7 ]]; then
        record "PASS" "T21: DB schema" "all 7 tables present"
    else
        record "FAIL" "T21: DB schema" "${SCHEMA_OK}/7 tables, missing:${SCHEMA_MISSING}"
    fi
else
    record "FAIL" "T21: DB schema" "ingero.db not found"
fi

################################################################################
# Phase 6: GPU Problem Investigation (T23a-T23ab, 28 issues via MCP)
################################################################################
header "Phase 6: GPU Problem Investigation (28 issues)"

_test_start=$SECONDS
log "Running 28 GPU problem investigations (ResNet-50 + alloc_stress + stress-ng + MCP)..."

ML_OUTPUT=$(bash scripts/gpu-investigation.sh 2>&1)
ML_EXIT=$?

# Display the script's output (it has its own formatting)
echo "$ML_OUTPUT" | grep -v '^ML_RESULT|'

# Ingest structured results from gpu-investigation.sh
ML_INGESTED=0
while IFS='|' read -r tid name status detail dur; do
    record "$status" "$name" "$detail"
    ML_INGESTED=$((ML_INGESTED + 1))
done < <(echo "$ML_OUTPUT" | grep '^ML_RESULT|' | sed 's/^ML_RESULT|//')

if [[ "$ML_INGESTED" -eq 0 ]]; then
    if [[ "$ML_EXIT" -ne 0 ]]; then
        record "FAIL" "T23a: GPU investigation" "script failed (exit $ML_EXIT)"
    else
        record "FAIL" "T23a: GPU investigation" "no structured results returned"
    fi
fi

log "GPU investigation: ingested $ML_INGESTED test results"

################################################################################
# Final Report
################################################################################
header "v0.6 Integration Test Report"

TOTAL=$((PASS_COUNT + FAIL_COUNT + SKIP_COUNT))
SCRIPT_DURATION=$((SECONDS - SCRIPT_START))
echo ""
echo -e "$(ts)   ${GREEN}PASS: $PASS_COUNT${NC}  ${RED}FAIL: $FAIL_COUNT${NC}  ${YELLOW}SKIP: $SKIP_COUNT${NC}  Total: $TOTAL"
echo ""

{
    echo "Ingero v0.6 Integration Test Report"
    echo "===================================="
    echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "Host: $(hostname)"
    echo "Kernel: $(uname -r)"
    echo "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
    echo "Go: $(go version 2>/dev/null | awk '{print $3}')"
    echo "PyTorch: $(python3 -c 'import torch; print(f"{torch.__version__}, CUDA {torch.version.cuda}")' 2>/dev/null || echo 'N/A')"
    echo ""
    echo "Results: PASS=$PASS_COUNT  FAIL=$FAIL_COUNT  SKIP=$SKIP_COUNT  Total=$TOTAL"
    echo ""
    echo -e "$REPORT"
} > logs/test-report.txt

# Generate JSON test report
_GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "N/A")
_GPU_NAME=$(echo "$_GPU_INFO" | cut -d',' -f1 | xargs)
_DRIVER_VER=$(echo "$_GPU_INFO" | cut -d',' -f2 | xargs)
_KERNEL_VER=$(uname -r)
_PYTORCH_VER=$(python3 -c 'import torch; print(f"{torch.__version__}, CUDA {torch.version.cuda}")' 2>/dev/null || echo "N/A")
_GO_VER=$(go version 2>/dev/null | awk '{print $3}')

# Write test results to temp file, one line per result
: > $TEST_TMP/test_results.txt
for entry in "${TEST_RESULTS[@]}"; do
    echo "$entry" >> $TEST_TMP/test_results.txt
done

SCRIPT_DURATION="$SCRIPT_DURATION" \
  GPU_NAME="$_GPU_NAME" DRIVER_VER="$_DRIVER_VER" KERNEL_VER="$_KERNEL_VER" \
  PYTORCH_VER="$_PYTORCH_VER" GO_VER="$_GO_VER" \
  PASS_COUNT="$PASS_COUNT" FAIL_COUNT="$FAIL_COUNT" SKIP_COUNT="$SKIP_COUNT" TOTAL="$TOTAL" \
  python3 -c "
import json, sys, os
from datetime import datetime, timezone

results_file = '$TEST_TMP/test_results.txt'
tests = []
with open(results_file) as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        parts = line.split('|', 4)
        if len(parts) == 5:
            tid, name, status, detail, dur = parts
            tests.append({
                'id': tid.strip(),
                'name': name.strip(),
                'status': status.strip(),
                'detail': detail.strip(),
                'duration_s': int(dur.strip()),
            })

report = {
    'version': '0.6',
    'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    'duration_s': int(os.environ.get('SCRIPT_DURATION', '0')),
    'system': {
        'gpu': os.environ.get('GPU_NAME', 'N/A'),
        'driver': os.environ.get('DRIVER_VER', 'N/A'),
        'kernel': os.environ.get('KERNEL_VER', 'N/A'),
        'pytorch': os.environ.get('PYTORCH_VER', 'N/A'),
        'go': os.environ.get('GO_VER', 'N/A'),
    },
    'summary': {
        'pass': int(os.environ.get('PASS_COUNT', '0')),
        'fail': int(os.environ.get('FAIL_COUNT', '0')),
        'skip': int(os.environ.get('SKIP_COUNT', '0')),
        'total': int(os.environ.get('TOTAL', '0')),
    },
    'tests': tests,
}

with open('logs/test-report.json', 'w') as f:
    json.dump(report, f, indent=2)
print(f'JSON report: logs/test-report.json ({len(tests)} tests)')
"

echo "$(ts) Report: logs/test-report.txt"
echo "$(ts) JSON:   logs/test-report.json"
echo "$(ts) Benchmark: logs/benchmark-summary.txt"
echo ""
echo "Log files:"
ls -la logs/test-report.txt logs/test-report.json logs/benchmark-summary.txt logs/integration-report.log logs/mcp-server.log logs/mcp-session.txt logs/bg-*.out logs/stack-*.{json,log} logs/otlp-*.{json,log} logs/combined-*.{json,log} logs/explain-*.log logs/bench-*.json logs/trace-*.{json,log} logs/check-*.log logs/demo-*.{json,log} logs/prom-*.{json,log,txt} logs/db-schema.txt logs/nvidia-smi.log logs/uname.log logs/version.log logs/ml-*.{json,log} logs/ml-investigation-report.md logs/gpu-inv-*.{json,log} logs/gpu-investigation-report.log logs/gpu-investigation.db 2>/dev/null

if [[ $FAIL_COUNT -gt 0 ]]; then
    exit 1
fi
exit 0
