#!/bin/bash
################################################################################
# GPU Problem Investigation — 23 Issues via MCP
#
# Comprehensive GPU causal observability validation. Runs a 5-phase 120s trace
# with ResNet-50 training + alloc_stress + stress-ng, then investigates all 23
# GPU problems Ingero can detect through MCP tool calls (primarily run_sql).
#
# Architecture: Bash handles trace/workload lifecycle, Python handles MCP
# queries and analysis. This separation exists because:
#   - Bash: sudo, signals, process cleanup, MCP server lifecycle
#   - Python: JSON processing, statistical analysis, HTTP/MCP calls, reports
#
# Phases:
#   1 (0-20s):    Cold start + steady baseline (ResNet-50 from scratch)
#   2 (20-50s):   Allocation stress (alloc_stress.py in background)
#   3 (50-90s):   CPU contention (stress-ng saturates all cores)
#   4 (90-110s):  Recovery (stressors killed, training continues)
#   5 (110-120s): Continued clean training
#
# Output: 23 test results (T23a-T23w), investigation report, ML_RESULT lines
#
# Run standalone:   bash scripts/gpu-investigation.sh
# Run via suite:    bash scripts/gpu-test.sh   (Phase 6)
#
# Requires: GPU, PyTorch, stress-ng, Ingero binary at bin/ingero
################################################################################

set -uo pipefail

# Resolve paths — works from agent/ or agent/scripts/
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
if [[ "$(basename "$SCRIPT_DIR")" == "scripts" ]]; then
    cd "$SCRIPT_DIR/.." || exit 1
else
    cd "$SCRIPT_DIR" || exit 1
fi
INGERO_DIR="$(pwd)"

# Colors
if [ -t 1 ]; then
    RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'
    BLUE='\033[0;34m'; CYAN='\033[0;36m'; NC='\033[0m'
else
    RED=''; GREEN=''; YELLOW=''; BLUE=''; CYAN=''; NC=''
fi

PASS_COUNT=0; FAIL_COUNT=0; SKIP_COUNT=0
_test_start=$SECONDS

# Structured results for gpu-test.sh ingestion
declare -a ML_RESULTS  # "ID|name|status|detail|duration_s"

ts() { date -u '+%Y-%m-%d %H:%M:%S'; }

record() {
    local status="$1" name="$2" detail="$3"
    local elapsed=$((SECONDS - _test_start))
    local tid="${name%%:*}"
    ML_RESULTS+=("${tid}|${name}|${status}|${detail}|${elapsed}")

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

# PIDs to clean up on exit
cleanup_pids=()
ML_DB=""
MCP_PORT=""
cleanup() {
    # Kill sudo-spawned processes FIRST (pkill -f) — kill+wait on the sudo
    # wrapper PID can deadlock because sudo doesn't forward SIGTERM to children.
    if [[ -n "$ML_DB" ]]; then
        sudo pkill -f "ingero trace.*${ML_DB}" 2>/dev/null || true
        sudo pkill -f "ingero mcp.*${ML_DB}" 2>/dev/null || true
    fi
    sudo pkill -f 'stress-ng.*matrixprod' 2>/dev/null || true
    pkill -f 'alloc_stress.py' 2>/dev/null || true

    # Now kill and reap background jobs
    for pid in "${cleanup_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
    done
    for pid in "${cleanup_pids[@]}"; do
        wait "$pid" 2>/dev/null || true
    done

    if [[ -n "$ML_DB" ]]; then
        rm -f "${ML_DB}" "${ML_DB}-wal" "${ML_DB}-shm" 2>/dev/null || true
    fi
}
trap cleanup EXIT

################################################################################
# Preflight
################################################################################

if [[ ! -x bin/ingero ]]; then
    echo "ERROR: bin/ingero not found. Run 'make build' first."
    exit 1
fi

if ! command -v stress-ng &>/dev/null; then
    echo "ERROR: stress-ng not found. Install: sudo apt-get install -y stress-ng"
    exit 1
fi

if ! python3 -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
    echo "ERROR: PyTorch with CUDA not available."
    exit 1
fi

if [[ ! -f scripts/gpu-investigation-analysis.py ]]; then
    echo "ERROR: scripts/gpu-investigation-analysis.py not found."
    exit 1
fi

mkdir -p logs

################################################################################
# Setup: 5-Phase Trace (120s)
################################################################################

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo -e "${BLUE}  GPU Problem Investigation — 23 Issues via MCP${NC}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

ML_DB="/tmp/ingero_inv_$(head -c 6 /dev/urandom | xxd -p).db"
TRACE_DURATION=120
REPORT_FILE="logs/gpu-investigation-report.log"

echo -e "$(ts) ${CYAN}[SETUP]${NC} 5-phase trace: 120s total"
echo -e "$(ts)   Phase 1: 0-20s   Cold start + baseline (ResNet-50)"
echo -e "$(ts)   Phase 2: 20-50s  Allocation stress (alloc_stress.py)"
echo -e "$(ts)   Phase 3: 50-90s  CPU contention (stress-ng $(nproc) workers)"
echo -e "$(ts)   Phase 4: 90-110s Recovery (stressors killed)"
echo -e "$(ts)   Phase 5: 110-120s Continued clean training"

# Pre-download CIFAR-10 dataset so training starts GPU work immediately.
echo -e "$(ts)   Pre-downloading CIFAR-10 dataset..."
if ! python3 -c "
import torchvision
torchvision.datasets.CIFAR10(root='/tmp/cifar10', train=True, download=True)
print('CIFAR-10 ready')
" > logs/gpu-inv-dataset-download.log 2>&1; then
    echo -e "$(ts) ${RED}[ERROR]${NC} CIFAR-10 download failed. See logs/gpu-inv-dataset-download.log"
    cat logs/gpu-inv-dataset-download.log
    exit 1
fi
echo -e "$(ts)   $(tail -1 logs/gpu-inv-dataset-download.log)"

################################################################################
# Phase 1: Cold start + steady baseline (0-20s)
################################################################################

echo -e "$(ts) ${CYAN}[PHASE 1]${NC} Starting ResNet-50 training..."

# Start training (10 epochs — enough to cover 120s on fast GPUs like H100/GH200)
python3 tests/workloads/training/resnet50_cifar10.py \
    --epochs 10 --batch-size 64 > logs/gpu-inv-training.log 2>&1 &
TRAIN_PID=$!
cleanup_pids+=("$TRAIN_PID")

# Wait for CUDA init
echo -e "$(ts)   Waiting for training to reach GPU..."
sleep 10

if ! kill -0 "$TRAIN_PID" 2>/dev/null; then
    echo -e "$(ts) ${RED}[ERROR]${NC} Training process died. See logs/gpu-inv-training.log"
    cat logs/gpu-inv-training.log
    exit 1
fi

echo -e "$(ts)   Training PID: $TRAIN_PID"
echo -e "$(ts)   Starting trace (${TRACE_DURATION}s) with --record-all --stack..."

# Start trace — --record-all stores every event, --stack captures call stacks
sudo ./bin/ingero trace --db "$ML_DB" --record-all --stack --duration ${TRACE_DURATION}s \
    2> logs/gpu-inv-trace.log &
TRACE_PID=$!
cleanup_pids+=("$TRACE_PID")

# Record phase timestamps (epoch seconds) for Python analysis
PHASE1_START=$(date +%s.%N)

# Phase 1 runs for 20s (trace is already capturing)
echo -e "$(ts)   Phase 1: baseline (20s)..."
sleep 20

################################################################################
# Phase 2: Allocation stress (20-50s)
################################################################################

echo -e "$(ts) ${CYAN}[PHASE 2]${NC} Starting alloc_stress.py (30s)..."
PHASE2_START=$(date +%s.%N)

python3 tests/workloads/synthetic/alloc_stress.py > logs/gpu-inv-alloc-stress.log 2>&1 &
ALLOC_PID=$!
cleanup_pids+=("$ALLOC_PID")

# Phase 2 runs for 30s
sleep 30

################################################################################
# Phase 3: CPU contention (50-90s)
################################################################################

echo -e "$(ts) ${CYAN}[PHASE 3]${NC} Starting stress-ng (40s, $(nproc) workers)..."
PHASE3_START=$(date +%s.%N)

# Kill alloc_stress if still running (it should have finished by now)
kill "$ALLOC_PID" 2>/dev/null || true

NCPUS=$(nproc)
sudo stress-ng --cpu "$NCPUS" --cpu-method matrixprod --timeout 45s > /dev/null 2>&1 &
STRESS_PID=$!
cleanup_pids+=("$STRESS_PID")

# Phase 3 runs for 40s
sleep 40

################################################################################
# Phase 4: Recovery (90-110s)
################################################################################

echo -e "$(ts) ${CYAN}[PHASE 4]${NC} Killing stressors, recovery phase (20s)..."
PHASE4_START=$(date +%s.%N)

sudo kill "$STRESS_PID" 2>/dev/null || true

# Phase 4 + 5 run for the remaining 30s of the trace
echo -e "$(ts)   Waiting for trace to finish (~30s remaining)..."
wait "$TRACE_PID" 2>/dev/null
TRACE_EXIT=$?

if [[ "$TRACE_EXIT" -ne 0 ]]; then
    echo -e "$(ts) ${RED}[ERROR]${NC} Trace failed (exit $TRACE_EXIT). See logs/gpu-inv-trace.log"
    cat logs/gpu-inv-trace.log
    record "FAIL" "T23a: trace captured events" "trace exited $TRACE_EXIT"
    for entry in "${ML_RESULTS[@]}"; do echo "ML_RESULT|${entry}"; done
    exit 1
fi

# Kill training
kill "$TRAIN_PID" 2>/dev/null || true
wait "$TRAIN_PID" 2>/dev/null || true

echo -e "$(ts)   Trace complete. DB: $ML_DB"

# Copy DB to logs/ for transfer
sudo cp "$ML_DB" logs/gpu-investigation.db && sudo chmod 644 logs/gpu-investigation.db
sudo cp "${ML_DB}-wal" logs/gpu-investigation.db-wal 2>/dev/null && sudo chmod 644 logs/gpu-investigation.db-wal || true
sudo cp "${ML_DB}-shm" logs/gpu-investigation.db-shm 2>/dev/null && sudo chmod 644 logs/gpu-investigation.db-shm || true

################################################################################
# Start MCP Server
################################################################################

echo -e "$(ts) ${CYAN}[MCP]${NC} Starting MCP server..."

MCP_PORT=$(( 8443 + RANDOM % 1000 ))
./bin/ingero mcp --db "$ML_DB" --http ":${MCP_PORT}" > logs/gpu-inv-mcp.log 2>&1 &
MCP_PID=$!
cleanup_pids+=("$MCP_PID")

# Wait for MCP server to be ready
MCP_READY=0
for i in $(seq 1 15); do
    if curl -skf -o /dev/null "https://localhost:${MCP_PORT}/mcp" \
        -H 'Content-Type: application/json' \
        -H 'Accept: application/json, text/event-stream' \
        -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"gpu-investigation","version":"1.0"}}}' 2>/dev/null; then
        MCP_READY=1
        break
    fi
    sleep 0.5
done

if [[ "$MCP_READY" -eq 0 ]]; then
    echo -e "$(ts)   ${RED}MCP server not ready after 7.5s${NC}"
    echo -e "$(ts)   MCP log: $(tail -3 logs/gpu-inv-mcp.log)"
    record "FAIL" "T23a: MCP server" "MCP server failed to start"
    for entry in "${ML_RESULTS[@]}"; do echo "ML_RESULT|${entry}"; done
    exit 1
fi

echo -e "$(ts)   MCP server ready on :${MCP_PORT}"

################################################################################
# Run Python Analysis (23 Investigations)
################################################################################

echo ""
echo -e "$(ts) ${CYAN}[ANALYSIS]${NC} Running 23 GPU problem investigations via MCP..."

ANALYSIS_OUTPUT=$(python3 scripts/gpu-investigation-analysis.py \
    --mcp-url "https://localhost:${MCP_PORT}/mcp" \
    --db "$ML_DB" \
    --report "$REPORT_FILE" \
    --phase1-start "$PHASE1_START" \
    --phase2-start "$PHASE2_START" \
    --phase3-start "$PHASE3_START" \
    --phase4-start "$PHASE4_START" \
    2> logs/gpu-inv-analysis-stderr.log)
ANALYSIS_EXIT=$?

# Display investigation output (non-ML_RESULT lines)
echo "$ANALYSIS_OUTPUT" | grep -v '^ML_RESULT|'

# Kill MCP server
sudo kill "$MCP_PID" 2>/dev/null || true

################################################################################
# Ingest Results
################################################################################

# Parse ML_RESULT lines from Python output
INGESTED=0
while IFS='|' read -r tid name status detail dur; do
    record "$status" "$name" "$detail"
    INGESTED=$((INGESTED + 1))
done < <(echo "$ANALYSIS_OUTPUT" | grep '^ML_RESULT|' | sed 's/^ML_RESULT|//')

if [[ "$INGESTED" -eq 0 ]]; then
    if [[ "$ANALYSIS_EXIT" -ne 0 ]]; then
        record "FAIL" "T23a: GPU investigation" "analysis script failed (exit $ANALYSIS_EXIT)"
        echo "Analysis stderr:"
        cat logs/gpu-inv-analysis-stderr.log
    else
        record "FAIL" "T23a: GPU investigation" "no structured results returned"
    fi
fi

################################################################################
# Summary
################################################################################

echo ""
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
TOTAL=$((PASS_COUNT + FAIL_COUNT + SKIP_COUNT))
echo -e "  ${GREEN}PASS=${PASS_COUNT}${NC}  ${RED}FAIL=${FAIL_COUNT}${NC}  ${YELLOW}SKIP=${SKIP_COUNT}${NC}  Total=${TOTAL}"
echo -e "  Report: ${REPORT_FILE}"
echo -e "${BLUE}════════════════════════════════════════════════════════════${NC}"
echo ""

# Output structured results for gpu-test.sh ingestion
for entry in "${ML_RESULTS[@]}"; do
    echo "ML_RESULT|${entry}"
done

if [[ $FAIL_COUNT -gt 0 ]]; then
    exit 1
fi
exit 0
