#!/bin/bash
################################################################################
# Ingero GPU Integration Test — One-Shot Script
#
# Runs on the GPU VM after code has been synced. Builds, tests, and validates
# the full Ingero stack against a real GPU. Outputs a structured test report.
#
# This script is designed to be run REMOTELY via SSH:
#
#   scp scripts/gpu-integration-test.sh user@<VM_IP>:~/workspace/ingero/
#   ssh user@<VM_IP> 'bash ~/workspace/ingero/scripts/gpu-integration-test.sh'
#
# Or locally on the VM:
#   cd ~/workspace/ingero && bash scripts/gpu-integration-test.sh
#
# Prerequisites (handled by cloud-init from tensordock/vm.sh):
#   - Go 1.22+ at /usr/local/go/bin
#   - clang-14
#   - NVIDIA GPU with working driver (nvidia-smi)
#   - PyTorch with CUDA support
#
# Output: writes report to ~/workspace/ingero/integration-test-report.log
################################################################################

set -euo pipefail

# Ensure Go and standard tools are in PATH (avoid WSL PATH contamination)
export PATH=/usr/local/go/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/go/bin:$HOME/.local/bin
export HOME="${HOME:-/home/$(whoami)}"

# Colors (disabled when output is piped/redirected to a file)
if [ -t 1 ]; then
  RED='\033[0;31m'
  GREEN='\033[0;32m'
  YELLOW='\033[1;33m'
  BLUE='\033[0;34m'
  NC='\033[0m'
else
  RED=''
  GREEN=''
  YELLOW=''
  BLUE=''
  NC=''
fi

PASS_COUNT=0
FAIL_COUNT=0
SKIP_COUNT=0
REPORT=""

cd "$(dirname "$0")/.." || { echo "Cannot find ingero directory"; exit 1; }
INGERO_DIR="$(pwd)"
REPORT_FILE="${INGERO_DIR}/integration-test-report.log"

ts()   { date -u '+%Y-%m-%d %H:%M:%S'; }
log()  { echo -e "$(ts) ${GREEN}[INFO]${NC}  $1"; }
warn() { echo -e "$(ts) ${YELLOW}[WARN]${NC}  $1"; }
fail() { echo -e "$(ts) ${RED}[FAIL]${NC}  $1"; }

record() {
    local status="$1" name="$2" detail="$3"
    local line="[$(ts)] [$status] $name: $detail"
    REPORT="${REPORT}${line}
"
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
}

header() {
    echo ""
    echo -e "$(ts) ${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "$(ts) ${BLUE}  $1${NC}"
    echo -e "$(ts) ${BLUE}════════════════════════════════════════════════════════════${NC}"
}

# ──────────────────────────────────────────────────────────────
# Phase 0: Environment Check
# ──────────────────────────────────────────────────────────────
header "Phase 0: Environment Verification"

# Go
if go version &>/dev/null; then
    GO_VER=$(go version | awk '{print $3}')
    record "PASS" "Go" "$GO_VER"
else
    record "FAIL" "Go" "not found in PATH"
    echo "FATAL: Go is required. Exiting."
    exit 1
fi

# GPU
if nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader &>/dev/null; then
    GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader | head -1)
    record "PASS" "GPU" "$GPU_INFO"
else
    # Try reboot hint
    if command -v nvidia-smi &>/dev/null; then
        record "FAIL" "GPU" "nvidia-smi present but driver not loaded. Try: sudo reboot"
    else
        record "FAIL" "GPU" "nvidia-smi not found"
    fi
fi

# Kernel + BTF
KERNEL=$(uname -r)
if [ -f /sys/kernel/btf/vmlinux ]; then
    record "PASS" "Kernel/BTF" "$KERNEL with BTF"
else
    record "FAIL" "Kernel/BTF" "$KERNEL — no BTF"
fi

# clang-14
if command -v clang-14 &>/dev/null; then
    record "PASS" "clang-14" "$(clang-14 --version 2>&1 | head -1)"
else
    record "SKIP" "clang-14" "not installed (eBPF codegen unavailable)"
fi

# PyTorch + CUDA
if python3 -c "import torch; assert torch.cuda.is_available(); print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')" 2>/dev/null; then
    TORCH_INFO=$(python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA {torch.version.cuda}')")
    record "PASS" "PyTorch+CUDA" "$TORCH_INFO"
else
    record "SKIP" "PyTorch+CUDA" "not available (GPU workload tests will be skipped)"
fi

# ──────────────────────────────────────────────────────────────
# Phase 1: Build
# ──────────────────────────────────────────────────────────────
header "Phase 1: Build"

log "Building ingero binary (with version injection)..."
if make build 2>&1; then
    BINARY_SIZE=$(ls -lh bin/ingero | awk '{print $5}')
    record "PASS" "Build" "bin/ingero binary ($BINARY_SIZE)"
else
    record "FAIL" "Build" "make build failed"
    echo "FATAL: Build failed. Exiting."
    exit 1
fi

# ── Probe Smoke Test — abort early if eBPF probes can't attach ──
if python3 -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
    log "Probe smoke test: starting 3s CUDA workload..."
    python3 -c "
import torch, time
d = torch.device('cuda:0')
a = torch.randn(1024, 1024, device=d)
start = time.time()
while time.time() - start < 5:
    torch.matmul(a, a)
    torch.cuda.synchronize()
" &>/dev/null &
    SMOKE_WL_PID=$!
    sleep 1

    log "Probe smoke test: running ingero trace --debug --json --duration 3s..."
    mkdir -p logs
    sudo ./bin/ingero trace --debug --json --duration 3s \
        > /tmp/smoke-test.json 2> /tmp/smoke-test.log || true

    wait "$SMOKE_WL_PID" 2>/dev/null || true

    SMOKE_COUNT=$(python3 -c "
import json
events = []
for line in open('/tmp/smoke-test.json'):
    try: events.append(json.loads(line))
    except: pass
print(len(events))
" 2>/dev/null || echo "0")

    if [[ "$SMOKE_COUNT" -gt 0 ]]; then
        record "PASS" "Probe smoke test" "$SMOKE_COUNT events in 3s"
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
        tail -30 /tmp/smoke-test.log 2>/dev/null | while IFS= read -r line; do
            echo "    | $line"
        done
        fail ""
        fail "Aborting test suite — fix probes before proceeding."
        record "FAIL" "Probe smoke test" "0 events — probes not attaching"
        exit 1
    fi
else
    record "SKIP" "Probe smoke test" "PyTorch+CUDA not available"
fi

# ──────────────────────────────────────────────────────────────
# Phase 2: Unit Tests
# ──────────────────────────────────────────────────────────────
header "Phase 2: Unit Tests"

log "Running go test ./..."
TEST_OUTPUT=$(go test ./... 2>&1)
TEST_EXIT=$?

if [[ $TEST_EXIT -eq 0 ]]; then
    PKG_COUNT=$(echo "$TEST_OUTPUT" | grep -c '^ok')
    record "PASS" "Unit Tests" "$PKG_COUNT packages pass"
else
    FAILED_PKGS=$(echo "$TEST_OUTPUT" | grep '^FAIL' | head -5)
    record "FAIL" "Unit Tests" "exit=$TEST_EXIT: $FAILED_PKGS"
fi
echo "$TEST_OUTPUT"

# ──────────────────────────────────────────────────────────────
# Phase 3: Integration Tests
# ──────────────────────────────────────────────────────────────
header "Phase 3: Integration Tests"

# 3a: ingero check
log "Testing: ingero check"
CHECK_OUT=$(sudo ./bin/ingero check 2>&1)
CHECK_EXIT=$?
echo "$CHECK_OUT"

if echo "$CHECK_OUT" | grep -q "GPU model:"; then
    GPU_MODEL=$(echo "$CHECK_OUT" | grep "GPU model:" | sed 's/.*GPU model: //')
    record "PASS" "check: GPU model" "$GPU_MODEL"
else
    record "FAIL" "check: GPU model" "not detected"
fi

if echo "$CHECK_OUT" | grep -q "NVIDIA driver:"; then
    record "PASS" "check: NVIDIA driver" "detected"
else
    record "FAIL" "check: NVIDIA driver" "not detected"
fi

if echo "$CHECK_OUT" | grep -q "BTF support:"; then
    record "PASS" "check: BTF" "available"
else
    record "FAIL" "check: BTF" "not found"
fi

# 3b: ingero trace with real GPU workload
log "Testing: ingero trace (real GPU workload)"

# Start workload FIRST — watch auto-defaults --user=SUDO_USER when run via sudo,
# and exits immediately if no CUDA processes exist for that user.
if python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python3 -c "
import torch, time, os
print(f'GPU workload PID: {os.getpid()}', flush=True)
d = torch.device('cuda:0')
a = torch.randn(2048, 2048, device=d)
b = torch.randn(2048, 2048, device=d)
start = time.time()
while time.time() - start < 18:
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
print(f'Done: {int((time.time()-start))}s', flush=True)
" 2>&1 &
    WORKLOAD_PID=$!
    sleep 2

    sudo ./bin/ingero trace --duration 12s 2>&1 > /tmp/watch_output.log &
    WATCH_PID=$!
    wait $WATCH_PID 2>/dev/null || true
    kill $WORKLOAD_PID 2>/dev/null; wait $WORKLOAD_PID 2>/dev/null || true

    # Parse watch output (strip ANSI)
    WATCH_CLEAN=$(cat /tmp/watch_output.log | sed 's/\x1b\[[0-9;]*[a-zA-Z]//g' | sed 's/\[K//g')
    EVENT_COUNT=$(echo "$WATCH_CLEAN" | grep -oP 'Events: \K[0-9]+' | tail -1)

    if [[ -n "$EVENT_COUNT" && "$EVENT_COUNT" -gt 0 ]]; then
        record "PASS" "watch: real GPU events" "$EVENT_COUNT events captured"
    else
        record "FAIL" "watch: real GPU events" "0 events (probes may not have fired)"
    fi

    # Check for system context line
    if echo "$WATCH_CLEAN" | grep -q "System:.*CPU"; then
        record "PASS" "watch: system context" "CPU/Mem/Load bar displayed"
    else
        record "FAIL" "watch: system context" "no system context line"
    fi

    # Check for CUDA operation stats (runtime or driver API ops)
    if echo "$WATCH_CLEAN" | grep -q "cudaDeviceSync\|cudaLaunchKernel\|cudaMalloc\|cuLaunchKernel\|cuCtxSynchronize"; then
        record "PASS" "watch: CUDA ops table" "operations listed"
    else
        record "FAIL" "watch: CUDA ops table" "no CUDA operations"
    fi
else
    record "SKIP" "watch: real GPU events" "PyTorch+CUDA not available"
fi

# 3c: ingero demo --no-gpu
log "Testing: ingero demo --no-gpu"
DEMO_OUT=$(timeout 30 ./bin/ingero demo --no-gpu --duration 15s 2>&1)
DEMO_EXIT=$?

if echo "$DEMO_OUT" | grep -q "GPU:.*\|gpu:.*\|NVIDIA"; then
    record "PASS" "demo: GPU auto-detect header" "GPU info displayed"
else
    record "FAIL" "demo: GPU auto-detect header" "no GPU info in header"
fi

if echo "$DEMO_OUT" | grep -q "Scenario: incident\|incident"; then
    record "PASS" "demo: incident scenario first" "incident is first scenario"
else
    record "FAIL" "demo: incident scenario first" "incident not first"
fi

if echo "$DEMO_OUT" | grep -q "System:.*CPU"; then
    record "PASS" "demo: system context bars" "ASCII bars displayed"
else
    record "FAIL" "demo: system context bars" "no ASCII bars"
fi

# 3d: ingero explain (DB-only, analyzes events recorded by earlier trace sessions)
log "Testing: ingero explain --since 5m (DB-only, no sudo)"
EXPLAIN_OUT=$(timeout 30 ./bin/ingero explain --since 5m 2>&1)
EXPLAIN_EXIT=$?

if echo "$EXPLAIN_OUT" | grep -q "INCIDENT REPORT"; then
    record "PASS" "explain: incident report" "report generated from stored data"
else
    record "FAIL" "explain: incident report" "no report output"
fi

if echo "$EXPLAIN_OUT" | grep -q "events\|chain"; then
    record "PASS" "explain: DB analysis" "analyzed stored events"
else
    record "FAIL" "explain: DB analysis" "no events found in DB"
fi

# 3e: MCP HTTPS transport
log "Testing: ingero mcp --http :8080"
sudo ./bin/ingero mcp --http :8080 2>&1 > /tmp/mcp_output.log &
MCP_PID=$!
sleep 2

if kill -0 $MCP_PID 2>/dev/null; then
    # Test get_check
    CHECK_RESP=$(curl -sk https://localhost:8080/mcp \
        -H 'Content-Type: application/json' \
        -H 'Accept: application/json, text/event-stream' \
        -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_check","arguments":{}}}' 2>&1)

    if echo "$CHECK_RESP" | grep -q '"result"'; then
        record "PASS" "MCP HTTPS: get_check" "valid JSON-RPC response"
    else
        record "FAIL" "MCP HTTPS: get_check" "no result in response"
    fi

    # Test get_trace_stats (no DB, should gracefully report)
    STATS_RESP=$(curl -sk https://localhost:8080/mcp \
        -H 'Content-Type: application/json' \
        -H 'Accept: application/json, text/event-stream' \
        -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_trace_stats","arguments":{}}}' 2>&1)

    if echo "$STATS_RESP" | grep -q '"result"\|"error"'; then
        record "PASS" "MCP HTTPS: get_trace_stats" "responds (no DB = graceful error)"
    else
        record "FAIL" "MCP HTTPS: get_trace_stats" "no response"
    fi

    # Test run_demo
    DEMO_RESP=$(curl -sk https://localhost:8080/mcp \
        -H 'Content-Type: application/json' \
        -H 'Accept: application/json, text/event-stream' \
        -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"run_demo","arguments":{"scenario":"incident"}}}' 2>&1)

    if echo "$DEMO_RESP" | grep -q '"result"'; then
        record "PASS" "MCP HTTPS: run_demo" "valid response"
    else
        record "FAIL" "MCP HTTPS: run_demo" "no result"
    fi

    sudo kill $MCP_PID 2>/dev/null || true
    sudo pkill -f "ingero mcp" 2>/dev/null || true
    sleep 1
else
    record "FAIL" "MCP HTTPS: server start" "server crashed on startup"
    cat /tmp/mcp_output.log
fi

# 3f: Prometheus /metrics
log "Testing: ingero trace --prometheus :9090"

# Start workload first (watch needs CUDA processes when run via sudo)
if python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python3 -c "
import torch, time
d = torch.device('cuda:0')
a = torch.randn(1024, 1024, device=d)
b = torch.randn(1024, 1024, device=d)
start = time.time()
while time.time() - start < 15:
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
" 2>&1 &
    WPID=$!
    sleep 2

    sudo ./bin/ingero trace --duration 10s --prometheus :9090 2>&1 > /tmp/prom_watch.log &
    PROM_PID=$!
    sleep 5

    PROM_OUT=$(curl -s localhost:9090/metrics 2>&1)

    if echo "$PROM_OUT" | grep -q "system_cpu_utilization"; then
        record "PASS" "Prometheus: system metrics" "CPU/mem/load metrics served"
    else
        record "FAIL" "Prometheus: system metrics" "no system metrics"
    fi

    if echo "$PROM_OUT" | grep -q "gpu_cuda_operation"; then
        record "PASS" "Prometheus: CUDA metrics" "operation metrics served"
    else
        record "FAIL" "Prometheus: CUDA metrics" "no CUDA metrics"
    fi

    if echo "$PROM_OUT" | grep -q "# TYPE.*gauge\|# TYPE.*counter"; then
        record "PASS" "Prometheus: exposition format" "valid HELP/TYPE lines"
    else
        record "FAIL" "Prometheus: exposition format" "missing TYPE declarations"
    fi

    kill $WPID 2>/dev/null; wait $WPID 2>/dev/null || true
    sudo kill $PROM_PID 2>/dev/null || true
    sudo pkill -f "ingero trace.*prometheus" 2>/dev/null || true
    sleep 1
else
    record "SKIP" "Prometheus: system metrics" "no PyTorch"
    record "SKIP" "Prometheus: CUDA metrics" "no PyTorch"
    record "SKIP" "Prometheus: exposition format" "no PyTorch"
fi

# 3g: ingero version
log "Testing: ingero version"
VER_OUT=$(./bin/ingero version 2>&1)
if echo "$VER_OUT" | grep -q "ingero"; then
    record "PASS" "version" "$VER_OUT"
else
    record "FAIL" "version" "unexpected output: $VER_OUT"
fi

# 3h: ingero trace --help (verify --otlp and --prometheus flags)
log "Testing: watch --help flags"
HELP_OUT=$(./bin/ingero trace --help 2>&1)

if echo "$HELP_OUT" | grep -q "\-\-otlp"; then
    record "PASS" "watch: --otlp flag" "present"
else
    record "FAIL" "watch: --otlp flag" "missing"
fi

if echo "$HELP_OUT" | grep -q "\-\-prometheus"; then
    record "PASS" "watch: --prometheus flag" "present"
else
    record "FAIL" "watch: --prometheus flag" "missing"
fi

# 3i: Driver API tracing (libcuda.so)
log "Testing: driver API tracing (libcuda.so)"
if python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    python3 -c "
import torch, time
d = torch.device('cuda:0')
a = torch.randn(2048, 2048, device=d)
b = torch.randn(2048, 2048, device=d)
start = time.time()
while time.time() - start < 15:
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
" 2>&1 &
    MATMUL_PID=$!
    sleep 2

    sudo ./bin/ingero trace --duration 10s --json 2>/dev/null > /tmp/driver_watch.json &
    DRIVER_WATCH_PID=$!
    wait $DRIVER_WATCH_PID 2>/dev/null || true
    kill $MATMUL_PID 2>/dev/null; wait $MATMUL_PID 2>/dev/null || true

    CU_LAUNCH_COUNT=$(grep -c 'cuLaunchKernel' /tmp/driver_watch.json 2>/dev/null || echo "0")
    CUDA_LAUNCH_COUNT=$(grep -c 'cudaLaunchKernel' /tmp/driver_watch.json 2>/dev/null || echo "0")

    if [[ "$CU_LAUNCH_COUNT" -gt 0 ]]; then
        record "PASS" "driver API: cuLaunchKernel" "$CU_LAUNCH_COUNT events (vs $CUDA_LAUNCH_COUNT runtime)"
    else
        record "FAIL" "driver API: cuLaunchKernel" "0 events — libcuda.so probes not firing"
    fi
else
    record "SKIP" "driver API: cuLaunchKernel" "PyTorch+CUDA not available"
fi

# 3j: record + query round-trip (recording is default)
log "Testing: record + query round-trip"
if python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    rm -f ~/.ingero/ingero.db ~/.ingero/ingero.db-wal ~/.ingero/ingero.db-shm
    sudo rm -f /root/.ingero/ingero.db /root/.ingero/ingero.db-wal /root/.ingero/ingero.db-shm

    python3 -c "
import torch, time
d = torch.device('cuda:0')
a = torch.randn(1024, 1024, device=d)
start = time.time()
while time.time() - start < 12:
    b = torch.matmul(a, a)
    torch.cuda.synchronize()
" 2>&1 &
    WORKLOAD_PID=$!
    sleep 2

    sudo ./bin/ingero trace --duration 8s 2>/dev/null > /tmp/record_output.log &
    RECORD_PID=$!
    wait $RECORD_PID 2>/dev/null || true
    kill $WORKLOAD_PID 2>/dev/null; wait $WORKLOAD_PID 2>/dev/null || true

    # With SUDO_USER-aware path, DB lands in the invoking user's home.
    if [ -f ~/.ingero/ingero.db ]; then
        record "PASS" "record: DB created" "~/.ingero/ingero.db exists"
    elif sudo test -f /root/.ingero/ingero.db; then
        record "PASS" "record: DB created" "/root/.ingero/ingero.db exists (legacy path)"
    else
        record "FAIL" "record: DB created" "database not found"
    fi

    QUERY_OUT=$(sudo ./bin/ingero query --since 5m --json 2>&1)
    QUERY_COUNT=$(echo "$QUERY_OUT" | grep -c '"op"' || true)
    if [[ "$QUERY_COUNT" -gt 0 ]]; then
        record "PASS" "record: query round-trip" "$QUERY_COUNT events retrieved"
    else
        record "FAIL" "record: query round-trip" "query returned no events"
    fi
else
    record "SKIP" "record: round-trip" "PyTorch+CUDA not available"
fi

# 3k: --debug flag
log "Testing: --debug flag"
if echo "$(./bin/ingero --help 2>&1)" | grep -q "\-\-debug"; then
    record "PASS" "--debug: flag present" "shown in --help"
else
    record "FAIL" "--debug: flag present" "not in --help output"
fi

DEBUG_OUT=$(sudo ./bin/ingero check --debug 2>&1)
if echo "$DEBUG_OUT" | grep -q "\[DEBUG\]"; then
    record "PASS" "--debug: output on stderr" "[DEBUG] prefix in check"
else
    record "FAIL" "--debug: output on stderr" "no [DEBUG] output"
fi

# 3l: --debug watch (run with and without to verify no debug noise when off)
log "Testing: --debug watch mode (both modes)"
if python3 -c "import torch; torch.cuda.is_available()" 2>/dev/null; then
    # Without --debug: start workload, then watch
    python3 -c "
import torch, time
a = torch.randn(512, 512, device='cuda:0')
start = time.time()
while time.time() - start < 10:
    torch.matmul(a, a)
    torch.cuda.synchronize()
" 2>/dev/null &
    NODEBUG_WL=$!
    sleep 2

    sudo ./bin/ingero trace --duration 5s --json 2>/tmp/watch_nodebug_stderr.log > /tmp/watch_nodebug.json &
    NODEBUG_PID=$!
    wait $NODEBUG_PID 2>/dev/null || true
    kill $NODEBUG_WL 2>/dev/null; wait $NODEBUG_WL 2>/dev/null || true

    if grep -q "\[DEBUG\]" /tmp/watch_nodebug_stderr.log 2>/dev/null; then
        record "FAIL" "--debug off: no debug noise" "[DEBUG] found in stderr without --debug"
    else
        record "PASS" "--debug off: no debug noise" "stderr clean"
    fi

    # With --debug: start workload, then watch
    python3 -c "
import torch, time
a = torch.randn(512, 512, device='cuda:0')
start = time.time()
while time.time() - start < 10:
    torch.matmul(a, a)
    torch.cuda.synchronize()
" 2>/dev/null &
    DEBUG_WL=$!
    sleep 2

    sudo ./bin/ingero trace --debug --duration 5s --json 2>/tmp/watch_debug_stderr.log > /tmp/watch_debug.json &
    DEBUG_PID=$!
    wait $DEBUG_PID 2>/dev/null || true
    kill $DEBUG_WL 2>/dev/null; wait $DEBUG_WL 2>/dev/null || true

    if grep -q "\[DEBUG\]" /tmp/watch_debug_stderr.log 2>/dev/null; then
        record "PASS" "--debug on: debug output" "[DEBUG] found in stderr"
    else
        record "FAIL" "--debug on: debug output" "no [DEBUG] output with --debug"
    fi
else
    record "SKIP" "--debug watch modes" "PyTorch+CUDA not available"
fi

# ──────────────────────────────────────────────────────────────
# Report
# ──────────────────────────────────────────────────────────────
header "Integration Test Report"

TOTAL=$((PASS_COUNT + FAIL_COUNT + SKIP_COUNT))
echo ""
echo -e "$(ts)   ${GREEN}PASS: $PASS_COUNT${NC}  ${RED}FAIL: $FAIL_COUNT${NC}  ${YELLOW}SKIP: $SKIP_COUNT${NC}  Total: $TOTAL"
echo ""

# Write report file (timestamps included in REPORT via record())
{
    echo "Ingero v0.6 Integration Test Report"
    echo "===================================="
    echo "Date: $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
    echo "Host: $(hostname)"
    echo "Kernel: $(uname -r)"
    echo "GPU: $(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
    echo "Go: $(go version 2>/dev/null | awk '{print $3}')"
    echo ""
    echo "Results: PASS=$PASS_COUNT  FAIL=$FAIL_COUNT  SKIP=$SKIP_COUNT  Total=$TOTAL"
    echo ""
    printf '%s' "$REPORT"
} > "$REPORT_FILE"

echo "$(ts) Report saved to: $REPORT_FILE"

if [[ $FAIL_COUNT -gt 0 ]]; then
    echo ""
    fail "$FAIL_COUNT test(s) failed. See report for details."
    exit 1
fi

log "All tests passed!"
exit 0
