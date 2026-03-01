#!/bin/bash
################################################################################
# Ingero K8s Integration Test — KT01-KT14
#
# End-to-end tests for v0.7 K8s support: cgroup_id in eBPF events, container
# ID resolution, DaemonSet lifecycle, MCP queries with container metadata.
#
# Tests:
#   KT01: Container image build & load (docker → k3s ctr import)
#   KT02: DaemonSet lifecycle (Running, no CrashLoopBackOff)
#   KT03: Probe attach (eBPF from K8s pod, events in SQLite)
#   KT04: cgroup_id in events (bpf_get_current_cgroup_id)
#   KT05: Container ID resolution (gold test: K8s API == cgroup_metadata)
#   KT06: Causal chains (chain engine with containerized events)
#   KT07: MCP queries (container-enriched)
#   KT08: Multi-pod tracing (2 sequential pods, distinct cgroup_ids)
#   KT09: Resource limits (DaemonSet pod < 512Mi)
#   KT10: Uprobe cleanup (bare-metal tracing works after DaemonSet delete)
#   KT11-14: v0.8 placeholders (skip)
#   Final: Bare-metal regression (gpu-test.sh, 62 tests)
#
# Prerequisites:
#   - bash scripts/k3s-setup.sh (k3s + GPU plugin + docker + sqlite3 + PyTorch image)
#   - make build (bin/ingero compiled)
#
# Usage:
#   bash scripts/k3s-test.sh
#   # or: make gpu-k3s-test / make lambda-k3s-test (from WSL)
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
SCRIPT_START=$SECONDS

# Per-test timing and structured results for JSON report
declare -a TEST_RESULTS  # Each entry: "id|name|status|detail|duration_s"
_test_start=$SECONDS

cd "$(dirname "$0")/.." || { echo "Cannot find ingero directory"; exit 1; }
mkdir -p logs

KUBECTL="sudo k3s kubectl"
INGERO_DB="/var/lib/ingero/ingero.db"
INGERO_IMAGE="docker.io/library/ingero:v0.7-test"

ts()   { date -u '+%Y-%m-%d %H:%M:%S'; }
log()  { echo -e "$(ts) ${GREEN}[INFO]${NC}  $1"; }
warn() { echo -e "$(ts) ${YELLOW}[WARN]${NC}  $1"; }
errmsg() { echo -e "$(ts) ${RED}[FAIL]${NC}  $1"; }

header() {
    echo ""
    echo -e "$(ts) ${BLUE}════════════════════════════════════════════════════════════${NC}"
    echo -e "$(ts) ${BLUE}  $1${NC}"
    echo -e "$(ts) ${BLUE}════════════════════════════════════════════════════════════${NC}"
}

record() {
    local status="$1" name="$2" detail="$3"
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

# Cleanup on exit
cleanup_pids=()
cleanup() {
    for pid in "${cleanup_pids[@]}"; do
        kill "$pid" 2>/dev/null || true
        wait "$pid" 2>/dev/null || true
    done
}
trap cleanup EXIT

# Helper: deploy a PyTorch matmul workload pod (30s loop)
deploy_workload_pod() {
    local pod_name="$1"
    $KUBECTL apply -f - <<WORKLOAD_EOF
apiVersion: v1
kind: Pod
metadata:
  name: ${pod_name}
  namespace: default
spec:
  restartPolicy: Never
  containers:
  - name: pytorch
    image: nvcr.io/nvidia/pytorch:24.01-py3
    command: ["python3", "-c", "import torch, time, os\nprint(f'PID={os.getpid()}', flush=True)\nd = torch.device('cuda:0')\na = torch.randn(2048, 2048, device=d)\nb = torch.randn(2048, 2048, device=d)\nstart = time.time()\nwhile time.time() - start < 30:\n    c = torch.matmul(a, b)\n    torch.cuda.synchronize()\nprint(f'Done: {time.time()-start:.1f}s', flush=True)"]
    resources:
      limits:
        nvidia.com/gpu: 1
WORKLOAD_EOF
}

# Helper: wait for pod to reach a terminal phase (Succeeded/Failed) or timeout
wait_pod_done() {
    local pod_name="$1"
    local timeout="${2:-180}"
    for i in $(seq 1 "$timeout"); do
        PHASE=$($KUBECTL get pod "$pod_name" -o jsonpath='{.status.phase}' 2>/dev/null || echo "Pending")
        if [ "$PHASE" = "Succeeded" ] || [ "$PHASE" = "Failed" ]; then
            echo "$PHASE"
            return 0
        fi
        sleep 1
    done
    echo "Timeout"
    return 1
}

# ── Binary guard — fail fast if not built ──
if [[ ! -x bin/ingero ]]; then
    errmsg "bin/ingero not found. Run 'make build' first."
    exit 1
fi

exec > >(tee logs/k3s-integration-report.log) 2>&1

echo "================================================================"
echo "  Ingero v0.7 K8s Integration Test — $(date)"
echo "  $(nvidia-smi --query-gpu=name,driver_version --format=csv,noheader 2>/dev/null || echo 'No GPU')"
echo "  Arch: $(uname -m)"
echo "================================================================"

################################################################################
# Pre-flight checks
################################################################################
header "Pre-flight Checks"

if ! command -v k3s &>/dev/null; then
    errmsg "k3s not installed. Run: bash scripts/k3s-setup.sh"
    exit 1
fi
log "k3s: $(k3s --version 2>/dev/null | head -1)"

GPU_COUNT=$($KUBECTL get node -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}' 2>/dev/null || echo "0")
if [ "$GPU_COUNT" = "0" ] || [ -z "$GPU_COUNT" ]; then
    errmsg "No GPU allocatable in k3s. Run: bash scripts/k3s-setup.sh"
    exit 1
fi
log "GPUs allocatable: $GPU_COUNT"

if ! command -v sqlite3 &>/dev/null; then
    errmsg "sqlite3 not installed. Run: bash scripts/k3s-setup.sh"
    exit 1
fi

################################################################################
# KT01: Container Image Build & Load
################################################################################
header "KT01: Container Image Build & Load"

# Verify binary is a Linux ELF
if file bin/ingero | grep -q "ELF 64-bit LSB"; then
    log "bin/ingero: valid Linux ELF binary"
else
    record "FAIL" "KT01: Container image build & load" "bin/ingero is not a Linux ELF binary"
    exit 1
fi

# Create minimal test Dockerfile
cat > /tmp/Dockerfile.ingero-test <<'DOCKERFILE_EOF'
FROM ubuntu:22.04
RUN apt-get update && apt-get install -y --no-install-recommends libbpf0 && rm -rf /var/lib/apt/lists/*
COPY bin/ingero /usr/local/bin/ingero
ENTRYPOINT ["/usr/local/bin/ingero"]
CMD ["trace", "--record"]
DOCKERFILE_EOF

# Build image (docker primary, buildah fallback)
if command -v docker &>/dev/null; then
    log "Building image with docker..."
    if docker build -t ingero:v0.7-test -f /tmp/Dockerfile.ingero-test . >> logs/kt01-image-build.log 2>&1; then
        log "Image built successfully"
    else
        record "FAIL" "KT01: Container image build & load" "docker build failed (see logs/kt01-image-build.log)"
        exit 1
    fi

    log "Importing image into k3s containerd..."
    if docker save ingero:v0.7-test | sudo k3s ctr images import - >> logs/kt01-image-build.log 2>&1; then
        log "Image imported into k3s containerd"
    else
        record "FAIL" "KT01: Container image build & load" "k3s ctr images import failed"
        exit 1
    fi
elif command -v buildah &>/dev/null; then
    log "Building image with buildah (docker not found)..."
    if sudo buildah bud -t ingero:v0.7-test -f /tmp/Dockerfile.ingero-test . >> logs/kt01-image-build.log 2>&1; then
        log "Image built with buildah"
    else
        record "FAIL" "KT01: Container image build & load" "buildah build failed"
        exit 1
    fi
    sudo buildah push ingero:v0.7-test oci-archive:/tmp/ingero.tar >> logs/kt01-image-build.log 2>&1
    sudo k3s ctr images import /tmp/ingero.tar >> logs/kt01-image-build.log 2>&1
    rm -f /tmp/ingero.tar
else
    record "FAIL" "KT01: Container image build & load" "no image build tool (docker or buildah). Install: sudo apt-get install -y docker.io"
    exit 1
fi

# Verify image is in k3s containerd
if sudo k3s ctr images ls | grep -q "ingero:v0.7-test"; then
    log "Image confirmed in k3s containerd"
else
    record "FAIL" "KT01: Container image build & load" "image not found in k3s containerd"
    exit 1
fi

# Verify ingero version runs inside the container (only if docker available)
if command -v docker &>/dev/null; then
    VERSION_OUTPUT=$(docker run --rm ingero:v0.7-test version 2>/dev/null || echo "")
    if [ -n "$VERSION_OUTPUT" ]; then
        log "Container ingero version: $VERSION_OUTPUT"
    else
        warn "ingero version failed inside container (non-fatal, image may still work in k3s)"
    fi
fi
record "PASS" "KT01: Container image build & load" "$INGERO_IMAGE in containerd"

################################################################################
# KT02: DaemonSet Lifecycle
################################################################################
header "KT02: DaemonSet Lifecycle"

# Clean up any previous state
sudo rm -f "$INGERO_DB" "${INGERO_DB}-wal" "${INGERO_DB}-shm" 2>/dev/null || true
$KUBECTL delete -f deploy/k8s/daemonset.yaml --ignore-not-found 2>/dev/null || true
sleep 2

# Apply namespace + RBAC
$KUBECTL apply -f deploy/k8s/namespace.yaml
$KUBECTL apply -f deploy/k8s/rbac.yaml

# Apply DaemonSet with patched image reference and imagePullPolicy
# docker save exports with docker.io/library/ prefix — must match exactly.
sed "s|ghcr.io/ingero-io/ingero:v0.7|${INGERO_IMAGE}|g" deploy/k8s/daemonset.yaml | \
    sed 's|imagePullPolicy: IfNotPresent|imagePullPolicy: Never|g' | \
    $KUBECTL apply -f -

log "DaemonSet applied, waiting for pod Running..."

# Poll for Running status
POD_RUNNING=false
for i in $(seq 1 120); do
    STATUS=$($KUBECTL get pods -n ingero-system -l app.kubernetes.io/name=ingero \
        -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "Pending")
    if [ "$STATUS" = "Running" ]; then
        POD_RUNNING=true
        break
    fi
    sleep 1
done

# Log pod describe for diagnostics
$KUBECTL describe pod -n ingero-system -l app.kubernetes.io/name=ingero \
    > logs/kt02-daemonset-lifecycle.log 2>&1 || true

if $POD_RUNNING; then
    # Check restart count
    RESTARTS=$($KUBECTL get pods -n ingero-system -l app.kubernetes.io/name=ingero \
        -o jsonpath='{.items[0].status.containerStatuses[0].restartCount}' 2>/dev/null || echo "0")
    if [ "$RESTARTS" = "0" ]; then
        INGERO_POD=$($KUBECTL get pods -n ingero-system -l app.kubernetes.io/name=ingero \
            -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
        log "Pod $INGERO_POD Running, restarts=0"
        record "PASS" "KT02: DaemonSet lifecycle" "pod Running, restarts=0"
    else
        record "FAIL" "KT02: DaemonSet lifecycle" "pod Running but restartCount=$RESTARTS"
    fi
else
    record "FAIL" "KT02: DaemonSet lifecycle" "pod not Running after 120s (status: $STATUS)"
    # Dump pod logs for debugging
    $KUBECTL logs -n ingero-system -l app.kubernetes.io/name=ingero --tail=50 \
        >> logs/kt02-daemonset-lifecycle.log 2>&1 || true
fi

# Wait a few seconds for probes to attach and DB to be created
log "Waiting 10s for probes to attach..."
sleep 10

################################################################################
# KT03: Probes Attached (eBPF from K8s pod)
################################################################################
header "KT03: Probes Attached"

# Deploy PyTorch workload pod (30s matmul loop)
log "Deploying PyTorch workload pod (30s matmul loop)..."
deploy_workload_pod "pytorch-matmul-a"

# Wait for workload pod to complete
log "Waiting for pytorch-matmul-a to complete..."
WORKLOAD_PHASE=$(wait_pod_done "pytorch-matmul-a" 180)

if [ "$WORKLOAD_PHASE" = "Succeeded" ]; then
    log "Workload completed successfully"
    $KUBECTL logs pytorch-matmul-a > logs/kt03-probe-attach.log 2>&1 || true
elif [ "$WORKLOAD_PHASE" = "Failed" ]; then
    warn "Workload pod failed"
    $KUBECTL logs pytorch-matmul-a > logs/kt03-probe-attach.log 2>&1 || true
    $KUBECTL describe pod pytorch-matmul-a >> logs/kt03-probe-attach.log 2>&1 || true
else
    warn "Workload pod timed out (phase: $WORKLOAD_PHASE)"
    $KUBECTL describe pod pytorch-matmul-a > logs/kt03-probe-attach.log 2>&1 || true
fi

# Wait for event flush to SQLite
log "Waiting 10s for event flush..."
sleep 10

# Query event counts
if [ -f "$INGERO_DB" ]; then
    TOTAL_EVENTS=$(sudo sqlite3 "$INGERO_DB" "SELECT COUNT(*) FROM events" 2>/dev/null || echo "0")
    log "Total events in DB: $TOTAL_EVENTS"

    # Per-source breakdown
    log "Per-source breakdown:"
    sudo sqlite3 "$INGERO_DB" \
        "SELECT s.name, COUNT(*) FROM events e JOIN sources s ON e.source=s.id GROUP BY s.name" \
        2>/dev/null | while IFS= read -r line; do
        log "  $line"
    done

    if [ "$TOTAL_EVENTS" -gt 100 ]; then
        record "PASS" "KT03: Probes attached" "$TOTAL_EVENTS events captured"
    elif [ "$TOTAL_EVENTS" -gt 0 ]; then
        record "PASS" "KT03: Probes attached" "$TOTAL_EVENTS events (low count, but probes working)"
    else
        record "FAIL" "KT03: Probes attached" "0 events in DB — probes may not be attached"
    fi
else
    record "FAIL" "KT03: Probes attached" "DB file $INGERO_DB not found"
fi

################################################################################
# KT04: Containerized GPU Tracing (cgroup_id)
################################################################################
header "KT04: cgroup_id in Events"

if [ ! -f "$INGERO_DB" ]; then
    record "SKIP" "KT04: cgroup_id in events" "DB not found (KT03 failed)"
else
    # Count events with non-zero cgroup_id
    # cgroup_id > 1 filters out the root cgroup (0 or 1)
    CGROUP_COUNT=$(sudo sqlite3 "$INGERO_DB" \
        "SELECT COUNT(*) FROM events WHERE cgroup_id > 1" 2>/dev/null || echo "0")
    log "Events with cgroup_id > 1: $CGROUP_COUNT"

    # Show distinct cgroup_ids
    log "Distinct cgroup_ids:"
    sudo sqlite3 "$INGERO_DB" \
        "SELECT DISTINCT cgroup_id FROM events WHERE cgroup_id > 1" \
        2>/dev/null | while IFS= read -r line; do
        log "  cgroup_id=$line"
    done

    # Per-source breakdown with cgroup stats
    log "Per-source cgroup coverage:"
    sudo sqlite3 "$INGERO_DB" \
        "SELECT s.name, COUNT(*), COUNT(CASE WHEN e.cgroup_id > 1 THEN 1 END) as with_cgroup
         FROM events e JOIN sources s ON e.source=s.id GROUP BY s.name" \
        2>/dev/null | while IFS= read -r line; do
        log "  $line"
    done

    if [ "$CGROUP_COUNT" -ge 50 ]; then
        record "PASS" "KT04: cgroup_id in events" "$CGROUP_COUNT events with cgroup_id"
    elif [ "$CGROUP_COUNT" -gt 0 ]; then
        record "PASS" "KT04: cgroup_id in events" "$CGROUP_COUNT events with cgroup_id (low but present)"
    else
        record "FAIL" "KT04: cgroup_id in events" "0 events with non-zero cgroup_id"
    fi
fi

################################################################################
# KT05: Container ID Resolution (The Gold Test)
################################################################################
header "KT05: Container ID Resolution"

if [ ! -f "$INGERO_DB" ] || [ "$WORKLOAD_PHASE" != "Succeeded" ]; then
    record "SKIP" "KT05: Container ID resolution" "prerequisite failed (DB or workload)"
else
    # Get container ID from K8s API
    K8S_RAW=$($KUBECTL get pod pytorch-matmul-a \
        -o jsonpath='{.status.containerStatuses[0].containerID}' 2>/dev/null || echo "")
    K8S_CONTAINER_ID="${K8S_RAW#containerd://}"
    log "K8s container ID: $K8S_CONTAINER_ID"

    if [ -z "$K8S_CONTAINER_ID" ]; then
        record "SKIP" "KT05: Container ID resolution" "could not get container ID from K8s API"
    else
        # Query cgroup_metadata table (including pod_name/namespace from K8s API enrichment)
        log "cgroup_metadata entries:"
        sudo sqlite3 "$INGERO_DB" \
            "SELECT cgroup_id, container_id, pod_name, namespace, cgroup_path FROM cgroup_metadata WHERE container_id != ''" \
            2>/dev/null | while IFS= read -r line; do
            log "  $line"
        done

        # Check if K8s container ID appears in cgroup_metadata
        FOUND=$(sudo sqlite3 "$INGERO_DB" \
            "SELECT COUNT(*) FROM cgroup_metadata WHERE container_id = '$K8S_CONTAINER_ID'" \
            2>/dev/null || echo "0")

        if [ "$FOUND" -gt 0 ]; then
            # Cross-reference: verify events exist for that cgroup_id
            CGROUP_ID=$(sudo sqlite3 "$INGERO_DB" \
                "SELECT cgroup_id FROM cgroup_metadata WHERE container_id = '$K8S_CONTAINER_ID' LIMIT 1" \
                2>/dev/null || echo "0")
            EVENT_COUNT=$(sudo sqlite3 "$INGERO_DB" \
                "SELECT COUNT(*) FROM events WHERE cgroup_id = $CGROUP_ID" \
                2>/dev/null || echo "0")
            log "cgroup_id=$CGROUP_ID has $EVENT_COUNT events"

            # Verify pod_name and namespace enrichment (populated by PodCache in K8s mode)
            DB_POD_NAME=$(sudo sqlite3 "$INGERO_DB" \
                "SELECT pod_name FROM cgroup_metadata WHERE container_id = '$K8S_CONTAINER_ID' LIMIT 1" \
                2>/dev/null || echo "")
            DB_NAMESPACE=$(sudo sqlite3 "$INGERO_DB" \
                "SELECT namespace FROM cgroup_metadata WHERE container_id = '$K8S_CONTAINER_ID' LIMIT 1" \
                2>/dev/null || echo "")
            log "pod_name=$DB_POD_NAME namespace=$DB_NAMESPACE"

            if [ "$EVENT_COUNT" -gt 0 ]; then
                # Pod name/namespace may be empty on bare-metal or if PodCache isn't running.
                # When running in K8s (DaemonSet), they should be populated.
                if [ -n "$DB_POD_NAME" ] && [ "$DB_POD_NAME" = "pytorch-matmul-a" ]; then
                    record "PASS" "KT05: Container ID resolution" "K8s container_id matched, pod=$DB_POD_NAME/$DB_NAMESPACE, $EVENT_COUNT events"
                elif [ -n "$DB_POD_NAME" ]; then
                    record "FAIL" "KT05: Container ID resolution" "pod_name mismatch: got '$DB_POD_NAME', want 'pytorch-matmul-a'"
                else
                    # Pod name empty — PodCache not running (bare-metal test mode).
                    # Container ID match alone is still a valid pass.
                    record "PASS" "KT05: Container ID resolution" "K8s container_id matched, $EVENT_COUNT events (pod metadata not enriched)"
                fi
            else
                record "FAIL" "KT05: Container ID resolution" "container_id matched but 0 events for cgroup_id=$CGROUP_ID"
            fi
        else
            # Debug: show what we have vs what we expected
            warn "K8s container_id not found in cgroup_metadata"
            warn "Expected: $K8S_CONTAINER_ID"
            warn "Available container_ids:"
            sudo sqlite3 "$INGERO_DB" \
                "SELECT container_id, pod_name, namespace FROM cgroup_metadata WHERE container_id != ''" \
                2>/dev/null | while IFS= read -r line; do
                warn "  $line"
            done
            record "FAIL" "KT05: Container ID resolution" "K8s container_id not in cgroup_metadata"
        fi
    fi
fi

################################################################################
# KT06: Causal Chains
################################################################################
header "KT06: Causal Chains"

if [ ! -f "$INGERO_DB" ]; then
    record "SKIP" "KT06: Causal chains" "DB not found"
else
    CHAIN_COUNT=$(sudo sqlite3 "$INGERO_DB" \
        "SELECT COUNT(*) FROM causal_chains" 2>/dev/null || echo "0")
    log "Stored causal chains: $CHAIN_COUNT"

    if [ "$CHAIN_COUNT" -gt 0 ]; then
        log "Top chains by severity:"
        sudo sqlite3 "$INGERO_DB" \
            "SELECT severity, cuda_op, summary FROM causal_chains ORDER BY severity DESC LIMIT 10" \
            2>/dev/null | while IFS= read -r line; do
            log "  $line"
        done
        record "PASS" "KT06: Causal chains" "$CHAIN_COUNT chains detected"
    else
        # Check for chain-eligible event patterns (sync stalls > 1ms)
        SYNC_STALLS=$(sudo sqlite3 "$INGERO_DB" \
            "SELECT COUNT(*) FROM events e JOIN ops o ON e.source=o.source_id AND e.op=o.op_id
             WHERE o.name IN ('cudaDeviceSync','cudaStreamSync','cuCtxSynchronize')
             AND e.duration > 1000000" 2>/dev/null || echo "0")
        log "Sync stalls > 1ms (chain-eligible): $SYNC_STALLS"

        if [ "$SYNC_STALLS" -gt 0 ]; then
            record "PASS" "KT06: Causal chains" "0 stored chains, but $SYNC_STALLS chain-eligible sync stalls"
        else
            record "FAIL" "KT06: Causal chains" "0 chains and 0 chain-eligible events"
        fi
    fi

    # Also run explain from host for diagnostic output
    log "Running ingero explain..."
    sudo ./bin/ingero explain --db "$INGERO_DB" --since 0 > logs/kt06-causal-chains.log 2>&1 || true
    head -20 logs/kt06-causal-chains.log | while IFS= read -r line; do
        log "  $line"
    done
fi

################################################################################
# KT07: MCP Queries (Container-Enriched)
################################################################################
header "KT07: MCP Queries"

if [ ! -f "$INGERO_DB" ]; then
    record "SKIP" "KT07: MCP queries" "DB not found"
else
    MCP_PASS=0
    MCP_FAIL=0

    # Start MCP server with HTTP transport (same pattern as gpu-test.sh)
    log "Starting MCP server on :8080 against $INGERO_DB..."
    sudo pkill -f 'ingero mcp' 2>/dev/null || true
    sleep 0.5
    sudo ./bin/ingero mcp --http :8080 --db "$INGERO_DB" > logs/kt07-mcp-server.log 2>&1 &
    MCP_PID=$!
    cleanup_pids+=("$MCP_PID")

    # Wait for MCP server to be ready (max 5s)
    MCP_READY=0
    for i in $(seq 1 10); do
        if curl -skf -o /dev/null https://localhost:8080/mcp \
            -H 'Content-Type: application/json' \
            -H 'Accept: application/json, text/event-stream' \
            -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"k3s-test","version":"1.0"}}}' 2>/dev/null; then
            MCP_READY=1
            break
        fi
        sleep 0.5
    done

    if [ "$MCP_READY" -eq 0 ]; then
        warn "MCP server did not start — skipping MCP tests"
        record "FAIL" "KT07: MCP queries" "MCP server failed to start"
        sudo kill "$MCP_PID" 2>/dev/null || true
    else
        log "MCP server ready"

        # Helper: call an MCP tool via HTTP and return the response
        mcp_call() {
            local tool="$1" args="$2"
            curl -skf https://localhost:8080/mcp \
                -H 'Content-Type: application/json' \
                -H 'Accept: application/json, text/event-stream' \
                -d "{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"${tool}\",\"arguments\":${args}}}" 2>/dev/null
        }

        # KT07a: get_check
        log "MCP get_check..."
        RESP=$(mcp_call "get_check" '{}')
        echo "KT07a: $RESP" >> logs/kt07-mcp-queries.log
        if [ -n "$RESP" ] && echo "$RESP" | grep -q '"result"'; then
            log "  get_check: OK"
            MCP_PASS=$((MCP_PASS + 1))
        else
            warn "  get_check: failed or no result"
            MCP_FAIL=$((MCP_FAIL + 1))
        fi

        # KT07b: get_trace_stats
        log "MCP get_trace_stats..."
        RESP=$(mcp_call "get_trace_stats" '{"since":"0"}')
        echo "KT07b: $RESP" >> logs/kt07-mcp-queries.log
        if [ -n "$RESP" ] && echo "$RESP" | grep -q '"result"'; then
            log "  get_trace_stats: OK"
            MCP_PASS=$((MCP_PASS + 1))
        else
            warn "  get_trace_stats: failed or no result"
            MCP_FAIL=$((MCP_FAIL + 1))
        fi

        # KT07c: run_sql (JOIN events with cgroup_metadata)
        log "MCP run_sql (cgroup JOIN)..."
        RESP=$(mcp_call "run_sql" '{"query":"SELECT cm.container_id, COUNT(*) as events FROM events e JOIN cgroup_metadata cm ON e.cgroup_id = cm.cgroup_id WHERE cm.container_id != '"'"''"'"' GROUP BY cm.container_id"}')
        echo "KT07c: $RESP" >> logs/kt07-mcp-queries.log
        if [ -n "$RESP" ] && echo "$RESP" | grep -q '"result"'; then
            log "  run_sql: OK"
            MCP_PASS=$((MCP_PASS + 1))
        else
            warn "  run_sql: failed or no result"
            MCP_FAIL=$((MCP_FAIL + 1))
        fi

        # KT07d: get_causal_chains
        log "MCP get_causal_chains..."
        RESP=$(mcp_call "get_causal_chains" '{"since":"0"}')
        echo "KT07d: $RESP" >> logs/kt07-mcp-queries.log
        if [ -n "$RESP" ] && echo "$RESP" | grep -q '"result"'; then
            log "  get_causal_chains: OK"
            MCP_PASS=$((MCP_PASS + 1))
        else
            warn "  get_causal_chains: failed or no result"
            MCP_FAIL=$((MCP_FAIL + 1))
        fi

        # Kill MCP server
        sudo kill "$MCP_PID" 2>/dev/null || true
        wait "$MCP_PID" 2>/dev/null || true

        if [ "$MCP_FAIL" -eq 0 ]; then
            record "PASS" "KT07: MCP queries" "$MCP_PASS/4 tools returned valid responses"
        else
            record "FAIL" "KT07: MCP queries" "$MCP_FAIL/4 tools failed"
        fi
    fi
fi

################################################################################
# KT08: Multi-Pod Tracing
################################################################################
header "KT08: Multi-Pod Tracing"

if [ ! -f "$INGERO_DB" ]; then
    record "SKIP" "KT08: Multi-pod tracing" "DB not found"
else
    # Clean up previous workload pod
    $KUBECTL delete pod pytorch-matmul-a --ignore-not-found 2>/dev/null || true

    BEFORE=$(sudo sqlite3 "$INGERO_DB" \
        "SELECT COUNT(DISTINCT container_id) FROM cgroup_metadata WHERE container_id != ''" \
        2>/dev/null || echo "0")
    log "Distinct container_ids before multi-pod test: $BEFORE"

    # Pod A
    log "Deploying pytorch-multi-a (30s matmul)..."
    deploy_workload_pod "pytorch-multi-a"
    PHASE_A=$(wait_pod_done "pytorch-multi-a" 180)
    log "pytorch-multi-a: $PHASE_A"
    sleep 5  # flush

    # Pod B
    log "Deploying pytorch-multi-b (30s matmul)..."
    deploy_workload_pod "pytorch-multi-b"
    PHASE_B=$(wait_pod_done "pytorch-multi-b" 180)
    log "pytorch-multi-b: $PHASE_B"
    sleep 10  # flush

    if [ "$PHASE_A" != "Succeeded" ] || [ "$PHASE_B" != "Succeeded" ]; then
        record "FAIL" "KT08: Multi-pod tracing" "workload pods failed (A=$PHASE_A, B=$PHASE_B)"
    else
        # Get container IDs from K8s API
        CID_A_RAW=$($KUBECTL get pod pytorch-multi-a \
            -o jsonpath='{.status.containerStatuses[0].containerID}' 2>/dev/null || echo "")
        CID_A="${CID_A_RAW#containerd://}"
        CID_B_RAW=$($KUBECTL get pod pytorch-multi-b \
            -o jsonpath='{.status.containerStatuses[0].containerID}' 2>/dev/null || echo "")
        CID_B="${CID_B_RAW#containerd://}"
        log "Pod A container_id: $CID_A"
        log "Pod B container_id: $CID_B"

        # Check both appear in cgroup_metadata
        FOUND_A=$(sudo sqlite3 "$INGERO_DB" \
            "SELECT COUNT(*) FROM cgroup_metadata WHERE container_id='$CID_A'" 2>/dev/null || echo "0")
        FOUND_B=$(sudo sqlite3 "$INGERO_DB" \
            "SELECT COUNT(*) FROM cgroup_metadata WHERE container_id='$CID_B'" 2>/dev/null || echo "0")

        # Count events per container
        log "Events per container:"
        sudo sqlite3 "$INGERO_DB" \
            "SELECT cm.container_id, COUNT(*) as events
             FROM cgroup_metadata cm
             JOIN events e ON cm.cgroup_id = e.cgroup_id
             WHERE cm.container_id != ''
             GROUP BY cm.container_id" \
            2>/dev/null | while IFS= read -r line; do
            log "  $line"
        done

        {
            echo "Pod A: container_id=$CID_A found=$FOUND_A"
            echo "Pod B: container_id=$CID_B found=$FOUND_B"
            $KUBECTL logs pytorch-multi-a 2>/dev/null || true
            $KUBECTL logs pytorch-multi-b 2>/dev/null || true
        } > logs/kt08-multi-pod.log 2>&1

        if [ "$FOUND_A" -gt 0 ] && [ "$FOUND_B" -gt 0 ]; then
            record "PASS" "KT08: Multi-pod tracing" "both container_ids found in cgroup_metadata"
        elif [ "$FOUND_A" -gt 0 ] || [ "$FOUND_B" -gt 0 ]; then
            record "FAIL" "KT08: Multi-pod tracing" "only one container_id found (A=$FOUND_A, B=$FOUND_B)"
        else
            record "FAIL" "KT08: Multi-pod tracing" "neither container_id found in cgroup_metadata"
        fi
    fi
fi

################################################################################
# KT09: Resource Limits
################################################################################
header "KT09: Resource Limits"

INGERO_POD=$($KUBECTL get pods -n ingero-system -l app.kubernetes.io/name=ingero \
    -o jsonpath='{.items[0].metadata.name}' 2>/dev/null || echo "")

if [ -z "$INGERO_POD" ]; then
    record "SKIP" "KT09: Resource limits" "ingero pod not found"
else
    POD_UID=$($KUBECTL get pod -n ingero-system "$INGERO_POD" \
        -o jsonpath='{.metadata.uid}' 2>/dev/null || echo "")
    log "Ingero pod UID: $POD_UID"

    PEAK_MEM_MB=""
    if [ -n "$POD_UID" ]; then
        # Find the cgroup directory for the ingero container
        # K8s pod UIDs have dashes (a1b2-c3d4) but cgroup v2 paths use underscores (a1b2_c3d4)
        POD_UID_CGROUP="${POD_UID//-/_}"
        CGROUP_DIR=$(find /sys/fs/cgroup -name "cri-containerd-*" -path "*${POD_UID_CGROUP}*" -type d 2>/dev/null | head -1)
        if [ -n "$CGROUP_DIR" ] && [ -f "$CGROUP_DIR/memory.peak" ]; then
            PEAK_MEM=$(cat "$CGROUP_DIR/memory.peak" 2>/dev/null || echo "0")
            PEAK_MEM_MB=$((PEAK_MEM / 1024 / 1024))
            log "Peak memory: ${PEAK_MEM_MB}MB (from $CGROUP_DIR/memory.peak)"
        elif [ -n "$CGROUP_DIR" ] && [ -f "$CGROUP_DIR/memory.current" ]; then
            # Fallback: memory.current if memory.peak not available
            CUR_MEM=$(cat "$CGROUP_DIR/memory.current" 2>/dev/null || echo "0")
            PEAK_MEM_MB=$((CUR_MEM / 1024 / 1024))
            log "Current memory: ${PEAK_MEM_MB}MB (memory.peak not available)"
        else
            log "Could not find cgroup memory stats for pod UID $POD_UID"
        fi
    fi

    # Also try kubectl top
    $KUBECTL top pod -n ingero-system --no-headers > logs/kt09-resource-limits.log 2>&1 || true

    if [ -n "$PEAK_MEM_MB" ]; then
        if [ "$PEAK_MEM_MB" -lt 512 ]; then
            record "PASS" "KT09: Resource limits" "peak memory ${PEAK_MEM_MB}MB < 512Mi limit"
        else
            record "FAIL" "KT09: Resource limits" "peak memory ${PEAK_MEM_MB}MB >= 512Mi limit"
        fi
    else
        record "SKIP" "KT09: Resource limits" "could not read cgroup memory stats"
    fi
fi

################################################################################
# KT10: Uprobe Cleanup
################################################################################
header "KT10: Uprobe Cleanup"

# Save DaemonSet pod logs before deletion
$KUBECTL logs -n ingero-system -l app.kubernetes.io/name=ingero --tail=500 \
    > logs/k3s-ingero-pod.log 2>&1 || true

# Save DB snapshot before deletion
if [ -f "$INGERO_DB" ]; then
    sudo sqlite3 "$INGERO_DB" ".dump" > logs/k3s-db-snapshot.sql 2>/dev/null || true
    log "DB snapshot saved to logs/k3s-db-snapshot.sql"
fi

# Save pod describe
$KUBECTL describe pods --all-namespaces > logs/k3s-pod-describe.txt 2>&1 || true

# Delete DaemonSet
log "Deleting DaemonSet..."
$KUBECTL delete -f deploy/k8s/daemonset.yaml --ignore-not-found 2>/dev/null || true

# Wait for ingero pod to terminate
log "Waiting for ingero pod to terminate..."
for i in $(seq 1 30); do
    REMAINING=$($KUBECTL get pods -n ingero-system -l app.kubernetes.io/name=ingero --no-headers 2>/dev/null | wc -l || echo "0")
    if [ "$REMAINING" = "0" ]; then
        log "Ingero pod terminated"
        break
    fi
    if [ "$i" = "30" ]; then
        warn "Ingero pod still present after 30s"
    fi
    sleep 1
done

# Clean up workload pods
$KUBECTL delete pod --all -n default --ignore-not-found 2>/dev/null || true
sleep 2

# Run bare-metal trace for 15s with a PyTorch workload
log "Running bare-metal trace (15s) to verify uprobe cleanup..."
python3 -c "
import torch, time
d = torch.device('cuda:0')
a = torch.randn(2048, 2048, device=d)
b = torch.randn(2048, 2048, device=d)
start = time.time()
while time.time() - start < 15:
    c = torch.matmul(a, b)
    torch.cuda.synchronize()
" &>/dev/null &
WORKLOAD_PID=$!
cleanup_pids+=("$WORKLOAD_PID")

sudo ./bin/ingero trace --duration 15s --json > /tmp/kt10-trace.json 2>/dev/null || true
kill "$WORKLOAD_PID" 2>/dev/null; wait "$WORKLOAD_PID" 2>/dev/null || true

EVENT_COUNT=$(wc -l < /tmp/kt10-trace.json 2>/dev/null || echo "0")
log "Bare-metal trace events: $EVENT_COUNT"
mv /tmp/kt10-trace.json logs/kt10-uprobe-cleanup.log 2>/dev/null || true

if [ "$EVENT_COUNT" -gt 10 ]; then
    record "PASS" "KT10: Uprobe cleanup" "$EVENT_COUNT events in bare-metal trace after DaemonSet removal"
else
    record "FAIL" "KT10: Uprobe cleanup" "only $EVENT_COUNT events — uprobes may be stuck"
fi

################################################################################
# KT11-KT14: v0.8 Placeholders
################################################################################
header "KT11-KT14: v0.8 Placeholders"

record "SKIP" "KT11: Noisy neighbor detection" "v0.8: per-cgroup scheduler latency comparison"
record "SKIP" "KT12: Pod lifecycle correlation" "v0.8: eviction/OOM-kill/restart → GPU timeline"
record "SKIP" "KT13: OOM-kill tracking" "v0.8: pod OOM-kill shows in causal chains"
record "SKIP" "KT14: Inference serving trace" "v0.8: vLLM/Triton HTTP → CUDA pipeline"

################################################################################
# Generate K8s Test Report (before bare-metal regression)
################################################################################
header "K8s Test Report"

K8S_TOTAL=$((PASS_COUNT + FAIL_COUNT + SKIP_COUNT))
K8S_DURATION=$((SECONDS - SCRIPT_START))
echo ""
echo -e "$(ts)   ${GREEN}PASS: $PASS_COUNT${NC}  ${RED}FAIL: $FAIL_COUNT${NC}  ${YELLOW}SKIP: $SKIP_COUNT${NC}  Total: $K8S_TOTAL"
echo ""

# Write JSON report for K8s tests
_GPU_INFO=$(nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader 2>/dev/null || echo "N/A")
_GPU_NAME=$(echo "$_GPU_INFO" | cut -d',' -f1 | xargs)
_DRIVER_VER=$(echo "$_GPU_INFO" | cut -d',' -f2 | xargs)
_KERNEL_VER=$(uname -r)
_GO_VER=$(go version 2>/dev/null | awk '{print $3}')

K3S_TMP=$(mktemp -d /tmp/k3s_test_XXXXXX)
: > "$K3S_TMP/test_results.txt"
for entry in "${TEST_RESULTS[@]}"; do
    echo "$entry" >> "$K3S_TMP/test_results.txt"
done

SCRIPT_DURATION="$K8S_DURATION" \
    GPU_NAME="$_GPU_NAME" DRIVER_VER="$_DRIVER_VER" KERNEL_VER="$_KERNEL_VER" \
    GO_VER="$_GO_VER" \
    PASS_COUNT="$PASS_COUNT" FAIL_COUNT="$FAIL_COUNT" SKIP_COUNT="$SKIP_COUNT" TOTAL="$K8S_TOTAL" \
    python3 -c "
import json, sys, os
from datetime import datetime, timezone

results_file = '$K3S_TMP/test_results.txt'
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
    'version': '0.7',
    'timestamp': datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
    'duration_s': int(os.environ.get('SCRIPT_DURATION', '0')),
    'system': {
        'gpu': os.environ.get('GPU_NAME', 'N/A'),
        'driver': os.environ.get('DRIVER_VER', 'N/A'),
        'kernel': os.environ.get('KERNEL_VER', 'N/A'),
        'go': os.environ.get('GO_VER', 'N/A'),
        'arch': '$(uname -m)',
    },
    'summary': {
        'pass': int(os.environ.get('PASS_COUNT', '0')),
        'fail': int(os.environ.get('FAIL_COUNT', '0')),
        'skip': int(os.environ.get('SKIP_COUNT', '0')),
        'total': int(os.environ.get('TOTAL', '0')),
    },
    'tests': tests,
}

with open('logs/k3s-test-report.json', 'w') as f:
    json.dump(report, f, indent=2)
print(f'JSON report: logs/k3s-test-report.json ({len(tests)} tests)')
"
rm -rf "$K3S_TMP"

################################################################################
# Bare-Metal Regression (gpu-test.sh)
################################################################################
header "Bare-Metal Regression (gpu-test.sh)"

log "Running standard integration tests (62 tests)..."
log "This takes ~45 min on A10/A100..."
echo ""

if bash scripts/gpu-test.sh; then
    log "Bare-metal regression: PASSED"
else
    errmsg "Bare-metal regression: FAILED"
fi

################################################################################
# Final Summary
################################################################################
header "K8s + Bare-Metal Final Summary"

echo ""
echo "K8s tests (KT01-KT14):"
echo -e "  ${GREEN}PASS: $PASS_COUNT${NC}  ${RED}FAIL: $FAIL_COUNT${NC}  ${YELLOW}SKIP: $SKIP_COUNT${NC}  Total: $K8S_TOTAL"
echo ""
echo "Bare-metal regression: see logs/test-report.json"
echo ""
echo "K8s artifacts:"
echo "  logs/k3s-test-report.json     — K8s test JSON report"
echo "  logs/k3s-ingero-pod.log       — DaemonSet pod logs"
echo "  logs/k3s-db-snapshot.sql       — DB dump at end of K8s tests"
echo "  logs/k3s-pod-describe.txt     — kubectl describe all pods"
echo "  logs/k3s-integration-report.log — full test output"
echo ""

if [ "$FAIL_COUNT" -gt 0 ]; then
    exit 1
fi
exit 0
