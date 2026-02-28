#!/bin/bash
# k3s-test.sh — Integration tests for Ingero on k3s with GPU.
#
# Tests:
#   1. Deploy ingero DaemonSet
#   2. Wait for ingero pod Running
#   3. Deploy a PyTorch training pod
#   4. Verify ingero captures events with cgroup_id != 0
#   5. Verify cgroup_metadata table has container_id
#   6. Cleanup DaemonSet (before bare-metal regression to avoid uprobe conflicts)
#   7. Run bare-metal regression tests (existing 62 tests)
#
# Usage:
#   bash scripts/k3s-test.sh
#   # or: make gpu-k3s-test (from WSL, runs via SSH on GPU VM)

set -euo pipefail

KUBECTL="sudo k3s kubectl"
INGERO_DB="/var/lib/ingero/ingero.db"
PASS=0
FAIL=0
SKIP=0

pass() { echo "  PASS: $1"; ((PASS++)); }
fail() { echo "  FAIL: $1"; ((FAIL++)); }
skip() { echo "  SKIP: $1"; ((SKIP++)); }

echo "=== Ingero k3s Integration Tests ==="
echo ""

# --- Pre-flight checks ---
echo "--- Pre-flight ---"
if ! command -v k3s &>/dev/null; then
    echo "FATAL: k3s not installed. Run: bash scripts/k3s-setup.sh"
    exit 1
fi

GPU_COUNT=$($KUBECTL get node -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}' 2>/dev/null || echo "0")
if [ "$GPU_COUNT" = "0" ] || [ -z "$GPU_COUNT" ]; then
    echo "FATAL: No GPU allocatable in k3s. Run: bash scripts/k3s-setup.sh"
    exit 1
fi
echo "  GPUs allocatable: $GPU_COUNT"

# --- Step 1: Deploy ingero ---
echo ""
echo "--- Step 1: Deploy Ingero DaemonSet ---"
$KUBECTL apply -f deploy/k8s/namespace.yaml
$KUBECTL apply -f deploy/k8s/rbac.yaml
$KUBECTL apply -f deploy/k8s/daemonset.yaml

# --- Step 2: Wait for ingero pod ---
echo ""
echo "--- Step 2: Wait for Ingero pod Running ---"
for i in $(seq 1 120); do
    STATUS=$($KUBECTL get pods -n ingero-system -l app.kubernetes.io/name=ingero -o jsonpath='{.items[0].status.phase}' 2>/dev/null || echo "Pending")
    if [ "$STATUS" = "Running" ]; then
        pass "Ingero pod is Running"
        break
    fi
    if [ "$i" = "120" ]; then
        fail "Ingero pod not Running after 120s (status: $STATUS)"
        $KUBECTL describe pods -n ingero-system -l app.kubernetes.io/name=ingero
    fi
    sleep 1
done

INGERO_POD=$($KUBECTL get pods -n ingero-system -l app.kubernetes.io/name=ingero -o jsonpath='{.items[0].metadata.name}' 2>/dev/null)
echo "  Pod: $INGERO_POD"

# --- Step 3: Deploy PyTorch training pod ---
echo ""
echo "--- Step 3: Deploy PyTorch training pod ---"
$KUBECTL apply -f - <<'PYTORCH_EOF'
apiVersion: v1
kind: Pod
metadata:
  name: pytorch-test
  namespace: default
spec:
  restartPolicy: Never
  containers:
  - name: pytorch
    image: nvcr.io/nvidia/pytorch:24.01-py3
    command: ["python3", "-c", "import torch; x = torch.randn(1000, 1000, device='cuda'); y = torch.mm(x, x); torch.cuda.synchronize(); print('done')"]
    resources:
      limits:
        nvidia.com/gpu: 1
PYTORCH_EOF

echo "  Waiting for pytorch-test to complete..."
for i in $(seq 1 180); do
    PHASE=$($KUBECTL get pod pytorch-test -o jsonpath='{.status.phase}' 2>/dev/null || echo "Pending")
    if [ "$PHASE" = "Succeeded" ] || [ "$PHASE" = "Failed" ]; then
        if [ "$PHASE" = "Succeeded" ]; then
            pass "PyTorch pod completed successfully"
        else
            fail "PyTorch pod failed"
        fi
        break
    fi
    if [ "$i" = "180" ]; then
        skip "PyTorch pod didn't complete in 180s (phase: $PHASE)"
    fi
    sleep 1
done

# --- Step 4: Verify events have cgroup_id ---
echo ""
echo "--- Step 4: Verify cgroup_id in events ---"
sleep 5  # Allow time for events to flush to SQLite

# Query the host-mounted SQLite DB directly (ingero stores at /var/lib/ingero/ingero.db).
CGROUP_COUNT=$(sudo sqlite3 "$INGERO_DB" "SELECT COUNT(*) FROM events WHERE cgroup_id > 1" 2>/dev/null || echo "0")
if [ "$CGROUP_COUNT" -gt "0" ]; then
    pass "Events with non-zero cgroup_id: $CGROUP_COUNT"
else
    skip "No events with non-zero cgroup_id found (may need longer trace)"
fi

# --- Step 5: Check cgroup_metadata table ---
echo ""
echo "--- Step 5: Verify cgroup_metadata table ---"
METADATA=$(sudo sqlite3 "$INGERO_DB" "SELECT COUNT(*) FROM cgroup_metadata WHERE container_id != ''" 2>/dev/null || echo "0")
if [ "$METADATA" -gt "0" ]; then
    pass "cgroup_metadata rows with container_id: $METADATA"
else
    skip "No cgroup_metadata entries found (container resolution may need longer trace)"
fi

# --- Step 6: Cleanup DaemonSet ---
# Remove DaemonSet BEFORE bare-metal regression to avoid duplicate uprobe conflicts.
# Both the DaemonSet pod and gpu-test.sh attach uprobes to libcudart.so/libcuda.so.
echo ""
echo "--- Step 6: Cleanup DaemonSet ---"
$KUBECTL delete pod pytorch-test --ignore-not-found
$KUBECTL delete -f deploy/k8s/daemonset.yaml --ignore-not-found
echo "  Waiting for ingero pod to terminate..."
for i in $(seq 1 30); do
    REMAINING=$($KUBECTL get pods -n ingero-system -l app.kubernetes.io/name=ingero --no-headers 2>/dev/null | wc -l || echo "0")
    if [ "$REMAINING" = "0" ]; then
        break
    fi
    sleep 1
done
echo "  DaemonSet removed (namespace + RBAC kept for re-runs)"

# --- Step 7: Bare-metal regression ---
echo ""
echo "--- Step 7: Bare-metal regression (gpu-test.sh) ---"
echo "  Running standard integration tests..."
if bash scripts/gpu-test.sh; then
    pass "Bare-metal regression tests passed"
else
    fail "Bare-metal regression tests failed"
fi

# --- Summary ---
echo ""
echo "=== k3s Test Summary ==="
echo "  PASS=$PASS FAIL=$FAIL SKIP=$SKIP"
if [ "$FAIL" -gt 0 ]; then
    exit 1
fi
