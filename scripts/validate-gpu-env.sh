#!/bin/bash
################################################################################
# Ingero GPU Environment Validator
#
# Validates that all tools and dependencies required for Ingero GPU development
# and demo are correctly installed. Run after setup-gpu-instance.sh or on any
# existing VM to verify readiness.
#
# Usage:
#   bash scripts/validate-gpu-env.sh           # validate environment
#   bash scripts/validate-gpu-env.sh --build   # also build and test ingero
#
# Exit codes:
#   0 = all checks pass
#   1 = one or more checks failed
################################################################################

set -o pipefail

# Colors
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m'

PASS=0
FAIL=0
WARN=0
FAILURES=()

check() {
    local name="$1"
    local cmd="$2"
    local detail

    if detail=$(eval "$cmd" 2>&1); then
        echo -e "  ${GREEN}[✓]${NC} ${name}"
        if [[ -n "$detail" ]]; then
            echo -e "      ${detail}"
        fi
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}[✗]${NC} ${name}"
        if [[ -n "$detail" ]]; then
            echo -e "      ${detail}"
        fi
        FAIL=$((FAIL + 1))
        FAILURES+=("$name")
    fi
}

warn_check() {
    local name="$1"
    local cmd="$2"
    local detail

    if detail=$(eval "$cmd" 2>&1); then
        echo -e "  ${GREEN}[✓]${NC} ${name}"
        if [[ -n "$detail" ]]; then
            echo -e "      ${detail}"
        fi
        PASS=$((PASS + 1))
    else
        echo -e "  ${YELLOW}[~]${NC} ${name} (optional)"
        WARN=$((WARN + 1))
    fi
}

echo ""
echo -e "${BLUE}================================================================${NC}"
echo -e "${BLUE}  Ingero GPU Environment Validator${NC}"
echo -e "${BLUE}================================================================${NC}"
echo ""

# ---- Kernel & BTF ----
echo -e "${BLUE}Kernel & BTF${NC}"
check "Kernel 5.15+" \
    "uname -r | awk -F. '{if (\$1>5 || (\$1==5 && \$2>=15)) print \$0; else exit 1}'"
check "BTF support (/sys/kernel/btf/vmlinux)" \
    "[ -f /sys/kernel/btf/vmlinux ] && ls -lh /sys/kernel/btf/vmlinux | awk '{print \$5}'"
echo ""

# ---- NVIDIA ----
echo -e "${BLUE}NVIDIA GPU${NC}"
check "NVIDIA driver (nvidia-smi)" \
    "nvidia-smi --query-gpu=driver_version,name,memory.total --format=csv,noheader 2>/dev/null | head -1"
check "libcudart.so findable" \
    "python3 -c 'import nvidia.cuda_runtime, os; p=os.path.join(nvidia.cuda_runtime.__path__[0], \"lib\"); import glob; libs=glob.glob(os.path.join(p, \"libcudart.so*\")); print(libs[0]) if libs else exit(1)' 2>/dev/null || ldconfig -p 2>/dev/null | grep libcudart | head -1 | awk '{print \$NF}'"
echo ""

# ---- Build Tools ----
echo -e "${BLUE}Build Tools${NC}"
check "clang-14" \
    "clang-14 --version 2>/dev/null | head -1"
check "bpftool" \
    "bpftool version 2>/dev/null | head -1"
check "Go 1.22+" \
    "go version 2>/dev/null | awk '{print \$3}'"
check "make" \
    "make --version 2>/dev/null | head -1"
check "libbpf-dev headers" \
    "[ -f /usr/include/bpf/bpf_helpers.h ] && echo '/usr/include/bpf/bpf_helpers.h present'"
echo ""

# ---- Python & PyTorch ----
echo -e "${BLUE}Python & PyTorch${NC}"
check "python3" \
    "python3 --version 2>&1"
check "pip3" \
    "pip3 --version 2>&1 | head -1"
check "PyTorch importable" \
    "python3 -c 'import torch; print(f\"PyTorch {torch.__version__}\")'"
check "PyTorch CUDA available" \
    "python3 -c 'import torch; assert torch.cuda.is_available(), \"no CUDA\"; print(f\"CUDA {torch.version.cuda}, GPU: {torch.cuda.get_device_name(0)}\")'"
echo ""

# ---- Optional Tools ----
echo -e "${BLUE}Optional Tools${NC}"
warn_check "bpftrace" "bpftrace --version 2>&1 | head -1"
warn_check "nvcc (CUDA toolkit)" "nvcc --version 2>/dev/null | grep release"
warn_check "git" "git --version 2>/dev/null"
echo ""

# ---- Build & Test (if --build flag and repo is present) ----
if [[ "$1" == "--build" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

    if [[ -f "$PROJECT_DIR/Makefile" ]]; then
        echo -e "${BLUE}Build & Test${NC}"
        check "make vmlinux" \
            "cd '$PROJECT_DIR' && make vmlinux 2>&1 | tail -1"
        check "make generate" \
            "cd '$PROJECT_DIR' && make generate 2>&1 | tail -1"
        check "make build" \
            "cd '$PROJECT_DIR' && make build 2>&1 && echo 'Binary: $(ls -lh '$PROJECT_DIR'/bin/ingero | awk \"{print \\$5}\")'"
        check "make test" \
            "cd '$PROJECT_DIR' && make test 2>&1 | tail -3"
        warn_check "ingero check" \
            "cd '$PROJECT_DIR' && sudo ./bin/ingero check 2>&1 | tail -5"

        # Probe attachment smoke test (requires CUDA workload)
        if python3 -c 'import torch; assert torch.cuda.is_available()' 2>/dev/null; then
            check "probe attachment (3s watch)" \
                "cd '$PROJECT_DIR' && python3 -c 'import torch,time; d=torch.device(\"cuda:0\"); a=torch.randn(512,512,device=d); start=time.time()
while time.time()-start<5: torch.matmul(a,a); torch.cuda.synchronize()' &>/dev/null & WL=\$!; sleep 1; sudo ./bin/ingero trace --json --duration 3s > /tmp/smoke.json 2>/dev/null; wait \$WL 2>/dev/null; COUNT=\$(python3 -c \"import json; events=[json.loads(l) for l in open('/tmp/smoke.json') if l.strip()]; print(len(events))\" 2>/dev/null || echo 0); [ \"\$COUNT\" -gt 0 ] && echo \"\$COUNT events in 3s\" || (echo '0 events — probes not attaching; run make generate && make build' && exit 1)"
        else
            echo -e "  ${YELLOW}[~]${NC} probe attachment (skipped — no CUDA)"
            WARN=$((WARN + 1))
        fi
        echo ""
    else
        echo -e "${YELLOW}Skipping build checks — Makefile not found at ${PROJECT_DIR}${NC}"
        echo ""
    fi
fi

# ---- Summary ----
echo -e "${BLUE}================================================================${NC}"
echo -e "  Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}, ${YELLOW}${WARN} warnings${NC}"
echo -e "${BLUE}================================================================${NC}"

if [[ $FAIL -gt 0 ]]; then
    echo ""
    echo -e "${RED}Failed checks:${NC}"
    for f in "${FAILURES[@]}"; do
        echo -e "  ${RED}✗${NC} $f"
    done
    echo ""
    echo -e "Run ${BLUE}bash scripts/setup-gpu-instance.sh${NC} to fix missing dependencies."
    exit 1
fi

echo ""
echo -e "${GREEN}All required checks passed. Environment is ready for Ingero.${NC}"
exit 0
