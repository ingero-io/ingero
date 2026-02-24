#!/bin/bash
################################################################################
# Ingero WSL Development Environment Setup
#
# Sets up the local WSL (Ubuntu 22.04) environment for Ingero development.
# This installs everything needed for Go development, eBPF compilation,
# and unit testing — no GPU VM required for most development work.
#
# What it installs and verifies:
#   1. System packages (build-essential, clang-14, llvm-14)
#   2. eBPF compilation toolchain (libbpf-dev, linux-tools-generic for bpftool)
#   3. Go (bootstrap version — go.mod auto-downloads exact toolchain)
#   4. Go development tools (staticcheck, gofumpt, bpf2go, cobra-cli)
#   5. Generates vmlinux.h from WSL kernel BTF
#   6. Verifies the full pipeline: compile eBPF → generate Go bindings → build
#
# Prerequisites:
#   - WSL2 with Ubuntu 22.04 (wsl --install -d Ubuntu-22.04)
#   - Internet connection
#
# Usage:
#   chmod +x scripts/setup-wsl.sh
#   ./scripts/setup-wsl.sh
#
# KEY INSIGHT: WSL kernels (5.15+, 6.x) have BTF support, so you can compile
# eBPF programs and run bpf2go locally without a GPU VM. The GPU VM is only
# needed for: (a) attaching probes to NVIDIA drivers, (b) running CUDA
# workloads, (c) integration testing with actual GPUs.
################################################################################

set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

print_header() {
    echo ""
    echo -e "${BLUE}================================================================${NC}"
    echo -e "${BLUE}  $1${NC}"
    echo -e "${BLUE}================================================================${NC}"
}

print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

print_success() {
    echo -e "${GREEN}  ✓${NC} $1"
}

print_fail() {
    echo -e "${RED}  ✗${NC} $1"
}

# Track failures
FAILURES=0

# Check we're running in WSL
if [ ! -f /proc/version ] || ! grep -qi microsoft /proc/version; then
    print_error "This script must be run inside WSL (Windows Subsystem for Linux)."
    print_info "Install WSL: wsl --install -d Ubuntu-22.04"
    exit 1
fi

# Check not running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Do not run as root. Run as your normal user (sudo is used where needed)."
    exit 1
fi

print_header "Ingero WSL Development Environment Setup"
print_info "WSL Kernel: $(uname -r)"
print_info "OS: $(lsb_release -d 2>/dev/null | cut -f2 || cat /etc/os-release | grep PRETTY_NAME | cut -d= -f2 | tr -d '\"')"
print_info "User: $(whoami)"

# Detect repo root (script may be run from anywhere)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
print_info "Repo root: $REPO_ROOT"

################################################################################
# Step 1: System Packages
################################################################################
print_header "Step 1/6: System Packages"

print_info "Updating package lists..."
sudo apt-get update -qq

print_info "Installing build tools and eBPF toolchain..."
sudo apt-get install -y -qq \
    build-essential \
    make \
    clang-14 \
    llvm-14 \
    libbpf-dev \
    libelf-dev \
    linux-tools-generic \
    linux-tools-common \
    wget \
    curl \
    jq \
    git

# Verify each critical tool
echo ""
print_info "Verifying installations..."

if command -v clang-14 &>/dev/null; then
    print_success "clang-14: $(clang-14 --version | head -1)"
else
    print_fail "clang-14 not found"
    FAILURES=$((FAILURES + 1))
fi

if command -v llvm-config-14 &>/dev/null; then
    print_success "llvm-14: $(llvm-config-14 --version)"
else
    print_fail "llvm-14 not found"
    FAILURES=$((FAILURES + 1))
fi

if dpkg -s libbpf-dev &>/dev/null 2>&1; then
    print_success "libbpf-dev: installed (provides bpf_helpers.h, bpf_tracing.h)"
else
    print_fail "libbpf-dev not installed"
    FAILURES=$((FAILURES + 1))
fi

if dpkg -s libelf-dev &>/dev/null 2>&1; then
    print_success "libelf-dev: installed (ELF parsing for eBPF)"
else
    print_fail "libelf-dev not installed"
    FAILURES=$((FAILURES + 1))
fi

if command -v jq &>/dev/null; then
    print_success "jq: $(jq --version)"
else
    print_fail "jq not found (needed for TensorDock VM scripts)"
    FAILURES=$((FAILURES + 1))
fi

# bpftool: WSL has a version mismatch (ubuntu tools 5.15 vs kernel 6.x), but it works
BPFTOOL_BIN=""
if command -v bpftool &>/dev/null 2>&1; then
    BPFTOOL_BIN="bpftool"
else
    # Find the actual binary (WSL shim warns about version but the real binary exists)
    BPFTOOL_CANDIDATES=(
        /usr/lib/linux-tools/*/bpftool
    )
    for candidate in "${BPFTOOL_CANDIDATES[@]}"; do
        if [ -f "$candidate" ]; then
            BPFTOOL_BIN="$candidate"
            break
        fi
    done
fi

if [ -n "$BPFTOOL_BIN" ]; then
    BPFTOOL_VER=$($BPFTOOL_BIN version 2>/dev/null | head -1 || echo "found")
    print_success "bpftool: $BPFTOOL_VER"
    print_info "  Note: bpftool version may not match WSL kernel version — this is normal."
    print_info "  BTF dump still works correctly across kernel versions."
else
    print_fail "bpftool not found (needed for vmlinux.h generation)"
    print_info "  Try: sudo apt install linux-tools-generic"
    FAILURES=$((FAILURES + 1))
fi

# Verify BPF headers exist
if [ -f /usr/include/bpf/bpf_helpers.h ]; then
    BPF_HEADERS=$(ls /usr/include/bpf/*.h 2>/dev/null | wc -l)
    print_success "BPF headers: $BPF_HEADERS files in /usr/include/bpf/"
else
    print_fail "BPF headers not found at /usr/include/bpf/"
    FAILURES=$((FAILURES + 1))
fi

################################################################################
# Step 2: Go Installation
################################################################################
print_header "Step 2/6: Go Installation"

GO_BOOTSTRAP_VERSION="1.26.0"
GO_NEEDED=0

if command -v go &>/dev/null; then
    EXISTING_GO=$(go version | awk '{print $3}')
    print_success "Go already installed: $EXISTING_GO"
    print_info "  go.mod specifies the exact toolchain — auto-downloaded on first build."
else
    GO_NEEDED=1
fi

if [ "$GO_NEEDED" -eq 1 ]; then
    print_info "Installing Go $GO_BOOTSTRAP_VERSION (bootstrap — go.mod auto-downloads exact version)..."
    wget -q "https://go.dev/dl/go${GO_BOOTSTRAP_VERSION}.linux-amd64.tar.gz" -O /tmp/go.tar.gz
    sudo rm -rf /usr/local/go
    sudo tar -C /usr/local -xzf /tmp/go.tar.gz
    rm /tmp/go.tar.gz

    # Add to PATH permanently if not already
    if ! grep -q "/usr/local/go/bin" ~/.bashrc 2>/dev/null; then
        cat >> ~/.bashrc << 'GOEOF'

# Go language
export PATH=$PATH:/usr/local/go/bin:$HOME/go/bin
GOEOF
        print_info "Added Go to ~/.bashrc PATH"
    fi

    # Make available in current session
    export PATH=$PATH:/usr/local/go/bin:$HOME/go/bin

    if command -v go &>/dev/null; then
        print_success "Go installed: $(go version)"
    else
        print_fail "Go installation failed"
        FAILURES=$((FAILURES + 1))
    fi
fi

################################################################################
# Step 3: Go Development Tools
################################################################################
print_header "Step 3/6: Go Development Tools"

# Ensure PATH includes go/bin
export PATH=$PATH:/usr/local/go/bin:$HOME/go/bin

print_info "Installing required Go tools..."

# bpf2go (eBPF code generator — compiles C to Go bindings)
go install github.com/cilium/ebpf/cmd/bpf2go@latest 2>&1 | tail -1 || true
if command -v bpf2go &>/dev/null; then
    print_success "bpf2go: installed (eBPF C → Go code generator)"
else
    # bpf2go doesn't have --version, just check it exists
    if [ -f "$HOME/go/bin/bpf2go" ]; then
        print_success "bpf2go: $HOME/go/bin/bpf2go"
    else
        print_fail "bpf2go installation failed"
        FAILURES=$((FAILURES + 1))
    fi
fi

# staticcheck (linter — used by make lint)
go install honnef.co/go/tools/cmd/staticcheck@latest 2>&1 | tail -1 || true
if command -v staticcheck &>/dev/null; then
    print_success "staticcheck: $(staticcheck --version 2>&1 | head -1)"
else
    print_fail "staticcheck installation failed"
    FAILURES=$((FAILURES + 1))
fi

# gofumpt (formatter — project convention)
go install mvdan.cc/gofumpt@latest 2>&1 | tail -1 || true
if command -v gofumpt &>/dev/null; then
    print_success "gofumpt: installed"
else
    print_fail "gofumpt installation failed"
    FAILURES=$((FAILURES + 1))
fi

# cobra-cli (CLI scaffolding)
go install github.com/spf13/cobra-cli@latest 2>&1 | tail -1 || true
if command -v cobra-cli &>/dev/null; then
    print_success "cobra-cli: installed"
else
    print_fail "cobra-cli installation failed"
    FAILURES=$((FAILURES + 1))
fi

################################################################################
# Step 4: BTF & vmlinux.h
################################################################################
print_header "Step 4/6: BTF Support & vmlinux.h Generation"

if [ -f /sys/kernel/btf/vmlinux ]; then
    BTF_SIZE=$(ls -lh /sys/kernel/btf/vmlinux | awk '{print $5}')
    print_success "BTF available: /sys/kernel/btf/vmlinux ($BTF_SIZE)"

    # Create headers directory
    mkdir -p "$REPO_ROOT/bpf/headers"

    if [ -n "$BPFTOOL_BIN" ]; then
        if [ -f "$REPO_ROOT/bpf/headers/vmlinux.h" ]; then
            EXISTING_LINES=$(wc -l < "$REPO_ROOT/bpf/headers/vmlinux.h")
            print_info "vmlinux.h already exists ($EXISTING_LINES lines). Regenerating..."
        fi

        print_info "Generating vmlinux.h from WSL kernel BTF..."
        $BPFTOOL_BIN btf dump file /sys/kernel/btf/vmlinux format c > "$REPO_ROOT/bpf/headers/vmlinux.h"

        VMLINUX_LINES=$(wc -l < "$REPO_ROOT/bpf/headers/vmlinux.h")
        print_success "vmlinux.h generated: $VMLINUX_LINES lines"
    else
        print_warn "Skipping vmlinux.h generation (bpftool not found)"
        print_info "Install bpftool and run: make vmlinux"
    fi
else
    print_warn "BTF not available in this WSL kernel"
    print_info "vmlinux.h can be generated on the GPU VM instead (make vmlinux)"
fi

################################################################################
# Step 5: Download Go Dependencies
################################################################################
print_header "Step 5/6: Go Module Dependencies"

cd "$REPO_ROOT"
print_info "Downloading Go module dependencies..."
go mod download 2>&1 | tail -3 || true
print_success "Go modules downloaded"

################################################################################
# Step 6: Verification — Full Build Pipeline
################################################################################
print_header "Step 6/6: Full Pipeline Verification"

cd "$REPO_ROOT"
echo ""

# Test 1: Go build
print_info "Testing: go build ./..."
if go build ./... 2>&1; then
    print_success "Go build: all packages compile"
else
    print_fail "Go build failed"
    FAILURES=$((FAILURES + 1))
fi

# Test 2: Go tests
print_info "Testing: go test ./..."
TEST_OUTPUT=$(go test ./... 2>&1)
if [ $? -eq 0 ]; then
    TEST_COUNT=$(echo "$TEST_OUTPUT" | grep -c "^ok" || echo "0")
    print_success "Go tests: $TEST_COUNT package(s) passed"
else
    print_fail "Go tests failed"
    FAILURES=$((FAILURES + 1))
fi

# Test 3: eBPF compilation (bpf2go)
if [ -f "$REPO_ROOT/bpf/headers/vmlinux.h" ] && [ -f "$REPO_ROOT/bpf/cuda_trace.bpf.c" ]; then
    print_info "Testing: make generate (eBPF compilation via bpf2go)..."
    if go generate ./internal/ebpf/... 2>&1; then
        print_success "eBPF compilation: bpf2go succeeded"

        # Verify generated files
        if [ -f "$REPO_ROOT/internal/ebpf/cuda/cudatrace_bpfel.go" ]; then
            print_success "Generated Go bindings: cudatrace_bpfel.go"
        fi
        if [ -f "$REPO_ROOT/internal/ebpf/cuda/cudatrace_bpfel.o" ]; then
            OBJ_SIZE=$(ls -lh "$REPO_ROOT/internal/ebpf/cuda/cudatrace_bpfel.o" | awk '{print $5}')
            print_success "Compiled eBPF bytecode: cudatrace_bpfel.o ($OBJ_SIZE)"
        fi
    else
        print_fail "eBPF compilation failed"
        FAILURES=$((FAILURES + 1))
    fi
else
    print_warn "Skipping eBPF compilation test (vmlinux.h or cuda_trace.bpf.c missing)"
fi

# Test 4: Final binary build
print_info "Testing: make build (full binary with version injection)..."
if make build 2>&1; then
    if [ -f "$REPO_ROOT/bin/ingero" ]; then
        BINARY_SIZE=$(ls -lh "$REPO_ROOT/bin/ingero" | awk '{print $5}')
        BINARY_VERSION=$("$REPO_ROOT/bin/ingero" version 2>&1 || echo "unknown")
        print_success "Binary built: bin/ingero ($BINARY_SIZE)"
        print_success "Version: $BINARY_VERSION"
    fi
else
    print_fail "Binary build failed"
    FAILURES=$((FAILURES + 1))
fi

################################################################################
# Summary
################################################################################
print_header "Setup Summary"

echo ""
echo "  System Packages:"
echo "    clang-14:        $(clang-14 --version 2>/dev/null | head -1 | awk '{print $4}' || echo 'NOT FOUND')"
echo "    llvm-14:         $(llvm-config-14 --version 2>/dev/null || echo 'NOT FOUND')"
echo "    libbpf-dev:      $(dpkg -s libbpf-dev 2>/dev/null | grep Version | awk '{print $2}' || echo 'NOT FOUND')"
echo "    bpftool:         $([ -n "$BPFTOOL_BIN" ] && $BPFTOOL_BIN version 2>/dev/null | head -1 || echo 'NOT FOUND')"
echo "    BPF headers:     $(ls /usr/include/bpf/*.h 2>/dev/null | wc -l) files in /usr/include/bpf/"
echo ""
echo "  Go Toolchain:"
echo "    go:              $(go version 2>/dev/null | awk '{print $3}' || echo 'NOT FOUND')"
echo "    bpf2go:          $([ -f "$HOME/go/bin/bpf2go" ] && echo 'installed' || echo 'NOT FOUND')"
echo "    staticcheck:     $(staticcheck --version 2>/dev/null | head -1 || echo 'NOT FOUND')"
echo "    gofumpt:         $([ -f "$HOME/go/bin/gofumpt" ] && echo 'installed' || echo 'NOT FOUND')"
echo ""
echo "  eBPF Pipeline:"
echo "    BTF:             $([ -f /sys/kernel/btf/vmlinux ] && echo 'available' || echo 'NOT AVAILABLE')"
echo "    vmlinux.h:       $([ -f "$REPO_ROOT/bpf/headers/vmlinux.h" ] && echo "$(wc -l < "$REPO_ROOT/bpf/headers/vmlinux.h") lines" || echo 'NOT GENERATED')"
echo "    Generated .go:   $([ -f "$REPO_ROOT/internal/ebpf/cuda/cudatrace_bpfel.go" ] && echo 'present' || echo 'not yet')"
echo "    Generated .o:    $([ -f "$REPO_ROOT/internal/ebpf/cuda/cudatrace_bpfel.o" ] && echo 'present' || echo 'not yet')"
echo ""
echo "  WSL Kernel:"
echo "    Version:         $(uname -r)"
echo "    BTF size:        $(ls -lh /sys/kernel/btf/vmlinux 2>/dev/null | awk '{print $5}' || echo 'N/A')"
echo ""

if [ "$FAILURES" -gt 0 ]; then
    print_error "$FAILURES verification(s) failed. Review the output above."
    exit 1
else
    print_header "All Checks Passed!"
    echo ""
    print_info "Your WSL environment is ready for Ingero development."
    echo ""
    echo "  Next steps:"
    echo "    ${BLUE}source ~/.bashrc${NC}          # Refresh PATH (if Go was just installed)"
    echo "    ${BLUE}cd $REPO_ROOT${NC}"
    echo "    ${BLUE}make build${NC}                # Build the ingero binary"
    echo "    ${BLUE}./bin/ingero check${NC}           # System diagnostics (no GPU expected in WSL)"
    echo "    ${BLUE}./bin/ingero version${NC}          # Verify version injection"
    echo ""
    print_info "For GPU testing, deploy a remote VM:"
    echo "    ${BLUE}make gpu-deploy${NC}           # TensorDock VM (requires API token in .env)"
    echo ""
fi
