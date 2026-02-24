#!/bin/bash
################################################################################
# Ingero GPU Instance Setup Script
#
# Run this script on your GPU instance (Lambda Labs, RunPod, etc.) to set up
# the complete eBPF development environment for NVIDIA GPU tracing.
#
# Prerequisites:
#   - Ubuntu 22.04 LTS
#   - NVIDIA GPU with driver installed
#   - Root/sudo access
#   - Internet connection
#
# Usage:
#   chmod +x setup-gpu-instance.sh
#   ./setup-gpu-instance.sh
#
# What it does:
#   1. Updates system packages
#   2. Installs eBPF toolchain (clang-14, llvm-14, libbpf, bpftool)
#   3. Verifies BTF support and generates vmlinux.h
#   4. Documents and pins NVIDIA driver version
#   5. Installs Go 1.26
#   6. Builds libbpf from source
#   7. Sets up workspace structure
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
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

# Check if running as root
if [ "$EUID" -eq 0 ]; then
    print_error "Do not run this script as root. Run as normal user with sudo access."
    exit 1
fi

# Detect OS
if [ ! -f /etc/os-release ]; then
    print_error "Cannot detect OS. This script requires Ubuntu 22.04 LTS."
    exit 1
fi

source /etc/os-release
if [[ "$ID" != "ubuntu" ]]; then
    print_error "This script is for Ubuntu. Detected: $ID"
    exit 1
fi

print_header "Ingero GPU Instance Setup"
print_info "OS: $PRETTY_NAME"
print_info "Kernel: $(uname -r)"
print_info "User: $(whoami)"

# Update system
print_header "Step 1/9: System Update"
print_info "Updating package lists..."
sudo apt update -qq

print_info "Skipping apt upgrade (can drop SSH on cloud VMs). Installing packages directly."

# Hold broken grub package on TensorDock VMs to prevent dpkg errors during installs
sudo apt-mark hold grub-efi-amd64-signed 2>/dev/null || true

# Install eBPF toolchain
print_header "Step 2/9: Installing eBPF Toolchain"
print_info "Installing clang-14, llvm-14, libbpf, bpftool..."

sudo apt install -y -qq \
    build-essential \
    clang-14 \
    llvm-14 \
    libbpf-dev \
    libelf-dev \
    libelf1 \
    zlib1g-dev \
    linux-tools-common \
    linux-tools-generic \
    make \
    pkg-config \
    git \
    wget \
    curl \
    vim \
    jq \
    sqlite3 \
    stress-ng

# bpftool: prefer kernel-specific package (avoids BTF version mismatch where an
# old bpftool from linux-tools-generic can't parse a newer kernel's BTF data).
print_info "Installing linux-tools-$(uname -r) for kernel-matching bpftool..."
sudo apt install -y -qq "linux-tools-$(uname -r)" 2>/dev/null || true

# Resolve best bpftool: kernel-matching > /usr/sbin/bpftool > any linux-tools glob
resolve_bpftool() {
    local bt
    bt=$(ls "/usr/lib/linux-tools/$(uname -r)/bpftool" 2>/dev/null) && echo "$bt" && return
    bt=$(which bpftool 2>/dev/null) && echo "$bt" && return
    bt=$(ls /usr/lib/linux-tools/*/bpftool 2>/dev/null | head -1) && echo "$bt" && return
    echo ""
}
BPFTOOL_BIN=$(resolve_bpftool)
if [ -n "$BPFTOOL_BIN" ]; then
    print_success "bpftool found: $BPFTOOL_BIN"
else
    print_warn "bpftool not found. Will try to continue without it."
fi

# Verify clang
if ! command -v clang-14 &> /dev/null; then
    print_error "clang-14 installation failed"
    exit 1
fi

CLANG_VERSION=$(clang-14 --version | head -1)
print_success "Clang installed: $CLANG_VERSION"

# Install debugging tools
print_header "Step 3/9: Installing eBPF Debugging Tools"
sudo apt install -y -qq bpftrace trace-cmd

BPFTRACE_VERSION=$(bpftrace --version 2>&1 | head -1 || echo "unknown")
print_success "bpftrace installed: $BPFTRACE_VERSION"

# Check BTF support (CRITICAL for CO-RE)
print_header "Step 4/9: Verifying BTF Support"

if [ -f /sys/kernel/btf/vmlinux ]; then
    BTF_SIZE=$(ls -lh /sys/kernel/btf/vmlinux | awk '{print $5}')
    print_success "BTF support detected: vmlinux ($BTF_SIZE)"

    # Create workspace and bpf headers directory
    print_info "Creating workspace structure..."
    mkdir -p ~/workspace/ingero/bpf/headers

    # Generate vmlinux.h using the best available bpftool
    print_info "Generating vmlinux.h from BTF (this takes ~30 seconds)..."
    if [ -n "$BPFTOOL_BIN" ]; then
        "$BPFTOOL_BIN" btf dump file /sys/kernel/btf/vmlinux format c > ~/workspace/ingero/bpf/headers/vmlinux.h
    else
        print_warn "Skipping vmlinux.h generation (no bpftool). Run 'make vmlinux' after installing bpftool."
    fi

    if [ -f ~/workspace/ingero/bpf/headers/vmlinux.h ]; then
        VMLINUX_LINES=$(wc -l ~/workspace/ingero/bpf/headers/vmlinux.h | awk '{print $1}')
        VMLINUX_SIZE=$(ls -lh ~/workspace/ingero/bpf/headers/vmlinux.h | awk '{print $5}')
        print_success "vmlinux.h generated: $VMLINUX_LINES lines ($VMLINUX_SIZE)"
    fi
else
    print_warn "BTF not available at /sys/kernel/btf/vmlinux"
    print_info "Checking kernel configuration..."

    if [ -f /boot/config-$(uname -r) ]; then
        if grep -q "CONFIG_DEBUG_INFO_BTF=y" /boot/config-$(uname -r); then
            print_error "BTF is configured but vmlinux file is missing. Kernel issue?"
        else
            print_error "Kernel not compiled with CONFIG_DEBUG_INFO_BTF=y"
        fi
    fi

    print_info "Trying kernel headers as fallback..."
    sudo apt install -y linux-headers-$(uname -r)

    if [ $? -eq 0 ]; then
        print_success "Kernel headers installed"
    else
        print_error "Failed to install kernel headers. eBPF development will be limited."
        exit 1
    fi
fi

# Verify and document NVIDIA environment
print_header "Step 5/9: NVIDIA Environment"

if command -v nvidia-smi &> /dev/null && nvidia-smi --query-gpu=name --format=csv,noheader &> /dev/null; then
    print_success "NVIDIA driver detected and working"

    DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | tr -d '[:space:]' | head -1)
    CUDA_VERSION=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader | tr -d '[:space:]' | head -1)
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
    GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)

    print_info "GPU: $GPU_NAME ($GPU_MEMORY)"
    print_info "NVIDIA Driver: $DRIVER_VERSION"
    print_info "CUDA Version: $CUDA_VERSION"

    # Pin driver version to prevent auto-updates
    print_info "Pinning NVIDIA driver to prevent automatic updates..."
    sudo apt-mark hold nvidia-driver-* 2>/dev/null || true
    print_success "Driver version pinned to $DRIVER_VERSION"
elif command -v nvidia-smi &> /dev/null; then
    # nvidia-smi binary exists but can't communicate with driver.
    # Common cause 1: module not loaded (TensorDock after kernel updates).
    # Common cause 2: VM image has driver built for a different kernel version.
    print_warn "nvidia-smi found but driver not communicating — trying modprobe first..."
    sudo modprobe nvidia 2>/dev/null || true

    if nvidia-smi --query-gpu=name --format=csv,noheader &> /dev/null; then
        print_success "modprobe nvidia fixed the issue"
    else
        print_warn "modprobe didn't help — kernel module mismatch? Reinstalling driver..."
        sudo apt-get update -qq
        sudo apt-get install -y nvidia-dkms-550 2>&1 | tail -5
        sudo apt-get install -y nvidia-driver-550 2>&1 | tail -5
    fi

    if nvidia-smi --query-gpu=name --format=csv,noheader &> /dev/null; then
        DRIVER_VERSION=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | tr -d '[:space:]' | head -1)
        CUDA_VERSION=$(nvidia-smi --query-gpu=cuda_version --format=csv,noheader | tr -d '[:space:]' | head -1)
        GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
        GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader | head -1)

        print_success "NVIDIA driver fixed: $GPU_NAME ($GPU_MEMORY)"
        print_info "NVIDIA Driver: $DRIVER_VERSION"
        print_info "CUDA Version: $CUDA_VERSION"
    else
        print_error "NVIDIA driver still broken after install. Try rebooting."
        exit 1
    fi

    # Save environment info
    mkdir -p ~/workspace
    cat > ~/workspace/nvidia_env.log << EOF
NVIDIA Environment - Documented $(date)
========================================
GPU: $GPU_NAME
Memory: $GPU_MEMORY
NVIDIA Driver: $DRIVER_VERSION
CUDA Version: $CUDA_VERSION
Kernel: $(uname -r)
OS: $PRETTY_NAME
Hostname: $(hostname)
========================================

IMPORTANT: eBPF kprobes depend on specific NVIDIA driver
function signatures. If you upgrade the driver, you may need
to adjust probe points in bpf/*.bpf.c files.

To find NVIDIA symbols:
  sudo cat /proc/kallsyms | grep nvidia | grep -i ioctl

To prevent driver updates:
  sudo apt-mark hold nvidia-driver-$DRIVER_VERSION
EOF
    print_success "Environment info saved to ~/workspace/nvidia_env.log"
else
    print_error "NVIDIA driver not found! This should be a GPU instance."
    print_error "Please verify: nvidia-smi"
    exit 1
fi

# Verify CUDA toolkit
if command -v nvcc &> /dev/null; then
    NVCC_VERSION=$(nvcc --version | grep release | awk '{print $5}' | tr -d ',')
    print_success "CUDA Toolkit: $NVCC_VERSION"
else
    print_warn "CUDA Toolkit (nvcc) not found in PATH"
    print_info "This is OK if CUDA is installed elsewhere (e.g., /usr/local/cuda)"
fi

# Install Go (bootstrap version — go.mod toolchain directive auto-downloads the exact version needed)
GO_BOOTSTRAP_VERSION="1.26.0"
print_header "Step 6/9: Installing Go ${GO_BOOTSTRAP_VERSION}"

if command -v go &> /dev/null; then
    EXISTING_GO=$(go version | awk '{print $3}')
    print_info "Go already installed: $EXISTING_GO"

    if [[ "$EXISTING_GO" == "go1.25"* ]] || [[ "$EXISTING_GO" == "go1.26"* ]] || [[ "$EXISTING_GO" > "go1.25" ]]; then
        print_success "Go version is sufficient (go.mod toolchain will auto-download exact version)"
    else
        print_warn "Go version is older than 1.25, upgrading..."
        GO_INSTALL_NEEDED=1
    fi
else
    GO_INSTALL_NEEDED=1
fi

if [ ! -z "$GO_INSTALL_NEEDED" ]; then
    print_info "Downloading Go ${GO_BOOTSTRAP_VERSION}..."
    cd /tmp
    wget -q "https://go.dev/dl/go${GO_BOOTSTRAP_VERSION}.linux-amd64.tar.gz"

    print_info "Installing Go..."
    sudo rm -rf /usr/local/go
    sudo tar -C /usr/local -xzf "go${GO_BOOTSTRAP_VERSION}.linux-amd64.tar.gz"
    rm "go${GO_BOOTSTRAP_VERSION}.linux-amd64.tar.gz"

    # Add Go to PATH for interactive shells (.bashrc)
    if ! grep -q "/usr/local/go/bin" ~/.bashrc; then
        cat >> ~/.bashrc << 'EOF'

# Go language + clean PATH (avoid WSL/Windows PATH contamination)
export PATH=/usr/local/go/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/go/bin:$HOME/.local/bin:$PATH
export GOCACHE=$HOME/workspace/.cache/go-build
export GOPATH=$HOME/workspace/.go
EOF
        print_info "Go PATH added to ~/.bashrc"
    fi

    # Add Go to PATH for NON-INTERACTIVE sessions (ssh user@host 'make build').
    # /etc/profile.d/ is sourced by login shells; /etc/environment is read by
    # PAM for all sessions including non-interactive SSH commands.
    print_info "Adding Go to system-wide PATH (/etc/profile.d + /etc/environment)..."
    sudo tee /etc/profile.d/go-path.sh > /dev/null << 'EOF'
# Installed by Ingero setup-gpu-instance.sh
export PATH=/usr/local/go/bin:$PATH
EOF
    sudo chmod +x /etc/profile.d/go-path.sh

    # /etc/environment is a simple KEY=VALUE file read by PAM — it applies to
    # ALL sessions, including non-interactive SSH commands like:
    #   ssh user@host 'go version'
    # This is the definitive fix for "go: not found" in remote make targets.
    if [ -f /etc/environment ]; then
        if ! grep -q "/usr/local/go/bin" /etc/environment; then
            if grep -q "^PATH=" /etc/environment; then
                # Prepend /usr/local/go/bin to existing PATH (handles both quoted and unquoted)
                sudo sed -i 's|^PATH="\(.*\)"|PATH="/usr/local/go/bin:\1"|; s|^PATH=\([^"]\)|PATH=/usr/local/go/bin:\1|' /etc/environment
            else
                echo 'PATH="/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"' | sudo tee -a /etc/environment > /dev/null
            fi
            print_info "Go added to /etc/environment (non-interactive SSH)"
        fi
    else
        echo 'PATH="/usr/local/go/bin:/usr/local/sbin:/usr/local/bin:/usr/sbin:/usr/bin:/sbin:/bin"' | sudo tee /etc/environment > /dev/null
        print_info "Created /etc/environment with Go PATH"
    fi

    # Source for current session
    export PATH=/usr/local/go/bin:$PATH:$HOME/go/bin
    export GOCACHE=$HOME/workspace/.cache/go-build
    export GOPATH=$HOME/workspace/.go

    GO_VERSION=$(go version)
    print_success "Go installed: $GO_VERSION"
else
    print_success "Go installation skipped (already installed)"
fi

# Install libbpf from source
print_header "Step 7/9: Installing libbpf from Source"

if [ -d ~/workspace/libbpf ]; then
    print_info "libbpf directory exists, updating..."
    cd ~/workspace/libbpf
    git pull -q origin master
else
    print_info "Cloning libbpf from GitHub..."
    cd ~/workspace
    git clone -q --depth 1 https://github.com/libbpf/libbpf.git
fi

print_info "Building libbpf (this takes ~1 minute)..."
cd ~/workspace/libbpf/src
make -j$(nproc) > /dev/null 2>&1

print_info "Installing libbpf..."
sudo make install > /dev/null 2>&1
sudo ldconfig

LIBBPF_VERSION=$(pkg-config --modversion libbpf 2>/dev/null || echo "unknown")
print_success "libbpf installed: version $LIBBPF_VERSION"

# Setup workspace structure
print_header "Step 8/9: Setting up Workspace"

cd ~
mkdir -p workspace/{.cache,.go,data}
print_success "Workspace structure created:"
print_info "  ~/workspace/ingero/     - Git repository (clone separately)"
print_info "  ~/workspace/.cache/     - Build cache"
print_info "  ~/workspace/.go/        - Go packages"
print_info "  ~/workspace/data/       - Test data and traces"
print_info "  ~/workspace/libbpf/     - libbpf source"

# Install Python packages for test workloads
print_header "Step 9/9: Installing Test Workload Dependencies"

# Ensure pip3 is available
if ! command -v pip3 &>/dev/null; then
    print_info "Installing pip3..."
    sudo apt install -y -qq python3-pip
fi

if command -v pip3 &>/dev/null; then
    # Install PyTorch + torchvision + numpy with CUDA 12.1 wheels
    print_info "Installing PyTorch + torchvision + numpy with CUDA 12.1 support..."
    pip3 install --quiet torch torchvision numpy --index-url https://download.pytorch.org/whl/cu121 2>&1 | tail -5
    print_success "PyTorch + torchvision + numpy installed"

    # Install ML dependencies for test workloads
    print_info "Installing ML dependencies (transformers, datasets, diffusers, accelerate)..."
    pip3 install --quiet transformers datasets diffusers accelerate 2>&1 | tail -3
    print_success "ML dependencies installed"

    # Install test workload requirements if repo is present
    if [ -f ~/workspace/ingero/tests/workloads/requirements.txt ]; then
        print_info "Installing test workload requirements..."
        pip3 install --quiet -r ~/workspace/ingero/tests/workloads/requirements.txt 2>&1 | tail -3
        print_success "Test workload dependencies installed"
    fi
else
    print_warn "pip3 not found after install attempt. Install manually: sudo apt install python3-pip"
fi

# Install Python debug symbols for DWARF-based CPython offset discovery.
# This enables Ingero to extract real struct offsets from libpython rather than
# relying on hardcoded upstream offsets (which differ on distro-patched builds).
print_info "Installing libpython debug symbols for DWARF offset discovery..."
PY_MINOR=$(python3 -c "import sys; print(sys.version_info.minor)" 2>/dev/null || echo "")
if [ -n "$PY_MINOR" ]; then
    # Try dbgsym first (from ddebs.ubuntu.com), then fallback to -dbg.
    if sudo apt install -y -qq "libpython3.${PY_MINOR}-dbg" 2>/dev/null; then
        print_success "libpython3.${PY_MINOR}-dbg installed (DWARF offsets available)"
    else
        print_info "Trying dbgsym from ddebs repository..."
        # Enable ddebs (Ubuntu debug symbol packages) if not already.
        if ! apt-cache policy 2>/dev/null | grep -q ddebs; then
            echo "deb http://ddebs.ubuntu.com $(lsb_release -cs) main restricted universe multiverse" | sudo tee /etc/apt/sources.list.d/ddebs.list >/dev/null 2>&1
            echo "deb http://ddebs.ubuntu.com $(lsb_release -cs)-updates main restricted universe multiverse" | sudo tee -a /etc/apt/sources.list.d/ddebs.list >/dev/null 2>&1
            sudo apt-key adv --keyserver keyserver.ubuntu.com --recv-keys F2EDC64DC5AEE1F6B9C621F0C8CAB6595FDFF622 2>/dev/null || true
            sudo apt update -qq 2>/dev/null
        fi
        if sudo apt install -y -qq "libpython3.${PY_MINOR}-dbgsym" 2>/dev/null; then
            print_success "libpython3.${PY_MINOR}-dbgsym installed (DWARF offsets available)"
        else
            print_warn "Could not install libpython debug symbols. T11 will use hardcoded offsets."
        fi
    fi
else
    print_warn "Python3 not found, skipping debug symbol install."
fi

# Verify torch + CUDA
if command -v python3 &>/dev/null; then
    TORCH_CUDA=$(python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch not importable")
    print_info "$TORCH_CUDA"
fi

# Validation
print_header "Validating Environment"

PASS=0
FAIL=0

check_tool() {
    local name="$1"
    local cmd="$2"
    if eval "$cmd" &>/dev/null; then
        echo -e "  ${GREEN}[PASS]${NC} $name"
        PASS=$((PASS + 1))
    else
        echo -e "  ${RED}[FAIL]${NC} $name"
        FAIL=$((FAIL + 1))
    fi
}

check_tool "Kernel BTF"           "[ -f /sys/kernel/btf/vmlinux ]"
check_tool "NVIDIA driver"        "command -v nvidia-smi"
check_tool "clang-14"             "command -v clang-14"
check_tool "bpftool"              "command -v bpftool"
check_tool "Go 1.22+"             "go version"
check_tool "sqlite3"              "command -v sqlite3"
check_tool "stress-ng"            "command -v stress-ng"
check_tool "libbpf-dev headers"   "[ -f /usr/include/bpf/bpf_helpers.h ]"
check_tool "python3"              "command -v python3"
check_tool "PyTorch importable"   "python3 -c 'import torch'"
check_tool "PyTorch CUDA"         "python3 -c 'import torch; assert torch.cuda.is_available()'"
check_tool "libcudart.so"         "python3 -c 'import nvidia.cuda_runtime'"

echo ""
echo -e "  Results: ${GREEN}${PASS} passed${NC}, ${RED}${FAIL} failed${NC}"

if [ $FAIL -gt 0 ]; then
    print_warn "Some checks failed. Run scripts/validate-gpu-env.sh for details."
fi

# Final summary
print_header "Setup Complete!"

echo ""
print_success "Environment Summary:"
echo "  OS:           $(lsb_release -d | cut -f2)"
echo "  Kernel:       $(uname -r)"
echo "  Clang:        $(clang-14 --version | head -1 | awk '{print $1,$4}')"
echo "  LLVM:         $(llvm-config-14 --version 2>/dev/null || echo 'installed')"
echo "  libbpf:       $LIBBPF_VERSION"
echo "  Go:           $(go version 2>/dev/null | awk '{print $3,$4}' || echo 'run: source ~/.bashrc')"
echo "  bpftool:      $([ -n "$BPFTOOL_BIN" ] && "$BPFTOOL_BIN" --version 2>&1 | head -1 || echo 'not found')"
echo "  bpftrace:     $BPFTRACE_VERSION"
echo "  GPU:          $GPU_NAME ($GPU_MEMORY)"
echo "  NVIDIA:       Driver $DRIVER_VERSION, CUDA $CUDA_VERSION"
echo "  BTF:          $([ -f /sys/kernel/btf/vmlinux ] && echo 'Available' || echo 'Not available')"
echo "  vmlinux.h:    $([ -f ~/workspace/ingero/bpf/headers/vmlinux.h ] && echo 'Generated' || echo 'Not generated')"

echo ""
print_success "Next Steps:"
echo ""
echo "  1. Start a new shell (or source ~/.bashrc) to pick up PATH changes:"
echo "     ${BLUE}source ~/.bashrc${NC}"
echo ""
echo "  2. Clone Ingero and build:"
echo "     ${BLUE}cd ~/workspace${NC}"
echo "     ${BLUE}git clone git@github.com:ingero-io/ingero.git${NC}"
echo "     ${BLUE}cd ingero${NC}"
echo "     ${BLUE}make generate && make build${NC}"
echo ""
echo "  3. Validate environment + probe attachment:"
echo "     ${BLUE}bash scripts/validate-gpu-env.sh --build${NC}"
echo ""
echo "  4. Run (requires sudo):"
echo "     ${BLUE}make dev${NC}"
echo ""
echo "  Note: Go PATH is set system-wide (/etc/environment) — remote SSH"
echo "  commands like 'ssh user@host make build' will work without manual"
echo "  PATH exports."
echo ""

print_info "Documentation:"
echo "  • NVIDIA Environment:    ~/workspace/nvidia_env.log"
echo "  • Project README:        README.md"

echo ""
print_warn "Cost Reminder:"
echo "  Billing is active while the VM is running."
echo "  Always stop or terminate your instance when not actively developing!"
echo "  Use: make gpu-stop (TensorDock) or terminate from your provider dashboard."

echo ""
print_header "Setup Script Completed Successfully"
echo ""
