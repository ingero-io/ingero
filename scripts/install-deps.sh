#!/usr/bin/env bash
# Install all build dependencies for Ingero on Ubuntu 22.04/24.04.
# Run this BEFORE `make` on a fresh machine.
#
# Usage:
#   git clone https://github.com/ingero-io/ingero.git
#   cd ingero
#   bash scripts/install-deps.sh
#   make
set -euo pipefail

RED='\033[0;31m'
GREEN='\033[0;32m'
NC='\033[0m'

ok()   { echo -e "  ${GREEN}✓${NC} $1"; }
fail() { echo -e "  ${RED}✗${NC} $1"; exit 1; }

GO_VERSION="1.26.1"
GO_TARBALL="go${GO_VERSION}.linux-amd64.tar.gz"
GO_URL="https://go.dev/dl/${GO_TARBALL}"

install_go() {
    echo "  Downloading Go ${GO_VERSION}..."
    rm -f "$GO_TARBALL"
    wget -q --show-progress -O "$GO_TARBALL" "$GO_URL" \
        || fail "Failed to download Go from $GO_URL"
    # Verify we got a real gzip file, not an HTML error page
    if ! file "$GO_TARBALL" | grep -q gzip; then
        rm -f "$GO_TARBALL"
        fail "Downloaded file is not a valid gzip archive. Check that Go ${GO_VERSION} exists at $GO_URL"
    fi
    sudo rm -rf /usr/local/go
    sudo tar -C /usr/local -xzf "$GO_TARBALL"
    rm -f "$GO_TARBALL"
    ok "go ${GO_VERSION} installed"
}

echo "=== Ingero build dependency installer ==="
echo ""

# ---- 1. Go 1.26+ ----
echo "[1/7] Checking Go..."
if command -v go &>/dev/null; then
    GO_VER=$(go version | grep -oP 'go\K[0-9]+\.[0-9]+')
    GO_MAJOR=$(echo "$GO_VER" | cut -d. -f1)
    GO_MINOR=$(echo "$GO_VER" | cut -d. -f2)
    if [ "$GO_MAJOR" -ge 1 ] && [ "$GO_MINOR" -ge 26 ]; then
        ok "go $(go version | grep -oP 'go[0-9.]+') already installed"
    else
        echo "  Go $GO_VER found but 1.26+ required. Installing..."
        install_go
    fi
else
    echo "  Go not found. Installing..."
    install_go
fi

# Ensure Go is in PATH for the rest of this script
export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH

# Persist PATH if not already in .bashrc
if ! grep -q '/usr/local/go/bin' ~/.bashrc 2>/dev/null; then
    echo 'export PATH=/usr/local/go/bin:$HOME/go/bin:$PATH' >> ~/.bashrc
    ok "added Go to ~/.bashrc PATH"
fi

# ---- 2. eBPF toolchain (clang-14, llvm-14, libbpf-dev, libelf-dev) ----
echo ""
echo "[2/7] Installing eBPF toolchain..."
sudo apt-get update -qq

# Figure out which clang/llvm version to install.
# Ingero's bpf2go directives hardcode clang-14 and llvm-strip-14.
# If clang-14 is available as a real package, install it directly.
# Otherwise install whatever version is available and symlink.
if apt-cache show clang-14 &>/dev/null; then
    sudo apt-get install -y -qq clang-14 llvm-14 libbpf-dev libelf-dev
    ok "clang-14 + llvm-14 installed natively"
else
    # Install default clang/llvm and symlink to -14
    sudo apt-get install -y -qq clang llvm libbpf-dev libelf-dev
    CLANG_VER=$(ls /usr/bin/clang-* 2>/dev/null | grep -oP 'clang-\K[0-9]+' | head -1)
    if [ -n "$CLANG_VER" ] && [ "$CLANG_VER" != "14" ]; then
        sudo ln -sf /usr/bin/clang-"$CLANG_VER" /usr/bin/clang-14
        sudo ln -sf /usr/bin/llvm-strip-"$CLANG_VER" /usr/bin/llvm-strip-14
        ok "clang-$CLANG_VER symlinked as clang-14 / llvm-strip-14"
    else
        ok "clang-14 already present"
    fi
fi

# Verify
command -v clang-14 &>/dev/null || fail "clang-14 not found after install"
command -v llvm-strip-14 &>/dev/null || fail "llvm-strip-14 not found after install"
ok "clang-14 and llvm-strip-14 verified"

# ---- 3. bpftool (for vmlinux.h generation) ----
echo ""
echo "[3/7] Installing bpftool..."
if ! command -v bpftool &>/dev/null; then
    sudo apt-get install -y -qq linux-tools-common linux-tools-"$(uname -r)" 2>/dev/null \
        || sudo apt-get install -y -qq linux-tools-common bpftool 2>/dev/null \
        || true
fi

# bpftool might be at a kernel-specific path; the Makefile handles this,
# but verify at least one location works
BPFTOOL=""
if command -v bpftool &>/dev/null; then
    BPFTOOL="bpftool"
elif [ -x "/usr/lib/linux-tools/$(uname -r)/bpftool" ]; then
    BPFTOOL="/usr/lib/linux-tools/$(uname -r)/bpftool"
else
    # Try any linux-tools version
    BPFTOOL=$(find /usr/lib/linux-tools/ -name bpftool -type f 2>/dev/null | head -1)
fi
[ -n "$BPFTOOL" ] || fail "bpftool not found. Install linux-tools for kernel $(uname -r)"
ok "bpftool found at: $BPFTOOL"

# ---- 4. Generate vmlinux.h ----
echo ""
echo "[4/7] Generating vmlinux.h..."
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(dirname "$SCRIPT_DIR")"
VMLINUX_DIR="$REPO_DIR/bpf/headers"

if [ ! -f /sys/kernel/btf/vmlinux ]; then
    fail "Kernel BTF not available at /sys/kernel/btf/vmlinux. Need kernel 5.15+ with CONFIG_DEBUG_INFO_BTF=y"
fi

mkdir -p "$VMLINUX_DIR"
$BPFTOOL btf dump file /sys/kernel/btf/vmlinux format c > "$VMLINUX_DIR/vmlinux.h"
ok "vmlinux.h generated ($(wc -l < "$VMLINUX_DIR/vmlinux.h") lines)"

# ---- 5. staticcheck (Go linter) ----
echo ""
echo "[5/7] Installing staticcheck..."
# GOTOOLCHAIN=local prevents Go from downloading an older toolchain
# when staticcheck's go.mod declares an older Go version.
# Without this, staticcheck gets compiled with Go 1.25 and can't
# analyze Go 1.26 code.
GOTOOLCHAIN=local go install honnef.co/go/tools/cmd/staticcheck@latest 2>/dev/null \
    && ok "staticcheck installed" \
    || {
        echo "  WARN: staticcheck install failed (known Go version mismatch)"
        echo "  'make build' and 'make test' will work; 'make lint' will be skipped"
    }

# ---- 6. Symlink libcudart.so (GPU runtime) ----
echo ""
echo "[6/7] Checking libcudart.so..."
if ldconfig -p 2>/dev/null | grep -q libcudart.so; then
    ok "libcudart.so already discoverable"
elif [ -L /usr/lib/x86_64-linux-gnu/libcudart.so ]; then
    ok "libcudart.so symlink already exists"
else
    # Search common locations one at a time. Using a single find with
    # multiple paths fails silently when some paths don't exist.
    # The Deep Learning AMI buries libcudart.so deep inside the PyTorch
    # venv's site-packages/nvidia/ tree.
    CUDART_PATH=""
    for search_dir in /opt /usr/local/cuda /usr/lib; do
        if [ -d "$search_dir" ]; then
            CUDART_PATH=$(find "$search_dir" -name 'libcudart.so' -type f 2>/dev/null | head -1)
            [ -n "$CUDART_PATH" ] && break
        fi
    done
    if [ -n "$CUDART_PATH" ]; then
        sudo ln -sf "$CUDART_PATH" /usr/lib/x86_64-linux-gnu/libcudart.so
        ok "symlinked $CUDART_PATH → /usr/lib/x86_64-linux-gnu/libcudart.so"
    else
        echo "  SKIP: libcudart.so not found (not needed for build, needed for 'ingero check/trace')"
        echo "  If using a PyTorch venv, run: sudo ln -sf \$(find / -name libcudart.so -type f | head -1) /usr/lib/x86_64-linux-gnu/libcudart.so"
    fi
fi

# ---- 7. Verify everything ----
echo ""
echo "[7/7] Verification"
echo ""

ALL_OK=true
for cmd in go clang-14 llvm-strip-14; do
    if command -v "$cmd" &>/dev/null; then
        ok "$cmd: $(command -v $cmd)"
    else
        fail "$cmd: NOT FOUND"
        ALL_OK=false
    fi
done

if [ -f "$VMLINUX_DIR/vmlinux.h" ]; then
    ok "vmlinux.h: $VMLINUX_DIR/vmlinux.h"
else
    fail "vmlinux.h: NOT FOUND"
    ALL_OK=false
fi

if command -v staticcheck &>/dev/null; then
    ok "staticcheck: $(command -v staticcheck)"
else
    echo -e "  ${RED}!${NC} staticcheck: not installed (lint will fail, build/test will work)"
fi

echo ""
if [ "$ALL_OK" = true ]; then
    echo -e "${GREEN}All dependencies installed.${NC}"
    echo ""
    echo "Next steps:"
    echo "  source ~/.bashrc   # reload PATH (required once after install)"
    echo "  make               # build Ingero"
else
    echo -e "${RED}Some dependencies are missing. Check errors above.${NC}"
    exit 1
fi
