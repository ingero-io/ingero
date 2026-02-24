#!/bin/bash
# Quick fix script for cloud-init gaps — run on GPU VM
set -e

echo "=== Fixing environment ==="

# Ensure Go is on PATH (cloud-init installed it but .bashrc may not have it)
if ! command -v go &>/dev/null; then
    if [[ -x /usr/local/go/bin/go ]]; then
        export PATH=/usr/local/go/bin:$HOME/go/bin:$HOME/.local/bin:$PATH
        # Only add if not already in .bashrc
        if ! grep -q '/usr/local/go/bin' ~/.bashrc; then
            echo 'export PATH=/usr/local/go/bin:$HOME/go/bin:$HOME/.local/bin:$PATH' >> ~/.bashrc
        fi
        echo "Go: $(/usr/local/go/bin/go version)"
    else
        echo "ERROR: Go not installed at /usr/local/go/bin/"
        exit 1
    fi
else
    echo "Go: $(go version)"
fi

# Load NVIDIA driver module (kernel mismatch on fresh boot)
if ! nvidia-smi &>/dev/null; then
    echo "Loading NVIDIA kernel modules..."
    sudo modprobe nvidia
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
else
    nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
fi

# Install bpftool for current kernel
if ! command -v bpftool &>/dev/null; then
    echo "Installing bpftool..."
    sudo apt-get install -y -qq linux-tools-$(uname -r) 2>/dev/null \
      || sudo apt-get install -y -qq linux-tools-generic 2>/dev/null \
      || echo "WARNING: bpftool install failed"
fi

# Verify BTF
if [[ -f /sys/kernel/btf/vmlinux ]]; then
    echo "BTF: available ($(stat -c%s /sys/kernel/btf/vmlinux) bytes)"
else
    echo "WARNING: BTF not available at /sys/kernel/btf/vmlinux"
fi

# Summary
echo ""
echo "=== Environment Check ==="
echo "Kernel: $(uname -r)"
echo "Go: $(go version 2>/dev/null || echo MISSING)"
echo "clang-14: $(which clang-14 2>/dev/null || echo MISSING)"
echo "bpftool: $(which bpftool 2>/dev/null || echo MISSING)"
echo "Python3: $(python3 --version 2>/dev/null || echo MISSING)"
python3 -c "import torch; print(f'PyTorch: {torch.__version__}, CUDA: {torch.cuda.is_available()}')" 2>/dev/null || echo "PyTorch: MISSING"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo "GPU: UNAVAILABLE"
echo "=== Done ==="
