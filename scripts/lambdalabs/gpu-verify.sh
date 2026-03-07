#!/bin/bash
# Lambda Labs GPU verification script — run on remote.
# Works on any GPU type (H100, A100, A10, L40, L4, etc.)
export PATH=/usr/local/go/bin:/usr/bin:/bin:/usr/sbin:/sbin:$HOME/go/bin:$HOME/.local/bin:$PATH

echo "=== System ==="
uname -a
cat /etc/os-release | head -3

echo ""
echo "=== GPU ==="
nvidia-smi --query-gpu=name,driver_version,memory.total,persistence_mode --format=csv,noheader

echo ""
echo "=== CUDA Compute ==="
python3 -c "
import torch
print(f'PyTorch {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA version: {torch.version.cuda}')
print(f'GPU: {torch.cuda.get_device_name(0)}')
print(f'GPU Memory: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')
# Quick compute test
t = torch.randn(1024, 1024, device='cuda')
r = torch.mm(t, t)
torch.cuda.synchronize()
print('Compute test: PASS')
"

echo ""
echo "=== Build Tools ==="
echo "Go: $(go version 2>/dev/null || echo NOT FOUND)"
echo "Clang: $(clang-14 --version 2>/dev/null | head -1 || echo NOT FOUND)"
echo "Make: $(make --version 2>/dev/null | head -1 || echo NOT FOUND)"
echo "SQLite3: $(sqlite3 --version 2>/dev/null || echo NOT FOUND)"
echo "BTF: $(ls /sys/kernel/btf/vmlinux 2>/dev/null || echo NOT FOUND)"
echo "libbpf: $(ls /usr/include/bpf/bpf_helpers.h 2>/dev/null || echo NOT FOUND)"
echo "bpftool: $(bpftool version 2>/dev/null || echo NOT FOUND)"
echo "jq: $(jq --version 2>/dev/null || echo NOT FOUND)"
echo "stress-ng: $(stress-ng --version 2>/dev/null | head -1 || echo NOT FOUND)"

echo ""
echo "=== Cloud-init ==="
cat ~/workspace/cloud-init-done.txt 2>/dev/null || echo "NOT COMPLETE"
tail -5 ~/workspace/setup.log 2>/dev/null || echo "No setup.log"

echo ""
echo "=== Disk ==="
df -h / | tail -1

echo ""
echo "=== Memory ==="
free -h | head -2
