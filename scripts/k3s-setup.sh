#!/bin/bash
# k3s-setup.sh — Install k3s + NVIDIA container toolkit + GPU device plugin.
# Idempotent: safe to run multiple times.
#
# Prerequisites:
#   - NVIDIA GPU with driver 550+ installed
#   - Ubuntu 22.04/24.04
#
# Usage:
#   bash scripts/k3s-setup.sh
#   # or: make gpu-k3s-setup (from WSL, runs via SSH on GPU VM)

set -euo pipefail

echo "=== k3s GPU Setup ==="

# 1. Install k3s (single-node, no traefik/metrics-server — we only need the kubelet)
if command -v k3s &>/dev/null; then
    echo "[OK] k3s already installed: $(k3s --version | head -1)"
else
    echo "[INSTALL] Installing k3s..."
    curl -sfL https://get.k3s.io | INSTALL_K3S_EXEC="--disable=traefik,metrics-server" sh -
    echo "[OK] k3s installed"
fi

# Wait for k3s to be ready
echo "[WAIT] Waiting for k3s node to be Ready..."
for i in $(seq 1 60); do
    if sudo k3s kubectl get nodes 2>/dev/null | grep -q " Ready"; then
        echo "[OK] k3s node is Ready"
        break
    fi
    if [ "$i" = "60" ]; then
        echo "[FAIL] k3s node not ready after 60s"
        exit 1
    fi
    sleep 1
done

# 2. Configure k3s to use nvidia-container-runtime
# k3s uses containerd — we need to tell it to use nvidia runtime for GPU pods.
K3S_CONTAINERD_CONFIG="/var/lib/rancher/k3s/agent/etc/containerd/config.toml.tmpl"
if [ -f "$K3S_CONTAINERD_CONFIG" ] && grep -q "nvidia" "$K3S_CONTAINERD_CONFIG"; then
    echo "[OK] nvidia-container-runtime already configured in k3s containerd"
else
    echo "[CONFIGURE] Setting up nvidia-container-runtime for k3s..."
    sudo mkdir -p "$(dirname "$K3S_CONTAINERD_CONFIG")"
    sudo tee "$K3S_CONTAINERD_CONFIG" > /dev/null <<'CONTAINERD_EOF'
[plugins."io.containerd.grpc.v1.cri".containerd.runtimes."nvidia"]
  privileged_without_host_devices = false
  runtime_engine = ""
  runtime_root = ""
  runtime_type = "io.containerd.runc.v2"
  [plugins."io.containerd.grpc.v1.cri".containerd.runtimes."nvidia".options]
    BinaryName = "/usr/bin/nvidia-container-runtime"
CONTAINERD_EOF
    echo "[OK] nvidia runtime configured, restarting k3s..."
    sudo systemctl restart k3s
    sleep 5
fi

# 3. Install NVIDIA device plugin (makes nvidia.com/gpu resource available)
DEVICE_PLUGIN_URL="https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.0/deployments/static/nvidia-device-plugin.yml"
if sudo k3s kubectl get ds -n kube-system nvidia-device-plugin-daemonset &>/dev/null; then
    echo "[OK] NVIDIA device plugin already deployed"
else
    echo "[INSTALL] Deploying NVIDIA device plugin..."
    sudo k3s kubectl apply -f "$DEVICE_PLUGIN_URL"
    echo "[OK] NVIDIA device plugin deployed"
fi

# 4. Wait for device plugin to be ready and GPU to be allocatable
echo "[WAIT] Waiting for GPU to be allocatable..."
for i in $(seq 1 90); do
    GPU_COUNT=$(sudo k3s kubectl get node -o jsonpath='{.items[0].status.allocatable.nvidia\.com/gpu}' 2>/dev/null || echo "0")
    if [ "$GPU_COUNT" != "0" ] && [ -n "$GPU_COUNT" ]; then
        echo "[OK] GPU allocatable: $GPU_COUNT"
        break
    fi
    if [ "$i" = "90" ]; then
        echo "[WARN] GPU not allocatable after 90s — device plugin may still be starting"
    fi
    sleep 1
done

# 5. Label the node for ingero nodeSelector
NODE_NAME=$(sudo k3s kubectl get nodes -o jsonpath='{.items[0].metadata.name}')
sudo k3s kubectl label node "$NODE_NAME" nvidia.com/gpu.present=true --overwrite

# 6. Install docker (needed for building ingero test image in k3s-test.sh)
# Lambda Labs AMD64 VMs have docker via nvidia-container-toolkit; ARM64 (GH200) may not.
if command -v docker &>/dev/null; then
    echo "[OK] docker already installed: $(docker --version)"
else
    echo "[INSTALL] Installing docker.io (needed for image build)..."
    sudo apt-get update -qq && sudo apt-get install -y -qq docker.io
    echo "[OK] docker installed"
fi

# 7. Install sqlite3 (needed for test assertion queries against ingero.db)
if command -v sqlite3 &>/dev/null; then
    echo "[OK] sqlite3 already installed"
else
    echo "[INSTALL] Installing sqlite3..."
    sudo apt-get install -y -qq sqlite3
    echo "[OK] sqlite3 installed"
fi

# 8. Pre-pull PyTorch image (~15GB multi-arch, avoids timeout in test pods)
# Uses k3s's built-in ctr to put the image in the correct containerd namespace.
if sudo k3s ctr images ls -q | grep -q "nvcr.io/nvidia/pytorch:24.01-py3"; then
    echo "[OK] PyTorch image already pulled"
else
    echo "[PULL] Pulling nvcr.io/nvidia/pytorch:24.01-py3 (this may take 5-10 min)..."
    sudo k3s ctr images pull nvcr.io/nvidia/pytorch:24.01-py3
    echo "[OK] PyTorch image pulled"
fi

echo ""
echo "=== k3s GPU Setup Complete ==="
echo "  Node:    $NODE_NAME"
echo "  GPUs:    $GPU_COUNT"
echo ""
echo "Next steps:"
echo "  bash scripts/k3s-test.sh  # Run K8s integration tests"
echo "  # or: make gpu-k3s-test / make lambda-k3s-test (from WSL)"
