#!/usr/bin/env bash
# CPU contention: run stress-ng alongside a GPU training workload.
#
# This causes sched_switch storms — the GPU process gets preempted
# off-CPU, increasing cudaStreamSync latency.
#
# Exercises (v0.2): sched_switch, sched_wakeup correlation with CUDA sync
# Expected Ingero output: cudaStreamSync p99 spikes correlated with sched_switch events
#
# Usage: bash cpu_contention.sh [--stress-cpus 4] [--duration 60]

set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
STRESS_CPUS=${1:-4}
DURATION=${2:-60}

echo "cpu_contention: ${STRESS_CPUS} stress-ng workers for ${DURATION}s alongside GPU training"
echo ""

# Check stress-ng
if ! command -v stress-ng &>/dev/null; then
    echo "Installing stress-ng..."
    # Hold broken grub package on TensorDock VMs to prevent dpkg errors
    sudo apt-mark hold grub-efi-amd64-signed 2>/dev/null || true
    sudo apt-get install -y stress-ng 2>&1 | grep -v "grub-efi"
fi

# Start stress-ng in background
echo "Starting stress-ng with ${STRESS_CPUS} CPU workers..."
stress-ng --cpu "${STRESS_CPUS}" --cpu-method matrixprod --timeout "${DURATION}s" &
STRESS_PID=$!
echo "  stress-ng PID: ${STRESS_PID}"

# Run a training workload (ResNet-50 is quick and steady)
echo "Starting ResNet-50 training (will run concurrently with CPU stress)..."
echo ""
python3 "${SCRIPT_DIR}/../training/resnet50_cifar10.py" --epochs 1 --batch-size 64

# Clean up stress-ng if still running
if kill -0 "${STRESS_PID}" 2>/dev/null; then
    echo ""
    echo "Stopping stress-ng..."
    kill "${STRESS_PID}" 2>/dev/null || true
    wait "${STRESS_PID}" 2>/dev/null || true
fi

echo ""
echo "cpu_contention complete."
echo "Check Ingero output for cudaStreamSync p99 spikes correlated with sched_switch events."
