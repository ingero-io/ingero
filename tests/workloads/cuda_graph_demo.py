#!/usr/bin/env python3
"""CUDA Graph demo workload — "The Graph Re-Capture Mystery"

Simulates a real inference serving scenario:
  Phase 1: Steady-state graph replay at a fixed batch size (warm graphs)
  Phase 2: Batch size changes arrive — triggers graph re-capture (the vLLM pattern)
  Phase 3: Return to steady state with the new graph

When combined with CPU contention (stress-ng --cpu 2), this creates the
exact conditions where CUDA Graph dispatch stalls and re-capture latency
spikes appear — visible only through Ingero's eBPF-based causal chains.

Usage:
    # Terminal 1: start workload
    python cuda_graph_demo.py

    # Terminal 2: trace with Ingero
    sudo ingero trace --pid $(pgrep -f cuda_graph_demo) --db demo.db

    # Terminal 3 (optional): add CPU contention
    stress-ng --cpu 2 --timeout 30s

    # After: investigate
    sudo ingero explain --db demo.db

Requires: PyTorch 2.x with CUDA support.
"""

import argparse
import os
import sys
import time

import torch
import torch.nn as nn


def create_model(hidden=1024):
    """Small model suitable for graph capture."""
    return nn.Sequential(
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
    ).cuda()


def run_phase(compiled, x, duration, label):
    """Run inference and return (iterations, elapsed)."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    iters = 0
    with torch.no_grad():
        while True:
            _ = compiled(x)
            iters += 1
            if iters % 100 == 0:
                elapsed = time.perf_counter() - start
                if elapsed >= duration:
                    break
    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    rate = iters / elapsed
    print(f"  {label}: {iters} iters in {elapsed:.1f}s ({rate:.0f} iter/s)")
    return iters, elapsed


def main():
    parser = argparse.ArgumentParser(description="CUDA Graph demo workload")
    parser.add_argument("--phase1", type=int, default=10, help="Phase 1 duration (seconds)")
    parser.add_argument("--phase2", type=int, default=10, help="Phase 2 duration (seconds)")
    parser.add_argument("--phase3", type=int, default=10, help="Phase 3 duration (seconds)")
    parser.add_argument("--hidden", type=int, default=1024, help="Hidden layer size")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available", file=sys.stderr)
        sys.exit(1)

    props = torch.cuda.get_device_properties(0)
    pid = os.getpid()
    print(f"CUDA Graph Demo — {props.name} ({props.total_memory // 1024 // 1024} MB)")
    print(f"PID: {pid}")
    print(f"Tip: sudo ingero trace --pid {pid} --db demo.db")
    print()

    # --- Phase 1: steady-state with batch_size=64 ---
    print("[Phase 1] Steady-state inference (batch=64, graphs warm)")
    model = create_model(args.hidden)
    compiled_64 = torch.compile(model, mode="reduce-overhead")
    x_64 = torch.randn(64, args.hidden, device="cuda")

    # Warmup — triggers graph capture + instantiate
    print("  Warming up (graph capture)...")
    with torch.no_grad():
        for _ in range(20):
            _ = compiled_64(x_64)
    torch.cuda.synchronize()
    print("  Graphs captured. Entering steady state...")

    run_phase(compiled_64, x_64, args.phase1, "Phase 1")

    # --- Phase 2: batch size change → graph re-capture ---
    print()
    print("[Phase 2] Batch size change! (batch=128 → graph re-capture)")
    x_128 = torch.randn(128, args.hidden, device="cuda")
    compiled_128 = torch.compile(model, mode="reduce-overhead")

    # This warmup triggers NEW graph capture for the new input shape
    print("  New batch size → triggering graph re-capture...")
    with torch.no_grad():
        for _ in range(20):
            _ = compiled_128(x_128)
    torch.cuda.synchronize()
    print("  New graphs captured. Running with new batch size...")

    run_phase(compiled_128, x_128, args.phase2, "Phase 2")

    # --- Phase 3: back to original batch size ---
    print()
    print("[Phase 3] Return to original batch size (batch=64, cached graph)")
    run_phase(compiled_64, x_64, args.phase3, "Phase 3")

    print()
    print("Demo complete.")
    print(f"Investigate: sudo ingero explain --db demo.db")


if __name__ == "__main__":
    main()
