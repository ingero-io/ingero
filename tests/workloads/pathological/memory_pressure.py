#!/usr/bin/env python3
"""Create host memory pressure while running GPU workloads.

Allocates most of the host RAM, then runs GPU training. Host memory pressure
causes mm_page_alloc slowdowns which cascade to cudaMalloc latency.

Exercises (v0.2): mm_page_alloc correlation with cudaMalloc
Expected Ingero output: cudaMalloc p99 rises when host memory is pressured,
    mm_page_alloc shows high-order allocation failures
"""

import argparse
import os
import time
import torch
import torch.nn as nn
import torch.optim as optim


def get_available_ram_gb():
    """Get available RAM in GB (Linux only)."""
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                kb = int(line.split()[1])
                return kb / (1024 * 1024)
    return 16.0  # fallback


def allocate_host_memory(target_gb):
    """Allocate host memory to create pressure. Returns list of buffers."""
    buffers = []
    chunk_gb = 1.0
    allocated = 0.0

    print(f"  Allocating {target_gb:.1f} GB of host memory in {chunk_gb:.0f}GB chunks...")
    while allocated < target_gb:
        remaining = target_gb - allocated
        chunk = min(chunk_gb, remaining)
        try:
            n_bytes = int(chunk * 1024 * 1024 * 1024)
            buf = bytearray(n_bytes)
            # Touch pages to ensure physical allocation
            for i in range(0, n_bytes, 4096):
                buf[i] = 1
            buffers.append(buf)
            allocated += chunk
        except MemoryError:
            print(f"  MemoryError at {allocated:.1f} GB — stopping allocation")
            break

    print(f"  Allocated {allocated:.1f} GB of host memory")
    return buffers


def gpu_workload(device, iterations=200):
    """Run a GPU workload that needs cudaMalloc under memory pressure."""
    print(f"  Running GPU workload: {iterations} iterations of alloc+compute+free")

    model = nn.Sequential(
        nn.Linear(1024, 2048),
        nn.ReLU(),
        nn.Linear(2048, 2048),
        nn.ReLU(),
        nn.Linear(2048, 1024),
    ).to(device)

    optimizer = optim.Adam(model.parameters())
    criterion = nn.MSELoss()

    for i in range(iterations):
        # Each iteration allocates intermediate tensors via cudaMalloc
        x = torch.randn(256, 1024, device=device)
        target = torch.randn(256, 1024, device=device)

        output = model(x)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # Periodically allocate/free a large tensor (stress cudaMalloc)
        if i % 10 == 0:
            big = torch.randn(1024, 1024, 4, device=device)  # ~16MB
            del big

        if (i + 1) % 50 == 0:
            print(f"    Step {i+1}/{iterations}, loss: {loss.item():.4f}")

    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description="Host memory pressure + GPU workload")
    parser.add_argument("--pressure-pct", type=float, default=80,
                        help="Percent of available RAM to consume (default: 80)")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--iterations", type=int, default=200, help="GPU workload iterations")
    args = parser.parse_args()

    device = torch.device(args.device)
    available_gb = get_available_ram_gb()
    target_gb = available_gb * (args.pressure_pct / 100.0)

    print(f"memory_pressure: device={device}, GPU={torch.cuda.get_device_name(device)}")
    print(f"  Available RAM: {available_gb:.1f} GB")
    print(f"  Target pressure: {args.pressure_pct}% = {target_gb:.1f} GB")
    print()

    # Phase 1: Baseline GPU workload (no memory pressure)
    print("Phase 1: Baseline (no memory pressure)")
    t0 = time.time()
    gpu_workload(device, iterations=50)
    print(f"  Baseline done in {time.time() - t0:.1f}s\n")

    # Phase 2: Allocate host memory to create pressure
    print("Phase 2: Creating memory pressure")
    buffers = allocate_host_memory(target_gb)
    print()

    # Phase 3: GPU workload under memory pressure
    print("Phase 3: GPU workload under memory pressure")
    t0 = time.time()
    gpu_workload(device, iterations=args.iterations)
    print(f"  Pressured workload done in {time.time() - t0:.1f}s\n")

    # Release memory
    print("Releasing host memory...")
    del buffers

    print("memory_pressure complete.")
    print("Compare Phase 1 vs Phase 3 in Ingero — cudaMalloc p99 should be higher in Phase 3.")


if __name__ == "__main__":
    main()
