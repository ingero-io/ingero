#!/usr/bin/env python3
"""Produce high cudaStreamSync/cudaDeviceSync latency.

Exercises: cudaStreamSync, cudaDeviceSync
Tests: sync latency measurement, stall detection
Expected Ingero output: cudaStreamSync with high p50/p99, SLOW anomaly flags
"""

import argparse
import time
import torch


def long_running_kernel(device, size=8192, iterations=50):
    """Launch a large matmul that takes measurable time, then sync.

    Each sync waits for the GPU to finish — Ingero should see high
    cudaStreamSync latency.
    """
    print(f"  Long matmul + sync: {size}x{size} x {iterations} iterations")
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    for i in range(iterations):
        c = torch.mm(a, b)
        torch.cuda.synchronize()  # explicit sync — measured by Ingero


def async_then_sync(device, num_ops=100, size=4096):
    """Queue many async ops then sync — sync latency = total queue drain time."""
    print(f"  Async queue + sync: {num_ops} matmuls queued, then sync")
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    # Queue many operations without syncing
    for _ in range(num_ops):
        c = torch.mm(a, b)

    # Single sync waits for all queued operations
    t0 = time.time()
    torch.cuda.synchronize()
    dt = time.time() - t0
    print(f"    Sync drained {num_ops} ops in {dt*1000:.1f}ms")


def stream_sync_pattern(device, num_streams=4, ops_per_stream=50, size=2048):
    """Multiple streams with per-stream sync — tests cudaStreamSynchronize."""
    print(f"  Per-stream sync: {num_streams} streams x {ops_per_stream} ops")
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    for _ in range(ops_per_stream):
        for stream in streams:
            with torch.cuda.stream(stream):
                _ = torch.mm(a, b)

        # Sync each stream individually
        for stream in streams:
            stream.synchronize()


def event_sync_pattern(device, iterations=100, size=4096):
    """Record CUDA events and synchronize on them."""
    print(f"  Event sync: {iterations} record+wait cycles")
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    for _ in range(iterations):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        _ = torch.mm(a, b)
        end.record()
        end.synchronize()  # waits for the event


def sync_with_transfer(device, size_mb=512, iterations=20):
    """Large H->D transfer followed by compute and sync.

    Tests the full pattern: memcpy + kernel + sync.
    """
    n_floats = (size_mb * 1024 * 1024) // 4
    print(f"  Transfer + compute + sync: {size_mb}MB x {iterations} iterations")

    host_data = torch.randn(n_floats, dtype=torch.float32, pin_memory=True)

    for _ in range(iterations):
        # Transfer
        gpu_data = host_data.to(device, non_blocking=False)
        # Compute (reshape to square-ish and matmul)
        side = int(n_floats ** 0.5)
        mat = gpu_data[:side * side].view(side, side)
        _ = torch.mm(mat, mat)
        # Sync
        torch.cuda.synchronize()

        del gpu_data


def main():
    parser = argparse.ArgumentParser(description="CUDA sync stall test")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"sync_stall: device={device}, GPU={torch.cuda.get_device_name(device)}")
    print()

    torch.empty(1, device=device)  # warm up

    print("Phase 1: Long-running kernel + sync")
    t0 = time.time()
    long_running_kernel(device)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Phase 2: Async queue then single sync")
    t0 = time.time()
    async_then_sync(device)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Phase 3: Per-stream sync pattern")
    t0 = time.time()
    stream_sync_pattern(device)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Phase 4: Event-based sync")
    t0 = time.time()
    event_sync_pattern(device)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Phase 5: Transfer + compute + sync pipeline")
    t0 = time.time()
    sync_with_transfer(device)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("sync_stall complete.")


if __name__ == "__main__":
    main()
