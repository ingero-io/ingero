#!/usr/bin/env python3
"""Stress cudaMemcpy with varying sizes and directions.

Exercises: cudaMemcpy (H->D, D->H, D->D)
Tests: transfer latency at different sizes, bandwidth saturation
Expected Ingero output: cudaMemcpy stats split by direction, p99 at large sizes
"""

import argparse
import time
import torch


def transfer_sweep(device, direction, sizes_mb, iterations=10):
    """Transfer data at various sizes and measure throughput."""
    print(f"  {direction}: {len(sizes_mb)} sizes x {iterations} iterations")
    for size_mb in sizes_mb:
        n_floats = int(size_mb * 1024 * 1024) // 4

        if direction == "H->D":
            src = torch.randn(n_floats, dtype=torch.float32, pin_memory=True)
            for _ in range(iterations):
                dst = src.to(device, non_blocking=False)
                del dst
        elif direction == "D->H":
            src = torch.randn(n_floats, dtype=torch.float32, device=device)
            for _ in range(iterations):
                dst = src.cpu()
                del dst
        elif direction == "D->D":
            src = torch.randn(n_floats, dtype=torch.float32, device=device)
            for _ in range(iterations):
                dst = src.clone()
                del dst

        torch.cuda.synchronize()


def pinned_vs_pageable(device, size_mb=256, iterations=20):
    """Compare pinned vs pageable memory transfer speed.

    Pageable transfers go through an extra copy — Ingero should show
    higher cudaMemcpy latency for pageable.
    """
    n_floats = (size_mb * 1024 * 1024) // 4
    print(f"  Pinned vs pageable: {size_mb}MB x {iterations} iterations")

    # Pinned
    pinned = torch.randn(n_floats, dtype=torch.float32, pin_memory=True)
    t0 = time.time()
    for _ in range(iterations):
        _ = pinned.to(device, non_blocking=False)
    torch.cuda.synchronize()
    pinned_time = time.time() - t0

    # Pageable (regular CPU tensor)
    pageable = torch.randn(n_floats, dtype=torch.float32)
    t0 = time.time()
    for _ in range(iterations):
        _ = pageable.to(device, non_blocking=False)
    torch.cuda.synchronize()
    pageable_time = time.time() - t0

    print(f"    Pinned:   {pinned_time:.2f}s ({size_mb * iterations / pinned_time:.0f} MB/s)")
    print(f"    Pageable: {pageable_time:.2f}s ({size_mb * iterations / pageable_time:.0f} MB/s)")


def bidirectional_flood(device, size_mb=64, iterations=100):
    """Simultaneous H->D and D->H transfers using streams."""
    n_floats = (size_mb * 1024 * 1024) // 4
    print(f"  Bidirectional flood: {size_mb}MB x {iterations} iterations (2 streams)")

    stream_h2d = torch.cuda.Stream()
    stream_d2h = torch.cuda.Stream()

    host_src = torch.randn(n_floats, dtype=torch.float32, pin_memory=True)
    dev_src = torch.randn(n_floats, dtype=torch.float32, device=device)

    for _ in range(iterations):
        with torch.cuda.stream(stream_h2d):
            _ = host_src.to(device, non_blocking=True)
        with torch.cuda.stream(stream_d2h):
            _ = dev_src.cpu()

    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description="CUDA memcpy stress test")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"memcpy_stress: device={device}, GPU={torch.cuda.get_device_name(device)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    print()

    torch.empty(1, device=device)  # warm up

    sizes = [0.001, 0.01, 0.1, 1, 4, 16, 64, 256, 1024]  # MB

    print("Phase 1: H->D transfer sweep")
    t0 = time.time()
    transfer_sweep(device, "H->D", sizes)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Phase 2: D->H transfer sweep")
    t0 = time.time()
    transfer_sweep(device, "D->H", sizes)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Phase 3: D->D transfer sweep")
    t0 = time.time()
    transfer_sweep(device, "D->D", sizes)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Phase 4: Pinned vs pageable memory")
    t0 = time.time()
    pinned_vs_pageable(device)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Phase 5: Bidirectional flood")
    t0 = time.time()
    bidirectional_flood(device)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("memcpy_stress complete.")


if __name__ == "__main__":
    main()
