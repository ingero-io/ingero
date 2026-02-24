#!/usr/bin/env python3
"""Stress cudaMalloc/cudaFree with varying sizes and patterns.

Exercises: cudaMalloc, cudaFree
Tests: allocation latency, fragmentation detection, p99 spikes
Expected Ingero output: cudaMalloc stats with rising p99 in fragmentation phase
"""

import argparse
import time
import torch


def sequential_alloc(sizes_mb, device, rounds=3):
    """Allocate and free tensors sequentially at given sizes."""
    print(f"  Sequential alloc/free: {len(sizes_mb)} sizes x {rounds} rounds")
    for r in range(rounds):
        for size_mb in sizes_mb:
            n_floats = (size_mb * 1024 * 1024) // 4
            t = torch.empty(n_floats, dtype=torch.float32, device=device)
            del t
    torch.cuda.synchronize()


def fragmentation_pattern(device, num_tensors=50, rounds=5):
    """Allocate many small tensors, free every other one, then try large alloc.

    This creates memory fragmentation — cudaMalloc for the large tensor
    must coalesce free blocks, which is slower.
    """
    print(f"  Fragmentation: {num_tensors} small tensors, free odd, alloc large x {rounds} rounds")
    for r in range(rounds):
        # Allocate many small tensors (1MB each)
        tensors = []
        for i in range(num_tensors):
            n_floats = (1 * 1024 * 1024) // 4
            tensors.append(torch.empty(n_floats, dtype=torch.float32, device=device))

        # Free every other tensor (creates holes)
        for i in range(0, num_tensors, 2):
            tensors[i] = None

        # Try to allocate a large tensor (must find contiguous space)
        large_mb = num_tensors // 2
        n_floats = (large_mb * 1024 * 1024) // 4
        large = torch.empty(n_floats, dtype=torch.float32, device=device)
        del large

        # Clean up remaining
        del tensors
        torch.cuda.empty_cache()

    torch.cuda.synchronize()


def rapid_small_alloc(device, count=5000):
    """Rapidly allocate and free tiny tensors."""
    print(f"  Rapid small alloc: {count} x 4KB tensors")
    for _ in range(count):
        t = torch.empty(1024, dtype=torch.float32, device=device)
        del t
    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description="CUDA allocation stress test")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"alloc_stress: device={device}, GPU={torch.cuda.get_device_name(device)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    print()

    # Warm up CUDA context
    torch.empty(1, device=device)

    print("Phase 1: Sequential alloc/free (1MB to 512MB)")
    sizes = [1, 4, 16, 64, 128, 256, 512]
    t0 = time.time()
    sequential_alloc(sizes, device)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Phase 2: Fragmentation pattern")
    t0 = time.time()
    fragmentation_pattern(device)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("Phase 3: Rapid small allocations")
    t0 = time.time()
    rapid_small_alloc(device)
    print(f"  Done in {time.time() - t0:.1f}s\n")

    print("alloc_stress complete.")


if __name__ == "__main__":
    main()
