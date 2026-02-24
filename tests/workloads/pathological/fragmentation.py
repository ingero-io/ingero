#!/usr/bin/env python3
"""Create GPU memory fragmentation and measure its impact on allocation.

Allocates tensors in an adversarial pattern (many small, free alternating,
then request large), repeatedly. This fragments the CUDA memory allocator
and causes cudaMalloc latency to climb.

Exercises: cudaMalloc latency under fragmentation
Expected Ingero output: cudaMalloc p99 climbing over rounds, anomaly flags
"""

import argparse
import time
import torch


def measure_alloc_latency(device, size_mb, warmup=5, measure=20):
    """Measure cudaMalloc latency for a specific size."""
    n_floats = (size_mb * 1024 * 1024) // 4
    times = []

    for i in range(warmup + measure):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        t = torch.empty(n_floats, dtype=torch.float32, device=device)
        torch.cuda.synchronize()
        dt = time.perf_counter() - t0
        del t
        if i >= warmup:
            times.append(dt * 1000)  # ms

    times.sort()
    return {
        "p50": times[len(times) // 2],
        "p99": times[-1],
        "mean": sum(times) / len(times),
    }


def fragment_round(device, num_small=100, small_mb=4, large_mb=256):
    """One round of fragmentation: alloc many small, free half, alloc large."""
    n_small = (small_mb * 1024 * 1024) // 4
    n_large = (large_mb * 1024 * 1024) // 4

    # Allocate many small tensors
    tensors = []
    for _ in range(num_small):
        tensors.append(torch.empty(n_small, dtype=torch.float32, device=device))

    # Free every other tensor (creates holes)
    for i in range(0, num_small, 2):
        tensors[i] = None

    # Allocate a large tensor (must navigate fragmented space)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    large = torch.empty(n_large, dtype=torch.float32, device=device)
    torch.cuda.synchronize()
    large_alloc_ms = (time.perf_counter() - t0) * 1000

    # Partial cleanup (keep some fragmentation)
    del large
    for i in range(1, num_small, 4):
        if tensors[i] is not None:
            tensors[i] = None

    return large_alloc_ms, tensors


def main():
    parser = argparse.ArgumentParser(description="GPU memory fragmentation test")
    parser.add_argument("--rounds", type=int, default=10, help="Fragmentation rounds")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    args = parser.parse_args()

    device = torch.device(args.device)
    vram_gb = torch.cuda.get_device_properties(device).total_memory / 1e9

    print(f"fragmentation: device={device}, GPU={torch.cuda.get_device_name(device)}")
    print(f"  VRAM: {vram_gb:.1f} GB, rounds: {args.rounds}")
    print()

    # Baseline measurement
    print("Baseline cudaMalloc latency (clean memory):")
    baseline = measure_alloc_latency(device, 256)
    print(f"  256MB alloc: p50={baseline['p50']:.2f}ms, p99={baseline['p99']:.2f}ms")
    print()

    # Fragmentation rounds
    all_tensors = []  # keep references to maintain fragmentation
    print("Fragmentation rounds:")

    for r in range(args.rounds):
        large_ms, kept_tensors = fragment_round(device, num_small=80, small_mb=4, large_mb=256)
        all_tensors.extend([t for t in kept_tensors if t is not None])

        vram_used = torch.cuda.memory_allocated(device) / 1e9
        vram_reserved = torch.cuda.memory_reserved(device) / 1e9

        print(f"  Round {r+1}/{args.rounds}: 256MB alloc={large_ms:.2f}ms "
              f"(VRAM: {vram_used:.1f}GB used, {vram_reserved:.1f}GB reserved)")

    print()

    # Post-fragmentation measurement
    print("Post-fragmentation cudaMalloc latency:")
    # Free some space for the measurement
    half = len(all_tensors) // 2
    del all_tensors[half:]
    torch.cuda.empty_cache()

    post = measure_alloc_latency(device, 256)
    print(f"  256MB alloc: p50={post['p50']:.2f}ms, p99={post['p99']:.2f}ms")
    print(f"  Slowdown: p50={post['p50']/max(baseline['p50'], 0.001):.1f}x, "
          f"p99={post['p99']/max(baseline['p99'], 0.001):.1f}x")

    # Cleanup
    del all_tensors
    torch.cuda.empty_cache()

    print(f"\nfragmentation complete.")
    print("Check Ingero — cudaMalloc p99 should climb across rounds.")


if __name__ == "__main__":
    main()
