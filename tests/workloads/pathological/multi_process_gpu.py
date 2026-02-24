#!/usr/bin/env python3
"""Run multiple processes sharing the same GPU.

Launches N child processes, each running a GPU workload on the same device.
This forces GPU context switches and increases cudaStreamSync latency.

Exercises (v0.3): nvidia.ko context switch kprobes, multi-process GPU sharing
Expected Ingero output: Higher sync latency, GPU context switch events in driver layer
"""

import argparse
import multiprocessing
import time
import torch


def gpu_worker(worker_id, device_str, iterations, size):
    """Each worker runs a matmul loop on the shared GPU."""
    device = torch.device(device_str)
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)

    for i in range(iterations):
        c = torch.mm(a, b)
        torch.cuda.synchronize()

        if (i + 1) % 50 == 0:
            print(f"  Worker {worker_id}: {i+1}/{iterations} iterations")

    del a, b, c
    torch.cuda.synchronize()
    print(f"  Worker {worker_id}: done")


def main():
    parser = argparse.ArgumentParser(description="Multi-process GPU sharing")
    parser.add_argument("--workers", type=int, default=3, help="Number of GPU processes")
    parser.add_argument("--iterations", type=int, default=200, help="Iterations per worker")
    parser.add_argument("--size", type=int, default=2048, help="Matrix size")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    args = parser.parse_args()

    print(f"multi_process_gpu: {args.workers} workers sharing {args.device}")
    print(f"  GPU={torch.cuda.get_device_name(args.device)}")
    print(f"  {args.iterations} iterations x {args.size}x{args.size} matmul each")
    print()

    # Phase 1: Single process baseline
    print("Phase 1: Single process baseline")
    t0 = time.time()
    gpu_worker(0, args.device, args.iterations, args.size)
    single_time = time.time() - t0
    print(f"  Single process: {single_time:.1f}s\n")

    # Phase 2: Multiple processes competing
    print(f"Phase 2: {args.workers} concurrent processes")
    t0 = time.time()

    # Use spawn to avoid CUDA fork issues
    ctx = multiprocessing.get_context("spawn")
    processes = []
    for i in range(args.workers):
        p = ctx.Process(target=gpu_worker, args=(i, args.device, args.iterations, args.size))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()

    multi_time = time.time() - t0
    slowdown = multi_time / single_time
    print(f"\n  {args.workers} processes: {multi_time:.1f}s ({slowdown:.1f}x slowdown)")
    print(f"\nmulti_process_gpu complete.")
    print(f"Compare single vs multi in Ingero — cudaStreamSync p99 should increase with contention.")


if __name__ == "__main__":
    main()
