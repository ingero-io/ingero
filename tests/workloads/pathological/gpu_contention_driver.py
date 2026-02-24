#!/usr/bin/env python3
"""
GPU contention / time-slicing driver — Tier 3 pathological workload.

Launches multiple processes that all compete for the same GPU. Each process
runs a tight CUDA kernel loop. With NVIDIA MPS disabled (default), processes
time-slice on the GPU, creating measurable cudaLaunchKernel latency variance.

Ingero detects this via CUDA API timing patterns: per-process cudaLaunchKernel
p99 variance reveals GPU context switch overhead (~0.8ms per switch).

Usage:
    python3 gpu_contention_driver.py                  # 3 workers (default)
    python3 gpu_contention_driver.py --workers 5      # 5 workers
    python3 gpu_contention_driver.py --duration 30    # run for 30 seconds
    python3 gpu_contention_driver.py --matrix-size 2048  # larger matrices

While running, use Ingero to observe:
    sudo ingero trace          # see per-PID cudaLaunchKernel variance
    ingero explain             # auto-detect GPU contention pattern
"""

import argparse
import multiprocessing
import os
import signal
import sys
import time

def gpu_worker(worker_id: int, matrix_size: int, duration: int):
    """Run a tight CUDA kernel loop on the shared GPU."""
    try:
        import torch
    except ImportError:
        print(f"Worker {worker_id}: PyTorch not installed, skipping", flush=True)
        return

    if not torch.cuda.is_available():
        print(f"Worker {worker_id}: No CUDA GPU available, skipping", flush=True)
        return

    device = torch.device("cuda:0")
    print(f"Worker {worker_id} (PID {os.getpid()}): starting on {torch.cuda.get_device_name(0)}", flush=True)

    # Pre-allocate matrices.
    a = torch.randn(matrix_size, matrix_size, device=device)
    b = torch.randn(matrix_size, matrix_size, device=device)
    host_buf = torch.empty(matrix_size, matrix_size, dtype=torch.float32, pin_memory=True)

    start = time.time()
    iterations = 0

    while time.time() - start < duration:
        # Tight loop: matmul + sync forces GPU context switches between workers.
        c = torch.matmul(a, b)
        # D2H transfer every 10th iteration — exercises cudaMemcpy / cuMemcpy
        # under contention, adding diversity to traced CUDA ops.
        if iterations % 10 == 0:
            host_buf.copy_(c[:matrix_size, :matrix_size])
        torch.cuda.synchronize()
        iterations += 1

    elapsed = time.time() - start
    rate = iterations / elapsed if elapsed > 0 else 0
    print(f"Worker {worker_id} (PID {os.getpid()}): {iterations} iterations in {elapsed:.1f}s ({rate:.0f} iter/s)", flush=True)


def main():
    parser = argparse.ArgumentParser(description="GPU contention driver for Ingero testing")
    parser.add_argument("--workers", type=int, default=3, help="Number of competing GPU workers")
    parser.add_argument("--duration", type=int, default=20, help="Duration in seconds")
    parser.add_argument("--matrix-size", type=int, default=1024, help="Matrix dimension for matmul")
    args = parser.parse_args()

    print(f"GPU Contention Driver — {args.workers} workers, {args.duration}s, matrix {args.matrix_size}x{args.matrix_size}")
    print(f"Parent PID: {os.getpid()}")
    print()

    # Launch worker processes.
    processes = []
    for i in range(args.workers):
        p = multiprocessing.Process(
            target=gpu_worker,
            args=(i, args.matrix_size, args.duration),
        )
        p.start()
        processes.append(p)
        print(f"  Launched worker {i} as PID {p.pid}")

    print(f"\n  All {args.workers} workers running. Use 'sudo ingero trace' to observe GPU contention.\n")

    # Wait for all workers to complete.
    for p in processes:
        p.join()

    print("\nAll workers finished.")


if __name__ == "__main__":
    main()
