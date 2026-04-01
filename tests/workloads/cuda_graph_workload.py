#!/usr/bin/env python3
"""CUDA Graph validation workload for Ingero Story 6.4.

Uses torch.compile(mode="reduce-overhead") to trigger CUDA Graph capture,
instantiate, and replay. Generates high-frequency GraphLaunch events for
overhead measurement and pipeline validation.

Usage:
    python cuda_graph_workload.py [--duration 30] [--batch-size 64] [--report]

Requires: PyTorch 2.x with CUDA support.
"""

import argparse
import sys
import time

import torch
import torch.nn as nn


def detect_gpu():
    """Auto-detect GPU and return VRAM in MB."""
    if not torch.cuda.is_available():
        print("ERROR: No CUDA GPU available", file=sys.stderr)
        sys.exit(1)
    props = torch.cuda.get_device_properties(0)
    vram_mb = props.total_memory // (1024 * 1024)
    print(f"GPU: {props.name} ({vram_mb} MB VRAM)")
    return vram_mb


def create_model(hidden_size=1024):
    """Create a small model suitable for graph capture."""
    model = nn.Sequential(
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, hidden_size),
    ).cuda()
    return model


def warmup_compile(compiled_model, x, warmup_iters=10):
    """Warmup: triggers graph capture and instantiate."""
    print(f"Warming up ({warmup_iters} iters, triggers graph capture)...")
    with torch.no_grad():
        for _ in range(warmup_iters):
            _ = compiled_model(x)
    torch.cuda.synchronize()
    print("Warmup complete (graphs captured and instantiated)")


def benchmark(compiled_model, x, duration_sec):
    """Run forward passes and measure throughput."""
    torch.cuda.synchronize()
    start = time.perf_counter()
    iters = 0

    with torch.no_grad():
        while True:
            _ = compiled_model(x)
            iters += 1

            # Check time every 100 iterations to reduce overhead.
            if iters % 100 == 0:
                elapsed = time.perf_counter() - start
                if elapsed >= duration_sec:
                    break

    torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    return iters, elapsed


def main():
    parser = argparse.ArgumentParser(description="CUDA Graph validation workload")
    parser.add_argument("--duration", type=int, default=30, help="Run duration in seconds")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size")
    parser.add_argument("--hidden-size", type=int, default=1024, help="Hidden layer size")
    parser.add_argument("--report", action="store_true", help="Print machine-readable report")
    args = parser.parse_args()

    vram_mb = detect_gpu()
    print(f"Config: batch_size={args.batch_size}, hidden={args.hidden_size}, duration={args.duration}s")
    print(f"PID: {__import__('os').getpid()}")

    model = create_model(args.hidden_size)
    compiled = torch.compile(model, mode="reduce-overhead")
    x = torch.randn(args.batch_size, args.hidden_size, device="cuda")

    warmup_compile(compiled, x)

    print(f"\nRunning benchmark for {args.duration}s (GraphLaunch on every forward pass)...")
    iters, elapsed = benchmark(compiled, x, args.duration)

    rate = iters / elapsed
    print(f"\nResults:")
    print(f"  Iterations: {iters}")
    print(f"  Elapsed: {elapsed:.2f}s")
    print(f"  Throughput: {rate:.1f} iters/sec")
    print(f"  Approx GraphLaunch rate: {rate:.0f}/sec")

    if args.report:
        print(f"\nREPORT: iters={iters} elapsed={elapsed:.3f} rate={rate:.1f}")


if __name__ == "__main__":
    main()
