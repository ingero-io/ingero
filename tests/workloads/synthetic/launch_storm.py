#!/usr/bin/env python3
"""Launch thousands of kernels back-to-back to stress cudaLaunchKernel.

Exercises: cudaLaunchKernel
Tests: kernel launch latency, launch throughput, queue depth effects
Expected Ingero output: high cudaLaunchKernel count, low p50, p99 spikes under saturation
"""

import argparse
import time
import torch
import torch.nn.functional as F


def tiny_kernel_storm(device, count=10000):
    """Launch many tiny kernels (element-wise ops on small tensors)."""
    print(f"  Tiny kernels: {count} launches")
    x = torch.randn(64, device=device)
    for _ in range(count):
        x = x + 1.0  # each is a tiny CUDA kernel
    torch.cuda.synchronize()
    return x


def matmul_burst(device, size=1024, count=1000):
    """Burst of matrix multiplications (heavier kernels)."""
    print(f"  Matmul burst: {count} x ({size}x{size}) matmuls")
    a = torch.randn(size, size, device=device)
    b = torch.randn(size, size, device=device)
    for _ in range(count):
        c = torch.mm(a, b)
    torch.cuda.synchronize()
    return c


def conv_burst(device, count=500):
    """Burst of convolution kernels (complex kernel launches)."""
    print(f"  Conv burst: {count} conv2d operations")
    x = torch.randn(8, 64, 32, 32, device=device)
    w = torch.randn(128, 64, 3, 3, device=device)
    for _ in range(count):
        _ = F.conv2d(x, w, padding=1)
    torch.cuda.synchronize()


def mixed_kernel_pipeline(device, iterations=500):
    """Simulate a realistic kernel launch sequence: alloc, compute, reduce."""
    print(f"  Mixed pipeline: {iterations} iterations (alloc+matmul+reduce)")
    for _ in range(iterations):
        x = torch.randn(512, 512, device=device)  # alloc + fill kernel
        y = torch.mm(x, x)  # matmul kernel
        z = y.sum()  # reduce kernel
    torch.cuda.synchronize()


def multi_stream_launch(device, num_streams=4, ops_per_stream=1000):
    """Launch kernels across multiple CUDA streams concurrently."""
    print(f"  Multi-stream: {num_streams} streams x {ops_per_stream} ops")
    streams = [torch.cuda.Stream() for _ in range(num_streams)]
    tensors = [torch.randn(256, 256, device=device) for _ in range(num_streams)]

    for _ in range(ops_per_stream):
        for i, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                tensors[i] = torch.mm(tensors[i], tensors[i])

    torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description="CUDA kernel launch storm")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"launch_storm: device={device}, GPU={torch.cuda.get_device_name(device)}")
    print()

    torch.empty(1, device=device)  # warm up

    print("Phase 1: Tiny kernel storm (10k element-wise ops)")
    t0 = time.time()
    tiny_kernel_storm(device)
    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s ({10000/dt:.0f} launches/sec)\n")

    print("Phase 2: Matmul burst (1k 1024x1024 matmuls)")
    t0 = time.time()
    matmul_burst(device)
    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s ({1000/dt:.0f} launches/sec)\n")

    print("Phase 3: Conv burst (500 conv2d ops)")
    t0 = time.time()
    conv_burst(device)
    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s ({500/dt:.0f} launches/sec)\n")

    print("Phase 4: Mixed pipeline (alloc+matmul+reduce)")
    t0 = time.time()
    mixed_kernel_pipeline(device)
    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s ({500/dt:.0f} iterations/sec)\n")

    print("Phase 5: Multi-stream concurrent launch")
    t0 = time.time()
    multi_stream_launch(device)
    dt = time.time() - t0
    print(f"  Done in {dt:.1f}s ({4*1000/dt:.0f} total launches/sec)\n")

    print("launch_storm complete.")


if __name__ == "__main__":
    main()
