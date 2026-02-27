#!/usr/bin/env python3
"""Stress cudaMalloc/cudaFree with varying sizes and patterns.

Exercises: cudaMalloc, cudaFree
Tests: allocation latency, fragmentation detection, p99 spikes
Expected Ingero output: cudaMalloc stats with rising p99 in fragmentation phase

The --duration flag loops the workload for the specified number of seconds,
ensuring the stress runs long enough for the GPU investigation trace to capture.

Uses both PyTorch allocations (which may be served by the caching allocator)
and direct cudaMalloc/cudaFree via ctypes (which always hit the driver API).
"""

import argparse
import ctypes
import ctypes.util
import time
import torch


def _get_cuda_runtime():
    """Load libcudart.so for direct cudaMalloc/cudaFree calls."""
    path = ctypes.util.find_library("cudart")
    lib = None
    if path:
        lib = ctypes.CDLL(path)
    else:
        # Try common locations (find_library misses LD_LIBRARY_PATH on Python <3.12)
        for candidate in ["libcudart.so", "libcudart.so.12", "libcudart.so.11"]:
            try:
                lib = ctypes.CDLL(candidate)
                break
            except OSError:
                continue
    if lib:
        # Declare argtypes/restype for correct behavior on ARM64 (GH200) and
        # to prevent ctypes from guessing types via default C promotion rules.
        lib.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
        lib.cudaMalloc.restype = ctypes.c_int
        lib.cudaFree.argtypes = [ctypes.c_void_p]
        lib.cudaFree.restype = ctypes.c_int
    return lib


def direct_cuda_alloc_free(lib, sizes_bytes, rounds=3):
    """Call cudaMalloc/cudaFree directly via ctypes, bypassing caching allocator."""
    print(f"  Direct cudaMalloc/cudaFree: {len(sizes_bytes)} sizes x {rounds} rounds")
    ptr = ctypes.c_void_p()
    for _ in range(rounds):
        for size in sizes_bytes:
            ret = lib.cudaMalloc(ctypes.byref(ptr), ctypes.c_size_t(size))
            if ret != 0:
                print(f"    cudaMalloc({size} bytes) failed: error {ret}")
                continue
            if ptr.value:
                ret = lib.cudaFree(ptr)
                if ret != 0:
                    print(f"    cudaFree failed: error {ret}")
                ptr.value = None


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
    parser.add_argument("--duration", type=int, default=0,
                        help="Loop for this many seconds (0 = single pass)")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"alloc_stress: device={device}, GPU={torch.cuda.get_device_name(device)}")
    print(f"  VRAM: {torch.cuda.get_device_properties(device).total_memory / 1e9:.1f} GB")
    if args.duration > 0:
        print(f"  Duration: {args.duration}s (looping)")
    print()

    # Warm up CUDA context
    torch.empty(1, device=device)

    # Load libcudart for direct cudaMalloc/cudaFree (bypasses caching allocator)
    cudart = _get_cuda_runtime()
    if cudart:
        print("  Loaded libcudart for direct cudaMalloc/cudaFree")
    else:
        print("  WARNING: libcudart not found, using PyTorch only (caching allocator)")
    print()

    # Cap sizes to 25% of free VRAM to avoid OOM when running alongside training
    free_mem = torch.cuda.mem_get_info(device)[0]
    max_mb = int(free_mem / (1024 * 1024)) // 4
    sizes = [s for s in [1, 4, 16, 64, 128, 256, 512] if s <= max_mb]
    if not sizes:
        sizes = [1]
    print(f"  Free VRAM: {free_mem / 1e9:.1f} GB, max alloc: {max_mb} MB, sizes: {sizes}")
    print()
    deadline = time.time() + args.duration if args.duration > 0 else 0
    iteration = 0

    while True:
        iteration += 1
        prefix = f"[iter {iteration}] " if args.duration > 0 else ""

        # Direct cudaMalloc/cudaFree (bypasses caching allocator)
        if cudart:
            print(f"{prefix}Direct cudaMalloc/cudaFree")
            t0 = time.time()
            max_direct = max_mb * 1024 * 1024
            direct_sizes = [s for s in [1 * 1024 * 1024, 16 * 1024 * 1024,
                            64 * 1024 * 1024, 256 * 1024 * 1024] if s <= max_direct]
            if not direct_sizes:
                direct_sizes = [1 * 1024 * 1024]
            direct_cuda_alloc_free(cudart, direct_sizes, rounds=2)
            print(f"  Done in {time.time() - t0:.1f}s\n")

        print(f"{prefix}Phase 1: Sequential alloc/free (1MB to {sizes[-1]}MB)")
        t0 = time.time()
        try:
            sequential_alloc(sizes, device)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM: {e}")
            torch.cuda.empty_cache()
        print(f"  Done in {time.time() - t0:.1f}s\n")

        print(f"{prefix}Phase 2: Fragmentation pattern")
        t0 = time.time()
        try:
            fragmentation_pattern(device)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM: {e}")
            torch.cuda.empty_cache()
        print(f"  Done in {time.time() - t0:.1f}s\n")

        print(f"{prefix}Phase 3: Rapid small allocations")
        t0 = time.time()
        try:
            rapid_small_alloc(device)
        except torch.cuda.OutOfMemoryError as e:
            print(f"  OOM: {e}")
            torch.cuda.empty_cache()
        print(f"  Done in {time.time() - t0:.1f}s\n")

        if deadline == 0 or time.time() >= deadline:
            break

    torch.cuda.synchronize()
    print("alloc_stress complete.")


if __name__ == "__main__":
    main()
