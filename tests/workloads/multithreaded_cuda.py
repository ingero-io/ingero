#!/usr/bin/env python3
"""Multi-threaded cuda workload for exercising the eBPF Python frame
walker's per-thread state lookup.

The walker's find_thread_state() iterates PyThreadState.next, matching
each entry's native_thread_id against the kernel tid of the firing
uprobe. For single-threaded workloads the single-tstate fallback masks
any offset bugs; only a genuinely multi-threaded workload verifies that
TstateNativeThreadID is correct AND that CPython actually populates
native_thread_id for non-main threads on the build under test.

Each worker calls a distinctly-named function that invokes cuda from
its own thread, so the expected walker output is that every cuda event
carries python_frames[0].py_func == the worker's function name. If the
walker returns main()'s frame or another worker's frame, either the
native_tid match is broken or the fallback is incorrectly masking
multiple tstates as "single-threaded".

Usage:
    # Terminal 1 (or the same shell, background with &):
    sudo ingero trace --py-walker=ebpf --json --debug > /tmp/ingero.jsonl

    # Terminal 2 (or foreground after ingero is listening):
    /path/to/python tests/workloads/multithreaded_cuda.py

Assertions (post-run, against /tmp/ingero.jsonl):

  1. Distinct kernel TIDs appear in cuda events — not all events pinned
     to the main thread's tid:
       jq -r '.tid' /tmp/ingero.jsonl | sort -u | wc -l   # expect >= 3

  2. For each worker, at least one cuda event's top py_func matches
     the worker's function name:
       jq -r '.stack[]? | select(.py_func) | .py_func' /tmp/ingero.jsonl \
         | sort -u | grep -E 'worker_(a|b|c)_cuda_loop'

  3. thread_match_found counter > 0 (not just single_thread_fallback):
       grep thread_match_found /tmp/ingero.err
"""

import ctypes
import os
import sys
import threading
import time


# Locate libcudart.so.12 the same way ingero discovers it; prefer the
# system copy (/usr/lib/x86_64-linux-gnu/libcudart.so.12.*) to match
# the inode ingero's uprobes are attached to. Falls back to any
# libcudart.so.12 on LD_LIBRARY_PATH so the script still runs when the
# system copy is absent (e.g., uv distribution with no system CUDA).
CUDART_CANDIDATES = [
    "/usr/lib/x86_64-linux-gnu/libcudart.so.12",
    "/usr/lib/x86_64-linux-gnu/libcudart.so.12.0.146",
    "/usr/local/cuda/lib64/libcudart.so.12",
    "libcudart.so.12",
]


def load_cudart():
    for candidate in CUDART_CANDIDATES:
        try:
            return ctypes.CDLL(candidate)
        except OSError:
            continue
    raise SystemExit("libcudart.so.12 not found. "
                     "Install a CUDA 12 runtime or adjust CUDART_CANDIDATES.")


CUDART = load_cudart()
CUDART.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
CUDART.cudaMalloc.restype = ctypes.c_int
CUDART.cudaFree.argtypes = [ctypes.c_void_p]
CUDART.cudaFree.restype = ctypes.c_int
CUDART.cudaDeviceSynchronize.restype = ctypes.c_int


def _cuda_loop(label: str, iterations: int) -> None:
    """Shared inner body. Each worker wraps this in its own distinctly
    named function so the walker's emitted py_func differs per thread.
    """
    ptr = ctypes.c_void_p()
    for _ in range(iterations):
        CUDART.cudaMalloc(ctypes.byref(ptr), 4096)
        CUDART.cudaDeviceSynchronize()
        CUDART.cudaFree(ptr)


def worker_a_cuda_loop() -> None:
    tid = threading.get_native_id()
    print(f"worker_a native_tid={tid} running on thread {threading.current_thread().name}")
    _cuda_loop("a", 40)


def worker_b_cuda_loop() -> None:
    tid = threading.get_native_id()
    print(f"worker_b native_tid={tid} running on thread {threading.current_thread().name}")
    _cuda_loop("b", 40)


def worker_c_cuda_loop() -> None:
    tid = threading.get_native_id()
    print(f"worker_c native_tid={tid} running on thread {threading.current_thread().name}")
    _cuda_loop("c", 40)


def main() -> int:
    print(f"main pid={os.getpid()} native_tid={threading.get_native_id()}")
    print(f"python {sys.version}")
    # Give ingero time to attach uprobes to our libcudart inode (the
    # HostProcessExec hook races with cudaMalloc from earlier CUDA init).
    time.sleep(3)

    workers = [
        threading.Thread(target=worker_a_cuda_loop, name="worker-a"),
        threading.Thread(target=worker_b_cuda_loop, name="worker-b"),
        threading.Thread(target=worker_c_cuda_loop, name="worker-c"),
    ]
    for t in workers:
        t.start()
    for t in workers:
        t.join()
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
