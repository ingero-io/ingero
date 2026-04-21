#!/usr/bin/env python3
"""Per-version walker matrix workload.

Drives cuda via either torch or ctypes (argv[1] picks), running a fixed
outer -> middle -> inner call chain so the walker's emitted py_func set
is deterministic across Python versions and the matrix harness can
assert the same expected frames every run.

Usage (via the walker-matrix harness; for manual runs see below):
    python py_walker_matrix.py {torch|ctypes} <pre-run-sleep-secs>

The harness starts `ingero trace` first, then runs this workload, then
reads the JSON event stream to assert the chain is present.

Expected walker output (post-validation, once the walker is correct):

  On 3.10..3.14: at least one cuda event's stack contains the full
    chain: inner -> middle -> outer -> <module>
  On 3.9: at least one frame emitted (single-thread fallback path;
    chain depth depends on whether native_thread_id behaves as expected
    on the build under test).

The three worker function names are deliberately prefixed `mx_` so they
do not collide with any other test workload's function names and are
easy to grep for.
"""

import ctypes
import os
import sys
import time


CUDART_CANDIDATES = [
    "/usr/lib/x86_64-linux-gnu/libcudart.so.12",
    "/usr/lib/x86_64-linux-gnu/libcudart.so.12.0.146",
    "/usr/local/cuda/lib64/libcudart.so.12",
    "libcudart.so.12",
]


def _load_cudart():
    for c in CUDART_CANDIDATES:
        try:
            return ctypes.CDLL(c)
        except OSError:
            continue
    raise SystemExit("libcudart.so.12 not found")


def _make_ctypes_driver():
    cr = _load_cudart()
    cr.cudaMalloc.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_size_t]
    cr.cudaMalloc.restype = ctypes.c_int
    cr.cudaFree.argtypes = [ctypes.c_void_p]
    cr.cudaFree.restype = ctypes.c_int
    cr.cudaDeviceSynchronize.restype = ctypes.c_int

    def drive():
        ptr = ctypes.c_void_p()
        cr.cudaMalloc(ctypes.byref(ptr), 4096)
        cr.cudaDeviceSynchronize()
        cr.cudaFree(ptr)

    return drive


def _make_torch_driver():
    import torch  # deferred so ctypes mode still works if torch is absent
    x = torch.randn(512, 512, device="cuda")
    y = torch.randn(512, 512, device="cuda")

    def drive():
        z = x @ y
        torch.cuda.synchronize()
        return z.sum().item()

    return drive


def mx_inner(drive, iterations):
    for _ in range(iterations):
        drive()


def mx_middle(drive, iterations):
    mx_inner(drive, iterations)


def mx_outer(drive, iterations):
    mx_middle(drive, iterations)


def main() -> int:
    mode = sys.argv[1] if len(sys.argv) > 1 else "ctypes"
    wait = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    print(f"pid={os.getpid()} python={sys.version_info.major}.{sys.version_info.minor} mode={mode}")
    sys.stdout.flush()

    if mode == "torch":
        try:
            drive = _make_torch_driver()
        except Exception as e:
            print(f"torch driver unavailable ({e}), falling back to ctypes", file=sys.stderr)
            drive = _make_ctypes_driver()
            mode = "ctypes-fallback"
    else:
        drive = _make_ctypes_driver()

    print(f"mode_final={mode} ready, sleeping {wait}s for ingero attach")
    sys.stdout.flush()
    time.sleep(wait)

    mx_outer(drive, iterations=40)
    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
