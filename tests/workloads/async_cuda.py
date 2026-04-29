#!/usr/bin/env python3
"""Generator- and coroutine-based cuda workload for verifying the eBPF
Python frame walker handles non-standard frame ownership correctly.

CPython allocates _PyInterpreterFrame in several ways (the `owner` byte
at offset 70 of the frame struct identifies which):

    FRAME_OWNED_BY_THREAD        0   standard chunk-allocated frame
    FRAME_OWNED_BY_GENERATOR     1   embedded in a PyGenObject / PyCoroObject
    FRAME_OWNED_BY_FRAME_OBJECT  2   standalone, owned by a PyFrameObject
    FRAME_OWNED_BY_CSTACK        3   C-call entry frame on the C stack

For the walker, the important fact is that owner only describes who
allocated the storage, NOT whether the frame is in the
`tstate.current_frame -> previous` chain. A currently-executing
generator or coroutine has its embedded frame LINKED into the chain via
its `previous` pointer exactly like a normal frame; only a SUSPENDED
generator's frame is detached, and no cuda can fire from a suspended
generator by definition. So the existing walker should produce correct
frames for both patterns without any special-case code.

This workload exists to verify that empirically. Two scenarios:

  1. Plain-generator variant: gen_outer() yields from gen_middle()
     which yields from gen_inner() which calls cuda. Walker output
     should be: inner -> middle -> outer -> <module>.

  2. Coroutine variant: coro_outer() awaits coro_middle() awaits
     coro_inner() which calls cuda. The asyncio event loop drives it.
     Walker output should include: inner -> middle -> outer, plus
     event-loop frames above that.

Usage:
    # Terminal 1:
    sudo ingero trace --py-walker=ebpf --json --debug > /tmp/ingero.jsonl

    # Terminal 2:
    /path/to/python tests/workloads/async_cuda.py

Assertions (post-run against /tmp/ingero.jsonl):

  1. At least one cuda event's stack contains "gen_inner":
       jq -r '.stack[]? | select(.py_func) | .py_func' /tmp/ingero.jsonl \
         | grep -q '^gen_inner$'

  2. At least one cuda event's stack contains "coro_inner":
       jq -r '.stack[]? | select(.py_func) | .py_func' /tmp/ingero.jsonl \
         | grep -q '^coro_inner$'

  3. The three-frame generator chain is present (inner/middle/outer
     all appear at least once, from a generator-driven cuda event):
       jq -r '.stack[]? | select(.py_func) | .py_func' /tmp/ingero.jsonl \
         | sort -u | grep -E 'gen_(inner|middle|outer)' | wc -l  # expect 3
"""

import asyncio
import ctypes
import os
import sys
import time
from typing import Iterator


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


def _cuda_call() -> None:
    ptr = ctypes.c_void_p()
    CUDART.cudaMalloc(ctypes.byref(ptr), 4096)
    CUDART.cudaDeviceSynchronize()
    CUDART.cudaFree(ptr)


# Generator chain. Each level is a generator function; when the
# consumer (main) iterates the outer, Python drives the whole stack
# and cuda fires from deep inside gen_inner.
def gen_inner() -> Iterator[int]:
    for i in range(20):
        _cuda_call()
        yield i


def gen_middle() -> Iterator[int]:
    yield from gen_inner()


def gen_outer() -> Iterator[int]:
    yield from gen_middle()


# Coroutine chain. asyncio's run loop drives coro_outer, which awaits
# coro_middle, which awaits coro_inner. cuda fires inside coro_inner
# while all three coroutine frames are live in the current_frame chain.
async def coro_inner() -> None:
    for _ in range(20):
        _cuda_call()
        await asyncio.sleep(0)


async def coro_middle() -> None:
    await coro_inner()


async def coro_outer() -> None:
    await coro_middle()


def main() -> int:
    print(f"pid={os.getpid()} python={sys.version}")
    time.sleep(3)  # let ingero attach

    print("running generator chain")
    for _ in gen_outer():
        pass

    print("running coroutine chain")
    asyncio.run(coro_outer())

    print("done")
    return 0


if __name__ == "__main__":
    sys.exit(main())
