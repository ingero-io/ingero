#!/usr/bin/env python3
"""Straggler-harness workload for Fleet E2E tests.

Runs a PyTorch matmul loop on the first visible CUDA device, writing progress
to stdout. On receiving SIGUSR1 it injects degradation (sleep-per-iteration)
that drops kernel-launch throughput below whatever baseline the fleet-push
agent has formed. SIGUSR2 clears the degradation. Optional time-based
triggers let e2e scripts drive the same transitions without signals.

Usage examples
--------------

  # Signal-driven (manual lab runs):
  python3 straggler-harness.py

  # Time-driven for automation: healthy for 180s, degraded for 120s, recover:
  python3 straggler-harness.py --degrade-after 180 --degrade-duration 120

The harness exits on SIGINT / SIGTERM (Ctrl-C or `kill`).
"""

from __future__ import annotations

import argparse
import os
import signal
import sys
import time
from dataclasses import dataclass


@dataclass
class State:
    degrade: bool = False
    sleep_per_iter: float = 0.05
    matrix_size: int = 4096


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def install_signal_handlers(state: State) -> None:
    def _on_usr1(_sig, _frm):
        state.degrade = True
        log(f"SIGUSR1 received; degrading (sleep_per_iter={state.sleep_per_iter}s)")

    def _on_usr2(_sig, _frm):
        state.degrade = False
        log("SIGUSR2 received; restoring healthy throughput")

    signal.signal(signal.SIGUSR1, _on_usr1)
    signal.signal(signal.SIGUSR2, _on_usr2)


def maybe_time_trigger(state: State, start: float, degrade_after: float, degrade_duration: float) -> None:
    if degrade_after <= 0:
        return
    elapsed = time.monotonic() - start
    if not state.degrade and elapsed >= degrade_after:
        state.degrade = True
        log(f"time trigger: degrading at t={elapsed:.1f}s (sleep_per_iter={state.sleep_per_iter}s)")
    elif state.degrade and degrade_duration > 0 and elapsed >= degrade_after + degrade_duration:
        state.degrade = False
        log(f"time trigger: restoring at t={elapsed:.1f}s")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__ or "")
    p.add_argument("--matrix-size", type=int, default=4096, help="square matrix dimension for matmul")
    p.add_argument("--sleep-per-iter", type=float, default=0.05, help="seconds of time.sleep per iter when degraded")
    p.add_argument("--degrade-after", type=float, default=0.0, help="trigger degradation after N seconds (0 = signals only)")
    p.add_argument("--degrade-duration", type=float, default=0.0, help="auto-restore after N degraded seconds (0 = stay degraded)")
    p.add_argument("--log-interval", type=float, default=10.0, help="seconds between progress log lines")
    args = p.parse_args()

    try:
        import torch
    except ImportError:
        print("ERROR: PyTorch is required. Install in the session venv before running.", file=sys.stderr)
        return 2

    if not torch.cuda.is_available():
        print("ERROR: CUDA not available. Run on a GPU instance.", file=sys.stderr)
        return 2

    device = torch.device("cuda:0")
    log(f"harness starting on device={device} ({torch.cuda.get_device_name(0)})")
    log(f"pid={os.getpid()} matrix_size={args.matrix_size} sleep_per_iter={args.sleep_per_iter}s")

    state = State(sleep_per_iter=args.sleep_per_iter, matrix_size=args.matrix_size)
    install_signal_handlers(state)

    x = torch.randn(state.matrix_size, state.matrix_size, device=device)

    start = time.monotonic()
    last_log = start
    iters = 0
    running = True

    def _on_term(_sig, _frm):
        nonlocal running
        running = False
        log("SIGTERM/SIGINT received; exiting after current iter")

    signal.signal(signal.SIGINT, _on_term)
    signal.signal(signal.SIGTERM, _on_term)

    while running:
        y = torch.mm(x, x)
        torch.cuda.synchronize()
        del y
        iters += 1

        maybe_time_trigger(state, start, args.degrade_after, args.degrade_duration)

        if state.degrade:
            # The degradation path reduces cudaLaunchKernel rate because
            # Python-side sleep stalls the dispatch loop. The host CPU + GPU
            # memory signals stay healthy; only the throughput signal drops.
            # This is the cleanest way to produce a detectable straggler
            # signature without also tripping other health signals.
            time.sleep(state.sleep_per_iter)

        now = time.monotonic()
        if now - last_log >= args.log_interval:
            rate = iters / (now - start)
            log(f"iters={iters} avg_rate={rate:.1f}/s degraded={state.degrade}")
            last_log = now

    log(f"exiting after iters={iters}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
