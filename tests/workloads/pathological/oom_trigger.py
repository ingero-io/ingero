#!/usr/bin/env python3
"""Gradually increase batch size until CUDA OOM.

Starts with a safe batch size and doubles it each round until cudaMalloc
fails with cudaErrorMemoryAllocation. Ingero should show cudaMalloc p99
climbing before the crash.

Exercises: cudaMalloc failure detection, progressive memory pressure
Expected Ingero output: cudaMalloc p99 climbing across rounds, final allocation failure
"""

import argparse
import time
import torch
import torch.nn as nn


def training_step(model, batch_size, device, input_size=1024):
    """Run one forward+backward pass at the given batch size."""
    x = torch.randn(batch_size, input_size, device=device)
    target = torch.randn(batch_size, input_size, device=device)
    output = model(x)
    loss = nn.functional.mse_loss(output, target)
    loss.backward()
    return loss.item()


def main():
    parser = argparse.ArgumentParser(description="CUDA OOM trigger (escalating batch size)")
    parser.add_argument("--start-batch", type=int, default=64, help="Starting batch size")
    parser.add_argument("--max-batch", type=int, default=65536, help="Maximum batch size to try")
    parser.add_argument("--steps-per-round", type=int, default=10, help="Training steps per batch size")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    args = parser.parse_args()

    device = torch.device(args.device)
    vram_gb = torch.cuda.get_device_properties(device).total_memory / 1e9

    print(f"oom_trigger: device={device}, GPU={torch.cuda.get_device_name(device)}")
    print(f"  VRAM: {vram_gb:.1f} GB")
    print(f"  Start batch: {args.start_batch}, doubling until OOM or {args.max_batch}")
    print()

    # A model that uses significant memory per sample
    model = nn.Sequential(
        nn.Linear(1024, 4096),
        nn.ReLU(),
        nn.Linear(4096, 4096),
        nn.ReLU(),
        nn.Linear(4096, 1024),
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters())

    batch_size = args.start_batch
    round_num = 0

    while batch_size <= args.max_batch:
        round_num += 1
        vram_used = torch.cuda.memory_allocated(device) / 1e9
        vram_reserved = torch.cuda.memory_reserved(device) / 1e9

        print(f"Round {round_num}: batch_size={batch_size} "
              f"(VRAM: {vram_used:.1f}GB used, {vram_reserved:.1f}GB reserved)")

        try:
            t0 = time.time()
            for step in range(args.steps_per_round):
                optimizer.zero_grad()
                loss = training_step(model, batch_size, device)
                optimizer.step()

            dt = time.time() - t0
            print(f"  OK: {args.steps_per_round} steps in {dt:.1f}s, loss={loss:.4f}")

            # Clear cache before next round
            torch.cuda.empty_cache()

        except torch.cuda.OutOfMemoryError as e:
            vram_used = torch.cuda.memory_allocated(device) / 1e9
            print(f"  OOM at batch_size={batch_size}!")
            print(f"  VRAM at OOM: {vram_used:.1f}GB allocated")
            print(f"  Error: {str(e)[:200]}")
            torch.cuda.empty_cache()
            break

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM at batch_size={batch_size}!")
                print(f"  Error: {str(e)[:200]}")
                torch.cuda.empty_cache()
                break
            raise

        batch_size *= 2

    print(f"\noom_trigger complete.")
    if batch_size <= args.max_batch:
        print(f"OOM hit at batch_size={batch_size}.")
        print("Check Ingero — cudaMalloc p99 should show progressive increase before OOM.")
    else:
        print(f"Reached max batch size {args.max_batch} without OOM.")


if __name__ == "__main__":
    main()
