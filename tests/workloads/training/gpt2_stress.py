#!/usr/bin/env python3
"""GPT-2 stress test — 5-phase escalation for Ingero causal chain validation.

Runs a real GPT-2 (124M) fine-tuning workload through increasingly aggressive
scenarios designed to trigger every category of Ingero causal chain:

  Phase 1: Baseline        — small batch, normal training (clean reference)
  Phase 2: Memory pressure — escalate batch size + sequence length toward OOM
  Phase 3: CPU contention  — stress-ng alongside training (sched_switch storms)
  Phase 4: Multi-process   — 2+ GPT-2 processes on same GPU (noisy neighbor)
  Phase 5: I/O pressure    — large checkpoint saves during training (block I/O stalls)

Usage:
  python3 gpt2_stress.py                          # all 5 phases
  python3 gpt2_stress.py --phases 1 2             # baseline + memory only
  python3 gpt2_stress.py --phases 3 --stress-cpus 8
  python3 gpt2_stress.py --phases 4 --workers 4

Expected Ingero output per phase:
  1: Stable cudaLaunchKernel/cudaStreamSync, no anomalies
  2: cudaMalloc p99 climbing, mm_page_alloc, possible OOM kill chain
  3: sched_switch latency → cudaStreamSync stalls → causal chain
  4: Per-process contention, cgroup sched latency, noisy neighbor chain
  5: block_rq_complete latency → cudaStreamSync gaps → IO→GPU chain

VRAM: 8-24GB depending on phase | Time: ~15-30 minutes for all phases
"""

import argparse
import multiprocessing
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time

import torch
from torch.utils.data import DataLoader
from transformers import GPT2LMHeadModel, GPT2Tokenizer, DataCollatorForLanguageModeling
from datasets import load_dataset


# ── Shared setup ────────────────────────────────────────────────────────────

def banner(phase, title, detail=""):
    line = f"{'='*60}"
    print(f"\n{line}")
    print(f"  PHASE {phase}: {title}")
    if detail:
        print(f"  {detail}")
    print(f"{line}\n")


def gpu_stats(device):
    used = torch.cuda.memory_allocated(device) / 1e9
    reserved = torch.cuda.memory_reserved(device) / 1e9
    total = torch.cuda.get_device_properties(device).total_memory / 1e9
    print(f"  VRAM: {used:.1f}GB used / {reserved:.1f}GB reserved / {total:.1f}GB total")


def load_model_and_data(device, max_length=256, batch_size=8):
    """Load GPT-2 + WikiText-2. Returns (model, optimizer, dataloader, tokenizer)."""
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    dataset = dataset.filter(lambda x: len(x["text"].strip()) > 0)

    def tokenize_fn(examples):
        return tokenizer(examples["text"], truncation=True,
                         max_length=max_length, padding="max_length")

    tokenized = dataset.map(tokenize_fn, batched=True,
                            remove_columns=dataset.column_names)
    tokenized.set_format("torch", columns=["input_ids", "attention_mask"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    dataloader = DataLoader(tokenized, batch_size=batch_size, shuffle=True,
                            collate_fn=collator, num_workers=2, pin_memory=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.01)
    return model, optimizer, dataloader, tokenizer


def train_steps(model, optimizer, dataloader, device, steps, label=""):
    """Run N training steps. Returns (avg_loss, elapsed_seconds, steps_done)."""
    model.train()
    running_loss = 0.0
    t0 = time.time()

    for step, batch in enumerate(dataloader):
        if step >= steps:
            break

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()

        if (step + 1) % 25 == 0:
            avg = running_loss / (step + 1)
            elapsed = time.time() - t0
            rate = (step + 1) * dataloader.batch_size / elapsed
            prefix = f"  [{label}]" if label else "  "
            print(f"{prefix} step {step+1}/{steps}  loss={avg:.3f}  "
                  f"{rate:.0f} samples/sec")

    steps_done = min(step + 1, steps) if steps > 0 else 0
    dt = time.time() - t0
    avg_loss = running_loss / max(steps_done, 1)
    return avg_loss, dt, steps_done


# ── Phase 1: Baseline ──────────────────────────────────────────────────────

def phase_baseline(device, steps=50):
    banner(1, "BASELINE", "batch=4, seq_len=256, clean training")
    gpu_stats(device)

    model, optimizer, dataloader, _ = load_model_and_data(
        device, max_length=256, batch_size=4)
    avg_loss, dt, done = train_steps(model, optimizer, dataloader, device,
                                     steps, label="baseline")
    print(f"\n  Baseline complete: {done} steps in {dt:.1f}s, loss={avg_loss:.3f}")
    gpu_stats(device)

    del model, optimizer, dataloader
    torch.cuda.empty_cache()


# ── Phase 2: Memory pressure ───────────────────────────────────────────────

def phase_memory_pressure(device, steps_per_round=30):
    banner(2, "MEMORY PRESSURE",
           "Escalate batch size + seq length toward OOM")

    configs = [
        (16, 256, "warm-up"),
        (32, 256, "medium batch"),
        (64, 256, "large batch"),
        (32, 512, "long sequences"),
        (64, 512, "large batch + long seq"),
        (32, 1024, "very long sequences"),
        (64, 1024, "push toward OOM"),
    ]

    for batch_size, seq_len, desc in configs:
        print(f"\n  --- batch={batch_size}, seq_len={seq_len} ({desc}) ---")
        gpu_stats(device)

        try:
            model, optimizer, dataloader, _ = load_model_and_data(
                device, max_length=seq_len, batch_size=batch_size)
            avg_loss, dt, done = train_steps(
                model, optimizer, dataloader, device, steps_per_round,
                label=f"b{batch_size}/s{seq_len}")
            print(f"  OK: {done} steps in {dt:.1f}s, loss={avg_loss:.3f}")

        except torch.cuda.OutOfMemoryError:
            print(f"  OOM at batch={batch_size}, seq_len={seq_len}")
            print("  (This is expected — Ingero should show cudaMalloc p99 "
                  "climbing before this point)")
            torch.cuda.empty_cache()
            break

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"  OOM at batch={batch_size}, seq_len={seq_len}")
                torch.cuda.empty_cache()
                break
            raise

        finally:
            # Clean up between rounds
            for name in ("model", "optimizer", "dataloader"):
                if name in dir():
                    exec(f"del {name}")
            torch.cuda.empty_cache()

    gpu_stats(device)


# ── Phase 3: CPU contention ────────────────────────────────────────────────

def phase_cpu_contention(device, stress_cpus=0, steps=100):
    ncpus = stress_cpus or os.cpu_count() or 4
    banner(3, "CPU CONTENTION",
           f"stress-ng with {ncpus} CPU workers alongside GPT-2 training")

    if not shutil.which("stress-ng"):
        print("  stress-ng not found — install with: sudo apt-get install -y stress-ng")
        print("  Skipping phase 3.")
        return

    # Load model before starting stress (so model loading isn't contended)
    model, optimizer, dataloader, _ = load_model_and_data(
        device, max_length=256, batch_size=8)
    gpu_stats(device)

    # Start stress-ng
    print(f"  Starting stress-ng ({ncpus} matrix workers)...")
    stress_proc = subprocess.Popen(
        ["stress-ng", "--cpu", str(ncpus), "--cpu-method", "matrixprod",
         "--timeout", "300s"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"  stress-ng PID: {stress_proc.pid}")

    try:
        avg_loss, dt, done = train_steps(
            model, optimizer, dataloader, device, steps,
            label="cpu-contention")
        print(f"\n  Training under contention: {done} steps in {dt:.1f}s, "
              f"loss={avg_loss:.3f}")
        print("  (Compare timing with Phase 1 baseline — expect slower samples/sec)")
    finally:
        stress_proc.terminate()
        stress_proc.wait()
        print("  stress-ng stopped.")

    gpu_stats(device)
    del model, optimizer, dataloader
    torch.cuda.empty_cache()


# ── Phase 4: Multi-process (noisy neighbor) ─────────────────────────────────

def _worker_train(worker_id, device_str, steps):
    """Subprocess entry point: train GPT-2 independently."""
    device = torch.device(device_str)
    print(f"  [worker-{worker_id}] PID={os.getpid()} starting on {device}")

    model, optimizer, dataloader, _ = load_model_and_data(
        device, max_length=256, batch_size=8)
    avg_loss, dt, done = train_steps(
        model, optimizer, dataloader, device, steps,
        label=f"worker-{worker_id}")
    print(f"  [worker-{worker_id}] done: {done} steps in {dt:.1f}s, "
          f"loss={avg_loss:.3f}")


def phase_multi_process(device, workers=3, steps_per_worker=60):
    banner(4, "MULTI-PROCESS GPU CONTENTION",
           f"{workers} GPT-2 processes on same GPU (noisy neighbor)")
    gpu_stats(device)

    device_str = str(device)
    procs = []

    for i in range(workers):
        p = multiprocessing.Process(
            target=_worker_train, args=(i, device_str, steps_per_worker))
        p.start()
        procs.append(p)
        print(f"  Launched worker-{i} (PID={p.pid})")

    for p in procs:
        p.join()

    print(f"\n  All {workers} workers finished.")
    print("  Use: ./bin/ingero explain --per-process --since 120s")
    gpu_stats(device)
    torch.cuda.empty_cache()


# ── Phase 5: I/O pressure (checkpoint writes) ──────────────────────────────

def phase_io_pressure(device, steps=100, save_every=20):
    banner(5, "I/O PRESSURE",
           f"Save full checkpoint every {save_every} steps during training")

    tmpdir = tempfile.mkdtemp(prefix="ingero_ckpt_")
    print(f"  Checkpoint dir: {tmpdir}")

    model, optimizer, dataloader, tokenizer = load_model_and_data(
        device, max_length=256, batch_size=8)
    model.train()
    gpu_stats(device)

    running_loss = 0.0
    t0 = time.time()

    for step, batch in enumerate(dataloader):
        if step >= steps:
            break

        input_ids = batch["input_ids"].to(device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(device, non_blocking=True)
        labels = input_ids.clone()

        outputs = model(input_ids=input_ids, attention_mask=attention_mask,
                        labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
        running_loss += loss.item()

        # Save checkpoint — triggers block I/O while GPU waits
        if (step + 1) % save_every == 0:
            ckpt_path = os.path.join(tmpdir, f"ckpt_step_{step+1}")
            save_t0 = time.time()
            model.save_pretrained(ckpt_path)
            tokenizer.save_pretrained(ckpt_path)
            save_dt = time.time() - save_t0
            avg = running_loss / (step + 1)
            print(f"  step {step+1}/{steps}  loss={avg:.3f}  "
                  f"checkpoint saved ({save_dt:.1f}s)")

    dt = time.time() - t0
    steps_done = min(step + 1, steps)
    avg_loss = running_loss / max(steps_done, 1)
    print(f"\n  I/O pressure complete: {steps_done} steps in {dt:.1f}s, "
          f"loss={avg_loss:.3f}")
    print(f"  Checkpoints at: {tmpdir}")
    print("  (Ingero should show block_rq_complete latency correlated with "
          "cudaStreamSync gaps)")

    gpu_stats(device)
    del model, optimizer, dataloader
    torch.cuda.empty_cache()

    # Cleanup checkpoints
    shutil.rmtree(tmpdir, ignore_errors=True)
    print("  Checkpoints cleaned up.")


# ── Main ────────────────────────────────────────────────────────────────────

ALL_PHASES = {
    1: ("Baseline", phase_baseline),
    2: ("Memory pressure", phase_memory_pressure),
    3: ("CPU contention", phase_cpu_contention),
    4: ("Multi-process", phase_multi_process),
    5: ("I/O pressure", phase_io_pressure),
}


def main():
    parser = argparse.ArgumentParser(
        description="GPT-2 stress test for Ingero causal chain validation")
    parser.add_argument("--phases", type=int, nargs="+",
                        default=[1, 2, 3, 4, 5],
                        help="Which phases to run (default: all)")
    parser.add_argument("--device", default="cuda:0", help="CUDA device")
    parser.add_argument("--stress-cpus", type=int, default=0,
                        help="CPU workers for phase 3 (0=all cores)")
    parser.add_argument("--workers", type=int, default=3,
                        help="GPU processes for phase 4")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"gpt2_stress: device={device}, "
          f"GPU={torch.cuda.get_device_name(device)}")
    print(f"  Phases: {args.phases}")
    print(f"  PID: {os.getpid()}")
    gpu_stats(device)

    t_total = time.time()

    for phase_num in args.phases:
        if phase_num not in ALL_PHASES:
            print(f"\n  Unknown phase {phase_num}, skipping.")
            continue

        name, fn = ALL_PHASES[phase_num]
        t_phase = time.time()

        if phase_num == 3:
            fn(device, stress_cpus=args.stress_cpus)
        elif phase_num == 4:
            fn(device, workers=args.workers)
        else:
            fn(device)

        print(f"\n  Phase {phase_num} ({name}) elapsed: "
              f"{time.time() - t_phase:.1f}s")

    total = time.time() - t_total
    print(f"\n{'='*60}")
    print(f"  ALL PHASES COMPLETE — total: {total:.1f}s")
    print(f"{'='*60}")
    print(f"\nIngero commands to analyze:")
    print(f"  ./bin/ingero explain --debug --since {int(total)+10}s")
    print(f"  ./bin/ingero explain --per-process --since {int(total)+10}s")
    print(f'  ./bin/ingero query "SELECT op, count(*), '
          f'avg(duration_ns)/1e6 as avg_ms FROM events GROUP BY op"')


if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
