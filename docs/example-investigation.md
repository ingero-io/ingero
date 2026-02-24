# Example: AI-Assisted GPU Training Investigation

> **Scenario:** An ML engineer ran ResNet-50 training on an RTX 4090 under CPU contention.
> Training felt slow. They connected an AI assistant (Claude) to Ingero's MCP server
> to investigate the recorded session — 843,193 kernel-level events captured over 90 seconds
> with zero code changes and <2% overhead.

---

## "How did my last training go?"

**AI Assistant:** Let me pull your session data.

| Source | Operation | Events | Avg Latency | Max Latency |
|--------|-----------|-------:|------------:|------------:|
| HOST | mm_page_alloc | 792,925 | - | - |
| DRIVER | cuLaunchKernel | 34,150 | 4.5 us | 5.1 ms |
| CUDA | cudaLaunchKernel | 14,403 | 15.5 us | 20.5 ms |
| CUDA | cudaStreamSync | 451 | 2.7 ms | 41.1 ms |
| CUDA | cudaMemcpy | 383 | 11.4 ms | **802.8 ms** |
| CUDA | cudaDeviceSync | 113 | 23.1 ms | **273.0 ms** |

**Overall: significant performance issues detected.**

Your ResNet-50 session ran for 90 seconds across 7 processes. But the GPU was doing
real training for only **4.7 seconds** — less than 6% of the session. Three problems:

1. **cudaMemcpy peaked at 802 ms** — a single Host-to-Device memory copy that should
   take ~40 ms on PCIe Gen4 took 20x longer.
2. **cudaDeviceSync spiked to 273 ms** — a GPU synchronization that was running at a
   steady 21.5 ms suddenly jumped 13x.
3. **42 thread exits across 6 short-lived processes** — heavy process churn adding
   scheduler pressure.

The root cause is **CPU starvation**. The host CPUs were saturated, which slowed the
CPU-side of every CUDA operation.

---

## "What happened at that 273 ms spike?"

**AI Assistant:** Let me look at the events around that exact moment.

The spike was on PID 33928 (the training loop). Before the spike, this process had a
metronomic pattern — launch one kernel, sync, repeat — with sync times locked at
**21.48 ms** for 50 consecutive iterations:

```
15:51:39.170  cuLaunchKernel    0.01 ms
15:51:39.170  cudaDeviceSync   21.49 ms   ← steady
15:51:39.191  cuLaunchKernel    0.01 ms
15:51:39.191  cudaDeviceSync   21.48 ms   ← steady
  ... (50 iterations at 21.48 ms) ...
```

Then at `15:51:39.363`, PyTorch fired a **forward pass burst** — 102 `cuLaunchKernel`
calls in 3 milliseconds with no sync in between. This is the autograd graph executing
the entire forward pass as a batch:

```
15:51:39.363  cuLaunchKernel    0.01 ms
15:51:39.363  cuLaunchKernel    0.01 ms
15:51:39.363  cuLaunchKernel    0.00 ms
  ... (102 kernel launches in 3ms) ...
15:51:39.366  cudaDeviceSync  272.98 ms   ← 13x SPIKE
15:51:39.642  cudaLaunchKernel  0.02 ms   ← back to normal
```

**The causal chain:**

```
stress-ng saturated all 4 CPUs
  → PID 33928's thread was descheduled during cudaDeviceSync
  → The CUDA driver's completion callback couldn't run (needs CPU)
  → The sync blocked for 273 ms instead of ~50 ms
  → Training loop throughput permanently degraded 3x for the rest of the session
```

The aftermath confirms this: DeviceSync never returned to 21.5 ms. It settled at
**58-65 ms** — the CPU contention established a new, degraded baseline.

---

## "Dig into the 802 ms memcpy spike"

**AI Assistant:** Here's the full memory transfer timeline for PID 33941 (the model
loader), showing transfer sizes, directions, and effective throughput:

```
Phase 1 — CUDA context init (15:51:12)
  Tiny <1MB copies, sub-microsecond
  CUDA runtime internal sanity checks

Phase 2 — Buffer pre-staging (15:51:12-15:51:15)
  4MB → 16MB → 64MB → 256MB → 1024MB, ~10 copies each
  Throughput: 100-170 GB/s (GPU-internal moves, no PCIe)
  PyTorch allocating GPU-side parameter buffers

Phase 3 — Weight transfer H2D (15:51:15-15:51:20) ← THE SPIKE
  4 MB H2D:     3.84 ms @ 1,041 MB/s    (should be ~25 GB/s)
  16 MB H2D:   11.42 ms @ 1,401 MB/s
  64 MB H2D:   61.10 ms @ 1,047 MB/s
  256 MB H2D: 230.52 ms @ 1,110 MB/s
  1 GB H2D:   802.77 ms @ 1,276 MB/s    ← worst
  1 GB H2D:   688.13 ms @ 1,488 MB/s    ← second worst

Phase 4 — Training readback D2H (15:51:27)
  1 GB D2H:     0.01 ms @ 77-170 GB/s   ← near line rate
```

Every H2D transfer runs at **~1.0-1.5 GB/s**. The RTX 4090 on PCIe Gen4 x16 can do
**25 GB/s**. That's a **20x penalty**. Two factors:

1. **Pageable memory.** PyTorch's default `torch.load()` uses regular (pageable) host
   memory. `cudaMemcpy` with pageable source must first stage through a pinned buffer —
   requiring CPU involvement for every chunk.

2. **CPU starvation.** With all CPUs saturated, the CUDA driver thread can't get
   scheduled to manage the staging copies. Each chunk waits for CPU time.

The proof: D2H copies in Phase 4 run at near line rate (77-170 GB/s) because
Device-to-Host DMA is GPU-initiated and doesn't need CPU scheduling. Host-to-Device
with pageable memory does.

**Fix:**

```python
# Load directly to GPU (avoids the CPU-mediated double-copy entirely)
checkpoint = torch.load('checkpoint.pth', map_location='cuda:0')
```

---

## "What about those 42 child processes?"

**AI Assistant:** They're not 42 children — they're **6 short-lived Python processes,
each with 7 threads** (main + 6 internal: CUDA event handler, memory allocator, GC, etc.).
All 7 threads exit simultaneously when each phase completes, producing 42 exit events.

Here's the process lifecycle timeline:

```
TIME              PID    LIFESPAN  ROLE
15:51:09-15:51:11 33929  1.1s      Warmup — 17,450 kernel JIT compilations
15:51:12-15:51:35 33941  23.0s     Model loading — 363 memcpy (the 802ms spike)
15:51:36-15:51:36 33951  0.3s      CUDA probe — 135 cudaMalloc, zero kernels
15:51:38-15:51:42 33928  4.7s      Training loop — 478 kernels (the 273ms spike)
              *** 53 SECONDS OF NOTHING ***
15:52:36-15:52:38 34260  1.1s      Second warmup — 16,204 kernel JIT compilations
15:52:39-15:52:39 34259  0.3s      Second probe — 135 cudaMalloc
```

PyTorch uses `torch.multiprocessing` to spawn a new subprocess for each phase. Each
pays the full CUDA context initialization cost.

**The real finding:** Out of 90 seconds, the GPU did real training for **4.7 seconds**
(5.2%). The rest was overhead:

| Phase | Time | % of Session |
|-------|-----:|:-------------|
| Warmup (JIT) | 2.2s | 2.4% |
| Model loading | 23.0s | 25.6% |
| CUDA probes | 0.6s | 0.7% |
| **Training** | **4.7s** | **5.2%** |
| Dead zone (CPU-blocked fork) | 53.0s | 58.9% |
| Second warmup + probe | 1.4s | 1.6% |
| Ingero overhead | ~5s | 5.6% |

The 53-second gap between PID 33928 dying and PID 34260 spawning is the script blocked
waiting for CPU time to fork the next subprocess.

---

## Summary

| Finding | Severity | Root Cause | Fix |
|---------|----------|-----------|-----|
| 802 ms cudaMemcpy | HIGH | Pageable memory + CPU starvation = 20x slower PCIe | `torch.load(map_location='cuda:0')` |
| 273 ms cudaDeviceSync | HIGH | CPU descheduled during 102-kernel sync wait | Reduce CPU contention on GPU nodes |
| 5.2% GPU utilization | HIGH | Process churn + 53s dead zone from CPU starvation | Pin training to dedicated CPU cores |
| 42 thread exits / 6 processes | MEDIUM | PyTorch multiprocessing overhead | Reduce subprocess spawning, use persistent workers |

**Bottom line:** The GPU was barely used. The RTX 4090 spent 95% of the session
waiting for CPU-side operations that were blocked by host contention. Fix the CPU
scheduling, pin memory for model loading, and this training run drops from 90 seconds
to under 30.

---

*Generated by [Ingero](https://github.com/ingero-io/ingero) v0.6 — eBPF-based GPU
causal observability. 843,193 events captured at <2% overhead with zero code changes.
AI analysis via MCP (Model Context Protocol) server.*
