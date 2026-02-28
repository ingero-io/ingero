# ML Engineer AI-Assisted GPU Investigation Sessions

> Real investigation sessions using Ingero's MCP server against PyTorch training traces. The AI assistant uses **only** the Ingero MCP tools — no shell access, no direct database queries. This is how it works in production: the assistant connects to the MCP server on the remote GPU machine and answers the engineer's questions.

---

# Session 1: A100 SXM4 (x86_64)

**Setup**: `ingero mcp --db gpu-investigation.db --http :8090`
**DB size**: 7.7 MB (2.64M events traced, 50K stored via stack sampling)
**GPU**: NVIDIA A100 SXM4 40GB, Lambda Labs

---

## Q1: "How did my last training go? Any findings by Ingero?"

**MCP tools used**: `get_trace_stats`, `get_causal_chains`

**Answer**:

Ingero traced **2.64 million** CUDA/system events over your training run on the A100 GPU.

**The GPU itself performed well** — kernel launches were fast (cuLaunchKernel p50 = 43µs), and your memory allocation/free was balanced: **45,992 mallocs vs 45,916 frees** — essentially no memory leak.

**However, Ingero detected 285 causal chain issues (47 HIGH, 238 MEDIUM).** The recurring root cause is:

**CPU contention is throttling your CUDA calls.** The system load was elevated (load avg 10.9) with heavy context switching (600+ sched_switch events per window). This caused:

| CUDA Operation | p50 (normal) | p99 (worst) | Tail ratio | Impact |
|---|---|---|---|---|
| cudaStreamSync | 15µs | 7.4ms | **1,028x** | Worst hit — GPU sync stalls |
| cuLaunchKernel | 43µs | 7.0ms | 163x | Kernel launch delays |
| cudaMalloc | 130µs | 1.0ms | 8x | Allocation slowdowns |
| cuMemAlloc | 121µs | 1.1ms | 8.7x | Driver alloc delays |

**Root cause**: Your training process is getting descheduled by the OS (14,504 context switches observed). Background processes or competing workloads are stealing CPU time, and every time your process gets descheduled mid-CUDA-call, that call's latency spikes 100-1000x.

**Ingero's recommendations**:
1. Pin your training process to dedicated CPU cores with `taskset`
2. Lower priority of background jobs with `nice -n 19`

---

## Q2: "What code paths cause the cudaStreamSync stalls?"

**MCP tools used**: `get_stacks` (source=1, op="cudaStreamSync")

**Answer**:

Two code paths, one dominant offender:

### Path 1 — 99.1% of calls (2,899 of 2,924) — THE PROBLEM
```
libtorch_cuda.so
  → c10::TensorImpl::~TensorImpl()   (tensor destructor)
```
- **avg: 2.5ms**, max: **8.8ms**, min: 46µs
- Process: `python3`

This is PyTorch's **tensor cleanup path**. When a tensor goes out of scope, PyTorch destroys it, which calls `cudaStreamSync` to ensure all pending GPU work on that tensor's stream is complete before freeing memory. Under CPU contention (load avg 10.9), the process gets descheduled *while waiting for the sync*, inflating what should be a ~50µs call to 8.8ms.

### Path 2 — 0.9% of calls (25) — healthy
```
libtorch_cuda.so
  → c10::TensorImpl::~TensorImpl()   (same destructor, different callsite)
```
- **avg: 5µs**, max: 7µs — perfectly normal

Same destructor, but this path runs during low-contention windows and completes in microseconds as expected.

**Bottom line**: The cudaStreamSync stalls are not caused by slow GPU work. They're caused by **tensor destruction happening while the CPU is overloaded** — the thread gets descheduled mid-sync.

---

## Q3: "Show me the kernel launch paths and memory allocation patterns"

**MCP tools used**: `get_stacks` (cudaLaunchKernel, cuLaunchKernel, cudaMalloc)

**Answer**:

### Kernel Launches (1.69M total)

The model uses **two launch paths** — normal for PyTorch:

#### Driver API path (cuLaunchKernel) — 1.69M launches, 97% of kernels
cuBLAS/cuDNN calling the GPU directly, bypassing the runtime API:

| Operation | Calls | Avg | Max | Library |
|---|---|---|---|---|
| **BatchNorm forward** | 342 | 899µs | 6.0ms | `cudnnBatchNormalizationForwardTrainingEx` |
| **BatchNorm backward** | 140 | 234µs | 7.0ms | `cudnnBatchNormalization` (backward) |
| Tensor ops | 230 | 46µs | 4.0ms | `cudaLaunchKernel` → `libtorch_cuda.so` |

#### Runtime API path (cudaLaunchKernel) — 607K launches
PyTorch's element-wise CUDA kernels:

| Operation | Calls | Avg | Max | Code Path |
|---|---|---|---|---|
| **Tensor add (int64)** | 378 | 1.7ms | **7.1ms** | `gpu_kernel_impl_nocast<CUDAFunctorOnSelf_add<long>>` |
| **Tensor add (float)** | 116 | 1.4ms | 4.1ms | `gpu_kernel_impl_nocast<CUDAFunctor_add<float>>` |
| **Softmax forward** | 105 | 64µs | 4.0ms | `dispatch_softmax_forward` |
| **Softmax backward** | 101 | 23µs | 69µs | `dispatch_softmax_backward` — healthy |
| **Mean reduction** | 105 | 27µs | 135µs | `gpu_reduce_kernel<MeanOps>` — healthy |
| **Division** | 102 | 29µs | 230µs | `div_true_kernel_cuda` — healthy |
| **Max** | 102 | 85µs | 6.0ms | `max_launch_kernel` |

**Pattern**: Kernels with avg > 1ms (`add`, `BatchNorm`) suffer from CPU contention. Kernels with avg < 100µs (`softmax`, `mean`, `div`) complete before the scheduler can preempt.

### Memory Allocation (cudaMalloc) — full Python traces

| Code Path | Calls | Avg | Total Allocated | Source |
|---|---|---|---|---|
| `alloc_stress.py:sequential_alloc` line 63 | 9,724 | 973µs | 280 GB | Allocation benchmark |
| `alloc_stress.py:fragmentation_pattern` line 74 via `torch.cuda.memory.empty_cache` | 1,924 | 946µs | 46 GB | Cache flush + realloc |
| `alloc_stress.py:rapid_small_alloc` line 105 | 1,729 | 795µs | 181 GB | Small rapid allocations |

**Key insight**: Ingero traced all the way from the CUDA driver up through PyTorch C++ into the **exact Python source lines**.

---

## Q4: "Show me the memory alloc/free balance and any leaks"

**MCP tools used**: `get_stacks` (cudaFree), `run_sql` (aggregates for malloc/free counts and per-minute breakdown)

**Answer**:

### Overall Balance

| Operation | Calls | Total Allocated |
|---|---|---|
| **cudaMalloc** (runtime) | 45,992 | 638.5 GB |
| **cuMemAlloc** (driver) | 21,338 | 317.6 GB |
| **cudaFree** | 45,916 | — |
| **Net unfreed** | **76 calls** | — |

76 allocations were not freed during the session — but this is **not a real leak**. PyTorch's CUDA caching allocator intentionally holds onto GPU memory pools and releases them only at process exit. 76 out of 45,992 (0.17%) is perfectly normal for a training run.

### Per-Minute Breakdown

| Minute | Mallocs | Frees | Delta |
|---|---|---|---|
| Minute 1 | 16,648 | 16,562 | +86 |
| Minute 2 | 29,344 | 29,354 | -10 |
| **Total** | **45,992** | **45,916** | **+76** |

The pattern is healthy — allocs and frees track each other closely. Minute 2 actually freed 10 more than it allocated (cleaning up from minute 1).

### Where Frees Happen (by code path)

| Code Path | Frees | Avg Time |
|---|---|---|
| `alloc_stress.py:rapid_small_alloc` line 105 | 711 | 289µs |
| `torch.cuda.memory.empty_cache` via `fragmentation_pattern` line 74 | 118 | 263µs |
| `alloc_stress.py:direct_cuda_alloc_free` line 46 | 112 | 278-418µs |

**Bottom line**: No memory leak. Alloc/free pattern is balanced, PyTorch's caching allocator is working correctly, and the 76 unfreed allocations are expected cache retention.

---

## Q5: "Let's look at the scheduling storms"

**MCP tools used**: `get_causal_chains`, `run_sql` (scheduling events per minute, context switches per PID with process names and off-CPU time)

**Answer**:

### Scheduling Storm Timeline

| Minute | Context Switches | Wakeups | Intensity |
|---|---|---|---|
| Minute 1 | 3,106 | 20 | Moderate storm |
| **Minute 2** | **9,425** | **143** | **Peak storm — 3x worse** |
| Minute 3 | 1,973 | 5 | Calming down |

**Total**: 14,504 context switches in 120 seconds — the training process was descheduled ~121 times per second.

### Who's Getting Hit

| Process | PID | Switches | Off-CPU Time | Role |
|---|---|---|---|---|
| **python3** (main) | 17054 | **12,380** | **62 seconds** | Training loop — GPU owner |
| python3 (child) | 17056 | 1,466 | 25ms | Secondary Python process |
| pt_data_worker | 17130 | 166 | 17.3s | DataLoader worker |
| pt_data_worker | 17132 | 164 | 17.4s | DataLoader worker |
| pt_data_worker | 17131 | 159 | 17.4s | DataLoader worker |
| pt_data_worker | 17129 | 159 | 17.2s | DataLoader worker |

**The main training process (PID 17054) was off-CPU for 62 seconds out of a 120-second trace — it lost 51% of its CPU time to scheduling.**

### The Causal Chain

Ingero traced 18 distinct causal patterns (47 HIGH severity), all following the same chain:

```
[SYSTEM] CPU 100% + Load avg up to 16.9
    ↓
[HOST] 858–3,263 context switches per window (5–7 seconds off-CPU)
    ↓
[CUDA] cudaStreamSync p99 inflated 123x – 1,028x
       cudaMalloc p99 inflated 8x
       cudaFree p99 inflated 4.2x
       cuMemAlloc p99 inflated 8.7x
```

**The worst hit**: cudaStreamSync went from 7µs (p50) to **7.4ms** (p99) — a **1,028x inflation**. The GPU finished its work in microseconds but the CPU thread waiting for it was asleep.

### What's Causing the Load

Load average peaked at **16.9** — roughly 17 runnable threads competing for CPU cores. The training itself only has 6 processes (1 main + 1 child + 4 DataLoader workers). The remaining ~11 runnable threads are **other system processes** (monitoring agents, log shippers, cron jobs, or noisy neighbors on shared infrastructure).

### Recommendations

1. **`taskset -c 0-7 python3 train.py`** — Pin training to 8 dedicated cores
2. **`nice -n 19`** on all background processes — deprioritize them
3. **Check for noisy neighbors** — load avg 16.9 with only 6 training processes means ~11 competing threads from elsewhere
4. **Consider `num_workers=2`** instead of 4 — DataLoader workers spent 17s each off-CPU, competing with the main process

---

## Session Summary

| Metric | Value |
|---|---|
| Total events traced | 2.64 million |
| Events stored (sampled) | 50,505 |
| DB size | 7.7 MB |
| Unique stacks | 2,832 (100% resolved) |
| Causal chains | 285 (47 HIGH, 238 MEDIUM) |
| MCP calls used | 9 |
| Investigation time | ~5 minutes |

### Key Findings
1. **GPU is healthy** — kernel launches are fast, memory is balanced
2. **CPU is the bottleneck** — 62s of 120s spent off-CPU due to scheduling storms
3. **cudaStreamSync worst affected** — 1,028x tail latency inflation from context switches
4. **No memory leaks** — 45,992 mallocs vs 45,916 frees (76 = PyTorch cache, expected)
5. **Full Python→C++→CUDA traces** — exact source lines identified for every hot path

### MCP Tools Used
| Tool | Calls | Purpose |
|---|---|---|
| `get_trace_stats` | 1 | Session overview, per-op latency percentiles |
| `get_causal_chains` | 2 | Root cause analysis, severity breakdown |
| `get_stacks` | 4 | Call stack analysis for cudaStreamSync, cudaLaunchKernel, cuLaunchKernel, cudaMalloc, cudaFree |
| `run_sql` | 2 | Malloc/free aggregate balance, scheduling events per process |

---

# Session 2: GH200 (ARM64/aarch64)

**Setup**: `ingero mcp --db gpu-investigation.db --http :8090`
**DB size**: 22 MB (1.71M events traced, 265K stored via stack sampling)
**GPU**: NVIDIA GH200 480GB (Grace Hopper, ARM64), Lambda Labs

---

## Q1: "How did my training go?"

**MCP tools used**: `get_trace_stats`, `get_causal_chains`

**Answer**:

Ingero traced **1.71 million** events during the training run, storing 265K sampled events for investigation.

**The GPU was fast** — driver-level kernel launches (cuBLAS/cuDNN path) clocked at p50 = **19µs** with 1.07M total launches. Memory sync operations were healthy (cudaDeviceSync p50 = 18µs).

**But Ingero flagged 138 causal chain issues — 47 HIGH severity.** All pointing to the same root cause:

**CPU contention is hammering CUDA calls.** The GH200 has 64 vCPUs, but load average hit **31.0**. This caused:

| CUDA Operation | p50 (normal) | p99 (worst) | Tail ratio | Impact |
|---|---|---|---|---|
| cudaLaunchKernel | 21µs | **3.3ms** | **160x** | Kernel launches delayed |
| cudaMalloc | 1.5ms | **20.7ms** | 13.5x | Memory allocs stalling |
| cuMemAlloc | 1.5ms | **20.6ms** | 13.6x | Driver allocs too |
| cuLaunchKernel | 6µs | 37µs | 6.5x | Driver launches less affected |
| cudaFree | 43µs | 257µs | 6x | Frees slowed |

Ingero also spotted a memory pressure chain: **10,408 page allocations (1.2 GB)** coinciding with CPU saturation and kernel launch delays.

**Memory balance**: 10,937 mallocs vs 10,937 frees — **perfectly balanced**, no leaks.

---

## Q2: "Show me the call stacks for cudaLaunchKernel"

**MCP tools used**: `get_stacks` (source=1, op="cudaLaunchKernel")

**Answer**:

Top 10 `cudaLaunchKernel` code paths, ranked by call frequency:

| # | PyTorch Operation | Calls | Avg | Max | Status |
|---|---|---|---|---|---|
| **1** | **Tensor add (int64)** — `at::native::add_kernel` | 536 | 1.2ms | **16ms** | Worst hit |
| **2** | **ReLU activation** — `at::native::relu_` | 520 | 977µs | 11ms | Bad |
| **3** | **ReLU backward** — `threshold_backward` | 296 | 1.7ms | 6ms | Bad |
| **4** | **Tensor add (float)** — `at::native::add_kernel` | 241 | 647µs | 9.4ms | Moderate |
| **5** | **Tensor add (float)** — different callsite | 166 | 1.1ms | 6ms | Moderate |
| **6** | **Tensor add (int64)** — different callsite | 134 | 358µs | 4ms | Lighter |
| **7** | **Tensor add (int64)** — third callsite | 121 | 408µs | 7.6ms | Spikey |
| **8** | **Softmax forward** — `log_softmax_cuda` | 113 | 139µs | 7ms | Fast normally, spikes |
| **9** | **Mean reduction** — `gpu_reduce_kernel<MeanOps>` | 113 | 187µs | 4ms | OK |
| **10** | **Fill (zero_grad)** — `fill_kernel_cuda` | 112 | 146µs | 4ms | OK |

The training loop is dominated by **tensor addition** (gradient accumulation) and **ReLU forward/backward** passes — the ResNet-50 signature. All paths go through `python3` → `_PyEval_EvalFrameDefault` → `libtorch_python.so` → `libtorch_cuda.so` — standard PyTorch eager-mode on ARM64.

---

## Q3: "What lines of my code caused issues?"

**MCP tools used**: `get_stacks` (cudaMalloc, cudaLaunchKernel, cudaFree)

**Answer**:

### `alloc_stress.py` — 3 functions, all rooted at `main()` line 114

| File | Line | Function | Impact | Avg Latency | Max Latency |
|---|---|---|---|---|---|
| **alloc_stress.py** | **46** | `direct_cuda_alloc_free()` | 621 mallocs, **41ms spikes** | 9ms | 41.3ms |
| **alloc_stress.py** | **74** | `fragmentation_pattern()` | 346 mallocs + 438 frees, **22ms spikes** | 12ms | 22.4ms |
| **alloc_stress.py** | **63** | `sequential_alloc()` | 328 frees (healthy) | 42µs | 66µs |
| **alloc_stress.py** | **114** | `main()` | Orchestrator — calls all 3 above | — | — |
| torch/cuda/memory.py | 210 | `empty_cache()` | Called from line 74, triggers bulk realloc | 9ms | 41.4ms |

**Python frames confirmed working on ARM64** — Ingero's DWARF-based CPython frame walker resolved exact source files, function names, and line numbers on aarch64.

---

## Q4: "Show me the scheduling storms"

**MCP tools used**: `get_causal_chains`, `run_sql` (scheduling events per minute, context switches per PID)

**Answer**:

### Storm Timeline

| Minute | Context Switches | Wakeups | Page Allocs | Memory |
|---|---|---|---|---|
| Minute 1 (warmup) | 650 | 5 | 1,032 | 4 MB |
| **Minute 2 (training)** | **145,240** | **2,688** | **87,993** | **9.5 GB** |

A **223x increase** in context switches in minute 2.

### Who's Getting Hit

| Process | PID | Switches | Off-CPU Time | Role |
|---|---|---|---|---|
| **python3** (main) | 17074 | **237,394** | **62 seconds** | Training loop — GPU owner |
| python3 (child) | 17076 | 2,450 | 15ms | Secondary process |
| pt_data_worker | 17215 | 115 | 13.0s | DataLoader worker |
| pt_data_worker | 17217 | 110 | 12.8s | DataLoader worker |
| pt_data_worker | 17216 | 110 | 12.8s | DataLoader worker |
| pt_data_worker | 17214 | 110 | 12.9s | DataLoader worker |

**The main training process was descheduled 237,394 times — 1,977/sec — and spent 62 seconds off-CPU.**

### Storm Escalation (from causal chains)

| Phase | Sched Switches/window | Load Avg | Memory Pressure | CUDA Effect |
|---|---|---|---|---|
| Early | 44,605 | **31.0** | — | cudaMalloc 13.5x |
| Building | 53,853 | **27.7** | — | cudaLaunchKernel 160x |
| Peak CPU | 56,046 | **24.2** | + 2.2 GB page allocs | Dual pressure |
| **Peak storm** | **111,146** | 12.6 | — | All CUDA ops affected |
| Late + memory | 92,795 | — | + 1.2 GB page allocs | cudaFree 6x |

Load average *decreased* (31→12.6) as the storm intensified (44K→111K switches) — processes were finishing, but survivors fought harder for CPU.

---

## Q5: "Show me the memory alloc/free balance"

**MCP tools used**: `run_sql` (aggregate counts, per-minute breakdown), `get_stacks` (cudaMalloc with sum_arg0)

**Answer**:

### Overall Balance

| Operation | Total Calls | Total Allocated |
|---|---|---|
| **cudaMalloc** (runtime) | 10,937 | 151.1 GB |
| **cuMemAlloc** (driver) | 10,937 | 151.1 GB |
| **cudaFree** | 10,937 | — |

**Perfectly balanced — 10,937 mallocs, 10,937 frees. Zero leak.**

### Allocation by Code Path

| Function | Line | Calls | Avg Latency | Max Latency | Total Allocated |
|---|---|---|---|---|---|
| `direct_cuda_alloc_free()` | 46 | 621 | **9.0ms** | **41.3ms** | 61.1 GB |
| `fragmentation_pattern()` | 74 | 346 | **12.2ms** | 22.4ms | 48.5 GB |
| `empty_cache()` (via line 74) | 210 | 120 | **9.1ms** | **41.4ms** | 12.0 GB |
| `main()` | 114 | 4 | **39.8ms** | 40.3ms | 2.0 GB |

`fragmentation_pattern()` defeats the CUDA caching allocator by calling `empty_cache()` then reallocating from scratch — forcing fresh driver allocations during CPU contention.

---

## Session 2 Summary

| Metric | Value |
|---|---|
| Total events traced | 1.71 million |
| Events stored (sampled) | 265,514 |
| DB size | 22 MB |
| Unique stacks | 218 (100% resolved, 32 with Python frames) |
| Causal chains | 138 (47 HIGH, 91 MEDIUM) |
| MCP calls used | 8 |
| Architecture | **ARM64 (aarch64)** — GH200 Grace Hopper |

### Key Findings
1. **GPU is healthy** — kernel launches fast, memory perfectly balanced (10,937/10,937)
2. **CPU is the bottleneck** — 237K context switches, 62s off-CPU, load avg 31
3. **cudaLaunchKernel worst affected** — 160x tail latency inflation
4. **No memory leaks** — zero net unfreed allocations
5. **Python frames work on ARM64** — DWARF-based CPython walker resolves exact source lines on aarch64
6. **Memory pressure compounds CPU contention** — 9.5 GB page allocations during peak storm

### MCP Tools Used
| Tool | Calls | Purpose |
|---|---|---|
| `get_trace_stats` | 1 | Session overview, per-op latency percentiles |
| `get_causal_chains` | 2 | Root cause analysis, storm escalation |
| `get_stacks` | 3 | Call stacks for cudaLaunchKernel, cudaMalloc, cudaFree with Python frames |
| `run_sql` | 2 | Malloc/free aggregates, scheduling events per process |
