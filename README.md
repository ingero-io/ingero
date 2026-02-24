# Ingero — GPU Causal Observability

**Version: 0.6**

*"Why is my H100 at 98% utilization but training throughput dropped 30%?"*

Ingero is a production-grade eBPF agent that assembles **causal chains** to answer that question. It bridges the gap between the Linux kernel and your CUDA code — with **<2% overhead**, **zero code changes**, and **one binary**.

- **The "Why":** Correlate a `cudaStreamSync` spike with a `sched_switch` event — the host kernel preempted your training thread.
- **The "Where":** Map low-level CUDA calls back to **Python source lines** in your PyTorch `forward()` pass.
- **The "Hidden Kernels":** Trace the CUDA Driver API to see kernel launches by cuBLAS/cuDNN that bypass standard runtime profilers.

No ClickHouse, no PostgreSQL, no MinIO — just one statically linked Go binary and embedded SQLite.

## What It Does

Ingero uses eBPF to trace GPU workloads at three layers, reads system metrics from `/proc`, and assembles causal chains that explain root causes:

1. **CUDA Runtime uprobes** — traces `cudaMalloc`, `cudaLaunchKernel`, `cudaMemcpy`, `cudaMemcpyAsync`, `cudaStreamSync` / `cudaDeviceSynchronize` via uprobes on `libcudart.so`
2. **CUDA Driver uprobes** — traces `cuLaunchKernel`, `cuMemcpy`, `cuMemcpyAsync`, `cuCtxSynchronize`, `cuMemAlloc` via uprobes on `libcuda.so`. Captures kernel launches from cuBLAS/cuDNN that bypass the runtime API.
3. **Host tracepoints** — traces `sched_switch`, `sched_wakeup`, `mm_page_alloc`, `oom_kill`, `sched_process_exec/exit/fork` for CPU scheduling, memory pressure, and process lifecycle
4. **System context** — reads CPU utilization, memory usage, load average, and swap from `/proc` (no eBPF, no root needed)

**Why eBPF uprobes?** We evaluated NVIDIA CUPTI (5-30% overhead, single-subscriber, per-process injection) and bpftime (userspace eBPF runtime — promising but immature, extra deployment dependency). Standard kernel uprobes won: zero dependencies, <2% overhead, works on any Linux 5.15+ kernel.

The **causal engine** correlates events across layers by timestamp and PID to produce automated root cause analysis with severity ranking and fix recommendations.

```
$ sudo ingero trace

  Ingero Trace — Live CUDA Event Stream
  Target: PID 4821 (python3)
  Library: /usr/lib/x86_64-linux-gnu/libcudart.so.12
  CUDA probes: 12 attached
  Driver probes: 10 attached
  Host probes: 7 attached

  System: CPU [████████░░░░░░░░░░░░] 47% | Mem [██████████████░░░░░░] 72% (11.2 GB free) | Load 3.2 | Swap 0 MB

  CUDA Runtime API                                               Events: 11,028
  ┌──────────────────────┬────────┬──────────┬──────────┬──────────┬─────────┐
  │ Operation            │ Count  │ p50      │ p95      │ p99      │ Flags   │
  ├──────────────────────┼────────┼──────────┼──────────┼──────────┼─────────┤
  │ cudaLaunchKernel     │ 11,009 │ 5.2 µs   │ 12.1 µs  │ 18.4 µs  │         │
  │ cudaMalloc           │     12 │ 125 µs   │ 2.1 ms   │ 8.4 ms   │ ⚠ p99  │
  │ cudaDeviceSynchronize│      7 │ 684 µs   │ 1.2 ms   │ 3.8 ms   │         │
  └──────────────────────┴────────┴──────────┴──────────┴──────────┴─────────┘

  CUDA Driver API                                                Events: 17,525
  ┌──────────────────────┬────────┬──────────┬──────────┬──────────┬─────────┐
  │ Operation            │ Count  │ p50      │ p95      │ p99      │ Flags   │
  ├──────────────────────┼────────┼──────────┼──────────┼──────────┼─────────┤
  │ cuLaunchKernel       │ 17,509 │ 4.8 µs   │ 11.3 µs  │ 16.2 µs  │         │
  │ cuMemAlloc           │     16 │ 98 µs    │ 1.8 ms   │ 7.1 ms   │         │
  └──────────────────────┴────────┴──────────┴──────────┴──────────┴─────────┘

  Host Context                                                   Events: 258
  ┌─────────────────┬────────┬──────────────────────────────────────────┐
  │ Event           │ Count  │ Detail                                   │
  ├─────────────────┼────────┼──────────────────────────────────────────┤
  │ mm_page_alloc   │    251 │ 1.0 MB allocated (order-0: 251)         │
  │ process_exit    │      7 │ 7 processes exited                       │
  └─────────────────┴────────┴──────────────────────────────────────────┘

  ⚠ cudaStreamSync p99 = 142ms — correlated with 23 sched_switch events
    (GPU thread preempted during sync wait, avg 2.1ms off-CPU)
```

## What You'll Discover

Things no other GPU tool can show you.

**"cuBLAS was launching 17,509 kernels and you couldn't see any of them."** Most profilers trace only the CUDA Runtime API — but cuBLAS calls `cuLaunchKernel` (driver API) directly, bypassing the runtime. Ingero traces both layers: 11,009 runtime + 17,509 driver = complete visibility into every kernel launch.

**"Your training slowed because logrotate stole 4 CPU cores."** System Context shows CPU at 94%, Load 12.1. The CUDA table shows cudaStreamSync p99 jumping from 16ms to 142ms. The Host Context shows 847 sched_switch events. `ingero explain` assembles the full causal chain: logrotate preempted the training process → CUDA sync stalled → training throughput dropped 30%. Fix: `nice -n 19 logrotate`, or pin training to dedicated cores.

**"Your model spends 38% of wall-clock time on data movement, not compute."** nvidia-smi says "GPU utilization 98%", but the GPU is busy doing cudaMemcpy, not compute. Ingero's time-fraction breakdown makes this obvious. The fix (pinned memory, async transfers, larger batches) saves 30-50% wall-clock time.

**"Your host is swapping and your GPU doesn't know it."** System Context shows Swap 2.1 GB. cudaMalloc p99 rises from 0.02ms to 8.4ms. No GPU tool shows this — nvidia-smi says GPU memory is fine, but host-side CUDA bookkeeping is hitting swap.

**"Ask your AI."** Claude queries Ingero via MCP: "At 15:41:22, mm_page_alloc latency spiked while CPU was at 94%. cudaMalloc p99 rose 600x. Another process allocated 6GB of RAM." The engineer never reads logs again.

## See It In Action

```bash
ingero demo                 # run all 6 scenarios (auto-detects GPU)
ingero demo incident        # full causal chain in 30 seconds
ingero demo --no-gpu        # synthetic mode (no root, no GPU needed)
sudo ingero demo --gpu      # real GPU + eBPF tracing
```

### Scenarios

| Scenario | What It Reveals |
|----------|----------------|
| `incident` | CPU spike + sched_switch storm → cudaStreamSync 8.5x latency spike → full causal chain with root cause and fix |
| `cold-start` | First CUDA calls take 50-200x longer than steady state (CUDA context init) |
| `memcpy-bottleneck` | cudaMemcpy dominates wall-clock time (38%), not compute — nvidia-smi lies |
| `periodic-spike` | cudaMalloc spikes 50x every ~200 batches (PyTorch caching allocator) |
| `cpu-contention` | Host CPU preemption causes CUDA latency spikes |
| `gpu-steal` | Multi-process GPU time-slicing quantified via CUDA API timing patterns |

Every scenario prints a GPU auto-detect header showing GPU model and driver version, then displays real-time ASCII bar charts for system context.

## Install

```bash
# Build from source (requires clang-14, Linux kernel with BTF)
git clone https://github.com/ingero-io/ingero.git
cd ingero
make              # generates eBPF bindings, builds, tests, and lints — single command
sudo make install # copies binary to /usr/local/bin/ingero
```

Or add an alias if you prefer running from the build directory:

```bash
alias ingero='sudo ./bin/ingero'
```

> **One-line install** (when releases are available):
> ```bash
> curl -fsSL https://get.ingero.io | sh
> ```

## Requirements

- Linux kernel 5.15+ with BTF (`CONFIG_DEBUG_INFO_BTF=y`)
- NVIDIA driver 550+ with CUDA 11.x, 12.x, or 13.x
- Root / `CAP_BPF` + `CAP_PERFMON` (eBPF requires elevated privileges)
- Tested on: RTX 3090, RTX 3090 Ti, RTX 4090, A10, A100 SXM4, H100 PCIe

## Commands

### `ingero check`

Check if your system is ready for eBPF-based GPU tracing.

```bash
$ sudo ingero check

Ingero — System Readiness Check

  [✓] Kernel version: 5.15.0-144-generic
      need 5.15+
  [✓] BTF support: /sys/kernel/btf/vmlinux
      available (5242880 bytes)
  [✓] NVIDIA driver: 580.126.09
      open kernel modules (550+)
  [✓] GPU model: NVIDIA GeForce RTX 3090 Ti, 24564 MiB
  [✓] CUDA runtime: /usr/lib/x86_64-linux-gnu/libcudart.so.12
      loaded by 1 process(es)
  [✓] CUDA driver (libcuda.so): /usr/lib/x86_64-linux-gnu/libcuda.so.1
      available for driver API tracing
  [✓] CUDA processes: 1 found
      PID 4821 (python3)

All checks passed — ready to trace!
```

### `ingero trace`

Live event stream with rolling stats, system context, and anomaly detection. Events are recorded to SQLite by default (use `--record=false` to disable).

```bash
sudo ingero trace                      # auto-detect CUDA processes (traces current user, stacks on, recording on)
sudo ingero trace --pid 4821           # trace specific process
sudo ingero trace --user bob           # trace CUDA processes owned by a different user
sudo ingero trace --record=false       # disable SQLite recording
sudo ingero trace --duration 60s       # stop after 60 seconds
sudo ingero trace --json               # JSON output (pipe to jq)
sudo ingero trace --verbose            # show individual events
sudo ingero trace --stack=false        # disable stack traces (saves ~0.4-0.6% overhead)
sudo ingero trace --prometheus :9090   # expose Prometheus /metrics endpoint
sudo ingero trace --otlp localhost:4318 # push metrics via OTLP
```

`--user` defaults to the invoking user (via `SUDO_USER`). Use `--user root` to trace root-owned CUDA processes, or `--user bob` to trace another user's.

The trace display shows four sections:
1. **System Context** — CPU, memory, load, swap with ASCII bar charts (green/yellow/red)
2. **CUDA Runtime API** — per-operation p50/p95/p99 latency with anomaly flags (cudaMalloc, cudaLaunchKernel, etc.)
3. **CUDA Driver API** — driver-level operations (cuLaunchKernel, cuMemAlloc, etc.) that cuBLAS/cuDNN call directly
4. **Host Context** — scheduler, memory, OOM, and process lifecycle events

### `ingero explain`

Analyze recorded events from SQLite and produce an incident report with causal chains, root causes, and fix recommendations. Reads from the database populated by `ingero trace` — no root needed.

```bash
ingero explain                         # analyze last 5 minutes
ingero explain --since 1h             # last hour
ingero explain --last 100             # last 100 events
ingero explain --pid 4821             # filter by process
ingero explain --chains               # show stored causal chains (no re-analysis)
ingero explain --json                 # JSON output for pipelines
ingero explain --from "15:40" --to "15:45"  # absolute time range
```

```
INCIDENT REPORT — 2 causal chains found (1 HIGH, 1 MEDIUM)

[HIGH] cudaStreamSync p99=142ms (8.5x p50) — CPU contention
  Timeline:
    15:41:20  [SYSTEM]  CPU 94%, Load 12.1, Swap 2.1GB
    15:41:20  [HOST]    sched_switch: PID 8821 (logrotate) preempted PID 4821
    15:41:22  [CUDA]    cudaStreamSync 142ms (normally 16.7ms)

  Root cause: logrotate cron job preempted training process 847 times
  Fix: Add `nice -n 19` to logrotate cron, or pin training to dedicated cores
```

### `ingero query`

Query stored events by time range, PID, and operation type.

```bash
sudo ingero query --since 1h
sudo ingero query --since 30m --op cudaMemcpy --json
```

Storage uses SQLite with 7-day rolling retention. Data is stored locally at `~/.ingero/ingero.db` — nothing leaves your machine.

### `ingero mcp`

Start an MCP (Model Context Protocol) server for AI agent integration.

```bash
sudo ingero mcp                   # stdio (for Claude Code / MCP clients)
sudo ingero mcp --http :8080      # HTTPS/TLS 1.3 (self-signed cert)
```

**AI-first analysis**: MCP responses use telegraphic compression (TSC) by default, reducing token count by ~60%. Set `{"tsc": false}` per request for verbose output.

**MCP tools:**

| Tool | Description |
|------|-------------|
| `get_check` | System diagnostics (kernel, BTF, NVIDIA, CUDA, GPU model) |
| `get_trace_stats` | CUDA + host statistics with p50/p95/p99 latency |
| `query_events` | Query stored events by time range, PID, operation |
| `get_causal_chains` | Causal chains with severity ranking and root cause |
| `run_demo` | Run synthetic demo scenarios |
| `get_test_report` | GPU integration test report (JSON) |

**curl examples** (with `--http :8080`):

```bash
# System diagnostics (-k for self-signed cert)
curl -sk https://localhost:8080/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_check","arguments":{}}}' | jq

# Causal chains (TSC-compressed for AI)
curl -sk https://localhost:8080/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":2,"method":"tools/call","params":{"name":"get_causal_chains","arguments":{}}}' | jq

# Verbose output (TSC off)
curl -sk https://localhost:8080/mcp \
  -H 'Content-Type: application/json' \
  -H 'Accept: application/json, text/event-stream' \
  -d '{"jsonrpc":"2.0","id":3,"method":"tools/call","params":{"name":"get_trace_stats","arguments":{"tsc":false}}}' | jq
```

### `ingero demo`

```bash
ingero demo                  # all 6 scenarios (incident first)
ingero demo incident         # single scenario
ingero demo gpu-steal        # also: gpu-contention, contention
ingero demo --no-gpu         # synthetic mode
```

### `ingero version`

```bash
$ ingero version
ingero v0.6 (commit: adc2943, built: 2026-02-23)
```

## Stack Tracing

Stack tracing is **on by default** — every CUDA/Driver API event captures the full userspace call chain. Shows **who called cudaMalloc** — from the CUDA library up through PyTorch, your Python code, and all the way to `main()`. GPU-measured overhead is **0.4-0.6%** (within noise on RTX 3090 through H100). Disable with `--stack=false` if needed.

```bash
sudo ingero trace --json               # JSON with resolved stack traces (stacks on by default)
sudo ingero trace --debug              # debug output shows resolved frames on stderr
sudo ingero demo --json                # GPU demo with stack traces
ingero explain                         # post-hoc causal analysis from DB
sudo ingero trace --stack=false        # disable stacks if needed
```

**Maximum depth**: 64 native frames (eBPF `bpf_get_stack`). This covers deep call chains from CUDA → cuBLAS/cuDNN → PyTorch C++ → Python interpreter and up to `main()` / `_start`.

### Python Stack Attribution

For Python workloads (PyTorch, TensorFlow, etc.), Ingero extracts **CPython frame information** directly from process memory. When a native frame is inside libpython's eval loop, the corresponding Python source frames are injected into the stack:

```
[Python] train.py:47 in forward()
[Python] model.py:123 in Linear.__call__()
[Native] torch::autograd::Engine::execute+0x1a3c (libtorch_cuda.so)
[Native] cudaMalloc+0x1f (libcudart.so.12)
```

Supported Python versions: **3.10, 3.11, 3.12** (covers Ubuntu 22.04 default, conda default, and most production deployments). Version detection is automatic via `/proc/[pid]/maps`.

### JSON Output with `--stack`

```json
{
  "timestamp": "2026-02-21T15:41:22.123456789Z",
  "pid": 4821,
  "tid": 4821,
  "source": "cuda",
  "op": "cudaMalloc",
  "duration_ns": 8400000,
  "duration": "8.4ms",
  "stack": [
    {"ip": "0x7f1234560000", "symbol": "cudaMalloc+0x1f", "file": "libcudart.so.12"},
    {"ip": "0x7f1234000000", "symbol": "torch::CUDACachingAllocator::allocate+0x3a", "file": "libtorch_cuda.so"},
    {"ip": "0x7f1200000000", "py_file": "train.py", "py_func": "forward", "py_line": 47},
    {"ip": "0x7f1200000100", "py_file": "model.py", "py_func": "__call__", "py_line": 123}
  ]
}
```

### Debug Output with `--stack --debug`

When `--debug` is enabled, resolved stack frames are logged to stderr:

```
[DEBUG] stack trace for cudaMalloc (PID 4821, TID 4821, 6 frames):
[DEBUG]   [0] cudaMalloc+0x1f (libcudart.so.12)
[DEBUG]   [1] torch::CUDACachingAllocator::allocate+0x3a (libtorch_cuda.so)
[DEBUG]   [2] [Python] train.py:47 in forward()
[DEBUG]   [3] [Python] model.py:123 in __call__()
[DEBUG]   [4] _PyEval_EvalFrameDefault+0x2a1 (libpython3.10.so.1.0)
[DEBUG]   [5] _start (python3.10)
```

## OTEL Integration (Optional)

OTEL export is **off by default** — enabled only when you pass `--otlp` or `--prometheus`.

```bash
# Prometheus metrics endpoint (pull)
sudo ingero trace --prometheus :9090
curl localhost:9090/metrics

# OTLP push (HTTP JSON to any OTEL-compatible receiver)
sudo ingero trace --otlp localhost:4318
sudo ingero trace --otlp localhost:4318 --debug  # see OTLP push logs on stderr
```

OTLP uses the HTTP JSON transport (`POST /v1/metrics`). Compatible with: OpenTelemetry Collector, Grafana Alloy, Grafana Cloud, Datadog Agent, New Relic, and any OTLP-compatible receiver.

Metrics use OTEL semantic conventions: `gpu.cuda.operation.duration`, `gpu.cuda.operation.count`, `system.cpu.utilization`, `system.memory.utilization`, `ingero.anomaly.count`. Per-operation, per-source granularity.

Zero external dependencies — no OTEL SDK import. The JSON payload is constructed directly using Go's standard library.

## How It Works

```
┌───────────────────────────────────────────────────────────────┐
│  User Space                                                    │
│                                                                │
│  ┌─────────┐    ┌──────────────┐  ┌───────┐   ┌───────────┐ │
│  │  CUDA   │    │   ingero     │  │SQLite │   │MCP Server  │ │
│  │  App    │    │   agent      │─▶│  DB   │   │(stdio/HTTPS)│ │
│  │(PyTorch)│    │              │  └───────┘   └───────────┘ │
│  └──┬──┬───┘    │ ┌──────────┐│                              │
│     │  │        │ │ causal   ││  ┌───────────┐               │
│     │  │        │ │ engine   ││  │ OTLP /    │               │
│     │  │        │ └──────────┘│─▶│ Prometheus│               │
│     │  │        └──┬──┬──┬────┘  └───────────┘               │
│     │  │           │  │  │ ▲                                  │
│     │  │           │  │  │ │ ring buffers                     │
│  ───┼──┼───────────┼──┼──┼─┼─────────────────────────────────│
│     │  ▼           │  ▼  ▼ │                                  │
│     │ ┌─────────┐  │ ┌────────────────────┐                   │
│     │ │libcuda  │◄─┤ │  eBPF uprobes      │  (Driver API)    │
│     │ │  .so    │  │ │  cuLaunchKernel     │                  │
│     │ └─────────┘  │ │  cuMemcpy/Alloc     │                  │
│     ▼              │ └────────────────────┘                   │
│  ┌─────────┐       │ ┌────────────────────┐                   │
│  │libcudart│◄──────┘ │  eBPF uprobes      │  (Runtime API)   │
│  │  .so    │         │  cudaLaunchKernel   │                  │
│  └─────────┘         │  cudaMalloc/Memcpy  │                  │
│                      └────────────────────┘                   │
│  ┌─────────────────────────────────────────────────────────┐  │
│  │  eBPF tracepoints (sched_switch, mm_page_alloc, oom,    │  │
│  │  sched_process_exec/exit/fork)                          │  │
│  └─────────────────────────────────────────────────────────┘  │
│                                                                │
│  Kernel Space        /proc → CPU%, Mem%, Load, Swap            │
└───────────────────────────────────────────────────────────────┘
```

1. **Discover** — scans `/proc` for processes linked to `libcudart.so`, finds `libcuda.so` automatically
2. **Attach** — eBPF probes load onto CUDA runtime uprobes, driver uprobes, and host tracepoints
3. **Capture** — eBPF programs record PID, TID, timestamps into per-layer ring buffers
4. **System** — reads CPU/memory/load/swap from `/proc` once per second
5. **Stats** — computes rolling p50/p95/p99 per operation, flags anomalies
6. **Correlate** — assembles causal chains (SYSTEM + HOST + CUDA Runtime + CUDA Driver) by timestamp and PID
7. **Store** — writes events to SQLite with 7-day rolling retention (on by default, disable with `--record=false`)
8. **Export** — pushes metrics via OTLP or serves Prometheus `/metrics` (optional)
9. **Serve** — exposes diagnostics to AI agents via MCP (stdio or HTTPS/TLS 1.3)

## Integration Testing

Validated on 6 GPU models across 2 cloud providers (TensorDock, Lambda Labs):

Stack tracing is on by default. GPU-measured overhead: **0.4-0.6%** (within noise).

| GPU | VRAM | Tests | Pass | Fail | Skip | Throughput | Stack OH | Stack Cov |
|-----|------|-------|------|------|------|-----------|----------|-----------|
| RTX 3090 | 24 GB | 24 | 21 | 1* | 2 | 88,150/30s | -0.8% | 99.6% |
| RTX 4090 | 24 GB | 35 | 35 | 0 | 0 | 127,548/3s | +0.6% | 99.9% |
| A10 | 24 GB | 29 | 27 | 0 | 2 | 48,500/20s | +0.5% | 99.6% |
| A100 SXM4 | 40 GB | 29 | 27 | 0 | 2 | 58,902/20s | +0.4% | 99.7% |
| H100 PCIe | 80 GB | 29 | 27 | 0 | 2 | 44,673/20s | +1.7% | 99.5% |

\* RTX 3090 failure was a test infrastructure bug (fixed), not a product bug. SKIPs are CPython frame extraction and chain detection — expected, not regressions.

**Latest: RTX 4090, Ubuntu 22.04 (2026-02-23)** — Driver 580.126.09, CUDA 12.1, Kernel 5.15.0-144, Go 1.26.0, PyTorch 2.5.1+cu121 (TensorDock)

35/35 integration tests PASS — first clean sweep:

- **check**: All system checks pass, GPU model + driver + BTF + CUDA libs detected
- **trace**: 85,820 events; driver API: 23,383 cuLaunchKernel (0 cudaLaunchKernel — cuBLAS direct path)
- **explain**: 357,990 events collected, incident report generated with causal chains
- **MCP HTTPS**: get_check, get_trace_stats, run_demo validated (TLS 1.3, ephemeral self-signed cert)
- **Prometheus**: System + CUDA metrics in valid exposition format
- **record + query**: DB created, 10,000 events retrieved (default limit)
- **--debug**: Verified on/off isolation (no debug noise when off, [DEBUG] present when on)

**H100 PCIe 80GB, Ubuntu 22.04 (2026-02-22)** — Driver 570.148.08, CUDA 12.8, Kernel 6.8.0-60, 200GB RAM (Lambda Labs)

27/29 PASS, 0 FAIL, 2 SKIP. cuCtxSynchronize p50: 875us (fastest tested GPU).

**A100 SXM4 40GB, Ubuntu 22.04 (2026-02-22)** — Driver 570.148.08, CUDA 12.8 (Lambda Labs)

27/29 PASS, 0 FAIL, 2 SKIP. 58,902 events/20s baseline, +0.4% stack overhead, 99.7% stack coverage.

**A10 24GB, Ubuntu 22.04 (2026-02-22)** — Driver 570.148.08, CUDA 12.8 (Lambda Labs)

27/29 PASS, 0 FAIL, 2 SKIP. DWARF offset discovery validated for CPython frame extraction.

## Competitive Position

Ingero is the only tool that traces the **full causal chain from host scheduler through CUDA Runtime and Driver APIs** and generates automated root cause analysis with AI-first MCP integration.

| Tool | Depth | Driver API | Production-Safe | Root Cause | AI |
|------|-------|-----------|----------------|-----------|-----|
| nvidia-smi / DCGM | Surface counters | No | Yes | No | No |
| Datadog GPU | DCGM relay | No | Yes | No | No |
| ZymTrace | GPU-internal (SASS) | No | Yes | No | No |
| Nsight | Deep (kernel-level) | Yes | No (10-100x) | No | No |
| **Ingero** | **Cross-stack causal** | **Yes** | **Yes (<2%)** | **Yes** | **Yes (MCP)** |

**Zero infrastructure**: Ingero is a single binary with embedded SQLite. No ClickHouse, no PostgreSQL, no MinIO, no Kubernetes required.

## What's Next

- GPU-internal profiling (SM stalls, SASS, warp divergence)
- HTTP/gRPC inference serving tracing (vLLM, Triton)
- Noisy neighbor detection (per-cgroup scheduler latency)
- Container/K8s metadata enrichment (`/proc/[pid]/cgroup` → pod name)
- Block I/O tracing (block_rq_issue/complete)
- DNS monitoring

## FAQ

**Is it safe for production?**
Yes. eBPF programs are verified by the kernel before loading — they cannot crash the system. Probes add <2% overhead including stack tracing (0.4-0.6% measured across RTX 3090, RTX 4090, A10, A100, H100 with PyTorch workloads).

**Does it require code changes?**
No. Ingero attaches to `libcudart.so` and kernel tracepoints at the OS level. Your application code is untouched. Traces any language — Python, C++, Java — anything linked against libcudart.so.

**What GPUs are supported?**
Any NVIDIA GPU with driver 550+ and CUDA 11.x/12.x/13.x. Tested on RTX 3090, RTX 3090 Ti, RTX 4090, A10, A100 SXM4, H100 PCIe.

**Does it work in containers?**
Yes, with `--privileged` or appropriate BPF capabilities. The host kernel must have BTF enabled.

**Where is data stored?**
Locally in `~/.ingero/ingero.db` (SQLite). Nothing leaves your machine. 7-day rolling retention.

## License

Ingero uses a standard eBPF split-licensing model to ensure maximum enterprise compatibility while strictly adhering to Linux kernel requirements.

* **User-Space (Go Agent, CLI, Causal Engine, SQLite, MCP):** Licensed under the [Apache License, Version 2.0](LICENSE). This allows you to freely use, modify, and distribute the Ingero agent within your own proprietary infrastructure without risk of copyleft infection.
* **Kernel-Space (eBPF C Code in the `bpf/` directory):** Dual-licensed under `GPL-2.0 OR BSD-3-Clause` ([LICENSE-GPL2](bpf/LICENSE-GPL2)). GPL-2.0 is required by the Linux kernel's BPF subsystem; BSD-3-Clause permits embedding in non-GPL toolchains.

This separation guarantees that your proprietary application code remains untouched and unencumbered when traced by Ingero.
