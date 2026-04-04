# Ingero  -  GPU Causal Observability

[![Go Report Card](https://goreportcard.com/badge/github.com/ingero-io/ingero)](https://goreportcard.com/report/github.com/ingero-io/ingero)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/ingero-io/ingero)](https://github.com/ingero-io/ingero/releases)
[![CI](https://github.com/ingero-io/ingero/actions/workflows/ci.yml/badge.svg)](https://github.com/ingero-io/ingero/actions/workflows/ci.yml)
[![MCP](https://img.shields.io/badge/MCP-server-blue)](https://glama.ai/mcp/servers/ingero-io/ingero)

**Featured in:**
[awesome-ebpf](https://github.com/qmonnet/awesome-ebpf) ·
[awesome-observability](https://github.com/adriannovegil/awesome-observability) ·
[awesome-sre-tools](https://github.com/SquadcastHub/awesome-sre-tools) ·
[awesome-cloud-native](https://github.com/rootsongjc/awesome-cloud-native) ·
[awesome-profiling](https://github.com/msaroufim/awesome-profiling) ·
[Awesome-GPU](https://github.com/Jokeren/Awesome-GPU) ·
[awesome-devops-mcp-servers](https://github.com/rohitg00/awesome-devops-mcp-servers) ·
[MCP Registry](https://registry.modelcontextprotocol.io) ·
[Glama](https://glama.ai/mcp/servers/ingero-io/ingero) ·
[mcpservers.org](https://mcpservers.org)

**Version: 0.9.1**

**The only GPU observability tool your AI assistant can talk to.**

*"What caused the GPU stall?" → "`forward()` at `train.py:142`  -  cudaMalloc spiking 48ms during CPU contention. 9,829 calls, 847 scheduler preemptions."*

Ingero is a production-grade eBPF agent that traces the full chain  -  from Linux kernel events through CUDA API calls to your Python source lines  -  with **<2% overhead**, **zero code changes**, and **one binary**.

<img src="docs/assets/readme-demo-incident.gif" width="800" alt="ingero demo incident — CPU contention causes GPU latency spike, full causal chain diagnosis with root cause and fix recommendation">

## Quick Start

```bash
# Install (Linux amd64 — see below for arm64/Docker)
VERSION=0.9.0
curl -fsSL "https://github.com/ingero-io/ingero/releases/download/v${VERSION}/ingero_${VERSION}_linux_amd64.tar.gz" | tar xz
sudo mv ingero /usr/local/bin/

# Trace your GPU workload
sudo ingero trace

# Diagnose what happened
ingero explain --since 5m
```

- **The "Why":** Correlate a `cudaStreamSync` spike with `sched_switch` events  -  the host kernel preempted your thread.
- **The "Where":** Map CUDA calls back to **Python source lines** in your PyTorch `forward()` pass.
- **The "Hidden Kernels":** Trace the CUDA Driver API to see kernel launches by cuBLAS/cuDNN that bypass standard profilers.

No ClickHouse, no PostgreSQL, no MinIO  -  just one statically linked Go binary and embedded SQLite.

See a [real AI investigation session](docs/ml_eng_sample_investigation_session.md)  -  an AI assistant diagnosing GPU training issues on A100 and GH200 using only Ingero's MCP tools. No shell access, no manual SQL  -  just questions and answers.

## What It Does

Ingero uses eBPF to trace GPU workloads at three layers, reads system metrics from `/proc`, and assembles causal chains that explain root causes:

1. **CUDA Runtime uprobes**  -  traces `cudaMalloc`, `cudaFree`, `cudaLaunchKernel`, `cudaMemcpy`, `cudaMemcpyAsync`, `cudaStreamSync` / `cudaDeviceSynchronize` via uprobes on `libcudart.so`
2. **CUDA Driver uprobes**  -  traces `cuLaunchKernel`, `cuMemcpy`, `cuMemcpyAsync`, `cuCtxSynchronize`, `cuMemAlloc` via uprobes on `libcuda.so`. Captures kernel launches from cuBLAS/cuDNN that bypass the runtime API.
3. **CUDA Graph lifecycle uprobes**  -  traces `cudaStreamBeginCapture`, `cudaStreamEndCapture`, `cudaGraphInstantiate`, `cudaGraphLaunch` for graph capture/replay visibility in `torch.compile` and vLLM workloads
4. **Host tracepoints**  -  traces `sched_switch`, `sched_wakeup`, `mm_page_alloc`, `oom_kill`, `sched_process_exec/exit/fork` for CPU scheduling, memory pressure, and process lifecycle
5. **System context**  -  reads CPU utilization, memory usage, load average, and swap from `/proc` (no eBPF, no root needed)

The **causal engine** correlates events across layers by timestamp and PID to produce automated root cause analysis with severity ranking and fix recommendations.

```
$ sudo ingero trace

  Ingero Trace  -  Live CUDA Event Stream
  Target: PID 4821 (python3)
  Library: /usr/lib/x86_64-linux-gnu/libcudart.so.12
  CUDA probes: 14 attached
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
  │ mm_page_alloc   │    251 │ 1.0 MB allocated (order-0: 251)          │
  │ process_exit    │      7 │ 7 processes exited                       │
  └─────────────────┴────────┴──────────────────────────────────────────┘

  ⚠ cudaStreamSync p99 = 142ms  -  correlated with 23 sched_switch events
    (GPU thread preempted during sync wait, avg 2.1ms off-CPU)
```

## What You'll Discover

Things no other GPU tool can show you.

**"cuBLAS was launching 17,509 kernels and you couldn't see any of them."** Most profilers trace only the CUDA Runtime API  -  but cuBLAS calls `cuLaunchKernel` (driver API) directly, bypassing the runtime. Ingero traces both layers: 11,009 runtime + 17,509 driver = complete visibility into every kernel launch.

**"Your training slowed because logrotate stole 4 CPU cores."** System Context shows CPU at 94%, Load 12.1. The CUDA table shows cudaStreamSync p99 jumping from 16ms to 142ms. The Host Context shows 847 sched_switch events. `ingero explain` assembles the full causal chain: logrotate preempted the training process → CUDA sync stalled → training throughput dropped 30%. Fix: `nice -n 19 logrotate`, or pin training to dedicated cores.

**"Your model spends 38% of wall-clock time on data movement, not compute."** nvidia-smi says "GPU utilization 98%", but the GPU is busy doing cudaMemcpy, not compute. Ingero's time-fraction breakdown makes this obvious. The fix (pinned memory, async transfers, larger batches) saves 30-50% wall-clock time.

**"Your host is swapping and your GPU doesn't know it."** System Context shows Swap 2.1 GB. cudaMalloc p99 rises from 0.02ms to 8.4ms. No GPU tool shows this  -  nvidia-smi says GPU memory is fine, but host-side CUDA bookkeeping is hitting swap.

**"Your vLLM inference spiked because a new batch size triggered CUDA Graph re-capture."** Ingero traces `cudaStreamBeginCapture` / `cudaGraphLaunch` via eBPF uprobes  -  no CUPTI, no Nsight, no code changes. When GraphLaunch rate drops 50%, Ingero flags graph pool exhaustion. When capture overlaps with OOM, the causal chain explains why. Works with `torch.compile(mode="reduce-overhead")` and vLLM out of the box.

**"Rank 3 stalled for 200ms while ranks 0-2 waited  -  one query shows all 4 nodes."** With `ingero query --nodes`, one command fans out to every node in your cluster and merges the results. `ingero merge` combines offline databases for air-gapped analysis. `ingero export --format perfetto` produces a timeline you can open in Perfetto UI  -  one track per node/rank, immediately spotting the straggler. Clock skew between nodes is detected automatically.

**"Ask your AI: what line of my code caused the GPU stall?"** Your AI assistant calls Ingero's MCP server and answers in one shot: "The issue is in `forward()` at `train.py:142`, calling cudaMalloc through PyTorch. 9,829 calls, avg 3.1ms but spiking to 48.3ms during CPU contention." Resolved Python source lines, native symbols, timing stats  -  no logs, no manual SQL, no hex addresses. The engineer asks questions in plain English and gets production root causes back.

## See It In Action

<details><summary><code>sudo ingero check</code> — system readiness</summary>
<br>
<img src="docs/assets/readme-check.gif" width="800" alt="ingero check verifying kernel, BTF, NVIDIA driver, GPU model, CUDA libraries, and active processes">
</details>

<details><summary><code>sudo ingero trace</code> — live event stream</summary>
<br>
<img src="docs/assets/readme-trace.gif" width="800" alt="ingero trace showing live CUDA Runtime and Driver API statistics with rolling p50/p95/p99 latencies and host context">
</details>

<details><summary><code>ingero explain --since 5m</code> — automated diagnosis</summary>
<br>
<img src="docs/assets/readme-explain.gif" width="800" alt="ingero explain producing incident report with causal chains, root cause analysis, and fix recommendations">
</details>

<details><summary><code>sudo ingero trace</code> — CUDA Graph lifecycle events</summary>
<br>
<img src="docs/assets/cuda-graph-trace.gif" width="800" alt="ingero trace showing CUDA Graph capture, instantiate, and launch events alongside CUDA runtime and host events">
</details>

<details><summary><code>ingero explain</code> — graph causal chain diagnosis</summary>
<br>
<img src="docs/assets/cuda-graph-explain.gif" width="800" alt="ingero explain showing causal chain linking CUDA Graph launch to CPU contention with fix recommendations">
</details>

<details><summary><code>ingero demo --no-gpu incident</code> — try without a GPU</summary>
<br>
<img src="docs/assets/readme-demo-nogpu.gif" width="800" alt="ingero demo running in synthetic mode without GPU, showing full causal chain diagnosis">
</details>

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
| `memcpy-bottleneck` | cudaMemcpy dominates wall-clock time (38%), not compute  -  nvidia-smi lies |
| `periodic-spike` | cudaMalloc spikes 50x every ~200 batches (PyTorch caching allocator) |
| `cpu-contention` | Host CPU preemption causes CUDA latency spikes |
| `gpu-steal` | Multi-process GPU time-slicing quantified via CUDA API timing patterns |

Every scenario prints a GPU auto-detect header showing GPU model and driver version, then displays real-time ASCII bar charts for system context.

## Multi-Node Investigation Walkthrough

A complete example: diagnosing a distributed training stall across a 4-node GPU cluster using every multi-node feature. Sample databases are available in [`investigations/`](investigations/#multi-node-investigation-samples) to reproduce these examples locally.

### Setup: Tag Each Node

On each node, trace with a node identity. Rank is auto-detected from `torchrun` environment variables (`RANK`, `LOCAL_RANK`, `WORLD_SIZE`).

```bash
# Node 1 (rank 0)
sudo ingero trace --node gpu-node-01

# Node 2 (rank 1)
sudo ingero trace --node gpu-node-02

# Node 3 (rank 2)
sudo ingero trace --node gpu-node-03

# Node 4 (rank 3)
sudo ingero trace --node gpu-node-04
```

Events are tagged with node identity and rank. Event IDs are node-namespaced (`gpu-node-01:1`, `gpu-node-01:2`, ...) to prevent collisions.

### Step 1: Start Dashboards for Fleet Queries

On each node, start the dashboard API (plain HTTP on trusted VPC):

```bash
ingero dashboard --no-tls --addr :8080
```

Or configure once in `ingero.yaml`:

```yaml
fleet:
  nodes:
    - gpu-node-01:8080
    - gpu-node-02:8080
    - gpu-node-03:8080
    - gpu-node-04:8080
```

### Step 2: Fan-Out Query  -  Find the Straggler

<img src="docs/assets/fleet-query.gif" width="800" alt="ingero query --nodes fan-out across 3 GPU nodes showing events per node with node column prepended">

From any node, query the entire cluster with one command:

```bash
$ ingero query --nodes gpu-node-01:8080,gpu-node-02:8080,gpu-node-03:8080,gpu-node-04:8080 \
    "SELECT node, source, count(*) as cnt, avg(duration)/1000 as avg_us FROM events GROUP BY node, source"

node              node           source  cnt    avg_us
----------------  -------------  ------  -----  ------
gpu-node-01:8080  gpu-node-01    4       11009  5.2
gpu-node-01:8080  gpu-node-01    3       847    18400
gpu-node-02:8080  gpu-node-02    4       10892  5.1
gpu-node-02:8080  gpu-node-02    3       412    2100
gpu-node-03:8080  gpu-node-03    4       10847  5.3
gpu-node-03:8080  gpu-node-03    3       398    1900
gpu-node-04:8080  gpu-node-04    4       10901  5.0
gpu-node-04:8080  gpu-node-04    3       421    2200

  8 rows from 4 node(s)
```

Node 1 has 847 host events with 18.4ms average duration  -  much higher than the other nodes (~2ms). That's the straggler.

### Step 3: Fan-Out Explain  -  Cross-Node Causal Chains

<img src="docs/assets/fleet-explain.gif" width="800" alt="ingero explain --nodes showing causal chains from multiple GPU nodes with severity and node attribution">

```bash
$ ingero explain --nodes gpu-node-01:8080,gpu-node-02:8080,gpu-node-03:8080,gpu-node-04:8080

FLEET CAUSAL CHAINS — 2 chain(s) from 4 node(s)

[HIGH] [gpu-node-01] cuLaunchKernel p99=843us (63.9x p50) — 847 sched_switch events + heavy block I/O
  Root cause: 847 sched_switch events + heavy block I/O
  Fix: Pin training process to dedicated cores with taskset; Add nice -n 19 to background jobs

[MEDIUM] [gpu-node-01] cuMemAlloc p99=932us (5.0x p50) — 855 sched_switch events + heavy block I/O
  Root cause: 855 sched_switch events + heavy block I/O
  Fix: Pin training process to dedicated cores with taskset
```

Both chains are on `gpu-node-01`  -  the other 3 nodes are healthy. The root cause is CPU contention from block I/O on node 1.

### Step 4: AI Fleet Investigation via MCP

Your AI assistant queries the fleet in one MCP tool call:

```
User: "Which node is causing the distributed training stall?"

AI calls: query_fleet(action="chains")

AI: "gpu-node-01 has two causal chains — HIGH severity cuLaunchKernel latency
spike (63.9x p50) caused by 847 scheduler preemptions and heavy block I/O.
The other 3 nodes are clean. Recommendation: pin the training process to
dedicated cores on gpu-node-01 and investigate the I/O source (likely
checkpoint writes or log rotation)."
```

### Step 5: Offline Merge for Air-Gapped Analysis

<img src="docs/assets/merge-export.gif" width="800" alt="ingero merge combining 3 node databases and ingero export producing a Perfetto timeline JSON">

SCP databases from each node and merge locally:

```bash
$ scp gpu-node-01:~/.ingero/ingero.db node-01.db
$ scp gpu-node-02:~/.ingero/ingero.db node-02.db
$ scp gpu-node-03:~/.ingero/ingero.db node-03.db
$ scp gpu-node-04:~/.ingero/ingero.db node-04.db

$ ingero merge node-01.db node-02.db node-03.db node-04.db -o cluster.db

  Merging node-01.db...
    47,003 events, 2 chains, 8 stacks
  Merging node-02.db...
    42,891 events, 0 chains, 6 stacks
  Merging node-03.db...
    41,204 events, 0 chains, 5 stacks
  Merging node-04.db...
    43,102 events, 0 chains, 6 stacks

  Merged 4 database(s) → cluster.db: 174,200 events, 2 chains, 12 unique stacks
```

The merged DB works with all standard tools:

```bash
ingero query -d cluster.db --since 1h
ingero explain -d cluster.db --chains
```

### Step 6: Perfetto Timeline  -  Visual Diagnosis

Export the merged database as a Perfetto trace:

```bash
$ ingero export --format perfetto -d cluster.db -o cluster-trace.json

  Exported 174,200 events + 2 chains → cluster-trace.json (16.2 MB)
```

Open `cluster-trace.json` in [ui.perfetto.dev](https://ui.perfetto.dev):

```
Process tracks:
  gpu-node-01 (rank 0)  ████████░░░░████████████████████░░████████  ← I/O gaps visible
  gpu-node-02 (rank 1)  ████████████████████████████████████████████  ← smooth
  gpu-node-03 (rank 2)  ████████████████████████████████████████████  ← smooth
  gpu-node-04 (rank 3)  ████████████████████████████████████████████  ← smooth

Causal chain markers:
  [HIGH] gpu-node-01: cuLaunchKernel 63.9x p50
  [MEDIUM] gpu-node-01: cuMemAlloc 5.0x p50
```

Each node appears as a separate process track. CUDA events are duration spans. Causal chains are severity-colored instant markers. The I/O gaps on node 1 are immediately visible in the timeline.

### Step 7: Clock Skew Detection

If nodes have drifted clocks, Ingero warns automatically:

```bash
$ ingero query --nodes gpu-node-01:8080,gpu-node-02:8080 --clock-skew-threshold 5ms \
    "SELECT node, count(*) FROM events GROUP BY node"

WARNING: gpu-node-02 is ~47ms ahead of gpu-node-01 (RTT: 2ms)
node              node           count(*)
----------------  -------------  --------
gpu-node-01:8080  gpu-node-01    47003
gpu-node-02:8080  gpu-node-02    42891

  2 rows from 2 node(s)
```

This prevents false causal conclusions  -  if you see "node-A's event happened 20ms before node-B's stall," the clock skew warning tells you whether that ordering is real or an artifact.

> **Note:** The multi-node features above (fan-out queries, offline merge, Perfetto export) are interim solutions for cross-node GPU investigation. A dedicated cluster-level observability and diagnostics tool with native multi-node support is coming soon.

## Install

### Binary Release (recommended)

Download a pre-built binary from [GitHub Releases](https://github.com/ingero-io/ingero/releases/latest).

Archive filenames include the version: `ingero_<version>_linux_<arch>.tar.gz`. Replace `VERSION` below with the latest release (e.g., `0.9.0`):

```bash
# Linux amd64
VERSION=0.9.0
curl -fsSL "https://github.com/ingero-io/ingero/releases/download/v${VERSION}/ingero_${VERSION}_linux_amd64.tar.gz" | tar xz
sudo mv ingero /usr/local/bin/

# Linux arm64 (GH200, Grace Hopper, Graviton)
VERSION=0.9.0
curl -fsSL "https://github.com/ingero-io/ingero/releases/download/v${VERSION}/ingero_${VERSION}_linux_arm64.tar.gz" | tar xz
sudo mv ingero /usr/local/bin/
```

### Docker Image

Multi-arch images (amd64 + arm64) are published to GHCR on every release:

```bash
# Pull the latest image
docker pull ghcr.io/ingero-io/ingero:latest

# Or pin to a specific version
docker pull ghcr.io/ingero-io/ingero:v0.9.0

# Quick test (no root, no GPU needed)
docker run --rm ghcr.io/ingero-io/ingero demo --no-gpu

# System readiness check
docker run --rm --privileged --pid=host ghcr.io/ingero-io/ingero check

# Live eBPF tracing (requires privileges + kernel mounts)
docker run --rm --privileged --pid=host \
  -v /sys/kernel/debug:/sys/kernel/debug \
  -v /sys/kernel/btf:/sys/kernel/btf:ro \
  -v /var/lib/ingero:/var/lib/ingero \
  ghcr.io/ingero-io/ingero trace --record
```

Minimum capabilities (alternative to `--privileged`): `--cap-add=BPF --cap-add=PERFMON --cap-add=SYS_ADMIN`.

> **Note:** eBPF tracing (`trace`, `demo --gpu`) requires `--privileged --pid=host` plus the kernel volume mounts shown above. Without these, only unprivileged commands work (`demo --no-gpu`, `check`, `version`, `explain`, `query`). The `--pid=host` flag shares the host's `/proc`  -  do **not** also bind-mount `-v /proc:/proc:ro` as this causes OCI runtime errors on Docker Desktop and WSL2.

**Data persistence:** The container stores the SQLite database at `/var/lib/ingero/ingero.db` by default. Mount `-v /var/lib/ingero:/var/lib/ingero` to persist data after the container stops. Without this mount, **all trace data is lost** when the container exits.

**Multiple databases:** Use `--db` or the `INGERO_DB` env var to work with different databases:

```bash
# Trace to a named database
docker run --rm --privileged --pid=host \
  -v /var/lib/ingero:/var/lib/ingero \
  -v /sys/kernel/debug:/sys/kernel/debug \
  -v /sys/kernel/btf:/sys/kernel/btf:ro \
  ghcr.io/ingero-io/ingero trace --db /var/lib/ingero/training-run-42.db

# Investigate a specific database
docker run --rm \
  -v /var/lib/ingero:/var/lib/ingero \
  ghcr.io/ingero-io/ingero explain --db /var/lib/ingero/training-run-42.db

# Compare databases from different runs
docker run --rm \
  -v /var/lib/ingero:/var/lib/ingero \
  ghcr.io/ingero-io/ingero query --db /var/lib/ingero/training-run-41.db --since 1h

docker run --rm \
  -v /var/lib/ingero:/var/lib/ingero \
  ghcr.io/ingero-io/ingero query --db /var/lib/ingero/training-run-42.db --since 1h
```

The image is ~10 MB (Alpine 3.20 + statically linked Go binary). When building the dev Dockerfile locally, pass version info via build args:

```bash
docker build -f deploy/docker/Dockerfile \
  --build-arg VERSION=0.9.0 \
  --build-arg COMMIT=$(git rev-parse --short HEAD) \
  --build-arg BUILD_DATE=$(date -u +%Y-%m-%dT%H:%M:%SZ) \
  -t ingero:local .
```

GHCR images have version info baked in automatically via GoReleaser. See `deploy/docker/Dockerfile` for details.

### Build from Source

```bash
# Quick setup: install all build dependencies (Go, clang, llvm) on Ubuntu 22.04/24.04
curl -fsSL https://raw.githubusercontent.com/ingero-io/ingero/main/scripts/install-deps.sh | bash

# Requires clang-14, Linux kernel with BTF
git clone https://github.com/ingero-io/ingero.git
cd ingero
make              # generates eBPF bindings, builds, tests, and lints  -  single command
sudo make install # optional  -  copies binary to /usr/local/bin/ingero
                  # or just use ./bin/ingero directly, or: alias ingero=$PWD/bin/ingero
```

## Requirements

- Linux kernel 5.15+ with BTF (`CONFIG_DEBUG_INFO_BTF=y`)
- NVIDIA driver 550+ with CUDA 11.x, 12.x, or 13.x
- Root / `CAP_BPF` + `CAP_PERFMON` (eBPF requires elevated privileges)
- Tested on: GH200, H100, A100, A10, RTX 4090, RTX 3090 (x86_64 and aarch64)

## Commands

### `ingero check`

Check if your system is ready for eBPF-based GPU tracing.

```text
$ ingero check

Ingero  -  System Readiness Check

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

All checks passed  -  ready to trace!
```

### `ingero trace`

Live event stream with rolling stats, system context, and anomaly detection. Events are recorded to SQLite by default (use `--record=false` to disable). The database is capped at 10 GB rolling storage and auto-purges old events when the limit is reached (see `--max-db`).

```bash
sudo ingero trace                           # auto-detect all CUDA processes for current user
sudo ingero trace --pid 4821               # trace specific process
sudo ingero trace --pid 4821,5032          # trace multiple specific processes
sudo ingero trace --user bob               # trace all CUDA processes owned by bob
sudo ingero trace --record=false           # disable SQLite recording
sudo ingero trace --duration 60s           # stop after 60 seconds
sudo ingero trace --json                   # JSON output (pipe to jq)
sudo ingero trace --verbose                # show individual events
sudo ingero trace --stack=false            # disable stack traces (saves ~0.4-0.6% overhead)
sudo ingero trace --max-db 10g             # limit DB to 10 GB (default), prunes oldest events
sudo ingero trace --max-db 500m            # limit DB to 500 MB (tight disk budget)
sudo ingero trace --max-db 0               # unlimited (no size-based pruning)
sudo ingero trace --deadband 5              # suppress idle snapshots (5% threshold)
sudo ingero trace --deadband 5 --heartbeat 30s  # deadband + force report every 30s
sudo ingero trace --prometheus :9090       # expose Prometheus /metrics endpoint
sudo ingero trace --otlp localhost:4318    # push metrics via OTLP
sudo ingero trace --node gpu-node-07      # tag events with node identity (for multi-node)
```

**Only `trace` needs sudo**  -  it attaches eBPF probes to the kernel. All other commands (`check`, `explain`, `query`, `mcp`, `demo`) run unprivileged. When you run `sudo ingero trace`, the database is written to your home directory (not `/root/`) and chown'd to your user, so non-sudo commands can read it.

**Process targeting:**
- **Default** (no flags): traces all CUDA processes owned by the invoking user (via `SUDO_USER`). On single-user boxes, this means all CUDA processes.
- **`--pid`**: target specific process(es), comma-separated (e.g., `--pid 1234,5678`).
- **`--user`**: target all CUDA processes owned by a specific user (`--user bob`, `--user root`).
- **Dynamic child tracking**: fork events auto-enroll child PIDs for host correlation.

The trace display shows five sections:
1. **System Context**  -  CPU, memory, load, swap with ASCII bar charts (green/yellow/red)
2. **CUDA Runtime API**  -  per-operation p50/p95/p99 latency with anomaly flags (cudaMalloc, cudaLaunchKernel, graphLaunch, etc.)
3. **CUDA Driver API**  -  driver-level operations (cuLaunchKernel, cuMemAlloc, etc.) that cuBLAS/cuDNN call directly
4. **Host Context**  -  scheduler, memory, OOM, and process lifecycle events
5. **CUDA Graph events**  -  graph capture, instantiate, and launch events (when graph-using workloads are traced)

### `ingero explain`

Analyze recorded events from SQLite and produce an incident report with causal chains, root causes, and fix recommendations. Reads from the database populated by `ingero trace`  -  no root needed.

```bash
ingero explain                         # analyze last 5 minutes
ingero explain --since 1h             # last hour
ingero explain --since 2d             # last 2 days
ingero explain --since 1h30m          # human-friendly durations (also: 1w, 3d12h)
ingero explain --last 100             # last 100 events
ingero explain --pid 4821             # filter by specific process
ingero explain --pid 4821,5032        # filter by multiple processes
ingero explain --chains               # show stored causal chains (no re-analysis)
ingero explain --json                 # JSON output for pipelines
ingero explain --from "15:40" --to "15:45"  # absolute time range
ingero explain --per-process              # per-process CUDA API breakdown
ingero explain --per-process --json       # JSON output for pipelines

# Multi-node fleet queries (fan-out to multiple Ingero dashboard APIs)
ingero explain --nodes host1:8080,host2:8080,host3:8080  # cross-node causal chains
```

#### Per-Process Breakdown

For multi-process GPU workloads (RAG pipelines, model serving with workers, multi-tenant GPU sharing), `--per-process` shows a CUDA API breakdown grouped by process:

```
$ ingero explain --per-process --since 5m

PER-PROCESS GPU API BREAKDOWN

  PID 4821 (vllm-worker)
    cuLaunchKernel      12,847 calls   p50=4.8µs   p95=11.2µs   p99=16.1µs
    cudaMemcpyAsync        892 calls   p50=38µs    p95=124µs    p99=891µs
    cudaMallocManaged       14 calls   p50=112µs   p95=2.1ms    p99=8.4ms

  PID 5032 (embedding-svc)
    cuLaunchKernel       3,201 calls   p50=5.1µs   p95=12.8µs   p99=19.4µs
    cudaMemcpy             448 calls   p50=42µs    p95=98µs     p99=412µs

  ⚠ Multi-process GPU contention: 2 processes sharing GPU with CUDA/Driver ops
```

This answers "which process is hogging the GPU?"  -  essential for diagnosing RAG pipeline contention where embedding, retrieval, and generation compete for GPU time.

```
INCIDENT REPORT  -  2 causal chains found (1 HIGH, 1 MEDIUM)

[HIGH] cudaStreamSync p99=142ms (8.5x p50)  -  CPU contention
  Timeline:
    15:41:20  [SYSTEM]  CPU 94%, Load 12.1, Swap 2.1GB
    15:41:20  [HOST]    sched_switch: PID 8821 (logrotate) preempted PID 4821
    15:41:22  [CUDA]    cudaStreamSync 142ms (normally 16.7ms)

  Root cause: logrotate cron job preempted training process 847 times
  Fix: Add `nice -n 19` to logrotate cron, or pin training to dedicated cores
```

### `ingero query`

Query stored events by time range, PID, and operation type. Supports multi-node fleet queries with `--nodes`.

```bash
ingero query --since 1h
ingero query --since 1h --pid 4821
ingero query --since 1h --pid 4821,5032
ingero query --since 30m --op cudaMemcpy --json

# Multi-node fleet queries (fan-out to multiple Ingero dashboard APIs)
ingero query --nodes host1:8080,host2:8080 "SELECT node, source, count(*) FROM events GROUP BY node, source"
ingero query --nodes host1:8080,host2:8080,host3:8080 "SELECT node, count(*) FROM events GROUP BY node"
```

Fleet queries fan out the SQL to each node's `/api/v1/query` endpoint, concatenate results with a `node` column prepended, and display a unified table. Partial failures return results from reachable nodes with warnings for unreachable ones. Clock skew between nodes is detected automatically (configurable via `--clock-skew-threshold`, default 10ms).

Configure default fleet nodes in `ingero.yaml` under `fleet.nodes` to avoid repeating `--nodes` on every command.

Storage uses SQLite with size-based pruning (default 10 GB via `--max-db`). Data is stored locally at `~/.ingero/ingero.db`  -  nothing leaves your machine.

### `ingero mcp`

Start an MCP (Model Context Protocol) server for AI agent integration.

```bash
ingero mcp                        # stdio (for Claude Code / MCP clients)
ingero mcp --http :8080           # HTTPS on port 8080 (TLS 1.3, auto-generated self-signed cert)
ingero mcp --http :8080 --tls-cert cert.pem --tls-key key.pem  # custom TLS certificate
```

> **Note:** The `--http` flag enables the Streamable HTTP transport  -  all connections use **TLS 1.3 only** (no plain HTTP). When no `--tls-cert`/`--tls-key` is provided, ingero auto-generates an ephemeral self-signed ECDSA P-256 certificate. Use `curl -k` to skip certificate verification for self-signed certs.

**AI-first analysis**: MCP responses use telegraphic compression (TSC) by default, reducing token count by ~60%. Set `{"tsc": false}` per request for verbose output.

**MCP tools:**

| Tool | Description |
|------|-------------|
| `get_check` | System diagnostics (kernel, BTF, NVIDIA, CUDA, GPU model) |
| `get_trace_stats` | CUDA + host statistics (p50/p95/p99 or aggregate fallback for large DBs) |
| `get_causal_chains` | Causal chains with severity ranking and root cause (deduplicated, top 10 by default) |
| `get_stacks` | Resolved call stacks for CUDA/driver operations (symbols, source files, timing) |
| `graph_lifecycle` | CUDA Graph lifecycle timeline for a PID: capture, instantiate, launch sequences |
| `graph_frequency` | Graph launch frequency per executable: hot/cold classification, pool saturation |
| `run_demo` | Run synthetic demo scenarios |
| `get_test_report` | GPU integration test report (JSON) |
| `run_sql` | Execute read-only SQL for ad-hoc analysis |
| `query_fleet` | Fan-out query across multiple Ingero nodes (chains, ops, overview, sql) with clock skew detection |

**MCP prompts:**

| Prompt | Description |
|--------|-------------|
| `/investigate` | Guided investigation workflow - walks the AI through stats, chains, and SQL to diagnose GPU issues. Works with any MCP client. |

**Works with any AI, not just Claude.** Use local open-source models via [ollmcp](https://github.com/jonigl/mcp-client-for-ollama) (Ollama MCP client):

```bash
# Install ollmcp and pull a model
pip install mcp-client-for-ollama
ollama pull minimax-m2.7:cloud

# Create a config pointing to Ingero's MCP server
cat > /tmp/ingero-mcp.json << 'EOF'
{"mcpServers":{"ingero":{"command":"ingero","args":["mcp","--db","trace.db"]}}}
EOF

# Start investigating - /investigate triggers the guided workflow
ollmcp -m minimax-m2.7:cloud -j /tmp/ingero-mcp.json
```

Tested with MiniMax M2.7 and Qwen 3.5 via Ollama on saved investigation databases. Also works with Claude Desktop, Cursor, and any MCP-compatible client.

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

### `ingero dashboard`

Start a browser-based GPU monitoring dashboard backed by the SQLite event store. Shows live system metrics, CUDA operation latencies, causal chains, and a capability manifest (grayed-out panels for metrics Ingero doesn't yet collect, with tooltips naming the required external tool). Requires `ingero trace` to be running (or to have run recently).

```bash
ingero dashboard                           # HTTPS on :8080 (self-signed TLS 1.3)
ingero dashboard --addr :9090              # custom port
ingero dashboard --db /path/to/ingero.db   # custom database
ingero dashboard --tls-cert cert.pem --tls-key key.pem  # custom TLS certificate
ingero dashboard --no-tls                  # plain HTTP (for fleet queries on trusted networks)

# Remote access via SSH tunnel:
ssh -L 8080:localhost:8080 user@gpu-vm
# Then open https://localhost:8080 in browser
```

**No sudo needed**  -  the dashboard reads from the SQLite database populated by `ingero trace`.

**Security:** TLS 1.3 only. Auto-generates an ephemeral self-signed ECDSA P-256 certificate (valid 24h) if no `--tls-cert`/`--tls-key` provided. DNS rebinding protection rejects requests from non-localhost Host headers.

**API endpoints:**

| Endpoint | Description |
|----------|-------------|
| `GET /api/v1/overview` | Event count, chain count, latest system snapshot, GPU info, top causal chain |
| `GET /api/v1/ops?since=5m` | Per-operation latency stats (percentile or aggregate mode) |
| `GET /api/v1/chains?since=1h` | Stored causal chains with severity, root cause, timeline |
| `GET /api/v1/snapshots?since=60s` | System metric time series (CPU, memory, swap, load) |
| `GET /api/v1/capabilities` | Metric availability manifest (available vs. grayed-out with required tool) |
| `GET /api/v1/graph-metrics` | CUDA Graph metrics: capture/launch rates, instantiation durations |
| `GET /api/v1/graph-events` | Recent CUDA Graph events with handles and durations |
| `POST /api/v1/query` | Execute read-only SQL (used by fleet fan-out queries) |
| `GET /api/v1/time` | Server wall-clock timestamp (used for clock skew detection) |

### `ingero merge`

Merge SQLite databases from multiple Ingero nodes into a single queryable database for offline cross-node analysis. Useful in air-gapped environments or when you prefer offline analysis over fan-out queries.

```bash
ingero merge node-a.db node-b.db node-c.db -o cluster.db       # merge 3 node databases
ingero merge old.db --force-node legacy-node -o merged.db       # assign node identity to legacy DBs

# Then use standard tools on the merged database
ingero query -d cluster.db --since 1h
ingero explain -d cluster.db --chains
ingero export --format perfetto -d cluster.db -o trace.json
```

Node-namespaced event IDs (`{node}:{seq}`) ensure zero collisions on merge. Stack traces are deduplicated by hash. Sessions are re-keyed. Clock skew between traces is detected and warned (configurable via `--clock-skew-threshold`, default 100ms).

### `ingero export`

Export event data to visualization formats. Currently supports Perfetto/Chrome Trace Event Format for timeline visualization in [ui.perfetto.dev](https://ui.perfetto.dev) or `chrome://tracing`.

```bash
# From a local or merged database
ingero export --format perfetto -d ~/.ingero/ingero.db -o trace.json
ingero export --format perfetto -d cluster.db -o trace.json --since 5m

# Fan-out mode (fetches from multiple nodes via fleet API)
ingero export --format perfetto --nodes node-1:8080,node-2:8080 -o trace.json
```

Opens in Perfetto UI with one process track per node/rank, CUDA events as duration spans, and causal chains as severity-colored instant markers. Multi-node traces show side-by-side timelines for spotting which rank stalled while others waited.

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
ingero v0.9.0 (commit: 676ab87, built: 2026-03-17)
```

## Stack Tracing

Stack tracing is **on by default**  -  every CUDA/Driver API event captures the full userspace call chain. Shows **who called cudaMalloc**  -  from the CUDA library up through PyTorch, your Python code, and all the way to `main()`. GPU-measured overhead is **0.4-0.6%** (within noise on RTX 3090 through H100). Disable with `--stack=false` if needed.

```bash
sudo ingero trace --json               # JSON with resolved stack traces (stacks on by default)
sudo ingero trace --debug              # debug output shows resolved frames on stderr
sudo ingero demo --gpu --json          # GPU demo with stack traces (needs sudo)
ingero explain                         # post-hoc causal analysis from DB (no sudo)
sudo ingero trace --stack=false        # disable stacks if needed
```

**Maximum depth**: 64 native frames (eBPF `bpf_get_stack`). This covers deep call chains from CUDA → cuBLAS/cuDNN → PyTorch C++ → Python interpreter and up to `main()` / `_start`.

### Python Stack Attribution

For Python workloads (PyTorch, TensorFlow, etc.), Ingero extracts **CPython frame information** directly from process memory. When a native frame is inside libpython's eval loop, the corresponding Python source frames are injected into the stack:

```
[Python] train.py:8 in train_step()
[Python] train.py:13 in main()
[Python] train.py:1 in <module>()
[Native] cublasLtSSSMatmul+0x1d4 (libcublasLt.so.12)
[Native] cublasSgemm_v2+0xa6 (libcublas.so.12)
[Native] (libtorch_cuda.so)
```

Supported Python versions: **3.10, 3.11, 3.12** (covers Ubuntu 22.04 default, conda default, and most production deployments). Version detection is automatic via `/proc/[pid]/maps`.

### JSON Output with `--stack`

Real output from a PyTorch ResNet-50 training run on A100 SXM4  -  a cuBLAS matmul kernel launch captured via Driver API uprobes, with the full call chain from Python through cuBLAS to the GPU:

```json
{
  "timestamp": "2026-02-25T12:06:24.753983243Z",
  "pid": 11435,
  "tid": 11435,
  "source": "driver",
  "op": "cuLaunchKernel",
  "duration_ns": 10900,
  "duration": "11us",
  "stack": [
    {"ip": "0x0", "py_file": "train.py", "py_func": "train_step", "py_line": 8},
    {"ip": "0x0", "py_file": "train.py", "py_func": "main", "py_line": 13},
    {"ip": "0x0", "py_file": "train.py", "py_func": "<module>", "py_line": 1},
    {"ip": "0x765bb62cfa44", "symbol": "cublasLtSSSMatmul+0x1d4", "file": "libcublasLt.so.12.8.4.1"},
    {"ip": "0x765be7734046", "symbol": "cublasSgemm_v2+0xa6", "file": "libcublas.so.12.8.4.1"},
    {"ip": "0x765c2517fa49", "file": "libtorch_cuda.so"}
  ]
}
```

This kernel launch is invisible to CUDA Runtime profilers  -  cuBLAS calls `cuLaunchKernel` directly. Only Ingero's Driver API uprobes capture it.

### Debug Output with `--stack --debug`

```
[DEBUG] stack trace for cuLaunchKernel (PID 11435, TID 11435, 6 frames):
[DEBUG]   [0] [Python] train.py:8 in train_step()
[DEBUG]   [1] [Python] train.py:13 in main()
[DEBUG]   [2] [Python] train.py:1 in <module>()
[DEBUG]   [3] cublasLtSSSMatmul+0x1d4 (libcublasLt.so.12)
[DEBUG]   [4] cublasSgemm_v2+0xa6 (libcublas.so.12)
[DEBUG]   [5] (libtorch_cuda.so)
```

## OTEL Integration (Optional)

OTEL export is **off by default**  -  enabled only when you pass `--otlp` or `--prometheus`.

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

Zero external dependencies  -  no OTEL SDK import. The JSON payload is constructed directly using Go's standard library.

## How It Works

```
┌────────────────────────────────────────────────────────────────┐
│  User Space                                                    │
│                                                                │
│  ┌─────────┐    ┌─────────────┐  ┌───────┐    ┌─────────────┐  │
│  │  CUDA   │    │   ingero    │  │SQLite │    │MCP Server   │  │
│  │  App    │    │   agent     │─►│  DB   │◄───│(stdio/HTTPS)│  │
│  │(PyTorch)│    │             │  │       │    └─────────────┘  │
│  │         │    │             │  │       │   ┌───────────┐     │
│  │         │    │             │  │       │◄──│ Dashboard │     │
│  │         │    │             │  └───────┘   │  (HTTPS)  │     │
│  └──┬──┬───┘    │ ┌──────────┐│              └───────────┘     │
│     │  │        │ │ causal   ││   ┌───────────┐                │
│     │  │        │ │ engine   ││   │ OTLP /    │                │
│     │  │        │ └──────────┘│──►│ Prometheus│                │
│     │  │        └──┬──┬──┬────┘   └───────────┘                │
│     │  │           │  │  │ ▲                                   │
│     │  │           │  │  │ │ ring buffers                      │
│─────┼──┼───────────┼──┼──┼─┼───────────────────────────────────│
│     │  ▼           │  ▼  ▼ │                                   │
│     │ ┌─────────┐  │ ┌────────────────────┐                    │
│     │ │libcuda  │◄─┤ │  eBPF uprobes      │  (Driver API)      │
│     │ │  .so    │  │ │  cuLaunchKernel    │                    │
│     │ └─────────┘  │ │  cuMemcpy/Alloc    │                    │
│     ▼              │ └────────────────────┘                    │
│  ┌─────────┐       │ ┌────────────────────┐                    │
│  │libcudart│◄──────┘ │  eBPF uprobes      │  (Runtime API)     │
│  │  .so    │◄────────│  cudaLaunchKernel  │                    │
│  └─────────┘         │  cudaMalloc/Memcpy │                    │
│                      │  Graph: Capture,   │                    │
│                      │  Instantiate,Launch│                    │
│                      └────────────────────┘                    │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │  eBPF tracepoints (sched_switch, mm_page_alloc, oom,    │   │
│  │  sched_process_exec/exit/fork)                          │   │
│  └─────────────────────────────────────────────────────────┘   │
│                                                                │
│  Kernel Space        /proc → CPU%, Mem%, Load, Swap            │
└────────────────────────────────────────────────────────────────┘
```

1. **Discover**  -  scans `/proc` for processes linked to `libcudart.so`, finds `libcuda.so` automatically
2. **Attach**  -  eBPF probes load onto CUDA runtime uprobes, driver uprobes, and host tracepoints
3. **Capture**  -  eBPF programs record PID, TID, timestamps into per-layer ring buffers
4. **System**  -  reads CPU/memory/load/swap from `/proc` once per second
5. **Stats**  -  computes rolling p50/p95/p99 per operation, flags anomalies
6. **Correlate**  -  assembles causal chains (SYSTEM + HOST + CUDA Runtime + CUDA Driver + CUDA Graph) by timestamp and PID
7. **Store**  -  writes events to SQLite with size-based pruning (`--max-db 10g` default). Disable recording with `--record=false`
8. **Export**  -  pushes metrics via OTLP or serves Prometheus `/metrics` (optional)
9. **Serve**  -  exposes diagnostics to AI agents via MCP (stdio or HTTPS/TLS 1.3)
10. **Dashboard**  -  browser-based HTTPS dashboard reads from SQLite, shows ops/chains/snapshots/capabilities with auto-polling
11. **Fleet**  -  fan-out queries across multiple nodes via dashboard API, merge offline databases, detect clock skew, export to Perfetto timeline

## Integration Testing

Validated on 6 GPU models across 3 cloud providers (TensorDock, Lambda Labs, Azure). Stack tracing is on by default. GPU-measured overhead: **0.4-1.7%** (within noise).

| GPU | VRAM | Tests | Pass | Fail | Warn | Stack OH | Stack Cov |
|-----|------|-------|------|------|------|----------|-----------|
| GH200 | 480 GB | 80 | 76 | 0 | 4 | +1.6% | 99.8% |
| A100 SXM4 | 40 GB | 80 | 76 | 0 | 4 | +0.9% | 99.4% |
| A10 | 24 GB | 80 | 76 | 0 | 4 | -0.1% | 99.2% |
| H100 (PCIe / SXM5) | 80 GB | 62 | 62 | 0 | 0 | +1.7% | 99.5% |
| RTX 4090 | 24 GB | 34 | 34 | 0 | 0 | +0.6% | 99.9% |
| RTX 3090 | 24 GB | 34 | 34 | 0 | 0 |  -  |  -  |

76/80 integration tests PASS (0 FAIL, 4 WARN) on GPUs tested with v0.8. Tested architectures: x86_64 and aarch64 (GH200 Grace Hopper).

## What Ingero Addresses Today

Ingero addresses 25 documented GPU problems across training, inference, and AI agent workloads:

| # | GPU Problem | Severity | How Ingero Detects It |
|---|-------------|----------|----------------------|
| 1 | NCCL hangs & distributed training deadlocks | CRITICAL | `sched_switch` shows blocked rank + CUDA sync timing. TCP retransmit tracing identifies network-caused hangs |
| 2 | GPU underutilization / data pipeline starvation | CRITICAL | Host scheduler + `cudaStreamSync` + `cudaMemcpy` pipeline bubble diagnosis. Block I/O shows DataLoader disk bottleneck |
| 3 | CUDA OOM & memory fragmentation | CRITICAL | `cudaMalloc`/`cuMemAlloc` allocation pattern tracing. `cudaMallocManaged` adds managed-memory over-subscription detection |
| 4 | Silent data corruption (SDC) | CRITICAL | Anomalous kernel timing as indirect signal (limited) |
| 5 | Inference cost explosion (multi-step agents) | CRITICAL | CUDA API burst/idle patterns per agent session |
| 6 | KV cache pressure & preemption cascades | CRITICAL | `cudaMalloc` patterns + `cudaStreamSync` spikes during preemption. Managed-memory page fault detection |
| 6b | CUDA Graph re-capture latency spikes (vLLM, torch.compile) | HIGH | Graph lifecycle tracing: capture/instantiate/launch rates, pool exhaustion detection, OOM during capture, CPU contention during launch |
| 7 | GPU hardware failures at scale | HIGH | `cudaMemcpy` baseline drift, `sched_switch` frequency anomalies |
| 8 | CPU bottleneck in GPU serving | HIGH | `sched_switch` on inference process + `cudaStreamSync` idle gaps |
| 9 | GPU idle waste during agent tool execution | HIGH | CUDA API silence periods correlated with host process activity. TCP tracing shows "GPU idle during 2s HTTP tool call" |
| 10 | GPU memory leaks in long-running services | HIGH | `cudaMalloc`/`cudaFree` imbalance tracking over time, per-container via cgroup |
| 11 | Mixed precision (AMP) instability | HIGH | Anomalous kernel timing (skipped updates = fast sync) |
| 12 | Goodput loss (training efficiency gap) | HIGH | Scheduler preemption, memcpy latency, pipeline bubbles. Block I/O shows checkpoint write + data read overhead |
| 13 | GPU scheduling & orchestration failures (K8s) | HIGH | Per-cgroup `sched_switch` latency + pod/namespace metadata. Auto-discovers `nvidia.com/gpu` pods |
| 14 | Model swapping latency (multi-model agents) | HIGH | `cudaMalloc` + `cudaMemcpy` patterns during model load. Block I/O shows disk→CPU transfer time |
| 15 | CUDA device-side asserts & illegal memory access | MEDIUM | CUDA API call sequence + stack traces before crash |
| 16 | NVIDIA driver / CUDA version incompatibility | MEDIUM | Uprobe attachment failure = library/driver mismatch signal |
| 17 | Thermal throttling & power limit throttling | MEDIUM | Kernel duration trending over time |
| 18 | Noisy neighbor / multi-tenant GPU interference | MEDIUM | Per-cgroup `sched_switch` latency + CUDA API latency correlation. Noisy neighbor detection via cgroup_schedstat |
| 19 | Cold start / model loading latency | MEDIUM | Full cold start sequence via CUDA API timing. Block I/O completes disk→CPU→GPU pipeline |
| 20 | Multi-GPU tensor parallel communication overhead | MEDIUM | Host-side straggler detection via `sched_switch` + CUDA sync. TCP retransmit tracing on NCCL ports |
| 21 | RAG pipeline GPU contention | MEDIUM | Per-process CUDA API breakdown (`explain --per-process`)  -  shows which process is hogging GPU time |
| 22 | Checkpoint save/load failures | MEDIUM | Memory spike detection + I/O blocking in `cudaStreamSync`. Block I/O shows actual write latency + NFS timeouts |
| 23 | PCIe bottleneck (KV cache swap, model loading) | MEDIUM | `cudaMemcpy` per-operation tracing with direction/size/duration. `cudaMallocManaged` page migration + Block I/O shows NVMe-PCIe contention |
| 24 | Loss spikes (non-AMP) | LOW-MED | System event correlation with loss timing |
| 25 | Triton Inference Server multi-GPU bugs | LOW-MED | CUDA API tracing on Triton processes |

## FAQ

**Is it safe for production?**
Yes. eBPF programs are verified by the kernel before loading  -  they cannot crash the system. Probes add <2% overhead including stack tracing (0.4-0.6% measured across RTX 3090, RTX 4090, A10, A100, H100 with PyTorch workloads).

**Does it require code changes?**
No. Ingero attaches to `libcudart.so` and kernel tracepoints at the OS level. Your application code is untouched. Traces any language  -  Python, C++, Java  -  anything linked against libcudart.so.

**What GPUs are supported?**
Any NVIDIA GPU with driver 550+ and CUDA 11.x/12.x. Tested on GH200 (aarch64), H100, A100, A10, RTX 4090, RTX 3090 (x86_64). Works on AWS Deep Learning AMIs (auto-discovers versioned `libcudart.so`).

**Does it work in containers?**
Yes. eBPF programs execute in kernel space  -  the container just loads them via syscalls. Run with `--privileged` (or `--cap-add=BPF,PERFMON,SYS_ADMIN`), `--pid=host`, and mount `/proc`, `/sys/kernel/debug`, and `/sys/kernel/btf`. The host kernel must have BTF enabled. Pre-built images are available at `ghcr.io/ingero-io/ingero`  -  see the [Docker Image](#docker-image) install section. This is the same pattern used by Falco, Tetragon, and other eBPF DaemonSets.

**Where is data stored?**
Locally in `~/.ingero/ingero.db` (SQLite). Nothing leaves your machine. Size-based pruning keeps the DB under 10 GB by default. With `--record-all`, this covers a few hours of heavy GPU load; with selective storage (default), it lasts much longer. Configure with `--max-db` (e.g., `--max-db 500m`, `--max-db 0` for unlimited). Use `--db /path/to/file.db` for a custom location.

**Does it check for updates?**
Yes. On interactive commands (`trace`, `demo`, `explain`, `check`), ingero checks GitHub Releases for newer versions (once per 24 hours, cached in `~/.ingero/update-check`). The check runs in the background and never delays your command. Set `INGERO_NO_UPDATE_NOTIFIER=1` to disable. Skipped for `query`, `mcp`, `version`, and dev builds.

## License

**Ingero is 100% free and open source.** Use it for anything  -  personal, commercial, enterprise, embed it in your product, modify it, redistribute it. No usage restrictions, no phone-home, no paid tiers required.

Dual-licensed following the standard eBPF split-licensing model (same as Cilium, Falco, and most eBPF projects):

* **User-Space** (Go agent, CLI, causal engine, SQLite, MCP): [Apache License 2.0](LICENSE)  -  maximum enterprise compatibility, no copyleft.
* **Kernel-Space** (eBPF C code in `bpf/`): [GPL-2.0](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html) OR [BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause)  -  GPL-2.0 is required by the Linux kernel's BPF subsystem; BSD-3-Clause permits embedding in non-GPL toolchains.
