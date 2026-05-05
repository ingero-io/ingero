# Ingero  -  GPU Causal Observability

[![Go Report Card](https://goreportcard.com/badge/github.com/ingero-io/ingero)](https://goreportcard.com/report/github.com/ingero-io/ingero)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![GitHub Release](https://img.shields.io/github/v/release/ingero-io/ingero)](https://github.com/ingero-io/ingero/releases)
[![CI](https://github.com/ingero-io/ingero/actions/workflows/ci.yml/badge.svg)](https://github.com/ingero-io/ingero/actions/workflows/ci.yml)
[![MCP](https://img.shields.io/badge/MCP-server-blue)](https://glama.ai/mcp/servers/ingero-io/ingero)

**Featured in:**
[awesome-ebpf](https://github.com/qmonnet/awesome-ebpf) ·
[awesome-observability](https://github.com/adriannovegil/awesome-observability) ·
[awesome-opentelemetry](https://github.com/magsther/awesome-opentelemetry) ·
[awesome-sre-tools](https://github.com/SquadcastHub/awesome-sre-tools) ·
[awesome-cloud-native](https://github.com/rootsongjc/awesome-cloud-native) ·
[awesome-profiling](https://github.com/msaroufim/awesome-profiling) ·
[Awesome-GPU](https://github.com/Jokeren/Awesome-GPU) ·
[awesome-gpu-engineering](https://github.com/goabiaryan/awesome-gpu-engineering) ·
[awesome-mcp-servers](https://github.com/punkpeye/awesome-mcp-servers) ·
[awesome-devops-mcp-servers](https://github.com/rohitg00/awesome-devops-mcp-servers) ·
[MCP Registry](https://registry.modelcontextprotocol.io) ·
[Glama](https://glama.ai/mcp/servers/ingero-io/ingero) ·
[mcpservers.org](https://mcpservers.org)

<!-- ingero-version:install-header product=ingero channel=stable -->
**Version: 0.14.0**

**The only GPU observability tool your AI assistant can talk to.**

*"What caused the GPU stall?" → "`forward()` at `train.py:142`  -  cudaMalloc spiking 48ms during CPU contention. 9,829 calls, 847 scheduler preemptions."*

Ingero is a production-grade eBPF agent that traces the full chain  -  from Linux kernel events through CUDA API calls to your Python source lines  -  with **<2% overhead**, **zero code changes**, and **one binary**.

<img src="docs/assets/readme-demo-incident.gif" width="800" alt="ingero demo incident — CPU contention causes GPU latency spike, full causal chain diagnosis with root cause and fix recommendation">

## Try it in 60 seconds

```bash
# Install (Linux amd64; arm64 / Docker / source variants are below)
# ingero-version:install-curl product=ingero channel=stable
VERSION=0.14.0
curl -fsSL "https://github.com/ingero-io/ingero/releases/download/v${VERSION}/ingero_${VERSION}_linux_amd64.tar.gz" | tar xz
sudo mv ingero /usr/local/bin/

# Trace your GPU workload
sudo ingero trace

# Explain what happened (no sudo)
ingero explain --since 5m
```

No GPU? `ingero demo --no-gpu incident` runs the full causal-chain diagnosis on
synthetic data; no root, no GPU, no ceremony.

After that:

- For a single node deployment: check out a real AI investigation session walked through end-to-end on A100 + GH200,
  see [`docs/ml_eng_sample_investigation_session.md`](docs/ml_eng_sample_investigation_session.md).
- For multi-node cluster deployment (Kubernetes / bare-metal / Docker,
  fleet-wide straggler detection, cluster architecture), see
  [`docs/quickstart_fleet.md`](docs/quickstart_fleet.md).

## What you get

- **Causal chains, not just metrics.** Ingero correlates CPU scheduler events, CUDA Runtime API calls, CUDA Driver API calls, CUDA Graph lifecycle, and NCCL collectives by timestamp + PID, then writes a timeline that explains *why* throughput dropped, not just *that* it did. `ingero explain` ships the chain with a fix recommendation.
- **Driver-level visibility most profilers miss.** cuBLAS / cuDNN / `torch.compile` call `cuLaunchKernel` directly via the Driver API and bypass the Runtime. Ingero traces both layers, so the kernels other tools never see show up here.
- **Python source attribution from `cudaMalloc`.** For PyTorch / TensorFlow workloads, Ingero extracts CPython frames from process memory and injects `[Python] file.py:line in func()` into the stack alongside native frames. CPython 3.10 / 3.11 / 3.12. The userspace walker is the default; an in-kernel eBPF walker handles `kernel.yama.ptrace_scope=3` hardened systems.
- **AI-queryable in one command.** `ingero mcp` exposes a Model Context Protocol server (stdio or HTTPS / TLS 1.3). Your AI assistant asks "what caused the GPU stall?" and gets back a resolved causal chain with Python source lines, no SQL, no hex addresses. Works with Claude Code, Cursor, and any local model via Ollama.
- **<2% overhead, zero code changes, single binary.** GPU-measured: 0.4-1.7% across RTX 3090 through H100 with stack tracing on. No SDK to import, no agent to run alongside your training process, no Helm chart to start with: one `curl` install, one `sudo ingero trace`, one `ingero explain`.

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

---

**This README covers single-node GPU tracing and investigation.** For multi-node distributed training diagnostics (fan-out queries across nodes, offline database merge, Perfetto timeline export, clock skew detection), see the [Multi-Node Investigation Walkthrough](investigations/README.md#multi-node-investigation-walkthrough).

---

## Install

### Binary Release (recommended)

Download a pre-built binary from [GitHub Releases](https://github.com/ingero-io/ingero/releases/latest).

Archive filenames include the version: `ingero_<version>_linux_<arch>.tar.gz`. Replace `VERSION` below with the latest release (e.g., `0.10.0`):

```bash
# Linux amd64
# ingero-version:install-archive-amd64 product=ingero channel=stable
VERSION=0.14.0
curl -fsSL "https://github.com/ingero-io/ingero/releases/download/v${VERSION}/ingero_${VERSION}_linux_amd64.tar.gz" | tar xz
sudo mv ingero /usr/local/bin/

# Linux arm64 (GH200, Grace Hopper, Graviton)
# ingero-version:install-archive-arm64 product=ingero channel=stable
VERSION=0.14.0
curl -fsSL "https://github.com/ingero-io/ingero/releases/download/v${VERSION}/ingero_${VERSION}_linux_arm64.tar.gz" | tar xz
sudo mv ingero /usr/local/bin/
```

### Docker Image

Multi-arch images (amd64 + arm64) are published to GHCR on every release:

```bash
# Pull the latest image
docker pull ghcr.io/ingero-io/ingero:latest

# Or pin to a specific version
# ingero-version:docker-pull-example product=ingero channel=stable
docker pull ghcr.io/ingero-io/ingero:v0.14.0

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
# ingero-version:docker-build-arg product=ingero channel=stable
docker build -f deploy/docker/Dockerfile \
  --build-arg VERSION=0.14.0 \
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

| Command | Purpose |
|-|-|
| `ingero check` | system readiness (kernel, BTF, driver, GPU, CUDA libs, processes) |
| `ingero trace` | live event stream + record to SQLite (the only command that needs sudo) |
| `ingero explain` | causal chains + root cause + fix recommendation from recorded events |
| `ingero query` | SQL over the local DB; `--nodes` fans out across multi-node fleets |
| `ingero mcp` | Model Context Protocol server for AI agents (stdio or HTTPS/TLS 1.3) |
| `ingero dashboard` | browser dashboard backed by SQLite (HTTPS/TLS 1.3) |
| `ingero merge` | merge SQLite DBs from multiple nodes into one queryable database |
| `ingero export` | export to Perfetto / Chrome Trace Event Format for timeline UIs |
| `ingero demo` | 6 synthetic scenarios; works without a GPU via `--no-gpu` |
| `ingero version` | print version |

For full flag reference, examples, and per-command output samples, see [`docs/commands.md`](docs/commands.md). For the AI / MCP integration deep-dive, see [`docs/commands.md#ingero-mcp`](docs/commands.md#ingero-mcp).

## Stack Tracing

Stack tracing is **on by default**: every CUDA / Driver API event captures the full userspace call chain. Shows **who called cudaMalloc**: from the CUDA library up through PyTorch, your Python code, and all the way to `main()`. GPU-measured overhead is **0.4-0.6%** (within noise on RTX 3090 through H100). Disable with `--stack=false` if needed.

For Python workloads, Ingero extracts CPython frame information directly from process memory (versions 3.10 / 3.11 / 3.12 supported) and injects `[Python]` source frames into the stack alongside `[Native]` frames.

See [`docs/stack_tracing.md`](docs/stack_tracing.md) for the userspace vs eBPF walker selection, JSON output examples, and `kernel.yama.ptrace_scope` troubleshooting.

## Integrations

| Integration | What it does | Reference |
|-|-|-|
| **MCP server** (AI agents) | `ingero mcp` exposes 10 tools + a `/investigate` prompt to any MCP-compatible client (Claude Code, Cursor, Ollama). Stdio or HTTPS / TLS 1.3. | [`docs/commands.md#ingero-mcp`](docs/commands.md#ingero-mcp) |
| **OTLP / Prometheus** | `--otlp HOST:PORT` (HTTP JSON) or `--prometheus :PORT` (pull). Standard semantic conventions; compatible with OTel Collector, Grafana Alloy / Cloud, Datadog, New Relic. | [`docs/otlp.md`](docs/otlp.md) |
| **Browser dashboard** | `ingero dashboard` serves an HTTPS dashboard backed by the SQLite trace DB. Live ops / chains / snapshots. | [`docs/commands.md#ingero-dashboard`](docs/commands.md#ingero-dashboard) |
| **Multi-node cluster** | Real-time cluster-wide straggler detection via the Ingero Fleet collector. Agents push OTLP via `ingero fleet-push`. K8s / bare-metal / Docker. | [`docs/quickstart_fleet.md`](docs/quickstart_fleet.md) |
| **Perfetto / Chrome tracing** | `ingero export --format perfetto` produces a timeline you open in [ui.perfetto.dev](https://ui.perfetto.dev) or `chrome://tracing`. One track per node / rank. | [`docs/commands.md#ingero-export`](docs/commands.md#ingero-export) |

## Architecture

The agent loads its own eBPF probes onto `libcudart.so` / `libcuda.so` uprobes and host tracepoints, drains ringbuffers from user space, assembles causal chains (SYSTEM + HOST + CUDA Runtime + CUDA Driver + CUDA Graph + NCCL), and writes results to local SQLite. The MCP server, Prometheus / OTLP exporter, and HTTPS dashboard all read from that SQLite. Pipeline: **discover → attach → capture → correlate → store → serve.**

For the full ASCII diagram, per-stage detail, and the cluster-mode (multi-node Fleet) architecture, see [`docs/architecture.md`](docs/architecture.md). For multi-node deployment, see [`docs/quickstart_fleet.md`](docs/quickstart_fleet.md).

Validated on 6 GPU models (GH200, A100 SXM4, A10, H100 PCIe/SXM5, RTX 4090, RTX 3090) across TensorDock, Lambda Labs, and Azure with x86_64 and aarch64. Stack-on overhead 0.4-1.7% measured.

## What Ingero detects (CRITICAL & HIGH)

| GPU Problem | Severity | How Ingero detects it |
|-|-|-|
| NCCL hangs & distributed-training deadlocks | CRITICAL | Direct `ncclAllReduce` / `ncclSend` / `ncclRecv` uprobes (v0.12.0+) measure per-collective wall time with rank / `comm_id_hash` / `nranks` correlation; `sched_switch` + TCP retransmit as cross-checks. |
| GPU underutilization / data pipeline starvation | CRITICAL | Host scheduler + `cudaStreamSync` + `cudaMemcpy` pipeline-bubble diagnosis; block-I/O shows DataLoader disk bottleneck. |
| CUDA OOM & memory fragmentation | CRITICAL | `cudaMalloc` / `cuMemAlloc` allocation patterns; managed-memory over-subscription via `cudaMallocManaged`. |
| KV-cache pressure & preemption cascades | CRITICAL | `cudaMalloc` patterns + `cudaStreamSync` spikes during preemption; managed-memory page-fault detection. |
| CPU bottleneck in GPU serving | HIGH | `sched_switch` on inference process + `cudaStreamSync` idle gaps. |
| Multi-process GPU contention (RAG, multi-tenant) | HIGH | Per-process CUDA API breakdown (`explain --per-process`); per-cgroup `sched_switch`. |
| GPU memory leaks in long-running services | HIGH | `cudaMalloc` / `cudaFree` imbalance tracking, per-container via cgroup. |
| GPU scheduling & orchestration failures | HIGH | Per-cgroup `sched_switch` + multi-orchestrator metadata: K8s, Slurm, ECS, Docker / containerd (v0.12.3+). |

Full list of detected issues: [`docs/detections.md`](docs/detections.md)

## More Reading

The combined entry point for: general questions, known issues, recurring
patterns Ingero auto-detects, troubleshooting cheatsheet, and advanced
tuning. Each subsection collapses on its own; click to expand.

<details>
<summary>General FAQ (production safety, container support, GPU support, storage)</summary>

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

</details>

<details>
<summary>Known issues + Walker roadmap</summary>

- **Multiprocess CUDA via `fork()`.** NVIDIA's CUDA driver doesn't support `fork()` after the parent has initialized a CUDA context — children can't use CUDA. The eBPF walker inherits the parent's walker state to the child synchronously on fork, but a child that can't call CUDA won't trigger the walker regardless. For multiprocess CUDA workloads, use `torch.multiprocessing.set_start_method('spawn')` (or Ray/torchrun spawn equivalents); fresh spawn-style processes initialize their own CUDA context and get full walker coverage through the normal dynamic-PID path.

- **Ubuntu 24.04 + distro-patched CPython 3.12 on the userspace walker.** Ubuntu's patched CPython has struct offsets that differ from upstream, and Ubuntu 24.04 doesn't ship `python3.12-dbgsym` in the main archive. The userspace walker falls back to hardcoded upstream offsets and produces garbage frame data. The **eBPF walker sidesteps this via its runtime offset harvester** — use `--py-walker=ebpf` on Ubuntu 24.04 until dbgsym becomes readily available. (Installing `python3.12-dbgsym` from `ddebs.ubuntu.com` also resolves the userspace-walker path.)

- **Trace-all mode at `kernel.yama.ptrace_scope=3`.** The eBPF walker works at ptrace_scope=3 with an explicit `--pid X` target (PID-specific uprobe attach). Without `--pid` (trace-all / dynamic-discovery mode), cuda/driver uprobes may not fire — a startup warning is logged. Workarounds: pass `--pid X` or lower `kernel.yama.ptrace_scope` to `1` (the Ubuntu default).

### Walker roadmap — userspace walker deprecation

The in-kernel eBPF walker (`--py-walker=ebpf`) is now the strategic path forward. It has runtime offset harvesting for patched distro builds (including Ubuntu 24.04 CPython 3.12), multi-library libcudart coverage, per-CUDA-tracer state broadcast, fork-inheritance for multiprocess workloads, and runs at any `ptrace_scope` value with `--pid`.

The **userspace walker — the current default — will be deprecated in an upcoming release.** Once the remaining items above are either fixed or accepted as narrow trade-offs, the eBPF walker will be promoted to the `auto` default. If you're setting up new deployments, prefer `--py-walker=ebpf` today. The `userspace` mode will remain available (via `--py-walker=userspace`) after the default flips, until the deprecation window closes.

</details>

<details>
<summary>Known patterns + troubleshooting + advanced configuration</summary>

Detected patterns (CUDA Graph capture, Python frames, dropped
events), operational cheat sheet, and power-user reference material
all live in [`docs/troubleshooting.md`](docs/troubleshooting.md).

</details>

## Reference

| Topic | Link |
|-|-|
| Full command reference (all 10 commands, every flag, output samples) | [`docs/commands.md`](docs/commands.md) |
| Architecture (single-node + cluster modes, ASCII diagram, pipeline stages) | [`docs/architecture.md`](docs/architecture.md) |
| Multi-node cluster deployment (K8s / bare-metal / Docker) | [`docs/quickstart_fleet.md`](docs/quickstart_fleet.md) |
| Stack tracing deep-dive (Python walker selection, JSON output, ptrace troubleshooting) | [`docs/stack_tracing.md`](docs/stack_tracing.md) |
| OTLP / Prometheus integration (transport, metric names, compatibility) | [`docs/otlp.md`](docs/otlp.md) |
| Full 25-problem detection catalog | [`docs/detections.md`](docs/detections.md) |
| AI investigation walkthrough (real session on A100 + GH200) | [`docs/ml_eng_sample_investigation_session.md`](docs/ml_eng_sample_investigation_session.md) |
| Fleet `ingero fleet-push` subcommand | [`docs/push_fleet.md`](docs/push_fleet.md) |
| Multi-node remediation protocol (PoC, experimental) | [`docs/remediation-protocol_fleet.md`](docs/remediation-protocol_fleet.md) |
| Test matrix | [`docs/test_matrix.md`](docs/test_matrix.md) |

## License

**Ingero is 100% free and open source.** Use it for anything  -  personal, commercial, enterprise, embed it in your product, modify it, redistribute it. No usage restrictions, no phone-home.

Dual-licensed following the standard eBPF split-licensing model (same as Cilium, Falco, and most eBPF projects):

* **User-Space** (Go agent, CLI, causal engine, SQLite, MCP): [Apache License 2.0](LICENSE)  -  maximum enterprise compatibility, no copyleft.
* **Kernel-Space** (eBPF C code in `bpf/`): [GPL-2.0](https://www.gnu.org/licenses/old-licenses/gpl-2.0.html) OR [BSD-3-Clause](https://opensource.org/licenses/BSD-3-Clause)  -  GPL-2.0 is required by the Linux kernel's BPF subsystem; BSD-3-Clause permits embedding in non-GPL toolchains.
