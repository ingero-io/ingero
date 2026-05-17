# Architecture

Ingero has two deployment modes: a single-node agent (the default; what
you get from `curl install + sudo ingero trace`) and a multi-node fleet
(agents pushing OTLP to a [Fleet collector](https://github.com/ingero-io/ingero-fleet) that classifies cluster-wide
stragglers in real time). Both share the same agent binary and the same
eBPF instrumentation; the cluster mode adds a Fleet collector and a
threshold feedback loop.

## Single-node mode

The agent loads its own eBPF probes, captures events into ringbuffers
that user-space drains, assembles causal chains, and writes results to
local SQLite. AI agents and dashboards read SQLite directly.

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

Pipeline:

1. **Discover** — scans `/proc` for processes linked to `libcudart.so`,
   finds `libcuda.so` automatically.
2. **Attach** — eBPF probes load onto CUDA runtime uprobes, driver
   uprobes, and host tracepoints.
3. **Capture** — eBPF programs record PID, TID, timestamps into
   per-layer ring buffers.
4. **System** — reads CPU / memory / load / swap from `/proc` once per
   second.
5. **Stats** — computes rolling p50/p95/p99 per operation, flags
   anomalies.
6. **Correlate** — assembles causal chains (SYSTEM + HOST + CUDA Runtime
   + CUDA Driver + CUDA Graph) by timestamp and PID.
7. **Store** — writes events to SQLite with size-based pruning
   (`--max-db 10g` default). Disable recording with `--record=false`.
8. **Export** — pushes metrics via OTLP or serves Prometheus `/metrics`
   (optional; see [`docs/otlp.md`](otlp.md)).
9. **Serve** — exposes diagnostics to AI agents via MCP (stdio or
   HTTPS/TLS 1.3).
10. **Dashboard** — browser-based HTTPS dashboard reads from SQLite,
    shows ops / chains / snapshots / capabilities with auto-polling.
11. **Fleet** — fan-out queries across multiple nodes via dashboard
    API, merge offline databases, detect clock skew, export to Perfetto
    timeline.

## Cluster mode

A multi-node deployment adds the **Fleet collector** — a small custom
OpenTelemetry Collector distribution that aggregates per-node health
scores and computes a peer-relative threshold the agents read back.
The agent's `ingero fleet-push` subcommand handles the OTLP push;
Fleet runs `ingeroprocessor` (peer threshold), `ncclprocessor`
(cross-rank barrier-wait), `providerlookupprocessor` (cloud /
node-label attribution), and a threshold extension that exposes the
classification back to agents over OTLP response headers.

The cluster diagram and per-component description live in
[`docs/quickstart_fleet.md`](quickstart_fleet.md), which is also the
deployment entry point. For the deeper Fleet architecture (data flow,
processor wiring, HA, cross-cluster federation), see
[ingero-fleet/docs/architecture_fleet.md](https://github.com/ingero-io/ingero-fleet/blob/main/docs/architecture_fleet.md).

## When to use which mode

| Question | Single-node | Cluster (Fleet) |
|-|-|-|
| One GPU host, ad-hoc investigation | yes | overkill |
| `ingero query --nodes` fan-out across N hosts after the fact | yes | yes (works either way) |
| Real-time cluster-wide straggler classification | no (Fleet does this) | yes |
| Multi-tenant K8s deployment with provider attribution | no | yes |
| AI assistant via MCP, single host | yes | also works in cluster mode |

The single-node mode is the default and the recommended starting point.
Add Fleet when you need the cluster itself to classify stragglers in
real time (rather than querying ad hoc across nodes), or when you're
running multi-tenant deployments that need cluster-wide attribution.
