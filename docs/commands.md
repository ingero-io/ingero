# Commands reference

Complete reference for the 10 `ingero` subcommands. The README has a
short summary table; this page has the full flag references and
output examples.

> **Only `trace` needs sudo**: it attaches eBPF probes to the kernel.
> All other commands (`check`, `explain`, `query`, `mcp`, `demo`,
> `dashboard`, `merge`, `export`, `version`) run unprivileged. When
> you run `sudo ingero trace`, the database is written to your home
> directory (not `/root/`) and chown'd to your user, so non-sudo
> commands can read it.

## `ingero check`

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

## `ingero trace`

Live event stream with rolling stats, system context, and anomaly
detection. Events are recorded to SQLite by default (use
`--record=false` to disable). The database is capped at 10 GB rolling
storage and auto-purges old events when the limit is reached
(see `--max-db`).

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
sudo ingero trace --otlp localhost:4318    # push metrics via OTLP (see docs/otlp.md)
sudo ingero trace --throttle-poll-interval 2s  # poll NVML clock-throttle reasons every 2s (default 5s; 0 disables)
sudo ingero trace --node gpu-node-07      # tag events with node identity (for multi-node)
sudo ingero trace --cuda-lib /opt/venv/lib/python3.11/site-packages/nvidia/cuda_runtime/lib/libcudart.so.12
                                           # explicit libcudart path (skips auto-discovery)
sudo ingero trace --ringbuf-size 32m       # override high-throughput ring buffer size (power of 2, min 4096)
sudo ingero trace --sampling-rate 0        # adaptive sampling (default: 1 = emit all; N>1 = 1-in-N)
sudo ingero trace --py-walker ebpf         # in-kernel CPython walker (works at ptrace_scope=3)
```

**Flag reference:**

- `--cuda-lib PATH`: Explicit path to `libcudart.so`. Skips
  auto-discovery. Useful for venv workloads where multiple `libcudart`
  copies exist.
- `--ringbuf-size SIZE`: Override ring buffer size for high-throughput
  probes (cuda, driver, host). Accepts `k`/`m`/`g` suffix. Must be a
  power of 2, minimum 4096. Default: compiled sizes (8MB cuda/driver,
  1MB host).
- `--sampling-rate N`: Event sampling rate. `0` = adaptive
  (auto-adjusts under sustained drops). `1` = emit all events
  (default behavior). `N > 1` = emit 1 in every N events. Applies to
  cuda/driver/graph probes only; host probes are never sampled.
- `--py-walker {auto,ebpf,userspace}`: Python frame walker selection.
  `auto` (default) uses the userspace walker. `ebpf` uses the
  in-kernel CPython walker (supports 3.10, 3.11, 3.12: no
  `/proc/pid/mem` required, works at `ptrace_scope=3`). `userspace`
  forces the classic walker.
- `--throttle-poll-interval DURATION`: NVML clock-throttle reason
  poll interval. Default `5s`; `0` disables. Emits the four
  `gpu.throttle.{power,thermal,sw,hw}_active` gauges per visible
  GPU. The interval is the bursting floor: throttle events shorter
  than the interval may be missed by design. See
  [`docs/otlp.md`](otlp.md) for the bit-to-bucket mapping table and
  metric semantics.

`ingero check` reports the current `kernel.yama.ptrace_scope` value
with actionable hints when it blocks Python source attribution.

**Process targeting:**

- **Default** (no flags): traces all CUDA processes owned by the
  invoking user (via `SUDO_USER`). On single-user boxes, this means
  all CUDA processes.
- **`--pid`**: target specific process(es), comma-separated
  (e.g., `--pid 1234,5678`).
- **`--user`**: target all CUDA processes owned by a specific user
  (`--user bob`, `--user root`).
- **Dynamic child tracking**: fork events auto-enroll child PIDs for
  host correlation.

The trace display shows five sections:

1. **System Context**: CPU, memory, load, swap with ASCII bar charts
   (green/yellow/red).
2. **CUDA Runtime API**: per-operation p50/p95/p99 latency with
   anomaly flags (cudaMalloc, cudaLaunchKernel, graphLaunch, etc.).
3. **CUDA Driver API**: driver-level operations (cuLaunchKernel,
   cuMemAlloc, etc.) that cuBLAS/cuDNN call directly.
4. **Host Context**: scheduler, memory, OOM, and process lifecycle
   events.
5. **CUDA Graph events**: graph capture, instantiate, and launch
   events (when graph-using workloads are traced).

## `ingero explain`

Analyze recorded events from SQLite and produce an incident report
with causal chains, root causes, and fix recommendations. Reads from
the database populated by `ingero trace`: no root needed.

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

### Per-Process Breakdown

For multi-process GPU workloads (RAG pipelines, model serving with
workers, multi-tenant GPU sharing), `--per-process` shows a CUDA API
breakdown grouped by process:

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

This answers "which process is hogging the GPU?": essential for
diagnosing RAG pipeline contention where embedding, retrieval, and
generation compete for GPU time.

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

## `ingero query`

Query stored events by time range, PID, and operation type. Supports
multi-node fleet queries with `--nodes`.

```bash
ingero query --since 1h
ingero query --since 1h --pid 4821
ingero query --since 1h --pid 4821,5032
ingero query --since 30m --op cudaMemcpy --json

# Multi-node fleet queries (fan-out to multiple Ingero dashboard APIs)
ingero query --nodes host1:8080,host2:8080 "SELECT node, source, count(*) FROM events GROUP BY node, source"
ingero query --nodes host1:8080,host2:8080,host3:8080 "SELECT node, count(*) FROM events GROUP BY node"
```

Fleet queries fan out the SQL to each node's `/api/v1/query` endpoint,
concatenate results with a `node` column prepended, and display a
unified table. Partial failures return results from reachable nodes
with warnings for unreachable ones. Clock skew between nodes is
detected automatically (configurable via `--clock-skew-threshold`,
default 10ms).

Configure default fleet nodes in `ingero.yaml` under `fleet.nodes` to
avoid repeating `--nodes` on every command.

Storage uses SQLite with size-based pruning (default 10 GB via
`--max-db`). Data is stored locally at `~/.ingero/ingero.db`: nothing
leaves your machine.

## `ingero mcp`

Start an MCP (Model Context Protocol) server for AI agent integration.

```bash
ingero mcp                        # stdio (for Claude Code / MCP clients)
ingero mcp --http :8080           # HTTPS on port 8080 (TLS 1.3, auto-generated self-signed cert)
ingero mcp --http :8080 --tls-cert cert.pem --tls-key key.pem  # custom TLS certificate
```

> **Note:** The `--http` flag enables the Streamable HTTP transport.
> All connections use **TLS 1.3 only** (no plain HTTP). When no
> `--tls-cert`/`--tls-key` is provided, ingero auto-generates an
> ephemeral self-signed ECDSA P-256 certificate. Use `curl -k` to
> skip certificate verification for self-signed certs.

**AI-first analysis**: MCP responses use telegraphic compression (TSC)
by default, reducing token count by ~60%. Set `{"tsc": false}` per
request for verbose output.

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
| `/investigate` | Guided investigation workflow: walks the AI through stats, chains, and SQL to diagnose GPU issues. Works with any MCP client. |

**Works with any AI, not just Claude.** Use local open-source models
via [ollmcp](https://github.com/jonigl/mcp-client-for-ollama) (Ollama
MCP client):

```bash
# Install ollmcp (minimax-m2.7:cloud routes to MiniMax API via Ollama Cloud,
# or use a local model like qwen3.5:32b via ollama pull qwen3.5:32b)
pip install mcp-client-for-ollama

# Create a config pointing to Ingero's MCP server
cat > /tmp/ingero-mcp.json << 'EOF'
{"mcpServers":{"ingero":{"command":"ingero","args":["mcp","--db","trace.db"]}}}
EOF

# Start investigating - /investigate triggers the guided workflow
ollmcp -m minimax-m2.7:cloud -j /tmp/ingero-mcp.json
```

Tested with MiniMax M2.7 and Qwen 3.5 via Ollama on saved investigation
databases. Also works with Claude Desktop, Cursor, and any
MCP-compatible client.

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

## `ingero dashboard`

Start a browser-based GPU monitoring dashboard backed by the SQLite
event store. Shows live system metrics, CUDA operation latencies,
causal chains, and a capability manifest (grayed-out panels for
metrics Ingero doesn't yet collect, with tooltips naming the required
external tool). Requires `ingero trace` to be running (or to have run
recently).

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

**No sudo needed**: the dashboard reads from the SQLite database
populated by `ingero trace`.

**Security:** TLS 1.3 only. Auto-generates an ephemeral self-signed
ECDSA P-256 certificate (valid 24h) if no `--tls-cert`/`--tls-key`
provided. DNS rebinding protection rejects requests from non-localhost
Host headers.

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

## `ingero merge`

Merge SQLite databases from multiple Ingero nodes into a single
queryable database for offline cross-node analysis. Useful in
air-gapped environments or when you prefer offline analysis over
fan-out queries.

```bash
ingero merge node-a.db node-b.db node-c.db -o cluster.db       # merge 3 node databases
ingero merge old.db --force-node legacy-node -o merged.db       # assign node identity to legacy DBs

# Then use standard tools on the merged database
ingero query -d cluster.db --since 1h
ingero explain -d cluster.db --chains
ingero export --format perfetto -d cluster.db -o trace.json
```

Node-namespaced event IDs (`{node}:{seq}`) ensure zero collisions on
merge. Stack traces are deduplicated by hash. Sessions are re-keyed.
Clock skew between traces is detected and warned (configurable via
`--clock-skew-threshold`, default 100ms).

## `ingero export`

Export event data to visualization formats. Currently supports
Perfetto/Chrome Trace Event Format for timeline visualization in
[ui.perfetto.dev](https://ui.perfetto.dev) or `chrome://tracing`.

```bash
# From a local or merged database
ingero export --format perfetto -d ~/.ingero/ingero.db -o trace.json
ingero export --format perfetto -d cluster.db -o trace.json --since 5m

# Fan-out mode (fetches from multiple nodes via fleet API)
ingero export --format perfetto --nodes node-1:8080,node-2:8080 -o trace.json
```

Opens in Perfetto UI with one process track per node/rank, CUDA events
as duration spans, and causal chains as severity-colored instant
markers. Multi-node traces show side-by-side timelines for spotting
which rank stalled while others waited.

## `ingero demo`

```bash
ingero demo                  # all 6 scenarios (incident first)
ingero demo incident         # single scenario
ingero demo gpu-steal        # also: gpu-contention, contention
ingero demo --no-gpu         # synthetic mode
```

## `ingero version`

```bash
$ ingero version
ingero v0.10.0 (commit: <sha>, built: <date>)
```
