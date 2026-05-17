[Ingero (hub)](../README.md) ·
[Architecture](architecture.md) ·
[Multi-node fleet](quickstart_fleet.md) ·
[`ingero fleet-push`](push_fleet.md) ·
[OTLP / Prometheus](otlp.md)

# Per-cohort straggler detection, per-cgroup CUDA metrics, OTLP traces

> **TL;DR:** Run one [Ingero Fleet collector](https://github.com/ingero-io/ingero-fleet) across many workloads
> without one noisy cohort polluting another, attribute every CUDA
> kernel launch and CPU-stall sample to the originating cgroup, and
> emit OTLP trace spans on detection edges. Two CLI surfaces
> (`ingero rates`, `pagerduty_trigger` MCP tool) ship alongside.

## Features

| Feature | Surface | Outcome for the operator |
|---------|---------|--------------------------|
| **Per-cohort straggler thresholds** | Fleet `ingeroprocessor` | One Fleet, many workloads. Threshold computed per `cgroup_path_hash` cohort: a slow training pod no longer drags the inference threshold down. |
| **Per-cgroup CUDA + CPU-stall metrics** | agent metric stream | Every CUDA kernel launch, memcpy, and CPU stall is attributed to the cgroup that produced it. Tells you *which container* is slow, not just *the host*. |
| **OTLP trace export from detection edges** | agent OTLP traces | Detection edges (HEALTHY <-> STRAGGLER) emit OTLP spans on the same endpoint as metrics. One pipeline for two signals. |
| **`ingero rates` CLI** | new subcommand | GPU hourly cost lookups embedded in the binary; refreshable from upstream. Useful for cost-of-stall numbers in incident reports. |
| **`pagerduty_trigger` MCP tool** | `ingero mcp serve` | Tool registered for AI-driven incident escalation. Falls back to a clear "not configured" error when no routing key is set. |
| **YAML config loader** | `ingero --config` | Load configuration from a YAML file; CLI flags continue to override file values. |

---

## Quick Start

### Prerequisites

- Linux x86_64, kernel >= 5.10
- For multi-node: one machine reachable as the Fleet collector
- For per-cgroup CUDA metrics + OTLP traces to fire live: an NVIDIA
  GPU with an active CUDA workload

### Single-node (3 minutes)

The single-node surfaces are `ingero rates`, the YAML loader, and the
`pagerduty_trigger` MCP tool.

```bash
# 1. Look up GPU hourly rates from the embedded table
ingero rates show

# 2. Refresh the table from upstream (falls back to embedded if offline)
ingero rates update

# 3. Start the MCP server with PagerDuty escalation enabled
ingero mcp serve --http :8080 \
  --pagerduty-routing-key "$PAGERDUTY_INTEGRATION_KEY"

# 4. (Optional) load the same config from YAML rather than flags
cat > /etc/ingero/ingero.yaml <<'YAML'
alerter:
  pagerduty:
    routing_key: ${PAGERDUTY_INTEGRATION_KEY}
mcp:
  http_addr: ":8080"
YAML
ingero --config /etc/ingero/ingero.yaml mcp serve
```

The cohort threshold, per-cgroup metrics, and OTLP trace export also
work on a single node, but they only become visible once you point the
agent at a Fleet collector. See multi-node below.

### Multi-node cluster (10 minutes)

Minimum cluster: 1 Fleet replica + N agents. Per-cohort thresholds,
per-cgroup metrics, and OTLP trace export all activate as soon as the
agent is pointed at Fleet with `--otlp-enabled`.

#### 1. Fleet collector config

`fleet.yaml` enables the `ingero` processor and a `traces` pipeline:

```yaml
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:4318
processors:
  ingero:
    threshold:
      k: 2.0           # MAD multiplier (lower = stricter)
      ema_alpha: 0.2
    quorum:
      statistical_min: 5
      coverage_fraction: 0.80
    push_interval: 10s
    ttl_multiplier: 5
extensions:
  ingero_threshold:
    agent_endpoint: 0.0.0.0:8080
exporters:
  debug:
    verbosity: detailed
service:
  telemetry:
    metrics:
      readers:
        - pull:
            exporter:
              prometheus:
                host: 0.0.0.0
                port: 9999
  extensions: [ingero_threshold]
  pipelines:
    metrics:
      receivers: [otlp]
      processors: [ingero]
      exporters: [debug]
    traces:
      receivers: [otlp]
      exporters: [debug]
```

Start it:

```bash
ingero-fleet --config fleet.yaml
```

#### 2. Agent on each node

Two-process pattern: `trace --record` captures eBPF events into a
local SQLite store; `fleet-push` derives a health score from that
store and pushes to Fleet. The flags that drive the new behavior are
`--fleet-workload-type` (cohort selection) and `--otlp-enabled`
(per-cgroup metrics + trace export):

```bash
# Terminal 1: capture eBPF events
sudo ingero trace --record --db /var/lib/ingero/agent.db

# Terminal 2: derive health, push to Fleet, emit OTLP metrics + traces
ingero fleet-push \
  --fleet-endpoint http://fleet:4318 \
  --fleet-cluster-id prod \
  --fleet-workload-type inference \
  --signal-db-path /var/lib/ingero/agent.db \
  --otlp-enabled \
  --otlp-endpoint http://fleet:4318
```

After about 10 seconds the agent transitions from `calibrating` to
`active` and starts pushing every interval.

#### 3. Verify the surfaces are live

```bash
curl -s http://fleet:9999/metrics | \
  grep -E '^ingero_fleet_(active_nodes|threshold|median|mad|straggler_count|coverage_low)'
```

Expected, with N >= 5 agents reporting (one cohort per
`cgroup_path_hash`):

```text
ingero_fleet_active_nodes{cluster_id="prod",cgroup_path_hash="..."} 5
ingero_fleet_median{cluster_id="prod",cgroup_path_hash="..."}       0.957
ingero_fleet_mad{cluster_id="prod",cgroup_path_hash="..."}          0.012
ingero_fleet_threshold{cluster_id="prod",cgroup_path_hash="..."}    0.900
ingero_fleet_coverage_low{...} 0
ingero_fleet_straggler_count{...} 0
```

When a node falls below `median - k * MAD * 1.4826`, `straggler_count`
increments and the next OTLP poll classifies that node as a straggler.

---

## How per-cohort thresholds work

Without cohorts, a single Fleet computes one cluster-wide threshold
across every reporting agent. That breaks down as soon as one Fleet
serves two workload shapes: a slow training step pulls down the median
used to flag inference stragglers.

Cohort-aware thresholds split the cluster into groups keyed by
`cgroup_path_hash`. Agents in the same systemd or kubelet cgroup hash
to the same cohort; agents in unrelated cgroups land in different
cohorts. Fleet computes a separate median, MAD, and threshold per
cohort, so a training-cohort straggler never affects the
inference-cohort floor.

Setting `--fleet-workload-type=inference` (vs `training`) on the agent
side is what materialises the split: the workload type is mixed into
the cgroup hash so the same Fleet can serve both classes side by side.

Detection formula, per cohort:

```text
threshold = median - k * MAD * 1.4826
```

`k` defaults to 2.0 and `1.4826` is the consistency constant that makes
MAD an unbiased estimator of standard deviation under normal data.
Stricter detection means a smaller `k`.

---

## Reference

### CLI flags

| Flag | Subcommand | Default | Purpose |
|------|------------|---------|---------|
| `--otlp-enabled` | `fleet-push` | `false` | Emit per-cgroup metrics + trace spans over OTLP |
| `--otlp-endpoint` | `fleet-push` | (none) | OTLP endpoint for the metrics + traces above |
| `--fleet-workload-type` | `fleet-push`, `trace` | `unknown` | Cohort key on the Fleet side; also picks correlator window (10s for training, 500ms for inference) |
| `--pagerduty-routing-key` | `mcp serve` | (none) | Enables the live `pagerduty_trigger` path; without it the tool returns "not configured" |

### Fleet metric labels

Every series on the `ingeroprocessor` carries a
`cgroup_path_hash="<16-char-hex>"` label. The empty-hash value
(`cgroup_path_hash=""`) is the backward-compatible cohort that catches
agents that did not pass `--fleet-workload-type`.

### Agent metrics

| Metric | Type | Attributes |
|--------|------|-----------|
| `ingero.node.cuda_kernel_launch_total` | counter | `cgroup_path_hash` |
| `ingero.node.cuda_memcpy_bytes_total` | counter | `cgroup_path_hash`, `ingero.memcpy.direction` |
| `ingero.node.cpu_stall_nanos_total` | counter | `cgroup_path_hash` |

### YAML config loader

The loader searches in this order, first hit wins:

1. `--config /path/to/file.yaml`
2. `$INGERO_CONFIG`
3. `./ingero.yaml`
4. `/etc/ingero/ingero.yaml`

CLI flags always override file values. The full schema lives in
[`internal/config/config.go`](../internal/config/config.go).

---

## Inference daemon mode (v0.16)

`ingero trace --inference` is the production daemon shape for inference workloads. It bundles five behaviors that are individually opt-in but most useful together:

1. **Sub-second causal window** (`--fleet-workload-type=inference`, 500ms instead of the 10s training default).
2. **Event sampler** attached to the SQLite store: 1% admission in healthy state, 100% under degradation, 30s cooldown back to healthy. Caps storage on inference workloads that emit kernel events at production QPS.
3. **JSON output** by default (no TUI), suitable for systemd / k8s log collectors.
4. **DB rollover** instead of in-place pruning: rotates `ingero.db` to `ingero.<UTC-timestamp>.db` when the file crosses the threshold (default 1g), keeps the last 6.
5. **Per-workload step-duration baseline + outlier detection.** Tracks the running mean (EMA) and 95th-percentile (P²) of step durations for each `(cgroup_path_hash, pid, stream_handle)` workload. Classifies each step against the workload's own p95 once warmed (default 30 healthy steps); emits `inference_outlier` events on the FOSS UDS socket and as rate-limited INFO logs.

### One-line invocation

```bash
sudo ingero trace --inference
```

Equivalent expanded form (every flag the umbrella sets, written explicitly):

```bash
sudo ingero trace \
  --fleet-workload-type=inference \
  --duration=0 \
  --json \
  --heartbeat=30s \
  --remediate \
  --max-db=0 \
  --db-rollover-size=1g \
  --db-rollover-keep=6
```

Any individually set flag wins over the umbrella default. For example `ingero trace --inference --json=false` keeps everything else and re-enables the TUI.

### Step-duration baseline

A "step" is the wall-clock interval between consecutive `cudaStreamSynchronize` (also `cudaDeviceSynchronize`, driver `ctxSynchronize`) events on the same `(pid, stream_handle)`. For continuous-batching servers like vLLM, SGLang, and TGI, **each step is one engine iteration, not one user-facing HTTP request** — every request typically produces multiple syncs. The metric name `ingero.infer.step_duration_ns` reflects this honestly.

Baseline updates pause while any HIGH-severity causal chain is active for the PID — see `--inference-pause-on-severity` to widen the gate (e.g. to `MEDIUM`). Outliers do not fold into the baseline, preserving cleanliness during sustained anomaly windows.

### Outlier buckets

Buckets are mutually exclusive at emission. A step that exceeds 3× of baseline-p95 increments the `3x` bucket only, not also `1.5x` and `2x`. SLO-style "exceeded 1.5x or higher" math happens at PromQL/Grafana time over the cumulative counter sums.

| Bucket | Trigger | Default action |
|---|---|---|
| `1.5x` | step >= 1.5 × baseline-p95 | log + UDS publish |
| `2x` | step >= 2.0 × baseline-p95 | log + UDS publish |
| `3x` | step >= 3.0 × baseline-p95 (configurable via `--inference-outlier-ratio`) | log + UDS publish + sampler bumped to 100% admission for the cooldown window |

The `3x` bucket also flips the sampler to admit 100% of events for the next 30s, so when the outlier resolves you have full-fidelity event history surrounding the spike. Configure with `--inference-sampler-degrade-on={1.5x,2x,3x,off}`.

### UDS protocol extension

When `--remediate` is on (default under `--inference`), each outlier emits a typed NDJSON message on `/tmp/ingero-remediate.sock`:

```json
{"type":"inference_outlier","node_id":"…","cluster_id":"…",
 "timestamp":"2026-05-09T14:30:55Z","event_id":"…",
 "cgroup_path_hash":"…","pid":12345,"stream_handle":18446744073709551615,
 "step_duration_ns":50000000,"baseline_p95_ns":12000000,
 "baseline_mean_ns":10500000,"bucket":"3x"}
```

The FOSS agent only publishes; consumers (the [ingero-ee orchestrator](https://github.com/ingero-io/ingero-ee), custom operator scripts) subscribe and react however they choose. Backward-compatible: existing consumers that decode by `type` ignore unknown variants.

### Tuning

| Flag / YAML key | Default | What it controls |
|---|---|---|
| `--inference-warmup` / `inference.baseline.warmup_samples` | 30 | Healthy steps before classification activates per workload |
| `--inference-outlier-ratio` / `inference.outlier.threshold_ratio` | 3.0 | Multiplier on baseline-p95 for the largest bucket |
| `--inference-pause-on-severity` / `inference.baseline.pause_on_severity` | `HIGH` | Lowest causal-chain severity that pauses baseline updates |
| `--inference-sampler-degrade-on` / `inference.outlier.sampler_degrade_on` | `3x` | Smallest bucket that bumps the sampler to 100% |
| `--db-rollover-size` / `inference.db_rollover.size` | `1g` | Trace DB file rollover threshold |
| `--db-rollover-keep` / `inference.db_rollover.keep` | `6` | Rolled-over files retained on disk |

### Phase-aware baselines (v0.16.1)

A naive single-baseline-per-stream design produces **false negatives** on heterogeneous-task streams. A vLLM continuous-batching server interleaves prefill (~200ms, kernel-heavy) and decode (~5ms, sparse-launch) on one hot-path stream. The mixed-bucket p95 lands near the prefill tail (~180ms), so a 10× slow decode (50ms vs 5ms baseline) gets absorbed and never fires.

v0.16.1 splits the per-`(cgroup, pid, stream)` baseline by **phase**, classifying each step from observable signals **before** the duration is compared to the baseline. The classifier is **duration-invariant**: a slow decode is still a decode (few launches, no memcpy, no NCCL), so it lands in the decode bucket and gets compared against the decode-phase p95.

Phase set: `prefill` / `decode` / `mixed` / `unknown`.

Rule order (first match wins; defaults LLM-tuned for 7B-70B serving):

1. **NCCL > 0** → `prefill` (distributed tensor-parallel allreduce)
2. **launches == 0 AND memcpy == 0** → `unknown` (idle-poll, not a real step)
3. **avg_kernel > 500us** → `prefill` (compute-heavy GEMM-style)
4. **launches > 200** → `prefill` (typical LLM attention/MLP layer count)
5. **launches < 50 AND memcpy < 1 MiB** → `decode`
6. **launches in [50, 200] OR memcpy >= 10 MiB** → `mixed`
7. (anything else) → `unknown`

`unknown`-classified steps participate in their own bucket but **do not** trigger sampler degradation — we lack workload context to know whether the slowdown is meaningful, so flipping to 100% admit on novel patterns would just burn storage.

### Tuning the phase classifier

| Flag / YAML key | Default | What it controls |
|---|---|---|
| `--inference-phase-classifier` / `inference.phase.classifier` | `rule` | `rule` (on) or `off` (revert to v0.16.0 single-baseline) |
| `--inference-phase-decode-max-launches` / `inference.phase.decode_max_launches` | 50 | Decode if launches < this (and memcpy small, no NCCL) |
| `--inference-phase-decode-max-memcpy` / `inference.phase.decode_max_memcpy` | `1m` | Above this, the step exits the decode bucket |
| `--inference-phase-prefill-min-launches` / `inference.phase.prefill_min_launches` | 200 | Prefill if launches > this OR avg-kernel > threshold |
| `--inference-phase-prefill-min-avg-kernel` / `inference.phase.prefill_min_avg_kernel` | `500us` | Prefill via fat-kernel branch |
| `--inference-phase-mixed-memcpy` / `inference.phase.mixed_memcpy` | `10m` | Bulk memcpy threshold for mixed |
| `--inference-phase-mixed-launch-low` / `inference.phase.mixed_launch_low` | 50 | Lower end of the mixed launch range (inclusive) |
| `--inference-phase-mixed-launch-high` / `inference.phase.mixed_launch_high` | 200 | Upper end of the mixed launch range (inclusive) |

Embedding, vision, and MoE workloads should tune individual thresholds — defaults are LLM-tuned and may misclassify other workloads (which fall to `unknown`, harmlessly).

### Engine /metrics scrape + OTel GenAI (v0.16.2)

The eBPF baseline answers "is this engine running about as fast as it usually does?" v0.16.2 layers on the engine's own canonical SLO metrics — TTFT (Time-To-First-Token), TPOT (Time-Per-Output-Token), prefill/decode latencies, token counts — by pulling the engine's `/metrics` endpoint and translating engine-specific Prometheus names to OTel GenAI semantic conventions.

**Auto-detection**: at startup, the agent reads `/proc/<pid>/cmdline` for each `--pid` target and matches:

| cmdline pattern | Engine | Default port |
|---|---|---|
| `vllm.entrypoints.openai.api_server` or `vllm serve` | vLLM | 8000 |
| `text-generation-launcher` | TGI | 8080 |
| `sglang.launch_server` | SGLang | 30000 |
| `tritonserver` | Triton | 8002 |

`--port`/`--http-port` flags on the cmdline override the default. NIM passes through vLLM's metric format unchanged, so it's covered by the vLLM detector.

**Output**: scraped metrics emit on the same OTLP endpoint as the eBPF metrics, using OTel GenAI semconv (v1.37):

| Engine name (vLLM example) | OTel GenAI canonical |
|---|---|
| `vllm:time_to_first_token_seconds` | `gen_ai.client.operation.time_to_first_token` |
| `vllm:inter_token_latency_seconds` | `gen_ai.server.time_per_output_token` |
| `vllm:e2e_request_latency_seconds` | `gen_ai.client.operation.duration` |
| `vllm:request_prefill_time_seconds` | `gen_ai.server.request.duration.prefill` |
| `vllm:request_decode_time_seconds` | `gen_ai.server.request.duration.decode` |
| `vllm:prompt_tokens_total` | `gen_ai.client.token.usage.input` |
| `vllm:generation_tokens_total` | `gen_ai.client.token.usage.output` |

**Engine-down behavior**: if the engine isn't responding (cold start, restart, network blip), the scraper logs at Debug, increments `MetricInferScrapeFailures`, and keeps ticking. Layer 1 (eBPF) continues uninterrupted — the agent stays useful regardless of engine health.

**One-line invocation** (vLLM example):

```bash
sudo ingero trace --inference --pid $(pgrep -f vllm.entrypoints) --otlp localhost:4318
```

Datadog Agent or OTel Collector receiving the OTLP exporter output sees both `ingero.infer.*` (eBPF baseline) and `gen_ai.*` (engine canonical SLO) metrics in the same ingestion. Existing Datadog LLM Observability dashboards (which speak OTel GenAI as of v1.37+) light up automatically with no Ingero-specific configuration.

**What v0.16.2 does NOT include**:
- TensorRT-LLM (no Prometheus endpoint; profiler-only)
- Ray Serve (no canonical /metrics format)
- Bespoke PyTorch endpoints (operator-defined; configure manually as a Prometheus scrape target)
- Dynamic engine re-discovery during a long trace run (engines detected at startup; restart trace if you start a new engine mid-run)

### What's NOT in the v0.16 umbrella

These are intentionally separate stories on the v0.16.x roadmap:
- **KV-cache block-level lineage** (vLLM PagedAttention scrape) — engine `/metrics` HTTP scrape primitive
- **Speculative-decoding accept-ratio** — engine `/metrics` extension
- **Quantization health basics** (FP8 staleness, NaN/Inf detection)
- **Memory-fragmentation precursors** (`mm_page_alloc_extfrag`, compaction tracepoints) — `internal/ebpf/memfrag` already shipped in v0.14; wiring into the trace path is a follow-up
- **Power/thermal probes** (NVML power-state polling)
- **OTLP histogram + counter emission** of `ingero.infer.*` — uses the v0.15 histogram encoder; follow-up

---

## Related

- [Multi-node Fleet quickstart](quickstart_fleet.md) (K8s, bare-metal, Docker)
- [`ingero fleet-push` reference](push_fleet.md)
- [OTLP / Prometheus integration](otlp.md)
- [Architecture overview](architecture.md)
- Companion repo: [ingero-fleet](https://github.com/ingero-io/ingero-fleet)
