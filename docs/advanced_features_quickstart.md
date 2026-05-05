[Ingero (hub)](../README.md) ·
[Architecture](architecture.md) ·
[Multi-node fleet](quickstart_fleet.md) ·
[`ingero fleet-push`](push_fleet.md) ·
[OTLP / Prometheus](otlp.md)

# Per-cohort straggler detection, per-cgroup CUDA metrics, OTLP traces

> **TL;DR:** Run one Ingero Fleet collector across many workloads
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

## Related

- [Multi-node Fleet quickstart](quickstart_fleet.md) (K8s, bare-metal, Docker)
- [`ingero fleet-push` reference](push_fleet.md)
- [OTLP / Prometheus integration](otlp.md)
- [Architecture overview](architecture.md)
- Companion repo: [ingero-fleet](https://github.com/ingero-io/ingero-fleet)
