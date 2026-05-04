# OTLP / Prometheus Integration

OTEL export is **off by default**. Enable with `--otlp` (push) or
`--prometheus` (pull).

## Prometheus (pull)

```bash
sudo ingero trace --prometheus :9090
curl localhost:9090/metrics
```

Exposes `/metrics` in Prometheus text format on the given listen
address. Compatible with any Prometheus / Grafana Cloud / Mimir
scraper.

## OTLP push

```bash
sudo ingero trace --otlp localhost:4318
sudo ingero trace --otlp localhost:4318 --debug   # see push logs on stderr
```

OTLP uses the **HTTP JSON transport** (`POST /v1/metrics`) and the
default OTEL HTTP receive port (4318). Compatible with:

- OpenTelemetry Collector
- Grafana Alloy
- Grafana Cloud
- Datadog Agent (OTLP receiver)
- New Relic (OTLP endpoint)
- Any OTLP-compatible HTTP receiver

## Metric names

Standard OTEL semantic conventions, per-operation, per-source granularity:

| Metric | Type | Notes |
|-|-|-|
| `gpu.cuda.operation.duration` | Histogram | Per CUDA op latency |
| `gpu.cuda.operation.count` | Counter | Per CUDA op call count |
| `system.cpu.utilization` | Gauge | Host CPU% from /proc |
| `system.memory.utilization` | Gauge | Host mem% from /proc |
| `ingero.anomaly.count` | Counter | Causal-chain anomalies detected |

## Implementation note

Zero external dependencies on the OTEL Go SDK. The JSON payload is
constructed directly using Go's standard library, so the agent binary
stays small and the OTEL surface stays auditable.
