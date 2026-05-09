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
| `gpu.throttle.power_active` | Gauge | NVML throttle reason: power-cap active (1) or not (0); per `gpu.uuid` |
| `gpu.throttle.thermal_active` | Gauge | NVML throttle reason: thermal active (1) or not (0); per `gpu.uuid` |
| `gpu.throttle.sw_active` | Gauge | NVML throttle reason: software-side active (1) or not (0); per `gpu.uuid` |
| `gpu.throttle.hw_active` | Gauge | NVML throttle reason: hardware umbrella (any HW reason); per `gpu.uuid` |
| `nccl.collective.duration_ms` | Histogram | Per-collective op latency (`op_type`, `comm_id_hash`, `rank`, `nranks`, `datatype`, `reduce_op`, `nccl.peer_rank` for Send/Recv) |
| `nccl.collective.barrier_wait_ms` | Histogram | Time between collective uretprobe and the matching `cudaStreamSynchronize` |

## NVML throttle-reason metrics

`gpu.throttle.{power,thermal,sw,hw}_active` come from
`nvmlDeviceGetCurrentClocksThrottleReasons` polled at
`--throttle-poll-interval` (default 5s, `0` disables). Each metric
emits `1` when the bucket is active and `0` otherwise. Metric names
are stable contract; dashboards may pin to them.

`hw_active` is the umbrella for any hardware reason: future NVML
bits not yet enumerated in the bucket table funnel here so dashboards
do not silently lose visibility on a newer driver.

Bit-to-bucket mapping (also documented in
`internal/nvml/decoder.go`):

| NVML bit | bucket(s) |
|----------|-----------|
| `nvmlClocksThrottleReasonGpuIdle` | (suppressed) |
| `nvmlClocksThrottleReasonApplicationsClocksSetting` | sw |
| `nvmlClocksThrottleReasonSwPowerCap` | power, sw |
| `nvmlClocksThrottleReasonHwSlowdown` | hw |
| `nvmlClocksThrottleReasonSyncBoost` | sw |
| `nvmlClocksThrottleReasonSwThermalSlowdown` | thermal, sw |
| `nvmlClocksThrottleReasonHwThermalSlowdown` | thermal, hw |
| `nvmlClocksThrottleReasonHwPowerBrakeSlowdown` | power, hw |
| `nvmlClocksThrottleReasonDisplayClockSetting` | sw |

The poll interval is the bursting floor: a throttle event shorter
than the interval may be missed by design. Choose the interval
based on the workload (5s is a reasonable default for steady-state
training; shorter intervals catch more bursty inference patterns
at the cost of one extra `nvidia-smi` call per tick).

Consumer GPUs that return `[Not Supported]` for the throttle field
are skipped per device; the agent logs the skip once per GPU at
info level instead of every tick.

This is the polling-based variant of W2; an event-driven BPF
version (issue #133) is on the roadmap for a future release.

## Implementation note

Zero external dependencies on the OTEL Go SDK. The JSON payload is
constructed directly using Go's standard library, so the agent binary
stays small and the OTEL surface stays auditable.

## Trust model: BPF-supplied register state

The cudaMemcpy direction tag (h2d / d2h / d2d / h2h / default / unknown)
is read from a userspace register at the uprobe entry. On a single-tenant
host this is always faithful: only the workload's own code can populate
that register before calling cudaMemcpy. On a multi-tenant host where the
agent runs across PIDs the agent does not own, a malicious process can
craft an in-register value that mislabels its own memcpy direction.

Concretely: the direction byte cannot be trusted as a security signal
across tenancy boundaries. Use it only for performance dashboarding and
percentile reporting. The same caveat applies to any BPF-supplied
register-derived attribute: kernel grid/block dims (v0.16 item M),
NCCL collective op-type (v0.14 item B), and IOCTL command numbers
(v0.16 items K/L). Cross-tenant correlation should rely on host-side
cgroup attribution (cgroup_path_hash, PID -> cgroup join), not on
register-trusted attributes.
