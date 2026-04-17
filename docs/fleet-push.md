# `ingero fleet-push`

The fleet-push command turns an ingero agent into a Fleet participant:
it derives a health score from local eBPF telemetry (or synthetic
signals in `--stub` mode), pushes it to a Fleet server via OTLP, reads
the piggyback threshold from the response, and classifies the node as
HEALTHY or STRAGGLER.

The companion `ingero trace --record` process produces the SQLite DB
that fleet-push reads from. Both are intended to run together, either
as a sidecar pod or as two DaemonSets sharing a hostPath volume.

## Quick start

```
# Terminal 1: record GPU activity to disk
sudo ingero trace --record --db /var/lib/ingero/ingero.db

# Terminal 2: derive health scores and push to Fleet
ingero fleet-push \
  --fleet-endpoint http://ingero-fleet.internal:4318 \
  --fleet-cluster-id prod-train-01 \
  --fleet-push-interval 5s \
  --signal-db-path /var/lib/ingero/ingero.db
```

After ~150 s (warmup), the agent transitions from `calibrating` to
`active` and begins contributing to the cluster threshold.

## Signal sources

Two modes:

- **Real (`--stub=false`, the default).** Reads aggregated CUDA events
  from the trace DB. The four signals are:
  - Throughput: `cudaLaunchKernel` rate over the rolling window
  - Compute: total CUDA wall-time as a fraction of the window
  - Memory: GPU memory headroom from `nvidia-smi` (falls back to host
    RAM if `nvidia-smi` is not on `PATH`)
  - CPU: 1 - host CPU utilization from `/proc/stat`

- **Stub (`--stub`).** Synthetic signals. Useful for endpoint smoke
  tests and `--endpoint=` verification. Never use in production.

The real collector queries the `event_aggregates_5s` table when
`--signal-window` is `<= 60s` (sub-minute reactivity, 10-min
retention) and the `event_aggregates` 1-minute table otherwise.

## Flag reference

Generated from `ingero fleet-push --help`:

| Flag | Default | Purpose |
| --- | --- | --- |
| `--fleet-endpoint` | (required) | OTLP HTTP endpoint, e.g. `http://fleet:4318` |
| `--fleet-cluster-id` | (required) | Cluster identifier for aggregation |
| `--fleet-node-id` | hostname | Node identifier |
| `--fleet-workload-type` | `""` | Freeform workload label (e.g. `training`, `inference`) |
| `--fleet-push-interval` | `5s` | OTLP push cadence |
| `--fleet-timeout` | `2s` | Per-request HTTP timeout |
| `--fleet-insecure` | `false` | Use http:// instead of https:// (incompatible with `--fleet-tls-*`) |
| `--fleet-tls-ca` | `""` | Path to CA cert for verifying Fleet |
| `--fleet-tls-cert` | `""` | Path to client cert for mTLS |
| `--fleet-tls-key` | `""` | Path to client key for mTLS |
| `--fleet-persist-path` | `~/.ingero/baseline.json` | Baseline persistence path |
| `--fleet-persist-stale-age` | `10m` | Discard a baseline older than this on startup |
| `--fleet-detection-mode` | `none` | Starting detection mode label |
| `--fleet-world-size` | `0` | Distributed training world size |
| `--fleet-node-rank` | `0` | Rank within the distributed group |
| `--stub` | `false` | Synthetic signal source (testing only) |
| `--fleet-threshold-url` | `""` | GET fallback URL; enables polling when set |
| `--fleet-poll-interval` | `10s` | GET polling cadence (±20% jitter) |
| `--fleet-classifier-hysteresis` | `0.02` | Dead-band to prevent straggler flapping |
| `--remediate` | `false` | Publish straggler events to the UDS |
| `--remediate-socket` | `/tmp/ingero-remediate.sock` | UDS path for `--remediate` |
| `--signal-db-path` | (store default) | SQLite DB populated by `trace --record` |
| `--signal-window` | `60s` | Rolling window for throughput/compute |
| `--signal-num-gpus` | `0` | 0 = autodetect |
| `--warmup-samples` | `30` | Samples before first non-calibrating push (`samples * push_interval` = warmup time) |

## Warmup

Agents observe `--warmup-samples` ticks before their first non-
calibrating push. Default is 30 samples × 5 s = 150 s. Shorten for
tests (`--warmup-samples 12` → 60 s), leave alone for production.

Persisted baselines restored by `--fleet-persist-path` skip warmup:
the state machine starts in `ACTIVE` and the first push already
counts. This is what makes rolling restarts non-disruptive.

## Detection modes

The agent progresses through a four-step fallback chain when Fleet
reachability or quorum degrades:

- `fleet` — live threshold from Fleet via piggyback or GET poll
- `fleet-cached` — Fleet unreachable but we have a recent
  piggyback-delivered threshold still inside its TTL
- `local-cached` — cache expired; we fall back to a threshold
  computed from the agent's own last baseline
- `local-baseline` — fresh local threshold from the EMA baseliner

Each transition is logged at Info level (`detection mode transition`).
Recovery is automatic on the next successful push.

## Straggler classification

When the state machine is ACTIVE and a threshold is available, the
classifier compares each tick's score:

- `score < threshold` → STRAGGLER. Edge emits one OTLP event with
  `ingero.node.straggler_event=1` + UDS message type
  `straggler_state`. Each subsequent STRAGGLER tick continues to emit.
- `score >= threshold + hysteresis` → HEALTHY. Edge emits one UDS
  message type `straggler_resolved`. No further emissions until the
  next STRAGGLER transition.

Edge transitions are logged at Info level (`straggler_state
transition`). Use this log line to detect transitions programmatically.

## UDS remediation stream

With `--remediate`, the agent opens a Unix domain socket (default
`/tmp/ingero-remediate.sock`) and streams NDJSON messages on each
straggler emission. Each line is a JSON object with a `type` field:

- `straggler_state` — agent is currently a straggler
- `straggler_resolved` — agent has returned to healthy

The `cmd/straggler-sink` reference consumer translates the stream into
Prometheus metrics. See its `main.go` doc comment for wire format.

## Minimum Fleet compatibility

- Fleet `0.10.0-dev` (`builder-config.yaml` `dist.version`) and later.
- The fleet-push contract (metric names, attribute names, header
  names) is stable; see `pkg/contract/contract.go` for the authoritative
  list.
