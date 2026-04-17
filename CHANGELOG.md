# Changelog

All notable changes to `ingero` (the agent) are tracked here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Fleet-side changes (the OTel Collector distribution) live in the
`ingero-fleet` repo's `CHANGELOG.md`.

## [Unreleased] - v1.0 cut candidate

### Added

- **Real GPU memory signal via `nvidia-smi`** (§2.7). The health
  score's memory signal is now `(gpu_total - gpu_used) / gpu_total`
  read from `nvidia-smi --query-gpu=memory.used,memory.total` across
  all visible GPUs. Falls back to the previous host-RAM proxy when
  `nvidia-smi` is not on `PATH` or errors out. State transitions
  (available → error, error → available) are logged once per change.
- **Sub-minute signal derivation** (§2.6). New `event_aggregates_5s`
  table on the trace DB stores 5-second buckets alongside the
  existing minute-bucket aggregates. `fleet-push --signal-window`
  values at or below 60 s now read the 5 s table, giving sub-minute
  reactivity. The 5 s table's rows are pruned after 10 min of
  retention; the 1 m table retains history bounded by `--max-db`.
- **Straggler transition log line.** `ingero fleet-push` emits an
  Info-level log line (`straggler_state transition`) on each
  HEALTHY↔STRAGGLER edge so operators and e2e scripts can watch for
  transitions without enabling Debug.
- **`cmd/straggler-sink`**. Reference consumer for the `--remediate`
  UDS. Reads NDJSON events from the socket, exposes Prometheus
  metrics (`ingero_sink_events_total`, `ingero_sink_active_stragglers`,
  `ingero_sink_connected`, `ingero_sink_last_event_timestamp_seconds`)
  on a configurable HTTP listener. Shipped for sidecar deployments;
  no new third-party dependencies.
- **Helm chart: optional `straggler-sink` sidecar.** `fleetPush.
  remediation.sinkEnabled` on `deploy/helm/ingero` wires the sink as
  a sidecar container with a shared `emptyDir` at `/tmp` so the agent
  and the sink see the same UDS. Off by default.
- **`fleet/e2e-straggler-test.sh` + `scripts/straggler-harness.py`**
  (§2.3). Real-workload E2E that drives a matmul loop to a degraded
  state via SIGUSR1 and asserts the agent logs a STRAGGLER transition
  and that the straggler-sink reports `active_stragglers=1`. Recovery
  path asserted on SIGUSR2.
- **`fleet/e2e-mtls-test.sh`** (§2.2). Generates a self-signed CA,
  server cert, and client cert; starts Fleet with mTLS on both the
  OTLP receiver and the threshold API; runs `fleet-push` without
  `--fleet-insecure` and verifies push success. Four negative checks
  assert that a missing client cert, a wrong-CA cert, a SAN mismatch,
  and an expired client cert all fail cleanly.
- **TLS material shared with the GET poller.** `ingero fleet-push`
  now applies `--fleet-tls-ca/--fleet-tls-cert/--fleet-tls-key` to
  the threshold GET fallback client in addition to the OTLP push
  client. Previously only the push path respected mTLS.
- **`LoadTLSConfig`** exported from `internal/health` (was
  `loadTLSConfig`). Callers that need a `*tls.Config` for a secondary
  client (poller, custom consumer) now use the same loader as the
  emitter.
- **`docs/fleet-push.md`**. Command reference, signal sources, flag
  table, warmup behavior, detection modes, straggler classification,
  UDS stream format.

### Known limitations in v1.0

- **Memory signal source:** GPU memory is read via the `nvidia-smi`
  subprocess. v1.1 will migrate to
  [`github.com/NVIDIA/go-nvml`](https://github.com/NVIDIA/go-nvml).
- **Sub-minute signal cadence:** reads come from a 5-second bucket
  table, so throughput derivation refreshes at a minimum of 5 s
  latency even if `--fleet-push-interval` is shorter. The EMA
  baseliner smooths this.
- **Multi-replica Fleet:** `replicaCount: 1` is the v1.0 baseline.
  Multi-replica deployments need an L7 LB with consistent-hash
  routing on `cluster_id`. See
  `ingero-fleet/docs/ARCHITECTURE.md`. Native consistent-hash routing
  is v1.1.
- **Kernel minimum:** 5.4 for CO-RE-compatible BTF. `ingero trace`
  may fail to attach on older kernels.
- **Stack trace capture (`trace --stack`)** fails on some kernel 6.x
  builds with `marshal value: []uint8 doesn't marshal to 8 bytes`.
  Use `--stack=false` (the default) as a workaround.

### Dependencies

No new third-party Go modules in v1.0 (the straggler-sink uses the
standard library for its Prometheus text output).

### Backward compatibility

- `ingero fleet-push` CLI flags are unchanged. All additions are
  new internal files or new Helm values.
- Older Fleet (`0.10.0-dev`) remains compatible with v1.0 agents.
- Older agents cannot consume the new `event_aggregates_5s` table,
  but no migration is required: the table is created automatically
  on DB open and is invisible to pre-v1.0 readers.
