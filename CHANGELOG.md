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

### Fixed

- **Stack trace capture (`trace --stack`)** no longer fails with
  `marshal value: []uint8 doesn't marshal to 8 bytes` on modern 6.x
  kernels. Cross-kernel validation on AL2023 (6.1), Ubuntu 22.04
  (6.8), and Ubuntu 24.04 (6.17) reproduced the failure on every
  tested 6.x kernel (not just Azure, as originally reported in
  [issue #24](https://github.com/ingero-io/ingero/issues/24)). The
  root cause was that `cuda_trace.bpf.c`'s `config_map.Put` was
  given a raw `[]byte` whose length had to match the kernel-side
  BTF-derived value size; on newer kernels that size is resolved
  differently. The cuda and driver tracers now pass a typed
  `*IngeroConfig` struct, which cilium/ebpf marshals via BTF rather
  than by length, making the write robust across kernel versions.

- **`--py-walker=ebpf` now emits frames for Python 3.10 and 3.11.**
  Previously the walker advertised support for 3.10/3.11/3.12 but
  empirically only 3.12 produced Python frames on mainstream builds:
  `find_thread_state` would walk the PyThreadState list looking for a
  match on `native_thread_id`, but Ubuntu's patched CPython 3.10/3.11
  leaves that field zero on the main thread. The walker now falls
  back to the first (and only) PyThreadState when exactly one thread
  is present and no match was found, mirroring the userspace walker's
  long-standing single-thread fallback. Multi-threaded workloads
  without a populated `native_thread_id` still return empty frames so
  we don't emit a wrong thread's stack. Hardcoded PyFrameObject and
  PyCodeObject offsets for 3.10 were also corrected (`FrameCode` 16 →
  32, `CodeFirstLineNo` 48 → 40).

### Added

- **eBPF Python walker support for CPython 3.9, 3.13, and 3.14.**
  The walker dispatcher gained `case 9` (routed to `walker_310`, same
  legacy PyFrameObject layout) and `case 13`/`case 14` (routed to
  `walker_312`, direct `_PyInterpreterFrame` layout; 3.13 dropped the
  `_PyCFrame` indirection). New offset tables `pyOffsets39`,
  `pyOffsets313`, and `pyOffsets314` cover the PyInterpreterState
  size growth (3.13's `interpreters.head` sits at byte 632 vs 3.12's
  40, 3.14's at byte 808) and the PyASCIIObject layout change in
  3.13+ (`UnicodeState` 20 → 32, `UnicodeData` 48 → 40). The runtime
  harvester's `InterpTstateHead` scan was widened from 128 to 1024
  slots (8 KiB) so it reaches 3.13+'s far-deep `threads.head` field.
  Harvester overlay is skipped for 3.13+ because its frame-walking
  heuristics were written against 3.12's `_PyInterpreterFrame` layout
  (f_executable at offset 0) and emit incorrect offsets on 3.13's
  reshuffled struct; the hardcoded tables and, eventually, the
  `_Py_DebugOffsets` reader are the authoritative sources on 3.13+.

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
- **`--py-walker=ebpf` Python 3.13 and 3.14 emit a single frame per
  cuda event instead of the full call chain.** The walker correctly
  identifies the current Python frame (filename + function name +
  line) but `_PyInterpreterFrame.previous` traversal across 3.13's
  new frame-stack allocator is not yet implemented; the chain walk
  stops after the top frame. 3.10/3.11/3.12 produce the full chain.
- **`--py-walker=ebpf` Python 3.9 dispatches to `walker_310` but
  returns zero frames on mainstream builds.** The PyThreadState and
  PyFrameObject offsets that worked for 3.10 do not carry over
  cleanly; needs a 3.9-specific validation pass on a vanilla build.
- **`--py-walker=ebpf` on CUDA 13 hosts (Ubuntu 24.04 DLAMI):** the
  CUDA runtime uprobes do not fire on `libcudart.so.13`, so the
  runtime-tracer-side walker integration never runs. Only driver
  events are captured on these hosts. Unrelated to the walker itself
  but masks it on that environment.

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
