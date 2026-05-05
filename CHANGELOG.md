# Changelog

All notable changes to `ingero` (the agent) are tracked here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Fleet-side changes (the OTel Collector distribution) live in the
`ingero-fleet` repo's `CHANGELOG.md`.

## [Unreleased]

## [0.14.0] - 2026-05-05

Skips v0.13.

### Added

- **libnccl process discovery**: periodic `/proc/PID/maps` scanner
  emits `gpu.nccl.process_loaded` (gauge=1 per discovered PID,
  labelled with `pid`, `comm`, `libnccl_path`, `libnccl_version`)
  and `gpu.nccl.processes_total`. New flag
  `--libnccl-discovery-interval` (default 10s; 0 disables).
- **Memcpy uprobes** for `cudaMemcpy2D`, `cudaMemcpy2DAsync`,
  `cudaMemcpyPeer`, `cudaMemcpyPeerAsync`. Per-direction labels
  (`h2h`, `h2d`, `d2h`, `d2d`, `default`, `unknown`). New gauges
  `gpu.memcpy.bytes_total{direction}` (cumulative counter) and
  `gpu.memcpy.duration_ms{direction}` (per-window average). 2D
  variants encode `direction=unknown` because the kind argument
  lives where libbpf's `PT_REGS_PARMn` macros cannot read it; the
  1D variants remain the precise direction-and-bytes signal.
- **W1 NVML-poll memfrag heuristic**: `gpu.memory.fragmentation_estimate`
  (gauge 0..1, derived from nvidia-smi memory.{used,free,total}) and
  per-PID `gpu.memory.process.allocated_bytes`. New flag
  `--memfrag-poll-interval` (default 10s; 0 disables). Polling-based;
  v0.15 will replace with an event-driven IOCTL kprobe variant.
- Per-cgroup CUDA + CPU-stall metric variants alongside existing
  per-process series. New `internal/health/cgroup_cache.go` builds
  the cgroup-path-hash identity.
- OTLP traces emission for detection events
  (`internal/tracing/tracer.go`).
- `ingero rates update` CLI subcommand. Live update with embedded
  Markdown fallback.
- `pagerduty_trigger` MCP tool. PagerDuty Events v2 trigger from
  the agent's detection-event shape. Off by default; opt in via
  `SetPagerDutyMCPEnabled(true)` after pairing the agent MCP with
  bearer auth.
- AWS validation harness: `scripts/aws/v0-13/{deploy,preflight,
  provision,teardown}.sh`.
- YAML config loader (`internal/config/config.go`).
- Trace-sampling policy (`internal/sampling/sampler.go`).
- Six new metric-name constants in `pkg/contract/contract.go`
  (`MetricGPUNCCLProcessLoaded`, `MetricGPUNCCLProcessesTotal`,
  `MetricGPUMemoryFragmentation`, `MetricGPUMemoryProcessAllocated`,
  `MetricGPUMemcpyBytesTotal`, `MetricGPUMemcpyDurationMS`); plus
  `ingero.cgroup.path_hash` attribute constant.
- `docs/troubleshooting.md`: patterns Ingero detects, operational
  cheat sheet, advanced configuration.

### Changed

- CHANGELOG backfilled with versioned sections for v0.10.1,
  v0.11.0, v0.12.1, v0.12.2, v0.12.4, v0.12.6, v0.12.7, v0.12.8,
  v0.12.9, and v0.12.10.
- `docs/otlp.md` adds the W2-poller bit-to-bucket mapping and the
  four `gpu.throttle.*_active` series; NCCL collective metrics also
  documented.
- `docs/commands.md` documents `--throttle-poll-interval`.
- README trimmed; technical reference content moved to
  `docs/troubleshooting.md`.
- `internal/store/store.go` schema extended with per-cgroup
  attribution columns (additive; older readers unaffected).
- PagerDuty client response body bounded by `io.LimitReader(64KiB)`
  before close.

### Fixed

- `cudaMemcpy2D` / `cudaMemcpy2DAsync` BPF probes encode
  `direction=5` (`unknown`) instead of `direction=0` (`h2h`).
  Earlier code relied on a userspace convention to interpret
  direction=h2h on a 2D op as unknown; that convention was not in
  the contract. The fix makes the metric label honest and prevents
  silent mixing of true H2H copies with 2D-with-unknown-direction.

## [0.12.10] - 2026-05-04

### Added

- **NVML clock-throttle reason metrics** (W2-poller). `ingero trace`
  now polls `nvmlDeviceGetCurrentClocksThrottleReasons` for every
  visible GPU at a configurable interval (default 5 s, override via
  `--throttle-poll-interval`; `0` disables) and emits four OTel
  gauges per device, labelled with `gpu.uuid`:
  - `gpu.throttle.power_active`
  - `gpu.throttle.thermal_active`
  - `gpu.throttle.sw_active`
  - `gpu.throttle.hw_active` (umbrella for any HW reason; future NVML
    bits not yet in the bucket table funnel here so dashboards do not
    silently lose visibility on a newer driver)

  Value is `1` when the bucket is active and `0` otherwise. The
  metric names are a stable contract: dashboards may pin to them.
  This is the polling-based variant of W2; the event-driven BPF
  version tracked in #133 remains for a future release alongside W1.
  Poll interval is the bursting floor: a throttle event shorter than
  the interval may be missed by design. Consumer GPUs that return
  `[Not Supported]` for the throttle field are skipped per device;
  the agent logs the skip once per GPU at info level instead of
  every tick. Full bit-to-bucket mapping documented in
  [`docs/otlp.md`](docs/otlp.md).

## [0.12.9] - 2026-05-04

### Changed

- **`/investigate` prompt aware of Echo federation.** The agent's MCP
  `/investigate` prompt now opens with a one-paragraph note for cases
  where it was reached via Ingero Echo's federated cluster-investigate
  flow: Echo has already ranked the node, so this prompt's job is the
  per-node "WHY" half (root cause + recommendation) while Echo
  handles the cluster-side "WHERE" half. Standalone usage is
  unchanged. In-tandem release alignment with Fleet's new Echo-side
  `/investigate` orchestration.

## [0.12.8] - 2026-05-03

In-tandem release with Fleet v0.12.8 (no agent code changes; tag
maintained for sync per the in-tandem release rule). Fleet shipped
the healthee OTel Collector extension and the quanthealth library.

## [0.12.7] - 2026-05-02

Doc-sync release synchronizing version pins to v0.12.6. No agent
code changes.

## [0.12.6] - 2026-05-02

Release-pipeline hotfix on top of v0.12.5. Same agent scope as
v0.12.5; the goreleaser pipeline fix landed at the fleet repo so
fleet artifacts could publish.

## [0.12.4] - 2026-05-02

### Changed

- **AWS provider detection hardened.** Instance-id shape validation
  added; the metadata URL is now injectable for tests.

### Internal

- Lint cleanup: silenced staticcheck S1000 / S1037 / U1000 / ST1013
  warnings in pre-existing code paths.

## [0.12.2] - 2026-05-02

### Added

- **NCCL probe per-tenant PID filter.** New BPF hash map
  `nccl_target_pids` (max 256 entries; sentinel at key 0 enables
  the filter; empty map = trace all PIDs). Userspace API:
  `Tracer.SetTargetPID(uint32)` / `Tracer.ClearTargetPIDs()`.
  `setupNCCLTracer` wires `--pid` through to the filter when set.
- **NCCL Send/Recv peer rank.** `ncclSend` / `ncclRecv` uprobes
  capture PARM4 as `peer_rank`; emitted as OTLP attribute
  `nccl.peer_rank` (only when non-zero). New contract constant
  `pkg/contract/contract.AttrNCCLPeerRank`.
- **Barrier-correlator refactor.** Package-scope state lifted to a
  `barrierCorrelator{mu, state, out}` struct so tests can construct
  isolated correlators. Default singleton preserves the agent-CLI
  wiring.
- **`HasCapBPF()` rewrite.** Replaces shell-out / capability-mask
  parsing with a direct `unix.Capget` syscall (capability v3
  header). Adds `capget` to the seccomp allowlist.
- **`setupNCCLTracer` factory.** Extracted from `traceRunE` with an
  injectable `ncclSetupParams` struct (geteuid, hasCapBPF,
  findLibForPID, findLibSystemwide, debugf, stderr) so paths can be
  unit-tested without spinning up cobra or real BPF.

### Fixed

- **NCCL probe attach-vs-PID-filter race.** Closed the window where
  target PIDs could change between `Attach()` and
  `SetTargetPIDs()`. `ClearTargetPIDs` errors now propagate to the
  caller instead of being silently swallowed.

## [0.12.1] - 2026-05-02

### Added

- **NCCL collective uprobe instrumentation.** 16 uprobes against
  libnccl.so (or libtorch_cuda.so for statically-linked PyTorch):
  `ncclCommInitRank`, `ncclCommDestroy`, `ncclAllReduce`,
  `ncclAllGather`, `ncclReduceScatter`, `ncclBcast`, `ncclSend`,
  `ncclRecv` (each with uprobe+uretprobe). Auto-discovers libnccl
  via `/proc/<pid>/maps`; falls back to libtorch_cuda.so or
  libtorch_global_deps.so.
- **`nccl.collective.duration_ms`, `barrier_wait_ms`, `bytes`**
  emitted per-collective with `op_type`, `comm_id_hash`, `rank`,
  `nranks`, `datatype`, `reduce_op`.
- **Agent-side barrier correlator** joins NCCL collective uretprobes
  with the next `cudaStreamSynchronize` on the same `(pid, stream)`
  and emits `nccl.collective.barrier_wait_ms`. 5-min correlation
  window with GC.
- **`--nccl` privilege gate**: surfaces a clear "requires root or
  CAP_BPF + CAP_PERFMON" warning instead of the libbpf attach error
  deep in the stack.
- **Stream-sync tap on the CUDA tracer channel** forks every
  `CUDAStreamSync` event into the barrier correlator without
  disturbing the merged event stream.
- **Contract docs**: `pkg/contract/contract.go` documents
  `world_size` (resource-level, job-wide) vs `nranks` (per-comm).
  Same number for single-comm jobs; diverge for FSDP+TP / MoE.

### Fixed

- **Reliability gaps from automated audit.** Hardening pass on
  agent paths flagged in v0.11.0's automated testing review.

## [0.11.0] - 2026-05-01

### Added

- **`ingero-alerter` sidecar.** Slack + PagerDuty backends for
  per-node straggler alerts. Reads NDJSON events from the agent's
  `--remediate` UDS; renders configurable templates.
- **`ingero check --support-bundle` flag.** Writes a diagnostic
  tarball (config, recent log, BPF program names, kernel/driver
  info) for support cases.
- **`ingero migrate` subcommand + framework.** First-class schema
  migrations for the trace DB; introduces a versioned migration
  registry so v0.11+ schema changes can apply forward without
  manual operator action.
- **Cost-of-problem gauges**: `ingero.node.info` (descriptor
  labels) and `ingero.node.world_size` (rank-count gauge per
  training job).
- **Per-feature integration assertion lines** in `gpu-test.sh`
  covering host kernel, block I/O, net syscalls, TCP retransmit.
- **CI matrix**: adds `ubuntu-22.04-arm` and `ubuntu-24.04-arm`
  runners.
- **Lambda GH200 hardware smoke** as an on-demand CI workflow.
- **Architecture diagram** added to README.

## [0.10.1] - 2026-05-01

### Fixed

- **arm64 release binaries now load BPF probes correctly.** The
  v0.10.0 release archives shipped a single set of pre-compiled
  BPF objects built for x86_64 and embedded that set into both the
  linux_amd64 and linux_arm64 binaries. On any aarch64 kernel the
  `pt_regs` CO-RE relocations resolved against the wrong struct
  layout and the verifier rejected the load with `bad CO-RE
  relocation: invalid func unknown`. bpf2go is now invoked with
  `-target amd64,arm64`; per-arch `.bpf.o` files are committed
  under `internal/ebpf/<pkg>/<name>_<arch>_bpfel.{go,o}` and
  selected at Go build time via build constraints. A new
  `internal/ebpf/parity` test package guards the per-arch output
  so the same regression cannot reach a release again. Reported by
  @saiyam1814 against DGX Spark (issue #35); affects every arm64
  release-binary user, not only Spark.
- **MCP server reports build-time version.** `ingero mcp` now
  reports the version embedded at build time (e.g. `v0.10.0`) in
  the MCP `Implementation.Version` field, instead of the
  previously hardcoded `"0.9.0"` string. Affects how MCP clients
  (Claude Desktop, IDE plugins, glama.ai) identify the server.
  The hardcode shipped in v0.10.0; binaries built from main HEAD
  or from any v0.10.1+ tag carry the fix.

### Internal

- New `bpf-freshness` CI job re-runs `make generate` on every PR
  and asserts no drift in `internal/ebpf/`, catching the original
  failure mode where committed BPF artifacts diverged from BPF C
  sources.
- `bpf/vmlinux.h` committed as canonical type catalog; Makefile
  vmlinux auto-regen dropped (`vmlinux.h` is now version-
  controlled). `user_pt_regs` shim added in `bpf/common.bpf.h` for
  arm64 register access.

## [0.10.0] - 2026-04-21

### Added

- **Real GPU memory signal via `nvidia-smi`.** The health score's memory
  signal is now `(gpu_total - gpu_used) / gpu_total` read from
  `nvidia-smi --query-gpu=memory.used,memory.total` across all visible
  GPUs. Falls back to the previous host-RAM proxy when `nvidia-smi` is
  not on `PATH` or errors out. State transitions (available → error,
  error → available) are logged once per change.
- **Sub-minute signal derivation.** New `event_aggregates_5s` table on
  the trace DB stores 5-second buckets alongside the existing
  minute-bucket aggregates. `fleet-push --signal-window` values at or
  below 60 s now read the 5 s table, giving sub-minute reactivity. The
  5 s table's rows are pruned after 10 min of retention; the 1 m table
  retains history bounded by `--max-db`.
- **Straggler transition log line.** `ingero fleet-push` emits an
  Info-level log line (`straggler_state transition`) on each
  HEALTHY↔STRAGGLER edge so operators and e2e scripts can watch for
  transitions without enabling Debug.
- **`cmd/straggler-sink`.** Reference consumer for the `--remediate`
  UDS. Reads NDJSON events from the socket, exposes Prometheus metrics
  (`ingero_sink_events_total`, `ingero_sink_active_stragglers`,
  `ingero_sink_connected`, `ingero_sink_last_event_timestamp_seconds`)
  on a configurable HTTP listener. Shipped for sidecar deployments;
  no new third-party dependencies.
- **Helm chart: optional `straggler-sink` sidecar.**
  `fleetPush.remediation.sinkEnabled` on `deploy/helm/ingero` wires the
  sink as a sidecar container with a shared `emptyDir` at `/tmp` so the
  agent and the sink see the same UDS. Off by default.
- **`fleet/e2e-straggler-test.sh` + `scripts/straggler-harness.py`.**
  Real-workload E2E that drives a matmul loop to a degraded state via
  SIGUSR1 and asserts the agent logs a STRAGGLER transition and the
  straggler-sink reports `active_stragglers=1`. Recovery path asserted
  on SIGUSR2.
- **`fleet/e2e-mtls-test.sh`.** Generates a self-signed CA, server
  cert, and client cert; starts Fleet with mTLS on both the OTLP
  receiver and the threshold API; runs `fleet-push` without
  `--fleet-insecure` and verifies push success. Four negative checks
  assert that a missing client cert, a wrong-CA cert, a SAN mismatch,
  and an expired client cert all fail cleanly.
- **TLS material shared with the GET poller.** `ingero fleet-push` now
  applies `--fleet-tls-ca` / `--fleet-tls-cert` / `--fleet-tls-key` to
  the threshold GET fallback client in addition to the OTLP push
  client. Previously only the push path respected mTLS.
- **`LoadTLSConfig`** exported from `internal/health` (was
  `loadTLSConfig`). Callers that need a `*tls.Config` for a secondary
  client (poller, custom consumer) now use the same loader as the
  emitter.
- **`docs/push_fleet.md`.** Command reference, signal sources, flag
  table, warmup behavior, detection modes, straggler classification,
  UDS stream format.
- **eBPF Python walker support for CPython 3.9, 3.13, and 3.14.** The
  walker dispatcher gained `case 9` (routed to `walker_310`, same
  legacy PyFrameObject layout) and `case 13` / `case 14` (routed to
  `walker_312`, direct `_PyInterpreterFrame` layout; 3.13 dropped the
  `_PyCFrame` indirection). New offset tables `pyOffsets39`,
  `pyOffsets313`, `pyOffsets314` cover the PyInterpreterState size
  growth (3.13's `interpreters.head` sits at byte 632 vs 3.12's 40,
  3.14's at byte 808) and the PyASCIIObject layout change in 3.13+
  (`UnicodeState` 20 → 32, `UnicodeData` 48 → 40). The runtime
  harvester's `InterpTstateHead` scan was widened from 128 to 1024
  slots (8 KiB) so it reaches 3.13+'s far-deep `threads.head` field.
  Harvester overlay is skipped for 3.13+ because its frame-walking
  heuristics emit incorrect offsets on the reshuffled struct; the
  hardcoded tables and `_Py_DebugOffsets` reader are authoritative on
  3.13+.

### Fixed

- **Stack trace capture (`trace --stack`)** no longer fails with
  `marshal value: []uint8 doesn't marshal to 8 bytes` on modern 6.x
  kernels. Cross-kernel validation on AL2023 (6.1), Ubuntu 22.04
  (6.8), and Ubuntu 24.04 (6.17) reproduced the failure on every
  tested 6.x kernel (not just Azure, as originally reported in
  [issue #24](https://github.com/ingero-io/ingero/issues/24)). The
  root cause was that `cuda_trace.bpf.c`'s `config_map.Put` was given
  a raw `[]byte` whose length had to match the kernel-side BTF-derived
  value size; on newer kernels that size is resolved differently. The
  cuda and driver tracers now pass a typed `*IngeroConfig` struct,
  which cilium/ebpf marshals via BTF rather than by length, making the
  write robust across kernel versions.
- **`--py-walker=ebpf` now emits frames for Python 3.10 and 3.11.**
  Previously the walker advertised support for 3.10/3.11/3.12 but
  empirically only 3.12 produced Python frames on mainstream builds:
  `find_thread_state` would walk the PyThreadState list looking for a
  match on `native_thread_id`, but Ubuntu's patched CPython 3.10/3.11
  leaves that field zero on the main thread. The walker now falls back
  to the first (and only) PyThreadState when exactly one thread is
  present and no match was found, mirroring the userspace walker's
  long-standing single-thread fallback. Multi-threaded workloads
  without a populated `native_thread_id` still return empty frames so
  we don't emit a wrong thread's stack. Hardcoded PyFrameObject and
  PyCodeObject offsets for 3.10 were also corrected (`FrameCode`
  16 → 32, `CodeFirstLineNo` 48 → 40).

### Limitations

- **Memory signal source:** GPU memory is read via the `nvidia-smi`
  subprocess. Future migration to
  [`github.com/NVIDIA/go-nvml`](https://github.com/NVIDIA/go-nvml)
  is planned.
- **Sub-minute signal cadence:** reads come from a 5-second bucket
  table, so throughput derivation refreshes at a minimum of 5 s
  latency even if `--fleet-push-interval` is shorter. The EMA
  baseliner smooths this.
- **Multi-replica Fleet:** `replicaCount: 1` is the recommended
  default. Multi-replica deployments need an L7 LB with consistent-hash
  routing on `cluster_id`. See `ingero-fleet/docs/ARCHITECTURE.md`.
- **Kernel minimum:** 5.4 for CO-RE-compatible BTF. `ingero trace`
  may fail to attach on older kernels.
- **`--py-walker=ebpf` Python 3.13 and 3.14** emit a single frame per
  cuda event instead of the full call chain. The walker correctly
  identifies the current Python frame (filename + function name +
  line) but `_PyInterpreterFrame.previous` traversal across 3.13's new
  frame-stack allocator is not yet implemented; the chain walk stops
  after the top frame. 3.10/3.11/3.12 produce the full chain.
- **`--py-walker=ebpf` Python 3.9** dispatches to `walker_310` but
  returns zero frames on mainstream builds. The PyThreadState and
  PyFrameObject offsets that worked for 3.10 do not carry over
  cleanly; needs a 3.9-specific validation pass on a vanilla build.
- **`--py-walker=ebpf` on CUDA 13 hosts (Ubuntu 24.04 DLAMI):** the
  CUDA runtime uprobes do not fire on `libcudart.so.13`, so the
  runtime-tracer-side walker integration never runs. Only driver
  events are captured on these hosts.

### Dependencies

No new third-party Go modules. The straggler-sink uses the standard
library for its Prometheus text output.

### Backward compatibility

- `ingero fleet-push` CLI flags are unchanged. All additions are new
  internal files or new Helm values.
- Older agents cannot consume the new `event_aggregates_5s` table,
  but no migration is required: the table is created automatically on
  DB open and is invisible to older readers.
