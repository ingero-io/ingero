# Changelog

All notable changes to `ingero` (the agent) are tracked here. Format
follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

Fleet-side changes (the OTel Collector distribution) live in the
`ingero-fleet` repo's `CHANGELOG.md`.

## [Unreleased]

## [0.17.0] - 2026-05-19

External annotation ingest: a recorded trace can be sliced by labels
that external workloads inject while it records.

### Added

- **`ingero annotate` and the `trace --annotate` ingest socket.**
  External workloads (training-loop callbacks, schedulers) write
  step / epoch / task labels into a live recorded trace over a local
  Unix-domain socket. The socket is owner-only by default;
  `trace --annotate-socket-gid` opts a group in. Input is taken from
  stdin or `--from-file`.
- **`query --annotations` and `explain --annotations`.** Join the
  injected labels to events by process incarnation and time window,
  so a recorded trace can be read or filtered per step / epoch / task.
- `examples/integrations/vllm/` - a docker-compose example that runs
  the agent colocated with a vLLM inference server.

## [0.16.0] - 2026-05-11

The inference release. `ingero trace --inference` adds an observability
mode for GPU serving workloads alongside the existing training-focused
tracing.

### Added

- **`ingero trace --inference` mode.** An umbrella mode that builds
  per-workload baselines for inference serving and flags outliers
  against them. Baselines are phase-aware: prefill and decode are
  measured separately, so a decode-heavy interval is not compared
  against a prefill baseline.
- **Inference engine detection.** The agent identifies the serving
  engine (vLLM, SGLang, TGI, Triton) and scrapes its metrics
  endpoint, re-detecting continuously so an engine that starts after
  the agent is still picked up.
- **KV-cache lineage tracking.** Follows KV-cache block allocations
  and emits an allocation-age histogram.
- **`ingero.infer.*` OTLP and Prometheus metrics**, labelled with
  `cluster_id` and `model` so cluster-side rollups can group by
  workload.
- **Per-outlier OTLP trace spans.** Inference outliers emit spans on
  the OTLP `/v1/traces` endpoint, with deterministic TraceID and
  EventID derivation.
- **Kernel-fingerprint workload key.** `cudaLaunchKernel` stream
  handles are captured so a workload can optionally be keyed by its
  kernel fingerprint.
- **Remediation server.** `internal/remediate` exposes a wire
  contract for remediation actions, with a UDS outlier protocol for
  local consumers.
- **Database rollover.** The local store rolls over on a size or age
  bound, with schema setup handled at rotation.
- **JSON daemon mode** wires the correlator and node identity into
  `--json` output.

### Changed

- Read-only SQL access is enforced at the engine level rather than by
  a query-string filter.
- `pkg/contract` gains a drift-guard and shared constants; export
  paths adopt the contract constants and propagate context.
- BPF probes: PID/TID naming, graph output handles, watchdog clock
  alignment.

### Security

- Safe defaults on external surfaces: secrets are read from the
  environment, TLS is required, and listeners bind to loopback unless
  configured otherwise.

### Fixed

- `demo --json` no longer panics: the system collector is initialized
  unconditionally in JSON mode.
- SGLang model extraction and `--inference` event routing corrected
  against real-engine validation.

### Dependencies

- Go 1.26.3; `golang.org/x/net` 0.53.0.

## [0.15.0] - 2026-05-07

### Added

- **Runtime libnccl uprobe attachment.** The agent now attaches NCCL
  collective uprobes against libnccl paths the libnccl-discovery
  scanner finds at runtime, in addition to the systemwide
  attachment that ran at startup. PyTorch + pip workloads ship
  libnccl in a venv that startup-time attach could never see, so
  the previous behavior captured zero NCCL events on bare cloud
  GPU VMs. Tracer API now exposes `Prepare()` + `AttachAt(libPath)`
  for the runtime-attach flow; `Attach(filterPIDs)` is a
  compatibility shim. New helper `Tracer.AttachedPaths()` for
  observability.
- **Prometheus running counters for NCCL collectives.** Vanilla
  Prometheus scrapers can now see NCCL activity on the agent:
  `gpu_nccl_collective_count{op_type}`,
  `gpu_nccl_collective_bytes_total{op_type}`,
  `gpu_nccl_collective_barrier_events{op_type}`. OTLP keeps the
  per-event gauge view as the canonical channel; the counters
  here are the pull-friendly slice. Cumulative across the agent
  process lifetime.
- **`--mcp-bearer-token <token>` flag** on `ingero mcp`. Requires
  `Authorization: Bearer <token>` for HTTPS MCP requests when set;
  empty disables auth (loopback-only deployments). Constant-time
  compare via SHA-256 padding so the wall-clock cost does not
  depend on input length. The `pagerduty_trigger` MCP tool is now
  registered automatically when (a) running on stdio (loopback by
  definition), or (b) running on HTTP with `--mcp-bearer-token`
  set; the v0.14 blanket default-off caveat is removed.
- **OTLP `Histogram` and `ExponentialHistogram` encoders.** The
  agent can now emit per-event distributions to OTLP receivers,
  not just Gauge and Sum. Encoder is direct JSON, no OTel-SDK
  dependency.
- **`gpu.memcpy.duration_ms` is now a per-event histogram** instead
  of the v0.14 per-window-average gauge. OTLP (`Histogram` data
  point) and Prometheus (`*_bucket` / `*_sum` / `*_count`) both
  updated. Bucket boundaries (ms): 0.1, 0.25, 0.5, 1, 2.5, 5, 10,
  25, 50, 100, 250, 500, 1000. Histogram is cumulative (matches
  Prometheus convention). Downstream queries reading the prior
  gauge migrate to histogram queries OR use fleet's updated
  `memcpy_bandwidth_summary` percentile output.

### Changed

- New `internal/auth` package: bearer parsing + constant-time
  compare + hardened TLS keypair loader (`auth.LoadTLSKeyPair`).
  Wires every TLS load site (MCP HTTPS, dashboard, fleet client,
  healthee EE poller). Refuses world-readable keys
  (`mode & 0o077 != 0`) unless `INGERO_TLS_ALLOW_LOOSE_KEY_PERMS=1`;
  rejects directories.
- `scripts/aws/v0-13/provision.sh`: validates operator public IP
  is a real IPv4 address before passing it to AWS as a
  SecurityGroup CidrIp; rejects HTML error bodies / captive-portal
  redirects from `checkip.amazonaws.com`.
- `scripts/aws/v0-13/deploy.sh`: pre-flight banner prints SSH user,
  SSH key, and target IPs before the first rsync; refuses to
  deploy on missing or non-IPv4 entries.
- Sampler doc note in `internal/sampling`: intentionally not
  cryptographically secure; this is a load-shedder, not a security
  control.
- `cgroup_path_hash` doc note in `internal/health/cgroup_cache.go`:
  the SHA-256-truncated digest is a stability tag for joining
  metric streams, not a confidentiality shield. Operators with
  sensitive cgroup paths should disable cgroup tagging or pass a
  per-cluster salt upstream.
- `docs/otlp.md` adds a trust-model section: BPF-derived register
  attributes (memcpy direction, NCCL op-type, IOCTL command numbers,
  kernel grid/block dims) cannot be trusted as security signals on
  multi-tenant hosts; cross-tenant correlation should rely on
  cgroup attribution.

### Added (experimental)

- **`--enable-experimental-kprobes` flag** on `ingero trace`
  (default off). When set, the agent reads
  `/proc/driver/nvidia/version` + `uname -r` and only loads the
  experimental probes if the pair is on the
  `internal/kprobe.DefaultAllowlist`. Off-allowlist hosts get a
  startup warning and the probes do NOT load. Allowlist seeded
  with three pairs: `{driver: 535.*, kernel: 5.15.*}` (Lambda A10
  prior baseline), `{driver: 535.*, kernel: 6.5.*}` (Lambda GH200
  prior baseline), `{driver: 570.*, kernel: 6.8.*}` (Lambda current
  Ubuntu 22.04 image, validated on A10 + 2x H100 amd64). New
  pairs are added only after a real-hardware run confirms the
  probes attach + fire correctly.
- **Memfrag IOCTL kprobe.** Hooks `nvidia_unlocked_ioctl` on the
  closed NVIDIA driver and emits per-cmd ringbuf events. v0.15
  records the IOCTL `cmd` field; argument-buffer decode (alloc
  size, virtual address) is a follow-up. Per-cmd counter
  `gpu.memfrag.ioctl_event_total{cmd}` exposed via Prometheus +
  OTLP. New `fleet.cluster.memfrag_hotspots` MCP tool reads it.
  Only fires when the experimental-kprobes flag is on AND the
  host is on the allowlist.
- **Throttle event counters via edge detection.**
  `gpu.throttle.{power,thermal,sw,hw}.event_total` counters
  layered on the existing nvidia-smi throttle poller. Each rising
  edge per (gpu_uuid, bucket) increments the counter once. The
  underlying poll floor is unchanged; sub-poll bursts are still
  missed by design (same caveat as the v0.12.10 W2 poller). Real
  event-driven kernel-level throttle detection requires a kprobe
  target NVIDIA does not publicly name or a libnvidia-ml.so cgo
  binding; both are deferred.
- **CUDA kernel launch dims uprobe.** Hooks `cuLaunchKernel` in
  libcuda.so and captures grid (X, Y, Z) + block (X, Y)
  dimensions per launch. Block Z requires PARM7 access (stack on
  amd64, register on arm64) and is a follow-up. New metrics:
  `gpu.kernel.launch.count`, `gpu.kernel.launch.threads_per_block`
  (histogram), `gpu.kernel.launch.grid_blocks` (histogram). New
  `fleet.cluster.kernel_launch_summary` MCP tool reads them. Only
  fires when the experimental-kprobes flag is on AND the host is
  on the allowlist.

### Fixed

- **NCCL ringbuf decode panic under Go 1.26.** The Event struct
  had unexported `_pad` / `_pad2` ABI padding fields that
  `binary.Read` panicked on under Go 1.26's stricter
  unexported-field reflect rules. Renamed to `Pad1` / `Pad2`;
  EventSize=104 and field offsets stay identical.
- **NCCL discovery dispatcher panic when running without `--nccl`.**
  A typed-nil `*ncclprobe.Tracer` widened to a non-nil
  `ncclAttacher` interface bypassed the naive `att == nil` check
  and segfaulted on the next method call. Added an explicit
  typed-nil guard in `dispatchNCCLAttach`. Triggered whenever the
  agent ran with `--prometheus` but without `--nccl` (the typical
  inference deployment shape).
- `tests/e2e/memcpy-1d-direction-matrix.sh` and
  `tests/e2e/memcpy-2d-peer-multigpu.sh` now compile the workload
  with `nvcc -cudart=shared` so the agent's uprobes against
  `/usr/lib/.../libcudart.so` see the workload's calls. Without
  this, nvcc statically links libcudart.
- `tests/e2e/nccl-abi-matrix.sh`: assertion now reads the
  Prometheus running counter `gpu_nccl_collective_count{op_type}`
  instead of the events-table SQL JOIN (the sources table has no
  `NCCL` row; NCCL events flow through a separate aggregation
  layer).

## [0.14.2] - 2026-05-06

### Added

- `.github/workflows/lambda-e2e-harness.yml`: workflow_dispatch CI
  job that provisions a Lambda Labs GPU instance, builds the
  agent, runs a curated subset of `tests/e2e/` scripts, captures
  per-script artifacts, posts a markdown summary, and terminates
  the VM. Default instance is gpu_1x_a10. Manual trigger only.

### Changed

- `tests/e2e/nccl-abi-matrix.sh`: assertion queries the trace
  SQLite DB directly for NCCL event count instead of scraping
  Prometheus (Prometheus exposition omits per-event
  `nccl.collective.*` gauges). The test now SKIPs when libnccl
  discovery sees a process but the per-version event count stays
  at zero, instead of FAILing on a known capability gap that is
  upstream of the test's contract.

## [0.14.1] - 2026-05-06

### Fixed

- Prometheus `/metrics` exporter now includes the v0.12.10 +
  v0.14.0 metric set (`gpu_throttle_*`, `gpu_nccl_process_loaded`,
  `gpu_nccl_processes_total`, `gpu_memory_{used,free,total}_bytes`,
  `gpu_memory_fragmentation_estimate`,
  `gpu_memory_process_allocated_bytes`,
  `gpu_memcpy_bytes_total{direction}`,
  `gpu_memcpy_duration_ms{direction}`). Same labels as the OTLP
  emission. Operators scraping Prometheus directly previously saw
  only the pre-v0.12.10 metric subset.

### Added

- Expanded end-to-end test harness under `tests/e2e/` covering ML
  engineer workflows (install-from-release, trace flag matrix,
  OTLP roundtrip, libnccl discovery assertions, memcpy/memfrag/
  throttle behavior, NCCL ABI matrix, k3s migration, soak) and
  data-plane completeness (event-coverage matrix, counter
  reconciliation, drop counter, critical-event guarantee, external
  OTel mirror, fan-in completeness, /investigate LLM behavior).
  See `tests/e2e/README.md` for how to run.
- Companion local-stack at `ingero-fleet/examples/local-stack/`
  for GPU-less fan-in validation on WSL or CI runners.
- `tests/e2e/quickstart-readme.sh`: regression test for the
  README's "Try it in 60 seconds" path (`--help`, `version`,
  `check`, `demo --no-gpu` for `incident`+`cold-start`,
  `mcp --help` tool advertisement). Runs CPU-only and rootless
  on every push via a new `README quickstart regression` step
  in `.github/workflows/ci.yml`.
- `tests/workloads/cuda_busy.cu`: small self-contained CUDA
  stresser (compute-bound kernel, no external deps), used by
  `throttle-induced.sh` as a PyTorch-free workload so the test
  runs on bare GPU VMs without `pip install torch`. Helper
  `_lib.sh::ensure_cuda_busy` builds it on demand and caches
  the binary in `/tmp`.

### Fixed (test harness)

- e2e cleanup: replaced the brittle `sudo kill "$AGENT_PID"`
  pattern (which captures the `sudo` PID, not the agent's, so
  the actual `ingero trace` process leaks across sequential
  test runs) with a `_lib.sh::kill_agent` helper that uses
  `pkill -f`. Applied to every script that boots the agent.

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
