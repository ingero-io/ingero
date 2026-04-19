# Ingero v1.0 Adversarial Review (Pre-Flight)

> Scope: new attack surface introduced by the v1.0 fleet branch (§2.2,
> §2.3, §2.4, §2.6, §2.7) on top of the 2026-04-16 review baseline.
>
> This pre-flight identifies the delta, documents what already holds,
> and lists the probes that still need a hands-on pass before the v1.0
> cut. It is not a substitute for running the attacks; it is the
> scoping document for that session.

## Delta vs the 2026-04-16 baseline

### New code paths

| File | Purpose | New surface |
| --- | --- | --- |
| `internal/health/gpu_memory.go` | nvidia-smi subprocess + parser | Subprocess spawn, CSV parsing from untrusted-ish source |
| `internal/health/signal_collector.go` | Memory-signal fallback, table selector | Context timeout handling, window-based branch |
| `internal/store/store.go` | `event_aggregates_5s` schema, dual-write, 10-min retention sweep | New table, new prepared-statement path, retention DELETE |
| `internal/cli/trace.go` | Dual-granularity aggregation maps | In-memory growth, 5s flush cadence |
| `internal/cli/fleet_push.go` | Poller TLS wiring | `LoadTLSConfig` share between emitter + poller |
| `internal/health/loop.go` | Straggler transition log | Low-volume Info log on classification edge |
| `internal/health/emitter.go` | `LoadTLSConfig` exported | No behavior change (rename only) |
| `cmd/straggler-sink/main.go` | UDS reader + HTTP metrics | Subprocess input parser, HTTP listener, metrics cardinality |
| `ingero-fleet/extension/ingerothresholdextension/config.go` + `extension.go` | HTTPS + mTLS for threshold API | TLS cert parsing, client cert verification |
| `deploy/helm/ingero/templates/daemonset.yaml` | Sidecar conditional + shared `/tmp` emptyDir | Helm-rendered Pod spec, privilege boundaries between containers |
| `helm/ingero-fleet/templates/configmap.yaml` | `middlewares:` directive | Operator visibility into what the collector actually runs |

### Defenses already in place

- **nvidia-smi parsing (`gpu_memory.go`):** parseNvidiaSMI rejects
  malformed columns, non-integer values, negative `used`, and
  non-positive `total`; empty output errors out. Subprocess runs with
  a 2 s timeout. Because the command + args are hardcoded and
  `exec.LookPath` discovers the binary at construction time, there is
  no shell-injection path and no attacker-controlled argv.
- **Straggler-sink parser (`cmd/straggler-sink/main.go`):**
  `bufio.Scanner` with a 64 KB max line length; malformed lines
  increment a counter and are logged but never crash the process.
  The `eventHeader` struct unmarshal only reads three fields, so
  oversized JSON objects (with other huge fields) still parse
  cheaply.
- **5s aggregate retention sweep (`store.go:RecordAggregates5s`):**
  single `DELETE FROM event_aggregates_5s WHERE bucket < ?` with a
  prepared-placeholder parameter; no SQL injection surface.
- **Extension HTTPS server (`extension.go`):** uses stdlib
  `crypto/tls` defaults, `MinVersion=TLS12`, `x509.NewCertPool`
  handles CA parsing, `tls.LoadX509KeyPair` validates the server
  keypair at load time. When `client_ca_file` is set,
  `ClientAuth=RequireAndVerifyClientCert` so any non-conforming
  client handshake is rejected by crypto/tls before any handler
  runs. Covered by `tls_test.go`.
- **Poller TLS wiring (`fleet_push.go`):** uses the same
  `LoadTLSConfig` path that the emitter does. No new file read
  patterns.

### Known residual risks (need hands-on probes)

Each of these deserves 30-90 minutes in an actual attack session
before the v1.0 tag. I scope them below; a security-focused run
should execute them and fold the results into this doc.

#### R1. nvidia-smi timeout race (gpu_memory.go)

A malicious `nvidia-smi` (PATH hijack on a compromised node) could
return a response that's valid-shaped but hostile — e.g., output
40 MiB of valid `used,total` rows to force the parser to sum an
attacker-controlled huge count. `bufio.Scanner` is bounded (64 KB
line) but `strings.Split` on the full output is not. If the output
is 2 GB of 8-byte lines, the Go process allocates the whole string
in RAM.

**Probe:** write a fake `nvidia-smi` that dumps 2 GB of valid CSV
to stdout; run with it first on `PATH`; measure the agent's RSS.

**Proposed fix if confirmed:** stream-parse via bufio.Scanner with
a hard line limit and a hard total-bytes limit (e.g., 64 KB total
is more than enough for 8 GPUs × 32-char row).

#### R2. Straggler-sink metric cardinality blow-up

`active_stragglers` is keyed by `{cluster_id, node_id}`. An
attacker that can feed crafted lines into the UDS (requires already
having code execution on the agent node, since the socket is chmod
0700) could enqueue millions of `straggler_state` lines with
distinct NodeIDs, growing the `stragglerState` map unboundedly.
The sink's memory would grow O(unique node IDs) until OOM.

**Probe:** feed 1M distinct-NodeID lines into the socket; measure
the sink's heap.

**Proposed fix if confirmed:** soft-cap the map (LRU or
`max_tracked_nodes=10000`) with a `dropped_total` counter.

Lower-severity than R1 because the UDS is owner-only — you need
code execution on the agent node before you can exploit this.

#### R3. TLS cert-rotation window bypass

`buildServerTLSConfig` loads cert + key once at extension start.
During cert rotation, a long-lived agent connection continues to
use the old handshake state for `KeepAlive` duration (Go default
~30 s). That's fine, but if the *CA* is compromised mid-run, there
is no mechanism to revoke trust until the extension restarts.

**Probe:** start Fleet with CA A, revoke A, issue B, swap the files
on disk, confirm no reload without restart.

**Proposed documentation:** already captured in
`ingero-fleet/docs/MTLS.md` (rotation section). This is a known
limitation, not a bug — but flag in the CHANGELOG as "cert rotation
requires restart".

#### R4. `RecordAggregates5s` retention sweep races

The opportunistic `DELETE` on every write could race with
`QueryAggregatePerOp5s` on the same connection pool. SQLite uses
serialized transactions with WAL mode, so this should just mean a
query may briefly see zero rows where it expected data, but worth
confirming with a stress test.

**Probe:** 8 goroutines hammering RecordAggregates5s + 8 hammering
QueryAggregatePerOp5s for 30 s. Confirm no data races, no panics,
no "database is locked" errors.

#### R5. Helm sidecar privilege escalation

The straggler-sink container shares `/tmp` via an emptyDir with the
privileged main container. The main container runs `ingero
fleet-push` with `securityContext.privileged: true` (inherited
from the existing chart, required for `/proc` eBPF access). The
sidecar does NOT need privileged — and should not have it.

**Probe:** render the chart with `sinkEnabled=true` and inspect the
sidecar's `securityContext`. Confirm it is NOT privileged,
does NOT mount `hostPath`s, and does NOT run as root.

**Status:** the current chart does not set a
`securityContext` block on the sidecar, so it inherits Pod-level
defaults. Pod has no Pod-level context, so the container runs as
whatever the image's USER is (unknown until the image is
published). Recommended: add `securityContext: {runAsNonRoot:
true, readOnlyRootFilesystem: true, capabilities: {drop: [ALL]}}`
to the sidecar before v1.0 tag.

#### R6. Straggler-state edge log leak

The new `straggler_state transition` log line (loop.go) includes
`score`, `threshold`, and `mode` as key-value pairs. These values
do not leak PII by themselves, but a curious tenant who can tail
the agent log on a shared node could infer when neighbor workloads
underperform. On single-tenant nodes this is fine; document as
expected behavior for multi-tenant.

**No probe needed; documentation is sufficient.**

## Re-run methodology

To formally close §4.6, run the 2026-04-16 methodology on v1.0:

1. Start at `internal/health/adv_*_test.go`; add v1.0-specific
   tests for R1, R2, R4.
2. Live-probe R1 with the fake-nvidia-smi approach.
3. Live-probe R2 with a tiny Go program that opens the UDS and
   blasts crafted lines.
4. Live-probe R3 with a cert-rotation script.
5. Fold findings into a new v1.0 report (same directory). Classify
   each as BLOCKER / MAJOR / MINOR / informational.
6. Ship fixes for any BLOCKER before tagging v1.0.

## Pre-flight verdict

No BLOCKER-class issue visible in static review. Two MAJOR
candidates (R1 nvidia-smi unbounded parse, R5 sidecar privilege
hardening) should be resolved before tag regardless of live-probe
results, since both are clear defensive improvements.

R2, R3, R4 should be confirmed hands-on; if any turn out non-trivial
to fix, each can either ship with a CHANGELOG note as a
v1.0 known limitation or be fixed in-release. Do not tag v1.0
without closing this loop.

## Fixes applied in this pass

- **R1 (nvidia-smi unbounded parse):** `parseNvidiaSMI` now rejects
  output larger than `maxNvidiaSMIOutput = 4 KiB` before any
  `strings.Split` allocation. Test:
  `TestGPUMem_RejectsOversizedOutput` in
  `internal/health/gpu_memory_test.go`.
- **R5 (sidecar privilege hardening):** the `straggler-sink`
  sidecar in `deploy/helm/ingero/templates/daemonset.yaml` now ships
  with `runAsNonRoot=true`, `runAsUser=65532`,
  `readOnlyRootFilesystem=true`, `allowPrivilegeEscalation=false`,
  `privileged=false`, `capabilities.drop=[ALL]`, and a RuntimeDefault
  seccomp profile. Verified by rendering the chart with
  `sinkEnabled=true` and inspecting the container spec.

R2 (sink cardinality), R3 (cert rotation window), and R4 (retention
sweep races) remain open for the hands-on session. Each is bounded
in blast radius and documented as known behavior in the CHANGELOG
until the session confirms or refutes exploitability.
