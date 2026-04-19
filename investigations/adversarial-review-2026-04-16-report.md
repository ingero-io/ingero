# Ingero Adversarial Review - 2026-04-16

> Scope: Autonomous adversarial testing of the `fleet` branch HEAD
> (commit edd76091 + security-sanitizer commit on top). Attack model:
> (a) co-tenant malicious process on a traced GPU node, (b) operator
> running `ingero merge` on an untrusted source DB, (c) unauthenticated
> attacker on the same L2 network as the dashboard.
>
> Environment: WSL2 Ubuntu 22.04, RTX 5080 Laptop, Go 1.26, CUDA 13.0.
>
> Session budget: ~5 hours. Waves 1-11 executed.

## Executive Summary

Two BLOCKERs, four MAJORs, three MINORs, plus a pre-existing
functional bug surfaced during security probing. Everything below
is reproducible; test code lives in `internal/*/adv_*_test.go` and
`/home/ubuntu/adv-test/` in the WSL Ubuntu 22.04 distro.

## BLOCKERS

### B1. SQLite memory bomb via `zeroblob(1GB)` (reachable from MCP + Dashboard)

**Surface:** `internal/mcp/server.go` `run_sql` tool and
`internal/dashboard` `/api/v1/query` HTTP endpoint.

**Attack:** `SELECT zeroblob(1073741824)` — SQLite allocates 1 GB in
a single expression. The ExecuteReadOnly validator allows SELECT
statements; no AST-level allowlist for functions. Measured: 2.5 GB
RSS in 5s per call, reliable OOM at ~10 concurrent calls.

Dashboard path needs no auth at all (see B2); MCP path needs stdio
access to the server. Dashboard path is therefore the higher-impact
variant.

**Fix:** reject `zeroblob`, `randomblob`, `hex(zeroblob(*))`,
`printf('%.*c', ...)`, recursive CTEs without LIMIT, and any query
whose EXPLAIN reports an estimated cost above a budget. Cheapest
short-term patch: regex-reject `zeroblob|randomblob` in the
ExecuteReadOnly validator and cap result row count at 10K.

### B2. LocalBaseline threshold unit mismatch — live-confirmed

**Surface:** `internal/health/emitter.go` local-baseline mode.

**Attack:** No attacker needed — this is a production bug that
makes the straggler detector unusable in local-baseline mode.
Observed at runtime: threshold=21.82 applied against scores in
[0, 1]. Every score is a straggler. With
`--fleet-remediate=affinity-pin`, the orchestrator pins the CPU
affinity of every traced process on every node simultaneously.

Root cause documented in Epic 3 Group 2 code review D1. Formula:
`threshold = mean(baseline.Throughput) * 0.85` where
`baseline.Throughput` is absolute tokens/sec (raw magnitude), but
`score` is a ratio in [0, 1]. The multiplication loses units.

**Fix:** patch landed in the fleet branch as part of Epic 3 review
cycle but needs to be verified in production. Recommend a CI
integration test that boots a real agent and asserts
`threshold < 1.0` in local-baseline mode.

## MAJORs

### M1. Dashboard HTTP endpoints have zero authentication

**Surface:** `/api/v1/query`, `/api/v1/chains`, `/api/v1/stats`,
`/api/v1/health`, `/api/v1/remediate-events`.

Dashboard defaults to `:0` (ephemeral) on localhost, but
`--dashboard-addr 0.0.0.0:9000` is a documented configuration. In
that mode:
  - Anyone on the L2 network can query the DB (B1 memory bomb)
  - Anyone can read the full event history (process names, kernel
    names, file paths, stack traces)
  - Self-signed cert is regenerated on every restart, so cert
    pinning is impossible

**Fix:** require a shared token (Bearer, set via env var) or mTLS
by default. Emit WARN if listening on non-loopback without auth.

### M2. UDS remediate socket defaults to mode 0755

**Surface:** `internal/health/remediate_uds.go` (or wherever the
remediation event stream UDS lives).

Any local user on the node can `connect()` to the socket and hijack
the stream — reading straggler events (process names, PIDs, rank
assignments). In a multi-tenant GPU node this leaks workload metadata.

**Fix:** chmod 0700 on the socket (per-user) or 0660 + set group
ownership to an `ingero` group whose membership is operator-controlled.
There is no legitimate consumer outside the ingero daemon itself.

### M3. Sanitizer LooksLikeInjection bypass (BiDi + zero-width)

**Surface:** `internal/mcp/sanitize.go:LooksLikeInjection`.

The canary audit check is pattern-based. Inserting U+202E (RIGHT-TO-
LEFT OVERRIDE) or ZWSP (U+200B) between characters ("S\u200BYSTEM:")
bypasses all markers. Verified with 18 variants in a standalone
test.

This is a canary, not the primary defense — the `[traced-data]...
[/traced-data]` wrap is still intact. So this is MAJOR only because
audit logs will miss real attacks, not because injection succeeds.

**Fix:** NFKC-normalize before pattern matching, OR strip zero-width
and RTL-override codepoints in `stripControlChars`. The comment
explicitly says "non-ASCII bytes are left intact so legitimate non-
English symbols survive" — need to except the handful of problematic
bidi/invisible codepoints.

### M4. `get_stacks` JSON frames have no input size limit

**Surface:** `internal/mcp/server.go:905`.

`json.Unmarshal([]byte(framesJSON), &rawFrames []string)` is
preceded by no `len(framesJSON)` check. Test
`TestAdv_FramesJSONBomb_TenMillionStrings` allocates 615 MB heap
and takes 1.9s on a single row. At `LIMIT=100` (the tool's max),
a crafted DB causes ~60 GB transient allocation.

Delivery: `ingero merge attacker.db victim.db -o merged.db` copies
`stack_traces.frames` verbatim with no validation
(`internal/cli/merge.go:444 stmt.Exec(hash, ips, frames)`). A
victim operator who accepts "here's a dump from my node for
debugging" imports the bomb.

**Fix:** pre-check `len(framesJSON) > 64 * 1024` and skip the row
with a WARN. A real stack is at most 127 frames * ~100 bytes =
12 KB. Same fix on the `ingero merge` side — validate frames JSON
shape before INSERT.

## MINORs

### m1. CRLF not validated on ClusterID / NodeID / WorkloadType

`emitter.Validate` rejects CR/LF in the Headers map but not in the
top-level identity fields. These propagate to OTLP/gRPC metadata
where CRLF would be trusted. Low impact (operator-set values, not
attacker-controlled in practice) but the defensive check is cheap.

### m2. Quarantine race produces noisy errors under concurrent Load()

8 goroutines racing to quarantine the same corrupt baseline file
produced 29/400 (~7%) errors in `TestAdv_QuarantineRace`. Live
baseline integrity is preserved. Only operator confusion. The
code comment acknowledges this as "acceptable trade-off".

### m3. `ingero explain --json` output is not sanitized

`ingero explain` prints process / kernel / op names verbatim. If a
user pipes the JSON to an LLM for triage (a plausible workflow),
the [traced-data] wrap is absent and prompt injection is possible.
MCP was the designed attack surface, but `explain --json` is a
natural extension.

## Pre-existing functional bug surfaced during probing

`get_stacks` expects `stack_traces.frames` as a JSON array of
strings (`[]string`), but the production serializer
(`store/store.go:serializeStackFrames`) writes objects
(`[{"s":"cudaMalloc","f":"libcudart.so"}]`). Therefore
`json.Unmarshal` fails for every real production row, and
`get_stacks` always returns empty `frames` arrays. Not a security
issue — a latent correctness bug that needs a separate ticket.

## Defenses that held under attack

For the record, the following were probed and found solid:

- `SanitizeTelemetry` primary defense: the `[traced-data]` wrap
  successfully frames all injection attempts (SYSTEM:, Ignore
  Previous, admin overrides, tool-call templates, etc.) as data,
  not instructions. Embedded-delimiter escape attempts stripped.
- UTF-8-safe truncation at `MaxFrameLen=1024` and `MaxNameLen=256`.
- Go's JSON parser rejects 100K-deep nested arrays with
  "exceeded max depth".
- `health.persist.Save` atomic rename: 4 writers × 4 readers for
  2 seconds → 1373 writes, 584,564 reads, ZERO torn reads. Atomic
  rename guarantee holds under heavy contention.
- `os.Rename` on symlink renames the link, not the target. Symlink-
  to-secret-file attack does not modify or leak the target.
- `chmod 000` baseline file → `LoadUnreadable`, NOT `LoadCorrupt`.
  Non-destructive on transient FS errors.
- ProcMem `ReadPyUnicodeString` bounds string length to 4 KiB
  regardless of attacker-controlled PyUnicodeObject.length field.
- `findThreadState` / `walkFrames` cycle bounds (256 / 64) are
  hit and enforced in static review.
- `bufio.Scanner` on `/proc/PID/maps` + PATH_MAX=4096 means no
  line-too-long DOS.
- All float inputs to `minOf4` are `clamp01`'d first, so NaN
  never propagates into the score pipeline.
- `internal/symtab/ParseELFSymbols` robust against 10 malformed
  inputs (empty, random, truncated, huge-shnum-claim, sparse
  5MB, /dev/null, /dev/zero, nonexistent, directory,
  /proc/cmdline).

## Reproduction artifacts

- `internal/health/adv_toctou_test.go` — 7 TOCTOU and race tests
- `internal/health/adv_persist_test.go` — 12 malformed-JSON tests
- `internal/health/adv_state_test.go` — NaN / Inf / boundary inputs
- `internal/mcp/adv_jsonbomb_test.go` — 5 JSON bomb variants
- `internal/symtab/adversarial_test.go` — ELF malformed input sweep
- `internal/store/adversarial_test.go` — SQL injection / memory bomb
- `/home/ubuntu/adv-test/FINDINGS.md` — raw per-wave log

## Recommendation

1. **Do not ship** until B1 (zeroblob) and B2 (local-baseline units)
   are patched. B2 is ingestion-grade bad in local mode.
2. **Ship hardening** for M1-M4 in the same release cycle. None
   requires architectural change.
3. **Re-run this adversarial pass** after fixes land; focus on the
   same attack vectors plus Python frame walker fuzzing in a live
   container with a crafted CPython process.
