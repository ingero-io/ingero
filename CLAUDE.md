# Ingero — Agent & Contributor Context

## Project Overview

Ingero is a production-grade, eBPF-based GPU causal observability agent. It traces CUDA Runtime and Driver APIs via uprobes, plus host OS events (CPU scheduling, memory, process lifecycle) via kernel tracepoints, to build causal chains explaining GPU latency. Designed for zero-config, <2% overhead, always-on production use.

Single Go binary, 7 commands: `check`, `trace`, `demo`, `explain`, `query`, `mcp`, `version`.

## Build & Verify Workflow

Agents and contributors MUST verify changes before proposing them:

```bash
make build        # Generate eBPF bindings + compile Go binary → bin/ingero
make test         # Run unit tests. Do not bypass failing tests.
make lint         # Go staticcheck
make              # All of the above (generate + build + test + lint)
./bin/ingero demo --no-gpu   # Synthetic smoke test (no GPU or root needed)
```

eBPF compilation requires Linux (or WSL). Real GPU tracing requires `sudo` and a NVIDIA GPU with driver 550+.

## Directory Layout

```
cmd/ingero/         CLI entry point
internal/           Go packages (cli, ebpf/cuda, ebpf/driver, ebpf/host,
                    correlate, store, mcp, export, stats, sysinfo, symtab,
                    cgroup, k8s, discover, update, version, ...)
bpf/                eBPF C programs (kernel-space, GPL-2.0 OR BSD-3-Clause)
pkg/events/         Shared event types
scripts/            GPU VM lifecycle, setup, integration tests
tests/              Integration tests and GPU workloads
```

## Architectural Constraints (CRITICAL)

* **eBPF uprobes, not CUPTI.** We trace CUDA calls via standard Linux kernel uprobes on `libcudart.so` and `libcuda.so`. Evaluated alternative: NVIDIA's CUPTI (per-process injection, 5-30% overhead, single-subscriber limit — unsuitable for production). Kernel uprobes are the production-proven choice: zero dependencies, <2% overhead, works on any 5.15+ kernel.
* **eBPF verifier.** Code in `bpf/` executes in the Linux kernel. It must pass the strict eBPF verifier: no unbounded loops, explicit memory bound checks, no sleeping.
* **Fully open-source, split license.** Ingero is 100% FOSS, following the same dual-license pattern used by Cilium, Falco, and most eBPF projects (GPL required by the kernel's BPF subsystem).
    * `bpf/` (kernel-space): `GPL-2.0 OR BSD-3-Clause` — every C file MUST have `// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause` on line 1.
    * Everything else (Go, docs): Apache 2.0. Do not cross-contaminate.
* **No CGO.** SQLite uses `modernc.org/sqlite` (pure Go). The binary must stay statically linkable.
* **eBPF structs mirrored.** Event structs in `bpf/common.bpf.h` must stay in sync with `pkg/events/types.go`.

## Remote Sync Check (MUST DO FIRST)

Before taking **any** action in this project — reading code, editing files, running builds, answering questions about the codebase — first check the remote repository for new commits or open PRs:

```bash
git fetch origin --quiet
git log HEAD..origin/main --oneline   # new commits on main
gh pr list --state open --limit 5     # open PRs
```

If there are new remote commits or relevant open PRs, **recommend pulling latest changes** (`git pull`) before proceeding. This avoids working on stale code, prevents merge conflicts, and ensures answers reflect the current state of the project.

## Testing Rules (CRITICAL)

A bad test is worse than no test. Every test must justify its existence.

* **No flaky tests.** If a test can't pass 100 times in a row, it doesn't belong. No sleeps, no timing-dependent assertions, no order-dependent state. If you need time control, inject a clock.
* **Test behavior, not coverage.** Do not write tests for the sake of coverage numbers. Each test must assert something meaningful — a real invariant, an edge case that broke before, or a contract between components. "This function was called" is not a useful assertion.
* **Real implementations over mocks.** No mock frameworks (`gomock`, `testify/mock`, `mockgen`, etc.). Use real implementations with controlled inputs: binary builders for eBPF structs, in-memory SQLite for storage, `httptest.NewServer` for external HTTP boundaries. Mock only at the process boundary (network, filesystem), never internal interfaces.
* **Table-driven tests with `t.Run()`.** This is the established pattern — use `[]struct{...}` with named subtests for all non-trivial test functions.
* **Delete tests that test nothing.** If a test only checks that a function "doesn't panic" or returns `nil` error on the happy path with no other assertions, remove it. That's false confidence.
* **Maintain the test matrix.** `docs/test_matrix.md` is the canonical registry of all unit tests. When adding or removing tests, update this file with the test number, description, and file path. Keep it in sync with the codebase.

## Internal Contributors

Ingero team members: full project context (architecture, dev environment, GPU workflow, teaching mode) is in the `ingero-io/internal` repo. Symlink it for Claude Code:

```bash
mkdir -p .claude && ln -s ../../internal/CLAUDE.md .claude/CLAUDE.md
```

Related private repos: `ingero-io/ingero-ee` (enterprise extensions), `ingero-io/internal` (strategy, docs, full CLAUDE.md).

## Progressive Context

Do not guess how the system works. Read the relevant source files when needed. All commits require DCO sign-off (`git commit -s`) — see `CONTRIBUTING.md`.
