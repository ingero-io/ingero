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
                    correlate, store, mcp, export, stats, sysinfo, symtab, ...)
bpf/                eBPF C programs (kernel-space, GPL-2.0 OR BSD-3-Clause)
pkg/events/         Shared event types
scripts/            GPU VM lifecycle, setup, integration tests
tests/              Integration tests and GPU workloads
```

## Architectural Constraints (CRITICAL)

* **eBPF uprobes, not CUPTI or bpftime.** We trace CUDA calls via standard Linux kernel uprobes on `libcudart.so` and `libcuda.so`. Evaluated alternatives: NVIDIA's CUPTI (per-process injection, 5-30% overhead, single-subscriber limit — unsuitable for production) and eunomia-bpf's bpftime userspace eBPF runtime (promising lower overhead but immature, limited verifier support, extra deployment dependency). Kernel uprobes are the production-proven choice: zero dependencies, <2% overhead, works on any 5.15+ kernel.
* **eBPF verifier.** Code in `bpf/` executes in the Linux kernel. It must pass the strict eBPF verifier: no unbounded loops, explicit memory bound checks, no sleeping.
* **Fully open-source, split license.** Ingero is 100% FOSS, following the same dual-license pattern used by Cilium, Falco, and most eBPF projects (GPL required by the kernel's BPF subsystem).
    * `bpf/` (kernel-space): `GPL-2.0 OR BSD-3-Clause` — every C file MUST have `// SPDX-License-Identifier: GPL-2.0 OR BSD-3-Clause` on line 1.
    * Everything else (Go, docs): Apache 2.0. Do not cross-contaminate.
* **No CGO.** SQLite uses `modernc.org/sqlite` (pure Go). The binary must stay statically linkable.
* **eBPF structs mirrored.** Event structs in `bpf/common.bpf.h` must stay in sync with `pkg/events/types.go`.

## Progressive Context

Do not guess how the system works. Read the relevant source files when needed. All commits require DCO sign-off (`git commit -s`) — see `CONTRIBUTING.md`.
