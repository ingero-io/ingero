# End-to-end test harness

Shell-based end-to-end tests that exercise the agent binary
against the public surface (CLI, Prometheus exposition, OTLP
emission, MCP) and the agent + fleet + echo data plane.

## What runs where

CPU-only and rootless tests run on every CI push:

- `quickstart-readme.sh` exercises the README's "Try it in 60
  seconds" path: subcommand inventory, `version`, `check`,
  `demo --no-gpu` for the documented scenarios, and `mcp --help`
  tool/prompt advertisement.

Local-stack tests run on any host with Docker (no GPU needed):

- `data-plane/34-fan-in-completeness.sh` brings up the
  fleet+echo+sim-agent stack at
  `ingero-fleet/examples/local-stack/`, asserts events fan in
  through fleet to echo's DuckDB.
- `data-plane/33-investigate-finds-everything.sh` drives the
  federated `/investigate` MCP prompt against that stack with a
  real LLM. Skips gracefully when `ANTHROPIC_API_KEY` is unset.

GPU-bound tests run via the `lambda-smoke.yml` workflow_dispatch
on a Lambda VM. The shipped scripts cover trace flags, OTLP/
Prometheus roundtrip, libnccl discovery, memcpy direction matrix,
memfrag emission, throttle behavior, NCCL ABI matrix, soak,
arm64 runtime, and data-plane completeness.

## Running a single script locally

```bash
# Build the agent binary first
go build -o bin/ingero ./cmd/ingero

# Most scripts take INGERO_BIN + REPO_ROOT
INGERO_BIN="$PWD/bin/ingero" REPO_ROOT="$PWD" \
  bash tests/e2e/<script>.sh

# GPU scripts need sudo for eBPF + nvidia-smi
sudo -E bash tests/e2e/<script>.sh
```

## Helpers

`_lib.sh` provides `wait_port_ready <host> <port> [timeout]` so
agent-boot races on slow runners do not flake the assertions.
Source from any script:

```bash
. "$(dirname "$0")/_lib.sh"
wait_port_ready 127.0.0.1 9090 30 || exit 1
```

## Adding a new script

- Source `_lib.sh` if you need port-readiness waits.
- Use `set -euo pipefail` and a `cleanup()` trap that kills the
  agent and cleans `/tmp` artifacts.
- Prefer `awk 'NR==1'` over `\| head -1` (head closes the pipe
  early; pipefail then propagates SIGPIPE from the producer).
- Skip with exit 0 when prerequisites (GPU, sudo, an API key)
  are missing instead of failing.
- Document expected runtime in the file header.
