#!/usr/bin/env bash
# quickstart-readme.sh
#
# Regression test for the README's "Try it in 60 seconds" path and
# the headline product surface advertised in README.md and the
# `docs/` getting-started pages. Runs CPU-only and rootless so the
# default GitHub Actions runner can execute it on every push.
#
# What this catches that the unit suite does not:
#   - the agent binary's CLI surface drifting away from the README
#     (renamed flags, removed subcommands, broken --version line)
#   - `ingero demo --no-gpu` regressing for the documented scenarios
#     (incident, cold-start). This is the explicit "no-GPU? try this"
#     entry point in README.md.
#   - `ingero mcp` startup banner / tool list drifting away from
#     the documentation. The README sells "AI-queryable in one
#     command"; if the MCP help text loses its tool/prompt section,
#     downstream MCP clients break and we want CI to catch it.
#
# What this does NOT cover (intentionally):
#   - `sudo ingero trace`: needs CAP_BPF + a kernel BPF subsystem;
#     unsafe to require on default CI runners. Covered by the
#     manually-triggered Lambda smoke (lambda-smoke.yml).
#   - The README's curl-tarball install flow: that needs a
#     published release; covered by tests/e2e/install-from-release.sh
#     post-tag.
#
# Runtime: ~15-30s on a cold runner. No external dependencies.
set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../.." && pwd)}"
INGERO_BIN="${INGERO_BIN:-$REPO_ROOT/bin/ingero}"
WORK=$(mktemp -d)
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

echo "==> [0/6] locate or build the agent binary"
if [[ ! -x "$INGERO_BIN" ]]; then
  echo "    (not at $INGERO_BIN; building)"
  ( cd "$REPO_ROOT" && go build -o "$INGERO_BIN" ./cmd/ingero ) 2>&1 | tail -3
fi
[[ -x "$INGERO_BIN" ]] || { echo "FAIL: agent binary missing"; exit 1; }
echo "OK: $INGERO_BIN"

# ---------------------------------------------------------------
# Test 1: --help advertises every README-promised subcommand.
# ---------------------------------------------------------------
echo "==> [1/6] --help lists every documented subcommand"
HELP=$("$INGERO_BIN" --help 2>&1)
for cmd in trace demo explain check mcp merge migrate dashboard; do
  if ! echo "$HELP" | grep -qE "^\s+$cmd\b"; then
    echo "FAIL: --help missing the '$cmd' subcommand (drifted from README?)"
    echo "$HELP"
    exit 1
  fi
done
echo "OK: trace, demo, explain, check, mcp, merge, migrate, dashboard all listed"

# ---------------------------------------------------------------
# Test 2: --version produces a parseable line. On a release build
# the version comes from -ldflags; on dev / source builds it falls
# back to "dev". Both are acceptable; what we want to catch is the
# command being removed entirely.
# ---------------------------------------------------------------
echo "==> [2/6] version subcommand prints something parseable"
VERSION_LINE=$("$INGERO_BIN" version 2>&1 | head -1)
if ! echo "$VERSION_LINE" | grep -qE "^ingero "; then
  echo "FAIL: 'ingero version' output unexpected: $VERSION_LINE"
  exit 1
fi
echo "OK: $VERSION_LINE"

# ---------------------------------------------------------------
# Test 3: ingero check runs without root and reports something
# coherent. On a host without a GPU this exits cleanly with a
# warning surface; on a GPU host it reports green. Either is OK -
# we only fail when the binary crashes or refuses to start.
# ---------------------------------------------------------------
echo "==> [3/6] ingero check runs and produces diagnostic output"
CHECK_OUT=$("$INGERO_BIN" check --debug 2>&1 || true)
if [[ -z "$CHECK_OUT" ]]; then
  echo "FAIL: ingero check produced no output"
  exit 1
fi
# Basic shape: should mention at least one of the readiness pillars
if ! echo "$CHECK_OUT" | grep -qiE "kernel|cuda|gpu|bpf|libbpf|driver|ebpf"; then
  echo "FAIL: ingero check did not surface any readiness pillar"
  echo "--- output ---"
  echo "$CHECK_OUT" | head -40
  exit 1
fi
echo "OK: ingero check surfaced readiness pillars"

# ---------------------------------------------------------------
# Test 4: demo --no-gpu incident produces non-trivial JSON event
# stream. This is the README's "No GPU? try this" command and the
# canonical first-impression path for new users on laptops / CI
# runners. A regression here breaks the headline demo silently.
# ---------------------------------------------------------------
echo "==> [4/6] demo --no-gpu incident produces JSON events"
"$INGERO_BIN" demo --no-gpu incident --json --duration 3s --speed 5 \
  > "$WORK/demo-incident.jsonl" 2> "$WORK/demo-incident.err" || {
    echo "FAIL: demo --no-gpu incident exited non-zero"
    cat "$WORK/demo-incident.err"
    exit 1
  }
EVENT_COUNT=$(grep -c '^{' "$WORK/demo-incident.jsonl" || echo 0)
if (( EVENT_COUNT < 50 )); then
  echo "FAIL: demo --no-gpu incident produced only $EVENT_COUNT JSON events (expected >= 50)"
  head -3 "$WORK/demo-incident.jsonl"
  exit 1
fi
# Sanity: events should reference at least one CUDA op the README
# headline mentions (cudaStreamSync is the canonical incident-demo
# spike op).
if ! grep -q "cudaStreamSync\|cudaLaunchKernel\|cudaMalloc" "$WORK/demo-incident.jsonl"; then
  echo "FAIL: demo --no-gpu incident events do not contain any documented CUDA op"
  head -5 "$WORK/demo-incident.jsonl"
  exit 1
fi
echo "OK: demo --no-gpu incident produced $EVENT_COUNT events with documented CUDA ops"

# ---------------------------------------------------------------
# Test 5: demo --no-gpu cold-start (the second example scenario in
# the README's --help text) also works. Catches per-scenario regression.
# ---------------------------------------------------------------
echo "==> [5/6] demo --no-gpu cold-start works"
"$INGERO_BIN" demo --no-gpu cold-start --json --duration 3s --speed 5 \
  > "$WORK/demo-coldstart.jsonl" 2> "$WORK/demo-coldstart.err" || {
    echo "FAIL: demo --no-gpu cold-start exited non-zero"
    cat "$WORK/demo-coldstart.err"
    exit 1
  }
COLD_COUNT=$(grep -c '^{' "$WORK/demo-coldstart.jsonl" || echo 0)
if (( COLD_COUNT < 5 )); then
  echo "FAIL: demo --no-gpu cold-start produced only $COLD_COUNT events"
  exit 1
fi
echo "OK: demo --no-gpu cold-start produced $COLD_COUNT events"

# ---------------------------------------------------------------
# Test 6: ingero mcp --help advertises the tool + prompt contract
# documented in README.md ("AI-queryable in one command"). The MCP
# tool descriptions are part of the public product surface; when
# they drift the README's headline claim becomes false.
# ---------------------------------------------------------------
echo "==> [6/6] mcp --help advertises tools + prompts"
MCP_HELP=$("$INGERO_BIN" mcp --help 2>&1)
for tool in get_trace_stats get_causal_chains run_sql get_stacks; do
  if ! echo "$MCP_HELP" | grep -qE "$tool"; then
    echo "FAIL: mcp --help missing tool '$tool'"
    echo "$MCP_HELP" | head -30
    exit 1
  fi
done
if ! echo "$MCP_HELP" | grep -qE "investigate"; then
  echo "FAIL: mcp --help missing /investigate prompt"
  exit 1
fi
echo "OK: mcp --help advertises all 4 spot-checked tools + investigate prompt"

echo
echo "PASS: README quickstart commands all behave as documented"
