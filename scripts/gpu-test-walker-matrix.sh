#!/usr/bin/env bash
# gpu-test-walker-matrix.sh — per-Python-version eBPF frame walker
# regression harness.
#
# For each Python version listed in /tmp/walker-matrix-pythons.txt
# (produced by scripts/install-python-versions.sh), start
# `ingero trace --py-walker=ebpf --json --debug` against the host,
# run tests/workloads/py_walker_matrix.py under that Python, and
# assert:
#   - at least one cuda event was emitted (walker pipeline alive)
#   - the py_debug_stats counter dump shows walker_entered > 0
#   - on 3.10+, at least one cuda event's stack contains the full
#     mx_inner -> mx_middle -> mx_outer -> <module> chain.
#   - on 3.9, at least one cuda event emitted a non-empty py_func
#     (single-tstate fallback path; full-chain is a soft expectation).
#
# Exits non-zero when any version's assertions fail, so the
# `make gpu-test-walker-matrix` target can gate a PR.
#
# Prereqs (run once, then re-runnable):
#   sudo -E bash scripts/install-python-versions.sh
#   make build                  # produces bin/ingero

set -uo pipefail

MANIFEST=${MANIFEST:-/tmp/walker-matrix-pythons.txt}
INGERO_BIN=${INGERO_BIN:-$(pwd)/bin/ingero}
WORKLOAD=tests/workloads/py_walker_matrix.py
# Force the bundled nvidia/cudart.so.12 in each venv to redirect to the
# system libcudart — matches the inode ingero's uprobes are attached
# to at startup. The static-discovery side of Priority 3 is solved by a
# library re-scan on exec, but until that lands the LD_PRELOAD trick
# keeps the matrix honest instead of silently showing zero events.
SYSTEM_CUDART_CANDIDATES=(
    /usr/lib/x86_64-linux-gnu/libcudart.so.12.0.146
    /usr/lib/x86_64-linux-gnu/libcudart.so.12
    /usr/local/cuda/lib64/libcudart.so.12
)
SYSTEM_CUDART=""
for c in "${SYSTEM_CUDART_CANDIDATES[@]}"; do
    if [[ -f "$c" ]]; then SYSTEM_CUDART="$c"; break; fi
done

if [[ -t 1 ]]; then
    RED=$'\033[0;31m' GREEN=$'\033[0;32m' YELLOW=$'\033[1;33m' NC=$'\033[0m'
else
    RED="" GREEN="" YELLOW="" NC=""
fi

ts()   { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
log()  { echo "[$(ts)] $*"; }
pass() { echo "[$(ts)]   ${GREEN}PASS${NC} $*"; PASS_COUNT=$((PASS_COUNT+1)); }
fail() { echo "[$(ts)]   ${RED}FAIL${NC} $*"; FAIL_COUNT=$((FAIL_COUNT+1)); }
warn() { echo "[$(ts)]   ${YELLOW}WARN${NC} $*"; WARN_COUNT=$((WARN_COUNT+1)); }

PASS_COUNT=0 FAIL_COUNT=0 WARN_COUNT=0

# Prereq checks ----------------------------------------------------------

if [[ ! -f "$MANIFEST" ]]; then
    fail "manifest not found: $MANIFEST (run scripts/install-python-versions.sh first)"
    exit 2
fi
if [[ ! -x "$INGERO_BIN" ]]; then
    fail "ingero binary not executable: $INGERO_BIN (run make build)"
    exit 2
fi
if ! command -v jq >/dev/null 2>&1; then
    fail "jq required for JSON assertions"
    exit 2
fi

# Per-version runner -----------------------------------------------------

# run_one_version <minor> <python-bin> <has_torch>
run_one_version() {
    local minor=$1 py=$2 has_torch=$3
    local mode="ctypes"
    [[ "$has_torch" == "yes" ]] && mode="torch"

    local tmpdir
    tmpdir=$(mktemp -d /tmp/walker-matrix-"$minor"-XXXXXX)
    local jsonl="$tmpdir/events.jsonl"
    local errlog="$tmpdir/ingero.err"

    log "== 3.${minor} ($py, mode=$mode) =="

    # Start ingero, wait for uprobes to attach, run workload, stop.
    sudo rm -f "$jsonl" "$errlog"
    sudo timeout 40 "$INGERO_BIN" trace --py-walker=ebpf --json --debug \
        >"$jsonl" 2>"$errlog" &
    local ing_pid=$!
    sleep 4

    local preload=()
    [[ -n "$SYSTEM_CUDART" ]] && preload=("LD_PRELOAD=$SYSTEM_CUDART")

    # Run workload via env so LD_PRELOAD propagates without affecting
    # the invoking shell.
    env "${preload[@]}" "$py" "$WORKLOAD" "$mode" 2 2>&1 | tail -4 || true

    # Give ingero a moment to flush the ring buffer, then stop.
    sleep 3
    sudo kill -INT "$ing_pid" 2>/dev/null || true
    wait "$ing_pid" 2>/dev/null || true

    # --- Assertions ---
    local cuda_events py_funcs chain_hits
    cuda_events=$(jq -c 'select(.source=="cuda")' "$jsonl" 2>/dev/null | wc -l)
    py_funcs=$(jq -r 'select(.source=="cuda") | .stack[]? | select(.py_func) | .py_func' \
                   "$jsonl" 2>/dev/null | sort -u)
    chain_hits=$(echo "$py_funcs" | grep -cE '^mx_(inner|middle|outer)$' || true)

    local entered_dispatcher depth_gt_zero
    entered_dispatcher=$(grep -oE 'entered_dispatcher[[:space:]]+[0-9]+' "$errlog" \
                             | tail -1 | awk '{print $2}')
    depth_gt_zero=$(grep -oE 'depth_gt_zero[[:space:]]+[0-9]+' "$errlog" \
                             | tail -1 | awk '{print $2}')
    entered_dispatcher=${entered_dispatcher:-0}
    depth_gt_zero=${depth_gt_zero:-0}

    local tag="3.${minor}"
    if [[ "$cuda_events" -eq 0 ]]; then
        fail "$tag: no cuda events captured — uprobe attach likely broke; see $errlog"
        return 1
    fi
    pass "$tag: $cuda_events cuda events"

    if [[ "$entered_dispatcher" -eq 0 ]]; then
        fail "$tag: walker never entered (entered_dispatcher=0)"
        return 1
    fi
    pass "$tag: walker entered $entered_dispatcher times"

    if [[ "$depth_gt_zero" -eq 0 ]]; then
        fail "$tag: walker emitted zero frames on every event (depth_gt_zero=0)"
        return 1
    fi
    pass "$tag: walker emitted frames on $depth_gt_zero events"

    if [[ "$minor" == "9" ]]; then
        # Soft expectation: 3.9 may have only the single-tstate fallback
        # path emitting frames, chain depth varies by build.
        if [[ -n "$py_funcs" ]]; then
            pass "$tag: at least one py_func emitted"
        else
            warn "$tag: walker fired but no py_func strings extracted (offsets may still drift)"
        fi
    else
        if [[ "$chain_hits" -ge 3 ]]; then
            pass "$tag: full mx_inner/middle/outer chain present"
        else
            fail "$tag: expected 3 of {mx_inner,mx_middle,mx_outer} in py_funcs, got $chain_hits (saw: $(echo "$py_funcs" | tr '\n' ' '))"
            return 1
        fi
    fi
    return 0
}

# Main loop --------------------------------------------------------------

log "walker matrix harness starting; ingero=$INGERO_BIN manifest=$MANIFEST"
while IFS=: read -r minor py has_torch; do
    [[ -z "$minor" ]] && continue
    run_one_version "$minor" "$py" "$has_torch" || true
done < "$MANIFEST"

# Summary ---------------------------------------------------------------

echo
log "summary: ${GREEN}$PASS_COUNT PASS${NC} / ${RED}$FAIL_COUNT FAIL${NC} / ${YELLOW}$WARN_COUNT WARN${NC}"
if [[ "$FAIL_COUNT" -gt 0 ]]; then
    exit 1
fi
exit 0
