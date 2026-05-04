#!/usr/bin/env bash
###############################################################################
# v0.13 Slice F deploy: copy repos to each node and run the build.
#
# Inputs:
#   --state-file <path>    JSON from provision.sh (required)
#
# What it does, per node:
#   - capture/peer1/peer2 (g4dn): rsync ingero + ingero-fleet + ingero-ee;
#     run `make vmlinux && make generate && make build` for ingero;
#     run `cargo build --release` for the orchestrator.
#   - fleet (t3.medium): rsync ingero-fleet only; run `make generate && make build`.
#
# Build failures:
#   - capture node: STOP. Capture is critical path.
#   - fleet node:   STOP. Fleet is critical for Slices B, C threshold/traces.
#   - peer node:    WARN, continue. Peers are quorum-only and tolerate spot reclaim
#                   anyway, so a build failure is no worse than a reclaim.
#
# Parallelism: SSH calls run in the background and we wait at the end. SSH
# multiplexing is intentionally NOT enabled because some validation phases
# benefit from independent connections.
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

STATE_FILE=""
SSH_USER="${SSH_USER:-ubuntu}"
SSH_OPTS=(-o StrictHostKeyChecking=no -o BatchMode=yes -o ConnectTimeout=10)
# Repo root layout: ingero-ee-2/ingero, ingero-ee-2/ingero-fleet, ingero-ee-2/ingero-ee.
# scripts/aws/v0-13 -> ../../.. lands in ingero/, then one more ../ lands in
# ingero-ee-2/. We want the parent of ingero/.
REPO_ROOT="${REPO_ROOT:-$SCRIPT_DIR/../../../..}"

stderr_log() {
    local level="$1"; shift
    printf '{"phase":"deploy","level":"%s","msg":%s}\n' \
        "$level" \
        "$(printf '%s' "$*" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')" 1>&2
}

abort() {
    stderr_log "fatal" "$1"
    exit 1
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --state-file)
            STATE_FILE="$2"
            shift 2
            ;;
        *)
            abort "unknown flag: $1"
            ;;
    esac
done

if [[ -z "$STATE_FILE" || ! -f "$STATE_FILE" ]]; then
    abort "--state-file required and must exist (got: '$STATE_FILE')"
fi

# Pull IPs from the state file using python (same dep we already use for JSON
# emission in provision.sh).
get_ip() {
    local role="$1"
    python3 -c "import json; print(json.load(open('$STATE_FILE'))['instances']['$role']['public_ip'])"
}

CAPTURE_IP=$(get_ip capture)
PEER1_IP=$(get_ip peer1)
PEER2_IP=$(get_ip peer2)
FLEET_IP=$(get_ip fleet)

###############################################################################
# rsync the three repos to ~/repos/ on the target. We exclude .git/objects
# and target build outputs to keep the transfer small; the remote build will
# regenerate them.
###############################################################################
sync_repos() {
    local ip="$1"
    local include_orchestrator="$2"
    local include_fleet="$3"
    local include_agent="$4"

    ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" "mkdir -p ~/repos"

    if [[ "$include_agent" == "1" ]]; then
        rsync -az --delete \
            --exclude='.git/objects' --exclude='target/' --exclude='bin/' \
            -e "ssh ${SSH_OPTS[*]}" \
            "$REPO_ROOT/ingero/" "$SSH_USER@$ip:~/repos/ingero/"
    fi
    if [[ "$include_fleet" == "1" ]]; then
        rsync -az --delete \
            --exclude='.git/objects' --exclude='target/' --exclude='bin/' \
            -e "ssh ${SSH_OPTS[*]}" \
            "$REPO_ROOT/ingero-fleet/" "$SSH_USER@$ip:~/repos/ingero-fleet/"
    fi
    if [[ "$include_orchestrator" == "1" ]]; then
        rsync -az --delete \
            --exclude='.git/objects' --exclude='target/' \
            -e "ssh ${SSH_OPTS[*]}" \
            "$REPO_ROOT/ingero-ee/" "$SSH_USER@$ip:~/repos/ingero-ee/"
    fi
}

###############################################################################
# Build commands per role. Output goes to a per-node log so that a failure
# does not lose context behind the parallel `wait`.
###############################################################################
build_g4dn_node() {
    local ip="$1"
    local log="$2"
    {
        sync_repos "$ip" 1 1 1
        ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" \
            'set -e; cd ~/repos/ingero && make vmlinux && make generate && make build'
        ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" \
            'set -e; cd ~/repos/ingero-ee/orchestrator && cargo build --release'
    } >"$log" 2>&1
}

build_fleet_node() {
    local ip="$1"
    local log="$2"
    {
        sync_repos "$ip" 0 1 0
        ssh "${SSH_OPTS[@]}" "$SSH_USER@$ip" \
            'set -e; cd ~/repos/ingero-fleet && make generate && make build'
    } >"$log" 2>&1
}

###############################################################################
# Run all four builds in parallel. Each write to its own log file under
# /tmp; on failure we cat the log into stderr_log so the caller sees what
# broke.
###############################################################################
LOG_DIR="${LOG_DIR:-/tmp/ingero-v0-13-deploy-$$}"
mkdir -p "$LOG_DIR"

declare -A pids
declare -A logs

start() {
    local role="$1" ip="$2" fn="$3"
    local log="$LOG_DIR/$role.log"
    logs[$role]=$log
    "$fn" "$ip" "$log" &
    pids[$role]=$!
    stderr_log "info" "deploy: $role pid=${pids[$role]} log=$log"
}

start capture "$CAPTURE_IP" build_g4dn_node
start peer1   "$PEER1_IP"   build_g4dn_node
start peer2   "$PEER2_IP"   build_g4dn_node
start fleet   "$FLEET_IP"   build_fleet_node

declare -A results
for role in capture peer1 peer2 fleet; do
    if wait "${pids[$role]}"; then
        results[$role]="ok"
    else
        results[$role]="fail"
    fi
done

###############################################################################
# Apply the failure policy.
###############################################################################
critical_failed=0
peer_failed=0
for role in capture fleet; do
    if [[ "${results[$role]}" == "fail" ]]; then
        critical_failed=1
        stderr_log "fatal" "$role build failed (critical) -- log: ${logs[$role]}"
        # Surface tail of the log so the operator does not have to SSH to read it.
        tail -n 60 "${logs[$role]}" 1>&2 || true
    fi
done
for role in peer1 peer2; do
    if [[ "${results[$role]}" == "fail" ]]; then
        peer_failed=1
        stderr_log "warn" "$role build failed (non-critical, continuing) -- log: ${logs[$role]}"
        tail -n 30 "${logs[$role]}" 1>&2 || true
    fi
done

if [[ "$critical_failed" -eq 1 ]]; then
    exit 1
fi

ok_count=0
for role in capture peer1 peer2 fleet; do
    [[ "${results[$role]}" == "ok" ]] && ok_count=$((ok_count + 1))
done

if [[ "$peer_failed" -eq 1 ]]; then
    echo "deploy partial: $ok_count/4 successful (peer build failed, continuing)"
else
    echo "deploy complete: $ok_count/4 successful"
fi
