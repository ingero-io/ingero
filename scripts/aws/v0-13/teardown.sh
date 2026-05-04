#!/usr/bin/env bash
###############################################################################
# v0.13 Slice F teardown: terminate by RunID tag, idempotent.
#
# Two input modes:
#   1. Env: TAG_RUN_ID=v0-13-1234567890 bash teardown.sh
#   2. Flag: bash teardown.sh --state-file path/to/state.json
#
# Invariants:
#   - Account guard runs first. We will not terminate anything in a non-allowed
#     account, even if a tag query somehow returns instances there.
#   - Security group deletion happens AFTER instances reach `terminated`.
#     EC2 refuses SG delete while instances reference it.
#   - Re-running on a clean RunID exits 0 ("no resources to clean up").
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

AWS_PROFILE_NAME="${AWS_PROFILE_NAME:-ingero}"
REGION="${REGION:-us-east-1}"
ALLOWED_ACCOUNT="873543243296"
STATE_FILE=""
STATE_DIR="${STATE_DIR:-$SCRIPT_DIR/../../../../_bmad-output}"

stderr_log() {
    local level="$1"; shift
    printf '{"phase":"teardown","level":"%s","msg":%s}\n' \
        "$level" \
        "$(printf '%s' "$*" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')" 1>&2
}

abort() {
    stderr_log "fatal" "$1"
    exit 1
}

###############################################################################
# Parse flags. --state-file overrides $TAG_RUN_ID; if both are given, the
# state-file wins because it is the more specific source of truth.
###############################################################################
while [[ $# -gt 0 ]]; do
    case "$1" in
        --state-file)
            STATE_FILE="$2"
            shift 2
            ;;
        --run-id)
            TAG_RUN_ID="$2"
            shift 2
            ;;
        *)
            abort "unknown flag: $1"
            ;;
    esac
done

if [[ -n "$STATE_FILE" ]]; then
    if [[ ! -f "$STATE_FILE" ]]; then
        # Treat missing state file as "already torn down". Idempotency.
        echo "teardown: state file $STATE_FILE missing, nothing to clean up"
        exit 0
    fi
    TAG_RUN_ID=$(python3 -c "import json,sys; print(json.load(open('$STATE_FILE'))['run_id'])")
fi

if [[ -z "${TAG_RUN_ID:-}" ]]; then
    abort "no RunID supplied (set TAG_RUN_ID env or pass --state-file or --run-id)"
fi

aws_cli() {
    aws --profile "$AWS_PROFILE_NAME" --region "$REGION" "$@"
}

###############################################################################
# Account guard. Same hard rule as preflight: only $ALLOWED_ACCOUNT is OK.
###############################################################################
account=$(aws_cli sts get-caller-identity --query Account --output text 2>&1) || \
    abort "sts get-caller-identity failed: $account"
if [[ "$account" != "$ALLOWED_ACCOUNT" ]]; then
    abort "active account is $account, expected $ALLOWED_ACCOUNT"
fi

###############################################################################
# Find instances by tag. We include all states except already-terminated
# because those have nothing left to terminate.
###############################################################################
instance_ids=$(aws_cli ec2 describe-instances \
    --filters "Name=tag:RunID,Values=$TAG_RUN_ID" \
              "Name=tag:Project,Values=ingero" \
              "Name=instance-state-name,Values=pending,running,stopping,stopped,shutting-down" \
    --query 'Reservations[].Instances[].InstanceId' \
    --output text 2>/dev/null || echo "")

count=0
if [[ -n "$instance_ids" ]]; then
    # shellcheck disable=SC2206 # word splitting is the goal here
    ids_array=($instance_ids)
    count=${#ids_array[@]}
    stderr_log "info" "terminating $count instance(s): ${ids_array[*]}"
    aws_cli ec2 terminate-instances --instance-ids "${ids_array[@]}" >/dev/null
    aws_cli ec2 wait instance-terminated --instance-ids "${ids_array[@]}"
fi

###############################################################################
# Delete the SG (only after instances are terminated; EC2 enforces that).
###############################################################################
sg_ids=$(aws_cli ec2 describe-security-groups \
    --filters "Name=tag:RunID,Values=$TAG_RUN_ID" "Name=tag:Project,Values=ingero" \
    --query 'SecurityGroups[].GroupId' --output text 2>/dev/null || echo "")

if [[ -n "$sg_ids" ]]; then
    # shellcheck disable=SC2206
    sg_array=($sg_ids)
    for sg in "${sg_array[@]}"; do
        # delete-security-group can fail transiently if dependency tracking
        # has not caught up; one short retry is enough in practice.
        if ! aws_cli ec2 delete-security-group --group-id "$sg" >/dev/null 2>&1; then
            sleep 5
            aws_cli ec2 delete-security-group --group-id "$sg" >/dev/null 2>&1 || \
                stderr_log "warn" "failed to delete security group $sg (manual cleanup needed)"
        fi
    done
fi

###############################################################################
# Delete the state file last, after AWS resources are gone. If a state file
# was passed in directly, use that path; otherwise assume the standard
# location based on RunID.
###############################################################################
if [[ -z "$STATE_FILE" ]]; then
    STATE_FILE="$STATE_DIR/v0-13-cluster-state-$TAG_RUN_ID.json"
fi
if [[ -f "$STATE_FILE" ]]; then
    rm -f "$STATE_FILE"
fi

if [[ "$count" -eq 0 ]]; then
    echo "teardown: no resources to clean up for $TAG_RUN_ID"
else
    echo "teardown complete: $count instance(s) terminated for $TAG_RUN_ID"
fi
