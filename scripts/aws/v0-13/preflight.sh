#!/usr/bin/env bash
###############################################################################
# v0.13 Slice F preflight: account guard, quota check, AMI availability.
#
# Hard rule from ~/.claude/CLAUDE.md: AWS account 873543243296 is the ONLY
# permitted account on this machine. Every consequential AWS call asserts that
# the active identity resolves to that account, otherwise we abort.
#
# The script is sourceable: provision.sh sources this file so the same checks
# run before any run-instances call. When sourced, we avoid `exit` on success
# so the caller stays alive; only failures abort.
###############################################################################
set -euo pipefail

# Allow callers to override profile/region via env. Defaults match the rest of
# the v0.13 harness (us-east-1 is where ami-0aad28499825d76c3 lives).
AWS_PROFILE_NAME="${AWS_PROFILE_NAME:-ingero}"
REGION="${REGION:-us-east-1}"
ALLOWED_ACCOUNT="873543243296"
EXPECTED_AMI="${EXPECTED_AMI:-ami-0aad28499825d76c3}"
KEY_NAME="${INGERO_AWS_SSH_KEY_NAME:-ingero-key}"

# Cluster math: 1x g4dn.xlarge on-demand (4 vCPU) + 2x g4dn.xlarge spot (8 vCPU
# total). We want headroom of 1 instance worth (4 vCPU) on the on-demand side
# in case a peer needs to be re-provisioned; the spot side does not need
# headroom because spot reclaim does not reduce the quota counter.
ON_DEMAND_G_MIN_VCPU=12
SPOT_G_MIN_VCPU=8

# Service Quotas codes for EC2 G/VT vCPU pools. These are stable across
# regions; if AWS renames either, the API call below returns NoSuchResource
# and we surface the failure rather than guessing at a new code.
ON_DEMAND_QUOTA_CODE="L-DB2E81BA"
SPOT_QUOTA_CODE="L-3819A6DF"

# stderr_log: structured JSON-friendly line so the run-validation driver can
# parse failures programmatically without regex acrobatics.
stderr_log() {
    local level="$1"; shift
    printf '{"phase":"preflight","level":"%s","msg":%s}\n' "$level" "$(printf '%s' "$*" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')" 1>&2
}

# Detect whether we are sourced; affects whether we exit vs return.
# (BASH_SOURCE[0] != $0 means sourced.)
_sourced=0
if [[ "${BASH_SOURCE[0]}" != "${0}" ]]; then
    _sourced=1
fi

abort() {
    stderr_log "fatal" "$1"
    if [[ "$_sourced" -eq 1 ]]; then
        return 1
    fi
    exit 1
}

###############################################################################
# Step 1: account identity assertion. This is the hard-stop rule.
###############################################################################
check_account() {
    local account
    if ! account=$(aws sts get-caller-identity --profile "$AWS_PROFILE_NAME" --query Account --output text 2>&1); then
        abort "aws sts get-caller-identity failed: $account"
        return 1
    fi
    if [[ "$account" != "$ALLOWED_ACCOUNT" ]]; then
        abort "active account is $account, expected $ALLOWED_ACCOUNT"
        return 1
    fi
    echo "preflight: account=$account profile=$AWS_PROFILE_NAME region=$REGION"
    return 0
}

###############################################################################
# Step 2: quota check. We refuse to provision if either pool is below threshold;
# better to fail fast here than to have run-instances reject the third launch.
###############################################################################
check_quota() {
    local code="$1" min_value="$2" label="$3"
    local value
    if ! value=$(aws service-quotas get-service-quota \
            --service-code ec2 \
            --quota-code "$code" \
            --profile "$AWS_PROFILE_NAME" \
            --region "$REGION" \
            --query 'Quota.Value' \
            --output text 2>&1); then
        abort "service-quotas lookup failed for $code ($label): $value"
        return 1
    fi
    # Service Quotas returns a float ("4.0"); compare with python because
    # bash arithmetic does not handle decimals.
    local ok
    ok=$(python3 -c "import sys; v=float('$value'); m=float('$min_value'); print('1' if v >= m else '0')")
    if [[ "$ok" != "1" ]]; then
        abort "$label quota is $value, need >= $min_value (code $code)"
        return 1
    fi
    echo "preflight: $label quota=$value (>= $min_value)"
    return 0
}

###############################################################################
# Step 3: AMI availability. The validation plan pins ami-0aad28499825d76c3 in
# us-east-1; if AWS deprecated/de-registered it, we need to know now.
###############################################################################
check_ami() {
    local found
    if ! found=$(aws ec2 describe-images \
            --image-ids "$EXPECTED_AMI" \
            --profile "$AWS_PROFILE_NAME" \
            --region "$REGION" \
            --query 'Images[0].ImageId' \
            --output text 2>&1); then
        abort "describe-images failed for $EXPECTED_AMI: $found"
        return 1
    fi
    if [[ "$found" != "$EXPECTED_AMI" ]]; then
        abort "AMI $EXPECTED_AMI not found in region $REGION (got: $found)"
        return 1
    fi
    echo "preflight: ami=$EXPECTED_AMI region=$REGION"
    return 0
}

###############################################################################
# Step 4: SSH key. Soft check; the operator can override via flag or env.
# We do not abort if the key is missing because passing one in via
# SSH_KEY_NAME at provision time is a valid workflow.
###############################################################################
check_ssh_key() {
    local out
    if ! out=$(aws ec2 describe-key-pairs \
            --key-names "$KEY_NAME" \
            --profile "$AWS_PROFILE_NAME" \
            --region "$REGION" \
            --query 'KeyPairs[0].KeyName' \
            --output text 2>&1); then
        # Surface as warning, not fatal; the operator can pass a different
        # key via the SSH_KEY_NAME env var into provision.sh.
        stderr_log "warn" "ssh key '$KEY_NAME' not found (override via INGERO_AWS_SSH_KEY_NAME env)"
        echo "preflight: ssh-key=MISSING ($KEY_NAME)"
        return 0
    fi
    echo "preflight: ssh-key=$out"
    return 0
}

###############################################################################
# Run all checks in sequence. Sequential because step 1's success is a
# precondition for steps 2-4 even running.
###############################################################################
preflight_run() {
    check_account || return 1
    check_quota "$ON_DEMAND_QUOTA_CODE" "$ON_DEMAND_G_MIN_VCPU" "on-demand-G-VT-vCPU" || return 1
    check_quota "$SPOT_QUOTA_CODE" "$SPOT_G_MIN_VCPU" "spot-G-VT-vCPU" || return 1
    check_ami || return 1
    check_ssh_key || return 1
    echo "preflight OK"
    return 0
}

# Only run automatically when invoked as a script. When sourced, the caller
# decides when to call preflight_run.
if [[ "$_sourced" -eq 0 ]]; then
    preflight_run
fi
