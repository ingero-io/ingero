#!/usr/bin/env bash
###############################################################################
# v0.13 Slice F provision: launch the 4-node cluster.
#
# Cluster shape (from spec-v0-13-aws-validation-and-gifs.md):
#   - capture (Role=capture): g4dn.xlarge ON-DEMAND. Recorder lives here;
#                             must not be reclaimed mid-recording.
#   - peer1   (Role=peer1):   g4dn.xlarge SPOT. Cohort quorum; reclaim OK.
#   - peer2   (Role=peer2):   g4dn.xlarge SPOT. Cohort quorum; reclaim OK.
#   - fleet   (Role=fleet):   t3.medium ON-DEMAND. Collector; no GPU.
#
# Idempotency: tags everything with RunID. A re-run with the same RunID will
# detect existing resources via tag filter and reuse them rather than launch
# duplicates.
###############################################################################
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Source preflight so account + quota + AMI checks happen before any launch.
# The preflight script returns non-zero on failure rather than exit when
# sourced, which lets us surface a structured error from this caller.
# shellcheck source-path=SCRIPTDIR
# shellcheck source=preflight.sh
source "$SCRIPT_DIR/preflight.sh"

###############################################################################
# Inputs (env-driven so the run-validation driver can wire them).
###############################################################################
SSH_KEY_NAME="${SSH_KEY_NAME:-${INGERO_AWS_SSH_KEY_NAME:-ingero-key}}"
# Local SSH private key file used by verify_gpu / wait_ssh_ready.
# When unset, ssh falls back to ssh-agent or ~/.ssh defaults; with
# BatchMode=yes that fails silently if neither is configured (observed
# 2026-05-04, where the GPU was healthy but verify_gpu spun for the
# full deadline because no auth path was available).
SSH_KEY_FILE="${INGERO_AWS_SSH_KEY_FILE:-}"
REGION="${REGION:-us-east-1}"
TAG_RUN_ID="${TAG_RUN_ID:-v0-13-$(date +%s)}"
AWS_PROFILE_NAME="${AWS_PROFILE_NAME:-ingero}"
AMI_ID="${EXPECTED_AMI:-ami-0aad28499825d76c3}"


# Output state file lives in _bmad-output so teardown.sh + run-validation.sh
# can read it. Path resolution: scripts/aws/v0-13 -> repo root -> sibling
# _bmad-output dir.
STATE_DIR="${STATE_DIR:-$SCRIPT_DIR/../../../../_bmad-output}"
STATE_FILE="$STATE_DIR/v0-13-cluster-state-$TAG_RUN_ID.json"

stderr_log() {
    local level="$1"; shift
    printf '{"phase":"provision","run_id":"%s","level":"%s","msg":%s}\n' \
        "$TAG_RUN_ID" "$level" \
        "$(printf '%s' "$*" | python3 -c 'import json,sys;print(json.dumps(sys.stdin.read()))')" 1>&2
}

abort() {
    stderr_log "fatal" "$1"
    exit 1
}

aws_cli() {
    aws --profile "$AWS_PROFILE_NAME" --region "$REGION" "$@"
}

###############################################################################
# Run preflight first. We re-call it here even when invoked from
# run-validation.sh (which calls preflight on its own) because provision.sh
# is also a valid standalone entry point and must not assume the caller did
# the checks.
###############################################################################
preflight_run || abort "preflight failed"

###############################################################################
# Security group: tagged with RunID so teardown can find + delete it. We
# auto-detect the operator's public IP for the SSH rule because the LLM
# agent may run from a non-static address.
###############################################################################
ensure_security_group() {
    local existing
    existing=$(aws_cli ec2 describe-security-groups \
        --filters "Name=tag:RunID,Values=$TAG_RUN_ID" "Name=tag:Project,Values=ingero" \
        --query 'SecurityGroups[0].GroupId' --output text 2>/dev/null || echo "None")
    if [[ "$existing" != "None" && -n "$existing" ]]; then
        echo "$existing"
        return 0
    fi
    local op_ip
    op_ip=$(curl -fsSL https://checkip.amazonaws.com 2>/dev/null | tr -d '[:space:]') || \
        abort "failed to detect operator public IP for SSH rule"
    if [[ -z "$op_ip" ]]; then
        abort "operator public IP came back empty"
    fi
    local sg_id
    sg_id=$(aws_cli ec2 create-security-group \
        --group-name "ingero-v0-13-$TAG_RUN_ID" \
        --description "ingero v0.13 slice F validation cluster $TAG_RUN_ID" \
        --tag-specifications "ResourceType=security-group,Tags=[{Key=Project,Value=ingero},{Key=Slice,Value=v0-13},{Key=RunID,Value=$TAG_RUN_ID}]" \
        --query 'GroupId' --output text)
    # SSH from operator IP only.
    aws_cli ec2 authorize-security-group-ingress \
        --group-id "$sg_id" \
        --ip-permissions "IpProtocol=tcp,FromPort=22,ToPort=22,IpRanges=[{CidrIp=${op_ip}/32,Description=operator}]" \
        >/dev/null
    # Intra-group OTLP + extension API. Self-referencing security group rule
    # so peers + capture + fleet can talk freely without needing public ports.
    for port in 4317 4318 8080; do
        aws_cli ec2 authorize-security-group-ingress \
            --group-id "$sg_id" \
            --ip-permissions "IpProtocol=tcp,FromPort=$port,ToPort=$port,UserIdGroupPairs=[{GroupId=$sg_id}]" \
            >/dev/null
    done
    echo "$sg_id"
}

###############################################################################
# Find an existing instance for a role under this RunID; returns "None" if
# absent. Lets us re-run provision.sh idempotently.
###############################################################################
find_existing_instance() {
    local role="$1"
    aws_cli ec2 describe-instances \
        --filters "Name=tag:RunID,Values=$TAG_RUN_ID" \
                  "Name=tag:Role,Values=$role" \
                  "Name=instance-state-name,Values=pending,running" \
        --query 'Reservations[0].Instances[0].InstanceId' \
        --output text 2>/dev/null || echo "None"
}

###############################################################################
# Launch one instance. Spot-vs-on-demand is selected by the caller via flags
# in $extra_args. Volume tags are baked in via tag-specifications so teardown
# does not have to chase orphan volumes.
###############################################################################
launch_instance() {
    local role="$1" instance_type="$2" volume_size="$3" sg_id="$4"
    shift 4
    local extra_args=("$@")

    local existing
    existing=$(find_existing_instance "$role")
    if [[ "$existing" != "None" && -n "$existing" ]]; then
        stderr_log "info" "reusing existing $role instance $existing"
        echo "$existing"
        return 0
    fi

    local tag_spec
    tag_spec="ResourceType=instance,Tags=[{Key=Project,Value=ingero},{Key=Slice,Value=v0-13},{Key=RunID,Value=$TAG_RUN_ID},{Key=Role,Value=$role},{Key=Name,Value=ingero-v0-13-$role-$TAG_RUN_ID}]"
    local volume_spec
    volume_spec="ResourceType=volume,Tags=[{Key=Project,Value=ingero},{Key=Slice,Value=v0-13},{Key=RunID,Value=$TAG_RUN_ID},{Key=Role,Value=$role}]"
    local block_device
    block_device="DeviceName=/dev/sda1,Ebs={VolumeSize=$volume_size,VolumeType=gp3,DeleteOnTermination=true}"

    local instance_id
    instance_id=$(aws_cli ec2 run-instances \
        --image-id "$AMI_ID" \
        --instance-type "$instance_type" \
        --key-name "$SSH_KEY_NAME" \
        --security-group-ids "$sg_id" \
        --block-device-mappings "$block_device" \
        --tag-specifications "$tag_spec" "$volume_spec" \
        "${extra_args[@]}" \
        --query 'Instances[0].InstanceId' --output text)
    echo "$instance_id"
}

###############################################################################
# Wait for instance to enter `running` AND become SSH-reachable. We use the
# AWS waiter for `running`, then a short TCP probe on port 22 because
# instance-running fires before sshd is accepting connections.
###############################################################################
wait_ssh_ready() {
    local instance_id="$1"
    aws_cli ec2 wait instance-running --instance-ids "$instance_id"
    local public_ip
    public_ip=$(aws_cli ec2 describe-instances \
        --instance-ids "$instance_id" \
        --query 'Reservations[0].Instances[0].PublicIpAddress' \
        --output text)
    if [[ -z "$public_ip" || "$public_ip" == "None" ]]; then
        abort "instance $instance_id has no public IP"
    fi
    # Short TCP probe loop. Bounded: 90s max (sshd typically up within ~30s
    # on the validated AMI). We use bash's /dev/tcp pseudo-device so no
    # extra deps.
    local deadline=$((SECONDS + 90))
    while (( SECONDS < deadline )); do
        if (echo > "/dev/tcp/$public_ip/22") 2>/dev/null; then
            echo "$public_ip"
            return 0
        fi
        sleep 3
    done
    abort "instance $instance_id ($public_ip) ssh not reachable in 90s"
}

###############################################################################
# Confirm GPU is up via nvidia-smi. NVIDIA driver init via cloud-init can
# trail SSH-readiness by 5-7 minutes on the bare Ubuntu 24.04 AMI we use,
# observed empirically across two dry runs (2026-05-03, 2026-05-04). 600s
# window covers the worst case we have seen; per-attempt progress log so
# operators see we are not hung. Wait for cloud-init to finish first --
# that is the strongest readiness signal AWS gives us, and it bounds the
# tail of driver init far better than polling nvidia-smi alone.
###############################################################################
verify_gpu() {
    local public_ip="$1"
    # shellcheck disable=SC2207  # word-splitting on ssh_args is intentional
    local ssh_key_args=()
    if [[ -n "$SSH_KEY_FILE" && -f "$SSH_KEY_FILE" ]]; then
        ssh_key_args=(-i "$SSH_KEY_FILE")
    fi
    # cloud-init status --wait blocks until cloud-init drains; bounds the
    # outer wait significantly. If cloud-init itself hangs we still bail
    # at the outer 600s deadline.
    ssh "${ssh_key_args[@]}" -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes \
        "ubuntu@$public_ip" 'sudo cloud-init status --wait' >/dev/null 2>&1 || true

    local deadline=$((SECONDS + 600))
    local attempt=0
    while (( SECONDS < deadline )); do
        attempt=$((attempt + 1))
        if ssh "${ssh_key_args[@]}" -o StrictHostKeyChecking=no -o ConnectTimeout=5 -o BatchMode=yes \
                "ubuntu@$public_ip" 'nvidia-smi -L' >/dev/null 2>&1; then
            stderr_log "info" "GPU ready on $public_ip after attempt $attempt"
            return 0
        fi
        if (( attempt % 6 == 0 )); then
            stderr_log "info" "still waiting for GPU on $public_ip (attempt $attempt, ~$((attempt * 10))s elapsed)"
        fi
        sleep 10
    done
    return 1
}

###############################################################################
# Main flow.
###############################################################################
main() {
    mkdir -p "$STATE_DIR"
    local sg_id
    sg_id=$(ensure_security_group)
    stderr_log "info" "security group $sg_id ready"

    # Spot pricing: max-price = current on-demand g4dn.xlarge rate so we get
    # the spot discount under contention but rarely lose to easy reclaim.
    # 0.526/hr is the us-east-1 on-demand rate as of 2025; AWS may change it.
    # If pricing drifts we will see spot launches fail with
    # InsufficientInstanceCapacity, which surfaces clearly in the API error.
    local spot_args=(--instance-market-options 'MarketType=spot,SpotOptions={MaxPrice=0.526,SpotInstanceType=one-time,InstanceInterruptionBehavior=terminate}')

    stderr_log "info" "launching capture (g4dn.xlarge on-demand)"
    local capture_id
    capture_id=$(launch_instance capture g4dn.xlarge 50 "$sg_id")
    stderr_log "info" "launching peer1 (g4dn.xlarge spot)"
    local peer1_id
    peer1_id=$(launch_instance peer1 g4dn.xlarge 50 "$sg_id" "${spot_args[@]}")
    stderr_log "info" "launching peer2 (g4dn.xlarge spot)"
    local peer2_id
    peer2_id=$(launch_instance peer2 g4dn.xlarge 50 "$sg_id" "${spot_args[@]}")
    stderr_log "info" "launching fleet (t3.medium on-demand)"
    local fleet_id
    fleet_id=$(launch_instance fleet t3.medium 30 "$sg_id")

    stderr_log "info" "waiting for SSH on all four instances"
    local capture_ip peer1_ip peer2_ip fleet_ip
    capture_ip=$(wait_ssh_ready "$capture_id")
    peer1_ip=$(wait_ssh_ready "$peer1_id")
    peer2_ip=$(wait_ssh_ready "$peer2_id")
    fleet_ip=$(wait_ssh_ready "$fleet_id")

    stderr_log "info" "verifying GPU on g4dn nodes"
    if ! verify_gpu "$capture_ip"; then
        abort "nvidia-smi failed on capture node $capture_ip"
    fi
    # Peer GPU failures are non-fatal: spot peers can be reclaimed at any
    # time and validation phases that need them already degrade to partial.
    verify_gpu "$peer1_ip" || stderr_log "warn" "nvidia-smi failed on peer1 $peer1_ip (continuing)"
    verify_gpu "$peer2_ip" || stderr_log "warn" "nvidia-smi failed on peer2 $peer2_ip (continuing)"

    # Emit state file. Used by teardown.sh + deploy.sh + run-validation.sh.
    python3 - "$STATE_FILE" \
        "$TAG_RUN_ID" "$REGION" "$AWS_PROFILE_NAME" "$sg_id" "$AMI_ID" \
        "$capture_id" "$capture_ip" \
        "$peer1_id" "$peer1_ip" \
        "$peer2_id" "$peer2_ip" \
        "$fleet_id" "$fleet_ip" <<'PY'
import json, sys
(out, run_id, region, profile, sg, ami,
 cap_id, cap_ip, p1_id, p1_ip, p2_id, p2_ip, fl_id, fl_ip) = sys.argv[1:]
state = {
    "run_id": run_id, "region": region, "profile": profile,
    "security_group_id": sg, "ami_id": ami,
    "instances": {
        "capture": {"id": cap_id, "public_ip": cap_ip, "role": "capture",
                    "instance_type": "g4dn.xlarge", "market": "on-demand"},
        "peer1":   {"id": p1_id,  "public_ip": p1_ip,  "role": "peer1",
                    "instance_type": "g4dn.xlarge", "market": "spot"},
        "peer2":   {"id": p2_id,  "public_ip": p2_ip,  "role": "peer2",
                    "instance_type": "g4dn.xlarge", "market": "spot"},
        "fleet":   {"id": fl_id,  "public_ip": fl_ip,  "role": "fleet",
                    "instance_type": "t3.medium",   "market": "on-demand"},
    },
}
with open(out, "w") as f:
    json.dump(state, f, indent=2)
PY

    echo "$STATE_FILE"
}

main
