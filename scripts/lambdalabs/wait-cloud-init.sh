#!/bin/bash
# Wait for cloud-init to complete on Lambda Labs instance.
# Reads IP from .lambdalabs-vm.json at the repo root.
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# 3 levels up from scripts/lambdalabs/ to reach mono-repo root.
PROJECT_DIR="$(cd "$SCRIPT_DIR/../../.." && pwd)"
STATE_FILE="$PROJECT_DIR/.lambdalabs-vm.json"

if [[ ! -f "$STATE_FILE" ]]; then
    echo "ERROR: No state file at $STATE_FILE — deploy first."
    exit 1
fi

IP=$(jq -r .ip "$STATE_FILE")
echo "Waiting for cloud-init on $IP..."
for i in $(seq 1 30); do
    sleep 10
    result=$(ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null \
        -i ~/.ssh/id_ed25519 ubuntu@"$IP" \
        'cat ~/workspace/cloud-init-done.txt 2>/dev/null' 2>/dev/null || true)
    if [ -n "$result" ]; then
        echo "Cloud-init done: $result"
        exit 0
    fi
    echo "  Still waiting... $((i*10))s"
done
echo "Timed out after 300s"
exit 1
