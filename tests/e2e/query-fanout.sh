#!/usr/bin/env bash
# Test 5: ingero query SQL fan-out across two recording nodes.
#
# Asserts:
#   - `ingero query --nodes node1:22,node2:22 "SELECT count(*) FROM events"`
#     returns rows from both nodes.
#   - Each node contributes a non-zero count.
#
# Hardware: two single-node hosts, each running an `ingero trace --record`
# session for at least 60s. SSH access from the driver host to both via the
# same key.
#
# Invoke (from a third host that holds the SSH key):
#   NODE1_HOST=10.0.0.1 NODE2_HOST=10.0.0.2 \
#   SSH_USER=ubuntu SSH_KEY=~/.ssh/id_rsa \
#   bash tests/e2e/query-fanout.sh
#
# Optional env:
#   INGERO_BIN       path to the agent binary on the driver host
#   REMOTE_BIN       path to the agent binary on each remote node
#                    (default: /usr/local/bin/ingero)
#   REMOTE_DB        per-node DB path (default: /var/lib/ingero/trace.db)
#
# Expected runtime: ~120s (60s record window + query).
set -euo pipefail

: "${NODE1_HOST:?NODE1_HOST must be set}"
: "${NODE2_HOST:?NODE2_HOST must be set}"
SSH_USER="${SSH_USER:-ubuntu}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_rsa}"
INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo ./ingero)}"
REMOTE_BIN="${REMOTE_BIN:-/usr/local/bin/ingero}"
REMOTE_DB="${REMOTE_DB:-/var/lib/ingero/trace.db}"

SSH_OPTS="-i $SSH_KEY -o StrictHostKeyChecking=no -o IdentitiesOnly=yes"

if [[ ! -x "$INGERO_BIN" ]]; then
  echo "FAIL: driver agent binary missing at $INGERO_BIN"
  exit 1
fi

WORK=$(mktemp -d)
PIDS=()

cleanup() {
  set +e
  for pid in "${PIDS[@]}"; do
    kill "$pid" 2>/dev/null
  done
  for h in "$NODE1_HOST" "$NODE2_HOST"; do
    ssh $SSH_OPTS "$SSH_USER@$h" "sudo pkill -f 'ingero trace' || true" 2>/dev/null
  done
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "=== Test 5: query-fanout (2 nodes) ==="

echo "==> [1/3] Boot agent on each node, recording for 60s"
for h in "$NODE1_HOST" "$NODE2_HOST"; do
  ssh $SSH_OPTS "$SSH_USER@$h" \
    "sudo $REMOTE_BIN trace --record --db $REMOTE_DB --duration 90s >/tmp/agent.log 2>&1 &" &
  PIDS+=("$!")
done

echo "==> [2/3] Wait 75s for agents to record"
sleep 75

echo "==> [3/3] Run fan-out query from driver host"
NODES_ARG="$NODE1_HOST:22,$NODE2_HOST:22"
RESULT="$WORK/result.txt"
if ! "$INGERO_BIN" query \
      --nodes "$NODES_ARG" \
      --ssh-user "$SSH_USER" \
      --ssh-key "$SSH_KEY" \
      --db "$REMOTE_DB" \
      "SELECT node_id, count(*) AS n FROM events GROUP BY node_id" \
      >"$RESULT" 2>&1; then
  echo "FAIL: query failed"
  cat "$RESULT"
  exit 1
fi

cat "$RESULT"

# Expect at least 2 distinct node rows.
NODE_LINES=$(grep -cE '^\s*\S+\s+\S+\s*$' "$RESULT" || true)
if (( NODE_LINES < 2 )); then
  echo "FAIL: query did not return rows from both nodes (saw $NODE_LINES)"
  exit 1
fi
echo "OK: fan-out returned rows from $NODE_LINES distinct nodes"

# Both counts must be > 0.
ZERO=$(awk 'NF>=2 && $NF==0' "$RESULT" || true)
if [[ -n "$ZERO" ]]; then
  echo "FAIL: at least one node returned zero events"
  echo "$ZERO"
  exit 1
fi
echo "OK: both nodes contributed non-zero counts"

echo "PASS: query-fanout"
