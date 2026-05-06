#!/bin/bash
# 34-fan-in-completeness.sh
#
# End-to-end fan-in completeness test on a host without GPUs.
# Brings up the local-stack (real fleet + real ingero-echo + 3
# sim-agents), waits for sim-agents to emit, then queries Echo's
# DuckDB through the run_analysis MCP tool to assert:
#
#   1. all 3 cluster_id/node_id tuples landed in DuckDB (fleet
#      did not drop / re-route any)
#   2. each node has at least N events in the window (no fan-in
#      starvation)
#   3. the planted straggler (gpu-node-03-bad with score 0.30)
#      shows the lowest minimum health-score
#
# This validates the substrate "agent (sim) -> fleet -> echo ->
# DuckDB -> MCP" wire path, on WSL/CI hosts without GPU.
#
# Substrate: ../../../../ingero-fleet/examples/local-stack/
# Companion: 33-investigate-finds-everything.sh (LLM behavior).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
FLEET_REPO="${FLEET_REPO:-$REPO_ROOT/ingero-fleet}"
STACK_DIR="$FLEET_REPO/examples/local-stack"

if [[ ! -d "$STACK_DIR" ]]; then
  echo "FAIL: local-stack not found at $STACK_DIR"
  echo "      set FLEET_REPO=/path/to/ingero-fleet"
  exit 2
fi

# Some hosts (notably WSL) do not put the invoking user in the
# docker group; use sudo when needed so the script is portable.
DOCKER="docker"
if ! docker info >/dev/null 2>&1; then
  if sudo -n docker info >/dev/null 2>&1; then
    DOCKER="sudo docker"
  else
    echo "SKIP: docker daemon not reachable (try: sudo usermod -aG docker $USER)"
    exit 0
  fi
fi
if ! $DOCKER compose version >/dev/null 2>&1; then
  echo "SKIP: docker compose plugin missing"
  exit 0
fi

cleanup() {
  set +e
  echo "==> [cleanup] tearing down local-stack"
  (cd "$STACK_DIR" && $DOCKER compose down -v >/dev/null 2>&1 || true)
}
trap cleanup EXIT

EMIT_WINDOW_S="${EMIT_WINDOW_S:-30}"
MIN_EVENTS_PER_NODE="${MIN_EVENTS_PER_NODE:-5}"

echo "==> [1/5] docker compose up --build"
( cd "$STACK_DIR" && $DOCKER compose up -d --build ) 2>&1 | tail -20

echo "==> [2/5] wait for echo health (up to 90s)"
DEADLINE=$(( SECONDS + 90 ))
until curl -sf http://localhost:18080/healthz >/dev/null 2>&1; do
  if (( SECONDS >= DEADLINE )); then
    echo "FAIL: echo did not become healthy within 90s"
    (cd "$STACK_DIR" && $DOCKER compose logs --tail=80 echo) 2>&1 | tail -100
    exit 1
  fi
  sleep 2
done
echo "OK: echo healthy"

echo "==> [3/5] wait ${EMIT_WINDOW_S}s for sim-agents to emit"
sleep "$EMIT_WINDOW_S"

echo "==> [4/5] query echo run_analysis via MCP (python urllib)"
# Inline MCP client in python: streamable-HTTP, single session,
# initialize -> tools/call(run_analysis). Avoids a Go build step
# and the pipefail-vs-set-e interactions that masked the build
# failure on earlier runs of this script.
QUERY_SQL="SELECT cluster_id, node_id, COUNT(*) AS events, MIN(value_double) AS min_score FROM events WHERE metric_name='ingero.node.health_score' GROUP BY cluster_id, node_id ORDER BY node_id"
export ECHO_MCP_URL="http://localhost:18081/"
export QUERY_SQL
RESULT=$(python3 <<'PYEOF'
import json, os, sys, urllib.request

URL = os.environ["ECHO_MCP_URL"]
SQL = os.environ["QUERY_SQL"]

def mcp(sid, method, params=None, rid=None):
    body = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if rid is not None:
        body["id"] = rid
    headers = {
        "content-type": "application/json",
        "accept": "application/json, text/event-stream",
    }
    if sid:
        headers["mcp-session-id"] = sid
    req = urllib.request.Request(URL, data=json.dumps(body).encode(),
                                 headers=headers, method="POST")
    with urllib.request.urlopen(req, timeout=30) as resp:
        new_sid = resp.headers.get("mcp-session-id") or sid
        ct = resp.headers.get("content-type", "")
        raw = resp.read().decode()
        if "text/event-stream" in ct:
            for line in raw.splitlines():
                if line.startswith("data: "):
                    return new_sid, json.loads(line[6:])
            return new_sid, None
        return new_sid, json.loads(raw) if raw else None

sid, _ = mcp(None, "initialize", {
    "protocolVersion": "2025-03-26",
    "capabilities": {},
    "clientInfo": {"name": "fanin-test", "version": "0.0.1"},
}, rid=1)
mcp(sid, "notifications/initialized")
_, tr = mcp(sid, "tools/call", {
    "name": "fleet.cluster.run_analysis",
    "arguments": {"sql": SQL, "limit": 1000},
}, rid=2)
result = (tr or {}).get("result", {})
sc = result.get("structuredContent")
if sc is not None:
    print(json.dumps(sc))
else:
    print(json.dumps({"rows": [], "columns": [], "raw": result}))
PYEOF
)
echo "$RESULT"

echo "==> [5/5] assertions"
ROW_COUNT=$(echo "$RESULT" | python3 -c "import sys,json; d=json.load(sys.stdin); print(len(d.get('rows', [])))")
if (( ROW_COUNT < 3 )); then
  echo "FAIL: expected 3 distinct (cluster,node) rows, got $ROW_COUNT"
  exit 1
fi
echo "OK: 3 nodes present"

# Each node should have at least MIN_EVENTS_PER_NODE events
LOW_NODE=$(echo "$RESULT" | python3 -c "
import sys,json
d=json.load(sys.stdin)
rows=d.get('rows',[])
for r in rows:
    if int(r[2]) < $MIN_EVENTS_PER_NODE:
        print(r[1]); sys.exit(0)
")
if [[ -n "$LOW_NODE" ]]; then
  echo "FAIL: node $LOW_NODE got fewer than $MIN_EVENTS_PER_NODE events"
  exit 1
fi
echo "OK: every node received >= $MIN_EVENTS_PER_NODE events"

# The planted straggler (gpu-node-03-bad) should have the lowest
# min_score
WORST=$(echo "$RESULT" | python3 -c "
import sys,json
d=json.load(sys.stdin)
rows=d.get('rows',[])
worst=min(rows, key=lambda r: float(r[3]))
print(worst[1])
")
if [[ "$WORST" != "gpu-node-03-bad" ]]; then
  echo "FAIL: lowest-scoring node is $WORST, expected gpu-node-03-bad"
  exit 1
fi
echo "OK: planted straggler (gpu-node-03-bad) has the lowest min_score"

echo
echo "PASS: fan-in completeness validated"
