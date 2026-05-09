#!/usr/bin/env bash
# 33-investigate-finds-everything.sh
#
# End-to-end LLM-behavior test for the federated /investigate prompt
# registered by ingero-echo. Drives an Anthropic Messages API
# conversation that exposes Echo's MCP tools as native tool-use,
# walks the /investigate recipe, and asserts the model:
#
#   1. Begins with fleet.cluster.summary (the recipe's Step 1).
#   2. Continues with find_stragglers OR find_outlier_nodes
#      (Step 2 / 2-fallback).
#   3. Names the planted straggler node id in its final answer.
#
# Substrate: ingero-fleet/examples/local-stack/ (real fleet + real
# echo + 3 sim-agents). The compose file plants gpu-node-03-bad as
# a low-health-score sim-agent so the recipe has something to
# surface.
#
# Skips gracefully if ANTHROPIC_API_KEY is missing or docker is
# not on PATH (returns 0 with a SKIP message).
#
# Companion: 34-fan-in-completeness.sh (substrate validation).

set -euo pipefail

REPO_ROOT="${REPO_ROOT:-$(cd "$(dirname "$0")/../../../.." && pwd)}"
FLEET_REPO="${FLEET_REPO:-$REPO_ROOT/ingero-fleet}"
STACK_DIR="$FLEET_REPO/examples/local-stack"
ANTHROPIC_MODEL="${ANTHROPIC_MODEL:-claude-opus-4-7}"
EXPECTED_BAD_NODE="${EXPECTED_BAD_NODE:-gpu-node-03-bad}"

if [[ -z "${ANTHROPIC_API_KEY:-}" ]]; then
  echo "SKIP: ANTHROPIC_API_KEY not set; cannot drive /investigate end-to-end"
  exit 0
fi
DOCKER="docker"
if ! docker info >/dev/null 2>&1; then
  if sudo -n docker info >/dev/null 2>&1; then
    DOCKER="sudo docker"
  else
    echo "SKIP: docker daemon not reachable"
    exit 0
  fi
fi
if ! $DOCKER compose version >/dev/null 2>&1; then
  echo "SKIP: docker compose plugin missing"
  exit 0
fi
if ! command -v python3 >/dev/null 2>&1; then
  echo "SKIP: python3 missing"
  exit 0
fi

cleanup() {
  set +e
  echo "==> [cleanup] tearing down local-stack"
  (cd "$STACK_DIR" && $DOCKER compose down -v >/dev/null 2>&1 || true)
}
trap cleanup EXIT

echo "==> [1/4] docker compose up --build"
( cd "$STACK_DIR" && $DOCKER compose up -d --build ) 2>&1 | tail -15

echo "==> [2/4] wait for echo health + sim-agent emissions (45s)"
DEADLINE=$(( SECONDS + 90 ))
until curl -sf http://localhost:18080/healthz >/dev/null 2>&1; do
  if (( SECONDS >= DEADLINE )); then
    echo "FAIL: echo did not become healthy"
    exit 1
  fi
  sleep 2
done
sleep 45  # let sim-agents emit ~22 events each at 2s interval

echo "==> [3/4] drive /investigate via Anthropic Messages API + MCP tool-use"
WORK=$(mktemp -d)
trap 'rm -rf "$WORK"; cleanup' EXIT
export ECHO_MCP_URL="http://localhost:18081/"
export ANTHROPIC_API_KEY ANTHROPIC_MODEL EXPECTED_BAD_NODE

python3 <<'PYEOF' > "$WORK/result.json"
import json, os, sys, urllib.request, urllib.error, uuid

API_KEY = os.environ["ANTHROPIC_API_KEY"]
MODEL = os.environ["ANTHROPIC_MODEL"]
MCP_URL = os.environ["ECHO_MCP_URL"]
EXPECTED_BAD = os.environ["EXPECTED_BAD_NODE"]

# Streamable MCP client (single-session). We re-implement the
# minimum slice of the protocol needed: initialize + tools/list +
# tools/call. Each request is a JSON-RPC POST; the server returns
# either application/json or text/event-stream. For the calls we
# make here a synchronous JSON response is sufficient.
def mcp(session_id, method, params=None, request_id=None):
    body = {"jsonrpc": "2.0", "method": method, "params": params or {}}
    if request_id is not None:
        body["id"] = request_id
    headers = {
        "content-type": "application/json",
        "accept": "application/json, text/event-stream",
    }
    if session_id:
        headers["mcp-session-id"] = session_id
    req = urllib.request.Request(
        MCP_URL,
        data=json.dumps(body).encode(),
        headers=headers,
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        sid = resp.headers.get("mcp-session-id") or session_id
        ct = resp.headers.get("content-type", "")
        raw = resp.read().decode()
        if "text/event-stream" in ct:
            for line in raw.splitlines():
                if line.startswith("data: "):
                    return sid, json.loads(line[6:])
            return sid, None
        return sid, json.loads(raw) if raw else None

# Initialize MCP session.
sid, init = mcp(None, "initialize", {
    "protocolVersion": "2025-03-26",
    "capabilities": {},
    "clientInfo": {"name": "investigate-llm-test", "version": "0.0.1"},
}, request_id=1)
mcp(sid, "notifications/initialized")
_, tl = mcp(sid, "tools/list", request_id=2)
tools = (tl or {}).get("result", {}).get("tools", [])
print(f"# discovered {len(tools)} tools", file=sys.stderr)

# Translate MCP tool descriptors to Anthropic tool-use schema.
anth_tools = []
for t in tools:
    schema = t.get("inputSchema") or {"type": "object", "properties": {}}
    anth_tools.append({
        "name": t["name"],
        "description": t.get("description", ""),
        "input_schema": schema,
    })

def anthropic_call(messages):
    body = {
        "model": MODEL,
        "max_tokens": 2048,
        "system": ("You are a fleet-wide GPU performance analyst. Use the "
                   "provided tools to walk the standard /investigate recipe: "
                   "start with fleet.cluster.summary, then "
                   "fleet.cluster.find_stragglers (or find_outlier_nodes if "
                   "no stragglers), then summarize WHERE/WHY for the worst "
                   "node. Always cite the node id in your final answer."),
        "messages": messages,
        "tools": anth_tools,
    }
    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=json.dumps(body).encode(),
        headers={
            "x-api-key": API_KEY,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=60) as resp:
        return json.loads(resp.read().decode())

# Conversation loop: at most 6 turns. Stop when the model emits an
# end_turn without further tool_use.
messages = [{"role": "user", "content": "/investigate"}]
tool_call_log = []  # ordered list of tool names called
final_text = ""
for turn in range(6):
    print(f"# turn {turn}", file=sys.stderr)
    resp = anthropic_call(messages)
    content = resp.get("content", [])
    messages.append({"role": "assistant", "content": content})
    tool_results = []
    for block in content:
        if block.get("type") == "tool_use":
            name = block["name"]
            tool_call_log.append(name)
            print(f"# tool_use {name} {json.dumps(block.get('input',{}))[:120]}", file=sys.stderr)
            # Execute the tool via the MCP session.
            try:
                _, tr = mcp(sid, "tools/call", {
                    "name": name,
                    "arguments": block.get("input") or {},
                }, request_id=100 + turn)
                tool_payload = (tr or {}).get("result", {})
                content_blocks = tool_payload.get("content", [])
                text_parts = [c.get("text", "") for c in content_blocks if c.get("type") == "text"]
                struct = tool_payload.get("structuredContent")
                tool_text = "\n".join(text_parts) if text_parts else ""
                if struct is not None:
                    tool_text += "\n\nstructured: " + json.dumps(struct)[:4000]
            except Exception as e:
                tool_text = f"tool error: {e}"
            tool_results.append({
                "type": "tool_result",
                "tool_use_id": block["id"],
                "content": tool_text or "(empty)",
            })
        elif block.get("type") == "text":
            final_text += block.get("text", "")
    if tool_results:
        messages.append({"role": "user", "content": tool_results})
    if resp.get("stop_reason") == "end_turn":
        break

result = {
    "tool_calls": tool_call_log,
    "final_text": final_text,
    "expected_bad": EXPECTED_BAD,
}
print(json.dumps(result))
PYEOF

cat "$WORK/result.json" | python3 -m json.tool

echo "==> [4/4] assertions"
python3 - "$WORK/result.json" "$EXPECTED_BAD_NODE" <<'PYEOF'
import json, sys
d = json.load(open(sys.argv[1]))
expected_bad = sys.argv[2]
calls = d["tool_calls"]
text = d["final_text"]
fail = []

if not calls or calls[0] != "fleet.cluster.summary":
    fail.append(f"first tool was {calls[0] if calls else 'NONE'}, expected fleet.cluster.summary")

if not any(c in calls for c in ("fleet.cluster.find_stragglers", "fleet.cluster.find_outlier_nodes")):
    fail.append(f"never called find_stragglers or find_outlier_nodes (calls={calls})")

if expected_bad not in text:
    fail.append(f"final answer does not name {expected_bad!r} (got first 400 chars: {text[:400]!r})")

if fail:
    for f in fail:
        print(f"FAIL: {f}")
    sys.exit(1)

print("OK: investigate recipe followed; planted straggler identified")
print(f"OK: tool sequence = {' -> '.join(calls)}")
PYEOF

echo
echo "PASS: investigate-finds-everything"
