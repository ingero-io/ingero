#!/usr/bin/env bash
# Test 32: Critical-event guarantee (data-plane gate 4).
#
# Triggers:
#   - 100 fork+exec+exit cycles (shell loop spawning /bin/true).
#   - 1 OOM via a Python process exceeding cgroup memory.limit_in_bytes.
#
# Asserts (HARD, 0% loss tolerance at every stage):
#   - Local trace DB has 100 fork + 100 exec + 100 exit + 1 oom event.
#   - Echo store has the same.
#   - External OTEL receiver (parallel mirror) has the same.
#
# Hardware: any A10. Linux with cgroup-v2 + sudo.
#
# Invoke:
#   sudo bash tests/e2e/data-plane/32-critical-events-guaranteed.sh
#
# Optional env:
#   INGERO_BIN
#   ECHO_DB           default /var/lib/ingero/echo.db
#   EXT_OTLP_PORT     default 4319
#
# Expected runtime: ~90s.
set -euo pipefail
. "$(dirname "$0")/../_lib.sh"

INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo ./ingero)}"
ECHO_DB="${ECHO_DB:-/var/lib/ingero/echo.db}"
EXT_OTLP_PORT="${EXT_OTLP_PORT:-4319}"

[[ -x "$INGERO_BIN" ]] || { echo "FAIL: agent missing at $INGERO_BIN"; exit 1; }
command -v sqlite3 >/dev/null || { echo "FAIL: sqlite3 missing"; exit 1; }
command -v docker >/dev/null || { echo "FAIL: docker missing"; exit 1; }

WORK=$(mktemp -d)
AGENT_PID=""
EXT_COLLECTOR="ingero-ext-test32-$$"
CG_DIR=""

cleanup() {
  set +e
  kill_agent
  docker rm -f "$EXT_COLLECTOR" >/dev/null 2>&1
  if [[ -n "$CG_DIR" && -d "$CG_DIR" ]]; then
    sudo rmdir "$CG_DIR" 2>/dev/null
  fi
  rm -rf "$WORK"
}
trap cleanup EXIT

echo "=== Test 32: critical-events-guaranteed ==="

echo "==> [setup] Boot external mirror on :$EXT_OTLP_PORT"
cat > "$WORK/ext.yaml" <<YAML
receivers:
  otlp:
    protocols:
      http:
        endpoint: 0.0.0.0:$EXT_OTLP_PORT
exporters:
  debug:
    verbosity: detailed
service:
  pipelines:
    metrics:
      receivers: [otlp]
      exporters: [debug]
    logs:
      receivers: [otlp]
      exporters: [debug]
YAML
docker run -d --rm --name "$EXT_COLLECTOR" --network host \
  -v "$WORK/ext.yaml:/etc/otelcol-contrib/config.yaml:ro" \
  otel/opentelemetry-collector-contrib:latest \
  --config=/etc/otelcol-contrib/config.yaml >/dev/null
sleep 4

echo "==> [setup] Boot agent"
T0=$(date -u +%s)
sudo "$INGERO_BIN" trace --record --db "$WORK/trace.db" \
  --duration 80s \
  --otlp-mirror "localhost:$EXT_OTLP_PORT" \
  >"$WORK/agent.log" 2>&1 &
AGENT_PID=$!
sleep 8

echo "==> [drive] 100 fork+exec+exit cycles"
for i in $(seq 1 100); do /bin/true; done

echo "==> [drive] 1 OOM via cgroup-v2 limited child"
CG_BASE="/sys/fs/cgroup/test32-$$.slice"
if [[ ! -w /sys/fs/cgroup ]]; then
  echo "WARN: /sys/fs/cgroup not writable; skipping OOM trigger"
  EXPECT_OOM=0
else
  sudo mkdir -p "$CG_BASE"
  CG_DIR="$CG_BASE"
  echo "+memory" | sudo tee /sys/fs/cgroup/cgroup.subtree_control >/dev/null 2>&1 || true
  echo $((128 * 1024 * 1024)) | sudo tee "$CG_BASE/memory.max" >/dev/null
  # Spawn a shell into the cgroup and run a memory-grower.
  sudo bash -c "echo \$\$ > $CG_BASE/cgroup.procs; python3 -c 'a=[]\nfor i in range(2000): a.append(b\"x\"*1024*1024)' || true" \
    >"$WORK/oom.log" 2>&1 || true
  EXPECT_OOM=1
fi

# Let the agent flush its window.
sleep 12
wait "$AGENT_PID" || true
T1=$(date -u +%s)
sleep 3
docker logs "$EXT_COLLECTOR" >"$WORK/ext.log" 2>&1

echo "==> [assert stage 1: local trace DB]"
DB="$WORK/trace.db"
count_in_db() {
  local kind="$1" db="$2"
  sqlite3 "$db" "SELECT count(*) FROM events WHERE metric_name LIKE 'proc.${kind}%' OR event_type='${kind}'" 2>/dev/null || echo 0
}
FORK=$(count_in_db fork "$DB")
EXEC=$(count_in_db exec "$DB")
EXIT=$(count_in_db exit "$DB")
OOM=$(sqlite3 "$DB"  "SELECT count(*) FROM events WHERE metric_name LIKE 'proc.oom%' OR event_type='oom'" 2>/dev/null || echo 0)

echo "local: fork=$FORK exec=$EXEC exit=$EXIT oom=$OOM"

assert_min() {
  local label="$1" actual="$2" expected="$3"
  if (( actual < expected )); then
    echo "FAIL: $label = $actual (expected >= $expected)"
    return 1
  fi
  echo "OK: $label = $actual >= $expected"
}

FAIL=0
assert_min "local fork" "$FORK" 100 || FAIL=$((FAIL+1))
assert_min "local exec" "$EXEC" 100 || FAIL=$((FAIL+1))
assert_min "local exit" "$EXIT" 100 || FAIL=$((FAIL+1))
if (( EXPECT_OOM == 1 )); then
  assert_min "local oom" "$OOM" 1 || FAIL=$((FAIL+1))
fi

echo "==> [assert stage 2: Echo store]"
if [[ -f "$ECHO_DB" ]]; then
  EFORK=$(sqlite3 "$ECHO_DB" "SELECT count(*) FROM events WHERE timestamp BETWEEN $T0 AND $T1 AND (metric_name LIKE 'proc.fork%' OR event_type='fork')" 2>/dev/null || echo 0)
  EEXEC=$(sqlite3 "$ECHO_DB" "SELECT count(*) FROM events WHERE timestamp BETWEEN $T0 AND $T1 AND (metric_name LIKE 'proc.exec%' OR event_type='exec')" 2>/dev/null || echo 0)
  EEXIT=$(sqlite3 "$ECHO_DB" "SELECT count(*) FROM events WHERE timestamp BETWEEN $T0 AND $T1 AND (metric_name LIKE 'proc.exit%' OR event_type='exit')" 2>/dev/null || echo 0)
  EOOM=$( sqlite3 "$ECHO_DB" "SELECT count(*) FROM events WHERE timestamp BETWEEN $T0 AND $T1 AND (metric_name LIKE 'proc.oom%'  OR event_type='oom')"  2>/dev/null || echo 0)
  echo "echo: fork=$EFORK exec=$EEXEC exit=$EEXIT oom=$EOOM"
  assert_min "echo fork" "$EFORK" 100 || FAIL=$((FAIL+1))
  assert_min "echo exec" "$EEXEC" 100 || FAIL=$((FAIL+1))
  assert_min "echo exit" "$EEXIT" 100 || FAIL=$((FAIL+1))
  if (( EXPECT_OOM == 1 )); then assert_min "echo oom" "$EOOM" 1 || FAIL=$((FAIL+1)); fi
else
  echo "WARN: ECHO_DB missing at $ECHO_DB; skipping stage 2"
fi

echo "==> [assert stage 3: external mirror]"
EXT_FORK=$(grep -c -E 'proc\.fork|event_type=fork'  "$WORK/ext.log" || true)
EXT_EXEC=$(grep -c -E 'proc\.exec|event_type=exec'  "$WORK/ext.log" || true)
EXT_EXIT=$(grep -c -E 'proc\.exit|event_type=exit'  "$WORK/ext.log" || true)
EXT_OOM=$( grep -c -E 'proc\.oom|event_type=oom'    "$WORK/ext.log" || true)
echo "ext: fork=$EXT_FORK exec=$EXT_EXEC exit=$EXT_EXIT oom=$EXT_OOM"
assert_min "ext fork" "$EXT_FORK" 100 || FAIL=$((FAIL+1))
assert_min "ext exec" "$EXT_EXEC" 100 || FAIL=$((FAIL+1))
assert_min "ext exit" "$EXT_EXIT" 100 || FAIL=$((FAIL+1))
if (( EXPECT_OOM == 1 )); then assert_min "ext oom" "$EXT_OOM" 1 || FAIL=$((FAIL+1)); fi

if (( FAIL > 0 )); then
  echo "FAIL: $FAIL critical-event assertion(s) failed (zero tolerance)"
  exit 1
fi
echo "PASS: critical-events-guaranteed"
