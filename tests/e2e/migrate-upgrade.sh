#!/usr/bin/env bash
# Test 6: Schema migration upgrade path (v0.10.x DB -> v0.14 schema).
#
# Asserts:
#   - `ingero migrate` exits 0 against a v0.10.x-shape trace DB fixture.
#   - "applied N migrations" line is present.
#   - Post-migration `ingero query` succeeds against the migrated DB.
#
# Hardware: any host. No GPU needed for this test.
#
# Invoke:
#   bash tests/e2e/migrate-upgrade.sh
#
# Optional env:
#   INGERO_BIN       path to the agent binary
#   FIXTURE_DB       path to a v0.10.x-shape DB fixture
#                    (default: tests/fixtures/db-v0.10.x.db)
#
# Expected runtime: ~10s.
#
# Notes on the fixture:
#   The fixture is a tiny SQLite database carrying the v0.10.x schema with a
#   handful of seed rows. If the fixture is absent, the script fabricates a
#   minimal one inline so the migration path can still be exercised in CI.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
INGERO_BIN="${INGERO_BIN:-$(command -v ingero || echo "$REPO_ROOT/ingero")}"
FIXTURE_DB="${FIXTURE_DB:-$REPO_ROOT/tests/fixtures/db-v0.10.x.db}"

if [[ ! -x "$INGERO_BIN" ]]; then
  echo "FAIL: agent binary missing at $INGERO_BIN"
  exit 1
fi
if ! command -v sqlite3 >/dev/null; then
  echo "FAIL: sqlite3 missing"
  exit 1
fi

WORK=$(mktemp -d)
cleanup() { rm -rf "$WORK"; }
trap cleanup EXIT

echo "=== Test 6: migrate-upgrade ==="

DB="$WORK/trace.db"
if [[ -f "$FIXTURE_DB" ]]; then
  cp "$FIXTURE_DB" "$DB"
  echo "OK: copied fixture from $FIXTURE_DB"
else
  echo "WARN: fixture missing at $FIXTURE_DB; fabricating minimal v0.10.x shape"
  sqlite3 "$DB" <<'SQL'
CREATE TABLE events (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  ts INTEGER NOT NULL,
  pid INTEGER,
  metric_name TEXT,
  value REAL
);
INSERT INTO events (ts, pid, metric_name, value) VALUES
  (1700000000, 1234, 'cuda.malloc', 1024),
  (1700000001, 1234, 'cuda.memcpy', 2048),
  (1700000002, 1235, 'nccl.allreduce.duration_ms', 5.5);
CREATE TABLE schema_version (version INTEGER NOT NULL);
INSERT INTO schema_version (version) VALUES (1);
SQL
fi

echo "==> [1/2] Run ingero migrate"
LOG="$WORK/migrate.log"
if ! "$INGERO_BIN" migrate "$DB" >"$LOG" 2>&1; then
  echo "FAIL: ingero migrate exited non-zero"
  cat "$LOG"
  exit 1
fi
cat "$LOG"

if ! grep -Eq 'applied [0-9]+ migration' "$LOG"; then
  echo "FAIL: 'applied N migrations' line missing"
  exit 1
fi
echo "OK: migrate reported applied migrations"

echo "==> [2/2] Query the migrated DB"
QLOG="$WORK/query.log"
if ! "$INGERO_BIN" query --db "$DB" "SELECT count(*) FROM events" >"$QLOG" 2>&1; then
  echo "FAIL: query against migrated DB failed"
  cat "$QLOG"
  exit 1
fi
cat "$QLOG"
echo "OK: query against migrated DB succeeded"

echo "PASS: migrate-upgrade"
