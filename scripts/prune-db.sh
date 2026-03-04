#!/usr/bin/env bash
# prune-db.sh — Remove non-CUDA sched_switch events from an Ingero SQLite DB.
#
# The sched_switch filter (v0.8.1+) prevents these rows from being stored
# during tracing. This script retroactively applies the same logic to
# databases created before the filter existed.
#
# What it deletes:
#   sched_switch events (source=3, op=1) where the PID has NO CUDA Runtime
#   (source=1) or Driver API (source=4) events in the same database.
#
# Note: the live code's trackedPIDs is slightly broader (includes IO/TCP/Net
# PIDs and fork children). This script uses the tighter CUDA-only definition
# because IO/TCP PIDs are system-wide kernel tracepoints that would widen
# the set to include system daemons, and fork children that never call CUDA
# have no sched_switch investigation value for GPU causal chains.
#
# What it keeps:
#   - sched_switch from CUDA-active PIDs (causal chain investigation value)
#   - All CUDA Runtime, Driver, I/O, TCP, Net events
#   - All process lifecycle events (exec/exit/fork/OOM)
#   - All system snapshots, causal chains, aggregates, stack traces
#
# Usage:
#   ./scripts/prune-db.sh <path-to-ingero.db>
#   ./scripts/prune-db.sh ~/.ingero/ingero.db
#   ./scripts/prune-db.sh logs/2026-03-04_*/ingero.db

set -euo pipefail

if [ $# -ne 1 ]; then
    echo "Usage: $0 <path-to-ingero.db>"
    exit 1
fi

DB="$1"

if [ ! -f "$DB" ]; then
    echo "Error: file not found: $DB"
    exit 1
fi

if ! command -v sqlite3 &>/dev/null; then
    echo "Error: sqlite3 not found. Install: sudo apt install sqlite3"
    exit 1
fi

# Show before state.
SIZE_BEFORE=$(stat -c%s "$DB")
SIZE_BEFORE_MB=$(awk "BEGIN{printf \"%.1f\", $SIZE_BEFORE/1048576}")

echo "Database: $DB"
echo "Size before: ${SIZE_BEFORE_MB} MB"
echo ""

sqlite3 "$DB" <<'SQL'
.mode column
.headers on
SELECT
    'total' AS category,
    COUNT(*) AS rows
FROM events
UNION ALL
SELECT
    'sched_switch (all)',
    COUNT(*)
FROM events WHERE source=3 AND op=1
UNION ALL
SELECT
    'sched_switch (CUDA PIDs)',
    COUNT(*)
FROM events WHERE source=3 AND op=1
    AND pid IN (SELECT DISTINCT pid FROM events WHERE source IN (1,4))
UNION ALL
SELECT
    'sched_switch (non-CUDA)',
    COUNT(*)
FROM events WHERE source=3 AND op=1
    AND pid NOT IN (SELECT DISTINCT pid FROM events WHERE source IN (1,4));
SQL

echo ""

# Count rows to delete.
TO_DELETE=$(sqlite3 "$DB" "SELECT COUNT(*) FROM events WHERE source=3 AND op=1 AND pid NOT IN (SELECT DISTINCT pid FROM events WHERE source IN (1,4));")

if [ "$TO_DELETE" -eq 0 ]; then
    echo "Nothing to prune — no non-CUDA sched_switch rows found."
    exit 0
fi

echo "Will delete ${TO_DELETE} non-CUDA sched_switch rows."
echo ""
read -rp "Proceed? [y/N] " confirm
if [[ ! "$confirm" =~ ^[yY]$ ]]; then
    echo "Aborted."
    exit 0
fi

# Create backup before destructive operation.
BACKUP="${DB}.bak"
echo ""
echo "Creating backup: ${BACKUP}"
cp "$DB" "$BACKUP"

echo "Deleting..."
sqlite3 "$DB" <<'SQL'
DELETE FROM events
WHERE source = 3
  AND op = 1
  AND pid NOT IN (SELECT DISTINCT pid FROM events WHERE source IN (1, 4));
SQL

ROWS_AFTER=$(sqlite3 "$DB" "SELECT COUNT(*) FROM events;")
echo "Events remaining: ${ROWS_AFTER}"

echo "Running VACUUM (reclaiming disk space)..."
sqlite3 "$DB" "VACUUM;"

SIZE_AFTER=$(stat -c%s "$DB")
SIZE_AFTER_MB=$(awk "BEGIN{printf \"%.1f\", $SIZE_AFTER/1048576}")
SAVED=$(awk "BEGIN{printf \"%.1f\", ($SIZE_BEFORE-$SIZE_AFTER)/1048576}")
PCT=$(awk "BEGIN{printf \"%.0f\", 100*($SIZE_BEFORE-$SIZE_AFTER)/$SIZE_BEFORE}")

echo ""
echo "Done."
echo "  Before: ${SIZE_BEFORE_MB} MB"
echo "  After:  ${SIZE_AFTER_MB} MB"
echo "  Saved:  ${SAVED} MB (${PCT}%)"
