package store

import (
	"database/sql"
	"fmt"
)

// applyAgentSchema runs the full agent-side schema-creation +
// migration sequence on a freshly-opened *sql.DB. The exact sequence
// previously lived inline in New(); it was extracted here so the
// rollover path (rollover.go) re-applies it byte-identically on the
// new DB after a roll, with no drift between the two callers.
//
// onDiskVersion is the value read from `PRAGMA user_version` BEFORE
// any schema work runs. It controls the user_version ratchet at the
// end (we only advance, never regress) so a future binary that
// already wrote a higher version doesn't get clobbered if an older
// binary somehow reaches this code.
//
// Idempotent: calling applyAgentSchema on an already-initialized DB
// is a no-op (the CREATE TABLE statements are IF NOT EXISTS, the
// ALTER TABLE statements use idempotent migrate* helpers that ignore
// "duplicate column" errors, and the INSERT statements use OR
// IGNORE / OR REPLACE).
func applyAgentSchema(db *sql.DB, onDiskVersion int) error {
	// Create schema.
	if _, err := db.Exec(schema); err != nil {
		return fmt.Errorf("creating schema: %w", err)
	}

	// Schema migrations for backward compatibility with older databases.
	// Both ALTER TABLEs are idempotent — they fail silently if the column
	// already exists. New databases get stack_hash from the schema and
	// stack_ips from this migration (unused but harmless — 0 bytes overhead).
	db.Exec(migrateAddStackHash)
	db.Exec(migrateAddStackIPs)

	// Create stack_traces table (deduplicated stack interning).
	if _, err := db.Exec(stackTracesSchema); err != nil {
		return fmt.Errorf("creating stack_traces table: %w", err)
	}

	// Add resolved frames column. Idempotent — no-op if column exists.
	db.Exec(migrateAddFramesColumn)

	// Migrate: if there are events with inline stack_ips, intern them into
	// the stack_traces table. No-op for new databases.
	migrateInlineStacks(db)

	// Create causal_chains table.
	if _, err := db.Exec(chainsSchema); err != nil {
		return fmt.Errorf("creating causal_chains table: %w", err)
	}

	// Create and populate static lookup tables (sources, ops, schema_info).
	if _, err := db.Exec(lookupSchema); err != nil {
		return fmt.Errorf("creating lookup tables: %w", err)
	}
	populateLookupTables(db)

	// Ensure schema_info reflects the running binary, even for databases
	// created by an older version (populateLookupTables skips inserts when
	// tables are already populated).
	db.Exec("INSERT OR REPLACE INTO schema_info (key, value) VALUES ('version', '0.8')")

	// v0.8 migration: add new sources and ops for existing databases.
	db.Exec("INSERT OR IGNORE INTO sources (id, name, description) VALUES (5, 'IO', 'Block I/O events')")
	db.Exec("INSERT OR IGNORE INTO sources (id, name, description) VALUES (6, 'TCP', 'TCP events')")
	db.Exec("INSERT OR IGNORE INTO sources (id, name, description) VALUES (7, 'NET', 'Network socket events')")
	db.Exec("INSERT OR IGNORE INTO ops (source_id, op_id, name, description) VALUES (1, 8, 'cudaMallocManaged', 'Unified Memory allocation')")
	db.Exec("INSERT OR IGNORE INTO ops (source_id, op_id, name, description) VALUES (4, 6, 'cuMemAllocManaged', 'Unified Memory allocation via driver API')")
	db.Exec("INSERT OR IGNORE INTO ops (source_id, op_id, name, description) VALUES (5, 1, 'block_read', 'Block device read request')")
	db.Exec("INSERT OR IGNORE INTO ops (source_id, op_id, name, description) VALUES (5, 2, 'block_write', 'Block device write request')")
	db.Exec("INSERT OR IGNORE INTO ops (source_id, op_id, name, description) VALUES (5, 3, 'block_discard', 'Block device discard/trim request')")
	db.Exec("INSERT OR IGNORE INTO ops (source_id, op_id, name, description) VALUES (6, 1, 'tcp_retransmit', 'TCP segment retransmission')")
	db.Exec("INSERT OR IGNORE INTO ops (source_id, op_id, name, description) VALUES (7, 1, 'net_send', 'Socket send/sendto syscall')")
	db.Exec("INSERT OR IGNORE INTO ops (source_id, op_id, name, description) VALUES (7, 2, 'net_recv', 'Socket recv/recvfrom syscall')")
	db.Exec("INSERT OR IGNORE INTO ops (source_id, op_id, name, description) VALUES (3, 10, 'pod_restart', 'K8s pod container restart detected')")
	db.Exec("INSERT OR IGNORE INTO ops (source_id, op_id, name, description) VALUES (3, 11, 'pod_eviction', 'K8s pod eviction detected')")
	db.Exec("INSERT OR IGNORE INTO ops (source_id, op_id, name, description) VALUES (3, 12, 'pod_oom_kill', 'K8s pod OOM kill detected')")
	db.Exec("INSERT OR IGNORE INTO schema_info (key, value) VALUES ('sessions_note', 'One row per ingero trace invocation. Correlate with events via time range.')")
	db.Exec("INSERT OR IGNORE INTO schema_info (key, value) VALUES ('process_names_note', 'PID-to-name mapping populated during trace. JOIN with events.pid for query enrichment.')")
	db.Exec("INSERT OR REPLACE INTO schema_info (key, value) VALUES ('event_aggregates_note', 'Per-minute aggregates. sum_arg0 tracks mm_page_alloc total bytes (chain engine threshold: >1GB). count-stored = discarded count.')")
	db.Exec("DELETE FROM schema_info WHERE key = 'stack_ips_note'")
	db.Exec("INSERT OR REPLACE INTO schema_info (key, value) VALUES ('stack_traces_note', 'Deduplicated stacks: events.stack_hash -> stack_traces.hash. frames column has resolved symbols. Use get_stacks MCP tool for call stack analysis.')")

	// Composite index for get_stacks MCP tool (GROUP BY source,op,stack_hash).
	db.Exec("CREATE INDEX IF NOT EXISTS idx_events_source_op_stack ON events(source, op, stack_hash)")

	// Drop redundant idx_events_source_op — the composite idx_events_source_op_stack
	// covers all (source, op) queries via SQLite leftmost-prefix matching.
	db.Exec("DROP INDEX IF EXISTS idx_events_source_op")

	// Create system_snapshots table.
	if _, err := db.Exec(snapshotsSchema); err != nil {
		return fmt.Errorf("creating system_snapshots table: %w", err)
	}
	db.Exec(snapshotsMigration)

	// Create sessions table.
	if _, err := db.Exec(sessionsSchema); err != nil {
		return fmt.Errorf("creating sessions table: %w", err)
	}

	// Create process_names table.
	if _, err := db.Exec(processNamesSchema); err != nil {
		return fmt.Errorf("creating process_names table: %w", err)
	}

	// Create event_aggregates table.
	if _, err := db.Exec(aggregatesSchema); err != nil {
		return fmt.Errorf("creating event_aggregates table: %w", err)
	}

	// Create event_aggregates_5s table.
	if _, err := db.Exec(aggregates5sSchema); err != nil {
		return fmt.Errorf("creating event_aggregates_5s table: %w", err)
	}

	// Add sum_arg0 column for mm_page_alloc total bytes.
	db.Exec(migrateAddSumArg0)

	// v0.7: Add cgroup_id column to events table.
	db.Exec(migrateAddCGroupID)
	db.Exec("CREATE INDEX IF NOT EXISTS idx_events_cgroup ON events(cgroup_id) WHERE cgroup_id != 0")

	// v0.7: Create cgroup_metadata table.
	if _, err := db.Exec(cgroupMetadataSchema); err != nil {
		return fmt.Errorf("creating cgroup_metadata table: %w", err)
	}
	db.Exec(migrateAddPodName)
	db.Exec(migrateAddNamespace)

	// v0.8: Create cgroup_schedstat table for noisy neighbor detection.
	if _, err := db.Exec(cgroupSchedstatSchema); err != nil {
		return fmt.Errorf("creating cgroup_schedstat table: %w", err)
	}

	db.Exec("INSERT OR REPLACE INTO schema_info (key, value) VALUES ('version', '0.8')")
	db.Exec("INSERT OR REPLACE INTO schema_info (key, value) VALUES ('cgroup_metadata_note', 'cgroup_id -> container_id mapping. Populated during K8s tracing. JOIN with events.cgroup_id for container context.')")

	// v0.9: Add node identity and rank columns. Idempotent.
	db.Exec(migrateAddEventsNode)
	db.Exec(migrateAddEventsRank)
	db.Exec(migrateAddEventsLocalRank)
	db.Exec(migrateAddEventsWorldSize)
	db.Exec(migrateAddSessionsNode)
	db.Exec(migrateAddSessionsRank)
	db.Exec(migrateAddSessionsLocalRank)
	db.Exec(migrateAddSessionsWorldSize)
	db.Exec(migrateAddChainsNode)

	db.Exec("INSERT OR REPLACE INTO schema_info (key, value) VALUES ('version', '0.9')")
	db.Exec("INSERT OR REPLACE INTO schema_info (key, value) VALUES ('node_note', 'Node identity for multi-node correlation. events.id format: {node}:{seq}. causal_chains.id format: {node}:{descriptor}.')")

	// v0.10: Add comm column to events table.
	db.Exec(migrateAddComm)
	if hasEventsCommColumn(db) {
		db.Exec("INSERT OR REPLACE INTO schema_info (key, value) VALUES ('version', '0.10')")
		db.Exec("INSERT OR REPLACE INTO schema_info (key, value) VALUES ('comm_note', 'events.comm captured kernel-side at event time. Empty for pre-v0.10 rows (fall back to process_names LEFT JOIN for legacy display).')")
	}

	// Ratchet PRAGMA user_version forward.
	if onDiskVersion < CurrentUserVersion {
		db.Exec(fmt.Sprintf("PRAGMA user_version = %d", CurrentUserVersion))
	}
	return nil
}
