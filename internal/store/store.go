// Package store provides SQLite-based persistent storage for events.
//
// Events are buffered in memory and batch-flushed to SQLite for performance.
// WAL mode enables concurrent reads while the writer flushes batches.
// Rolling 7-day retention keeps the database from growing unbounded.
package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"strings"
	"sync"
	"time"

	_ "modernc.org/sqlite"

	"github.com/ingero-io/ingero/pkg/events"
)

const (
	// DefaultBatchSize is the max events buffered before a flush.
	DefaultBatchSize = 1000

	// DefaultFlushInterval is the max time between flushes.
	DefaultFlushInterval = 100 * time.Millisecond

	// DefaultRetention is how long events are kept before pruning.
	DefaultRetention = 7 * 24 * time.Hour

	// DefaultPruneInterval is how often we check for expired events.
	DefaultPruneInterval = 1 * time.Hour
)

const schema = `
CREATE TABLE IF NOT EXISTS events (
	id         INTEGER PRIMARY KEY AUTOINCREMENT,
	timestamp  INTEGER NOT NULL,  -- unix nanos
	pid        INTEGER NOT NULL,
	tid        INTEGER NOT NULL,
	source     INTEGER NOT NULL,
	op         INTEGER NOT NULL,
	duration   INTEGER NOT NULL,  -- nanos
	gpu_id     INTEGER NOT NULL DEFAULT 0,
	arg0       INTEGER NOT NULL DEFAULT 0,
	arg1       INTEGER NOT NULL DEFAULT 0,
	ret_code   INTEGER NOT NULL DEFAULT 0,
	stack_ips  TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_pid ON events(pid);
CREATE INDEX IF NOT EXISTS idx_events_source_op ON events(source, op);
`

const chainsSchema = `
CREATE TABLE IF NOT EXISTS causal_chains (
	id              TEXT PRIMARY KEY,       -- e.g. "chain-001"
	detected_at     INTEGER NOT NULL,       -- unix nanos (when chain was detected)
	severity        TEXT NOT NULL,           -- HIGH, MEDIUM, LOW
	summary         TEXT NOT NULL,           -- one-line description
	root_cause      TEXT NOT NULL,           -- human-readable root cause
	explanation     TEXT NOT NULL,           -- paragraph explaining the chain
	recommendations TEXT NOT NULL DEFAULT '',-- JSON array of strings
	cuda_op         TEXT NOT NULL DEFAULT '',-- affected CUDA/driver op
	cuda_p99_us     INTEGER NOT NULL DEFAULT 0,
	cuda_p50_us     INTEGER NOT NULL DEFAULT 0,
	tail_ratio      REAL NOT NULL DEFAULT 0,
	timeline        TEXT NOT NULL DEFAULT '' -- JSON array of timeline events
);

CREATE INDEX IF NOT EXISTS idx_chains_detected_at ON causal_chains(detected_at);
CREATE INDEX IF NOT EXISTS idx_chains_severity ON causal_chains(severity);
`

const snapshotsSchema = `
CREATE TABLE IF NOT EXISTS system_snapshots (
	id         INTEGER PRIMARY KEY AUTOINCREMENT,
	timestamp  INTEGER NOT NULL,  -- unix nanos
	cpu_pct    REAL    NOT NULL,  -- CPU utilization %
	mem_pct    REAL    NOT NULL,  -- memory used %
	mem_avail  INTEGER NOT NULL,  -- available MB
	swap_mb    INTEGER NOT NULL,  -- swap used MB
	load_avg   REAL    NOT NULL   -- 1-minute load average
);
CREATE INDEX IF NOT EXISTS idx_snapshots_timestamp ON system_snapshots(timestamp);
`

const sessionsSchema = `
CREATE TABLE IF NOT EXISTS sessions (
	id         INTEGER PRIMARY KEY AUTOINCREMENT,
	started_at INTEGER NOT NULL,
	stopped_at INTEGER NOT NULL DEFAULT 0,
	gpu_model  TEXT NOT NULL DEFAULT '',
	gpu_driver TEXT NOT NULL DEFAULT '',
	cpu_model  TEXT NOT NULL DEFAULT '',
	cpu_cores  INTEGER NOT NULL DEFAULT 0,
	mem_total  INTEGER NOT NULL DEFAULT 0,
	kernel     TEXT NOT NULL DEFAULT '',
	os_release TEXT NOT NULL DEFAULT '',
	cuda_ver   TEXT NOT NULL DEFAULT '',
	python_ver TEXT NOT NULL DEFAULT '',
	ingero_ver TEXT NOT NULL DEFAULT '',
	pid_filter TEXT NOT NULL DEFAULT '',
	flags      TEXT NOT NULL DEFAULT ''
);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);
`

// lookupSchema creates static reference tables that make the DB self-describing.
// Any SQL client can JOIN against these to get human-readable names.
//
// Example:
//
//	SELECT e.id, s.name AS source, o.name AS op, e.duration/1000 AS dur_us, e.pid
//	FROM events e
//	JOIN sources s ON e.source = s.id
//	JOIN ops o ON e.source = o.source_id AND e.op = o.op_id
//	ORDER BY e.timestamp DESC LIMIT 20;
const lookupSchema = `
CREATE TABLE IF NOT EXISTS sources (
	id          INTEGER PRIMARY KEY,
	name        TEXT NOT NULL,
	description TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS ops (
	source_id   INTEGER NOT NULL,
	op_id       INTEGER NOT NULL,
	name        TEXT NOT NULL,
	description TEXT NOT NULL,
	PRIMARY KEY (source_id, op_id)
);

CREATE TABLE IF NOT EXISTS schema_info (
	key   TEXT PRIMARY KEY,
	value TEXT NOT NULL
);
`

// populateLookupTables inserts reference data if the tables are empty.
func populateLookupTables(db *sql.DB) {
	var count int
	db.QueryRow("SELECT COUNT(*) FROM sources").Scan(&count)
	if count > 0 {
		return // already populated
	}

	tx, err := db.Begin()
	if err != nil {
		return
	}

	// Sources — matches EVENT_SRC_* in bpf/common.bpf.h
	sources := []struct {
		id   int
		name string
		desc string
	}{
		{1, "CUDA", "CUDA Runtime API (libcudart.so) — uprobes"},
		{2, "NVIDIA", "NVIDIA kernel driver (nvidia.ko) — reserved, not currently traced"},
		{3, "HOST", "Host kernel events — tracepoints (scheduler, memory, process lifecycle)"},
		{4, "DRIVER", "CUDA Driver API (libcuda.so) — uprobes"},
	}
	for _, s := range sources {
		tx.Exec("INSERT INTO sources (id, name, description) VALUES (?, ?, ?)", s.id, s.name, s.desc)
	}

	// Ops — matches CUDA_OP_*, HOST_OP_*, DRIVER_OP_* in bpf/common.bpf.h
	ops := []struct {
		src  int
		op   int
		name string
		desc string
	}{
		// CUDA Runtime (source=1)
		{1, 1, "cudaMalloc", "GPU memory allocation"},
		{1, 2, "cudaFree", "GPU memory free"},
		{1, 3, "cudaLaunchKernel", "Launch GPU kernel via runtime API"},
		{1, 4, "cudaMemcpy", "Synchronous host<->device memory copy"},
		{1, 5, "cudaStreamSync", "Wait for all ops in a CUDA stream to complete"},
		{1, 6, "cudaDeviceSync", "Wait for all GPU work on device to complete"},
		// Host kernel (source=3)
		{3, 1, "sched_switch", "CPU context switch — thread was descheduled"},
		{3, 2, "sched_wakeup", "Thread woken up and enqueued to run queue"},
		{3, 3, "mm_page_alloc", "Kernel page allocation (arg0 = bytes)"},
		{3, 4, "oom_kill", "Out-of-memory killer invoked"},
		{3, 5, "process_exec", "New process executed (execve)"},
		{3, 6, "process_exit", "Process exited"},
		{3, 7, "process_fork", "Process forked (child auto-enrolled for tracing)"},
		// CUDA Driver (source=4)
		{4, 1, "cuLaunchKernel", "Launch GPU kernel via driver API (cuBLAS/cuDNN path)"},
		{4, 2, "cuMemcpy", "Synchronous memory copy via driver API"},
		{4, 3, "cuMemcpyAsync", "Asynchronous memory copy via driver API"},
		{4, 4, "cuCtxSynchronize", "Wait for all GPU work in context to complete"},
		{4, 5, "cuMemAlloc", "GPU memory allocation via driver API"},
	}
	for _, o := range ops {
		tx.Exec("INSERT INTO ops (source_id, op_id, name, description) VALUES (?, ?, ?, ?)", o.src, o.op, o.name, o.desc)
	}

	// Schema info — units, version, helpful context
	info := []struct{ k, v string }{
		{"version", "0.6"},
		{"timestamp_unit", "nanoseconds (Unix epoch)"},
		{"duration_unit", "nanoseconds"},
		{"arg0_note", "Operation-specific: byte size for alloc/memcpy, kernel function pointer for launch"},
		{"arg1_note", "Operation-specific: memcpy direction for cudaMemcpy, unused for most ops"},
		{"ret_code_note", "CUDA return code (0 = cudaSuccess). Host events always 0."},
		{"stack_ips_note", "JSON array of hex instruction pointers, e.g. [\"0x7f1234\",\"0x7f5678\"]. Empty if --stack not used."},
		{"example_query", "SELECT e.id, s.name AS source, o.name AS op, e.duration/1000 AS dur_us, e.pid FROM events e JOIN sources s ON e.source = s.id JOIN ops o ON e.source = o.source_id AND e.op = o.op_id ORDER BY e.timestamp DESC LIMIT 20"},
		{"system_snapshots_note", "System metrics sampled every 1s during recording. Replay with correlator for post-hoc causal chain analysis."},
		{"sessions_note", "One row per 'ingero trace' invocation. Correlate with events via time range."},
	}
	for _, kv := range info {
		tx.Exec("INSERT INTO schema_info (key, value) VALUES (?, ?)", kv.k, kv.v)
	}

	tx.Commit()
}

// migrateSchema adds columns that may be missing in older databases.
const migrateAddStackIPs = `ALTER TABLE events ADD COLUMN stack_ips TEXT NOT NULL DEFAULT ''`

// Store provides persistent event storage backed by SQLite.
type Store struct {
	db         *sql.DB
	dbPath     string
	insertCh   chan events.Event
	snapshotCh chan SystemSnapshot
	retention  time.Duration

	mu      sync.Mutex
	closed  bool
}

// QueryParams defines filters for querying stored events.
type QueryParams struct {
	Since  time.Duration // events from the last N duration
	From   time.Time     // events after this time (overrides Since if set)
	To     time.Time     // events before this time
	PID    uint32        // filter by single PID (0 = all). Deprecated: use PIDs.
	PIDs   []uint32      // filter by multiple PIDs (empty = all). Takes precedence over PID.
	Source uint8         // filter by source (0 = all)
	Op     uint8         // filter by op (0 = all, only used if Source > 0)
	Limit  int           // max results (0 = 10000)
}

// SystemSnapshot holds system metrics for a single point in time.
// Written once per second during --record, queried for post-hoc causal analysis.
type SystemSnapshot struct {
	Timestamp  time.Time
	CPUPercent float64
	MemUsedPct float64
	MemAvailMB int64
	SwapUsedMB int64
	LoadAvg1   float64
}

// Session records hardware/software metadata for a single 'ingero trace' run.
// One row per trace invocation. Events are correlated by time range
// (WHERE timestamp BETWEEN started_at AND stopped_at), not by FK.
type Session struct {
	ID        int64
	StartedAt time.Time
	StoppedAt time.Time // zero value = still running
	GPUModel  string
	GPUDriver string
	CPUModel  string
	CPUCores  int
	MemTotal  int64 // MB
	Kernel    string
	OSRelease string
	CUDAVer   string
	PythonVer string
	IngeroVer string
	PIDFilter string
	Flags     string
}

// DefaultDBPath returns the default database path (~/.ingero/ingero.db).
//
// When running via sudo, resolves the invoking user's home directory
// (via SUDO_USER) so that 'sudo ingero trace' and 'ingero explain'
// use the same database file. Without this, trace would write to
// /root/.ingero/ while explain (no sudo) reads from /home/<user>/.ingero/.
func DefaultDBPath() string {
	home, err := os.UserHomeDir()
	if err != nil {
		home = "/tmp"
	}

	// If running as root via sudo, use the invoking user's home instead
	// of /root, so non-sudo commands can read the same DB.
	if sudoUser := os.Getenv("SUDO_USER"); sudoUser != "" && os.Getuid() == 0 {
		if sudoHome := lookupHome(sudoUser); sudoHome != "" {
			home = sudoHome
		}
	}

	return filepath.Join(home, ".ingero", "ingero.db")
}

// chownForSudoUser changes ownership of path to the SUDO_USER when running
// as root via sudo. This ensures non-sudo commands can access the DB files.
// Best-effort — silently ignores errors.
func chownForSudoUser(path string) {
	sudoUser := os.Getenv("SUDO_USER")
	if sudoUser == "" || os.Getuid() != 0 {
		return
	}
	uid, gid := lookupUID(sudoUser)
	if uid >= 0 {
		os.Chown(path, uid, gid)
	}
}

// lookupUID resolves a username to UID and GID via /etc/passwd.
// Returns -1, -1 if the user cannot be found.
func lookupUID(username string) (int, int) {
	data, err := os.ReadFile("/etc/passwd")
	if err != nil {
		return -1, -1
	}

	prefix := username + ":"
	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, prefix) {
			continue
		}
		// Format: user:x:uid:gid:gecos:home:shell
		fields := strings.SplitN(line, ":", 7)
		if len(fields) >= 4 {
			uid := 0
			gid := 0
			fmt.Sscanf(fields[2], "%d", &uid)
			fmt.Sscanf(fields[3], "%d", &gid)
			return uid, gid
		}
	}
	return -1, -1
}

// lookupHome resolves a username to their home directory via /etc/passwd.
// Returns "" if the user cannot be found. Pure Go, no CGO dependency.
func lookupHome(username string) string {
	data, err := os.ReadFile("/etc/passwd")
	if err != nil {
		return ""
	}

	prefix := username + ":"
	for _, line := range strings.Split(string(data), "\n") {
		if !strings.HasPrefix(line, prefix) {
			continue
		}
		// Format: user:x:uid:gid:gecos:home:shell
		fields := strings.SplitN(line, ":", 7)
		if len(fields) >= 6 {
			return fields[5]
		}
	}
	return ""
}

// New opens or creates a SQLite database at dbPath.
// Use ":memory:" for in-memory testing.
func New(dbPath string) (*Store, error) {
	// Create parent directory if needed.
	if dbPath != ":memory:" {
		dir := filepath.Dir(dbPath)
		if err := os.MkdirAll(dir, 0o755); err != nil {
			return nil, fmt.Errorf("creating db directory %s: %w", dir, err)
		}
		// When running as root via sudo, chown the directory to the
		// invoking user so non-sudo commands can read/write the DB.
		chownForSudoUser(dir)
	}

	// For in-memory databases, use shared cache so all connections from the
	// database/sql pool share the same database.  Without this, each pooled
	// connection gets its own empty in-memory DB and won't see the schema
	// created by another connection (causes "no such table" under -race).
	dsn := dbPath
	if dsn == ":memory:" {
		dsn = "file::memory:?cache=shared"
	}

	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("opening database: %w", err)
	}

	// Enable WAL mode for concurrent reads during writes.
	if _, err := db.Exec("PRAGMA journal_mode=WAL"); err != nil {
		db.Close()
		return nil, fmt.Errorf("setting WAL mode: %w", err)
	}

	// Create schema.
	if _, err := db.Exec(schema); err != nil {
		db.Close()
		return nil, fmt.Errorf("creating schema: %w", err)
	}

	// Migrate: add stack_ips column if missing (older databases).
	db.Exec(migrateAddStackIPs) // Ignore error — column already exists is expected.

	// Create causal_chains table.
	if _, err := db.Exec(chainsSchema); err != nil {
		db.Close()
		return nil, fmt.Errorf("creating causal_chains table: %w", err)
	}

	// Create and populate static lookup tables (sources, ops, schema_info).
	if _, err := db.Exec(lookupSchema); err != nil {
		db.Close()
		return nil, fmt.Errorf("creating lookup tables: %w", err)
	}
	populateLookupTables(db)

	// Ensure schema_info reflects the running binary, even for databases
	// created by an older version (populateLookupTables skips inserts when
	// tables are already populated).
	db.Exec("INSERT OR REPLACE INTO schema_info (key, value) VALUES ('version', '0.6')")
	db.Exec("INSERT OR IGNORE INTO schema_info (key, value) VALUES ('sessions_note', 'One row per ingero trace invocation. Correlate with events via time range.')")

	// Create system_snapshots table (metrics sampled every 1s during recording).
	if _, err := db.Exec(snapshotsSchema); err != nil {
		db.Close()
		return nil, fmt.Errorf("creating system_snapshots table: %w", err)
	}

	// Create sessions table (one row per 'ingero trace' invocation).
	if _, err := db.Exec(sessionsSchema); err != nil {
		db.Close()
		return nil, fmt.Errorf("creating sessions table: %w", err)
	}

	// When running as root via sudo, chown the DB file to the invoking
	// user so non-sudo commands (explain, query) can open it.
	if dbPath != ":memory:" {
		chownForSudoUser(dbPath)
	}

	s := &Store{
		db:         db,
		dbPath:     dbPath,
		insertCh:   make(chan events.Event, DefaultBatchSize*2),
		snapshotCh: make(chan SystemSnapshot, 64), // 1 snapshot/sec, 64 = ~1 min buffer
		retention:  DefaultRetention,
	}

	return s, nil
}

// Record enqueues an event for async writing. Non-blocking — drops if buffer full.
func (s *Store) Record(evt events.Event) {
	select {
	case s.insertCh <- evt:
	default:
		// Buffer full — drop event rather than block the caller.
	}
}

// Run starts the background flush and prune loops. Blocks until ctx is cancelled.
func (s *Store) Run(ctx context.Context) {
	flushTicker := time.NewTicker(DefaultFlushInterval)
	defer flushTicker.Stop()

	pruneTicker := time.NewTicker(DefaultPruneInterval)
	defer pruneTicker.Stop()

	// Prune on startup.
	s.pruneOld()

	var batch []events.Event

	for {
		select {
		case evt := <-s.insertCh:
			batch = append(batch, evt)
			if len(batch) >= DefaultBatchSize {
				s.flushBatch(batch)
				batch = batch[:0]
			}

		case snap := <-s.snapshotCh:
			s.writeSnapshot(snap)

		case <-flushTicker.C:
			if len(batch) > 0 {
				s.flushBatch(batch)
				batch = batch[:0]
			}

		case <-pruneTicker.C:
			s.pruneOld()

		case <-ctx.Done():
			// Final flush.
			// Drain remaining from channels.
			for {
				select {
				case evt := <-s.insertCh:
					batch = append(batch, evt)
				case snap := <-s.snapshotCh:
					s.writeSnapshot(snap)
				default:
					goto done
				}
			}
		done:
			if len(batch) > 0 {
				s.flushBatch(batch)
			}
			return
		}
	}
}

// flushBatch writes a batch of events in a single transaction.
func (s *Store) flushBatch(batch []events.Event) {
	if len(batch) == 0 {
		return
	}

	tx, err := s.db.Begin()
	if err != nil {
		return
	}

	stmt, err := tx.Prepare(`INSERT INTO events (timestamp, pid, tid, source, op, duration, gpu_id, arg0, arg1, ret_code, stack_ips)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		tx.Rollback()
		return
	}
	defer stmt.Close()

	for _, evt := range batch {
		stackJSON := serializeStackIPs(evt.Stack)
		stmt.Exec(
			evt.Timestamp.UnixNano(),
			evt.PID,
			evt.TID,
			uint8(evt.Source),
			evt.Op,
			int64(evt.Duration),
			evt.GPUID,
			evt.Args[0],
			evt.Args[1],
			evt.RetCode,
			stackJSON,
		)
	}

	tx.Commit()
}

// pruneOld deletes events, system snapshots, and sessions older than the retention period.
func (s *Store) pruneOld() {
	cutoff := time.Now().Add(-s.retention).UnixNano()
	s.db.Exec("DELETE FROM events WHERE timestamp < ?", cutoff)
	s.db.Exec("DELETE FROM system_snapshots WHERE timestamp < ?", cutoff)
	s.db.Exec("DELETE FROM sessions WHERE started_at < ?", cutoff)
}

// appendPIDFilter appends a PID filter clause to a SQL query.
// Uses PIDs if set, falls back to deprecated PID field for backward compat.
// colPrefix is "" for unqualified column or "e." for table-qualified queries.
//
// Go idiom: variadic args via append — each placeholder "?" gets its own
// parameterized value, preventing SQL injection even with user-supplied PIDs.
func appendPIDFilter(query string, args []interface{}, q QueryParams, colPrefix string) (string, []interface{}) {
	pids := q.PIDs
	if len(pids) == 0 && q.PID > 0 {
		pids = []uint32{q.PID} // backward compat
	}
	if len(pids) == 1 {
		query += fmt.Sprintf(" AND %spid = ?", colPrefix)
		args = append(args, pids[0])
	} else if len(pids) > 1 {
		placeholders := make([]string, len(pids))
		for i, p := range pids {
			placeholders[i] = "?"
			args = append(args, p)
		}
		query += fmt.Sprintf(" AND %spid IN (%s)", colPrefix, strings.Join(placeholders, ","))
	}
	return query, args
}

// Query retrieves events matching the given parameters.
func (s *Store) Query(q QueryParams) ([]events.Event, error) {
	query := "SELECT timestamp, pid, tid, source, op, duration, gpu_id, arg0, arg1, ret_code, stack_ips FROM events WHERE 1=1"
	var args []interface{}

	// Time range.
	if !q.From.IsZero() {
		query += " AND timestamp >= ?"
		args = append(args, q.From.UnixNano())
	} else if q.Since > 0 {
		query += " AND timestamp >= ?"
		args = append(args, time.Now().Add(-q.Since).UnixNano())
	}

	if !q.To.IsZero() {
		query += " AND timestamp <= ?"
		args = append(args, q.To.UnixNano())
	}

	// PID filter (single or multi).
	query, args = appendPIDFilter(query, args, q, "")

	// Source filter.
	if q.Source > 0 {
		query += " AND source = ?"
		args = append(args, q.Source)
	}

	// Op filter (only if Source is set).
	if q.Source > 0 && q.Op > 0 {
		query += " AND op = ?"
		args = append(args, q.Op)
	}

	// Fetch the newest events first (DESC) so the LIMIT keeps the most
	// recent data rather than the oldest.  We reverse the slice afterward
	// to return chronological order.
	query += " ORDER BY timestamp DESC"

	limit := q.Limit
	if limit <= 0 {
		limit = 10000
	}
	query += " LIMIT ?"
	args = append(args, limit)

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("querying events: %w", err)
	}
	defer rows.Close()

	var result []events.Event
	for rows.Next() {
		var (
			tsNanos    int64
			pid, tid   uint32
			source, op uint8
			durNanos   int64
			gpuID      uint32
			arg0, arg1 uint64
			retCode    int32
			stackJSON  string
		)
		if err := rows.Scan(&tsNanos, &pid, &tid, &source, &op, &durNanos, &gpuID, &arg0, &arg1, &retCode, &stackJSON); err != nil {
			return nil, fmt.Errorf("scanning event row: %w", err)
		}
		evt := events.Event{
			Timestamp: time.Unix(0, tsNanos),
			PID:       pid,
			TID:       tid,
			Source:    events.Source(source),
			Op:        op,
			Duration:  time.Duration(durNanos),
			GPUID:     gpuID,
			Args:      [2]uint64{arg0, arg1},
			RetCode:   retCode,
		}
		evt.Stack = deserializeStackIPs(stackJSON)
		result = append(result, evt)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	// Reverse to chronological order (oldest first).
	for i, j := 0, len(result)-1; i < j; i, j = i+1, j-1 {
		result[i], result[j] = result[j], result[i]
	}

	return result, nil
}

// RichEvent is an event enriched with human-readable names from the lookup tables.
// Produced by QueryRich using SELECT JOIN against sources and ops tables.
type RichEvent struct {
	events.Event
	SourceName string
	SourceDesc string
	OpName     string
	OpDesc     string
}

// QueryRich retrieves events with JOIN against lookup tables for human-readable names.
// Use this for AI/MCP output where self-describing data matters.
func (s *Store) QueryRich(q QueryParams) ([]RichEvent, error) {
	query := `SELECT e.timestamp, e.pid, e.tid, e.source, e.op, e.duration,
		e.gpu_id, e.arg0, e.arg1, e.ret_code, e.stack_ips,
		COALESCE(s.name, 'SRC_' || e.source),
		COALESCE(s.description, ''),
		COALESCE(o.name, 'OP_' || e.op),
		COALESCE(o.description, '')
	FROM events e
	LEFT JOIN sources s ON e.source = s.id
	LEFT JOIN ops o ON e.source = o.source_id AND e.op = o.op_id
	WHERE 1=1`
	var args []interface{}

	if !q.From.IsZero() {
		query += " AND e.timestamp >= ?"
		args = append(args, q.From.UnixNano())
	} else if q.Since > 0 {
		query += " AND e.timestamp >= ?"
		args = append(args, time.Now().Add(-q.Since).UnixNano())
	}
	if !q.To.IsZero() {
		query += " AND e.timestamp <= ?"
		args = append(args, q.To.UnixNano())
	}
	query, args = appendPIDFilter(query, args, q, "e.")
	if q.Source > 0 {
		query += " AND e.source = ?"
		args = append(args, q.Source)
	}
	if q.Source > 0 && q.Op > 0 {
		query += " AND e.op = ?"
		args = append(args, q.Op)
	}

	query += " ORDER BY e.timestamp DESC"

	limit := q.Limit
	if limit <= 0 {
		limit = 10000
	}
	query += " LIMIT ?"
	args = append(args, limit)

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("querying events (rich): %w", err)
	}
	defer rows.Close()

	var result []RichEvent
	for rows.Next() {
		var (
			tsNanos    int64
			pid, tid   uint32
			source, op uint8
			durNanos   int64
			gpuID      uint32
			arg0, arg1 uint64
			retCode    int32
			stackJSON  string
			srcName, srcDesc, opName, opDesc string
		)
		if err := rows.Scan(&tsNanos, &pid, &tid, &source, &op, &durNanos,
			&gpuID, &arg0, &arg1, &retCode, &stackJSON,
			&srcName, &srcDesc, &opName, &opDesc); err != nil {
			return nil, fmt.Errorf("scanning rich event: %w", err)
		}
		re := RichEvent{
			Event: events.Event{
				Timestamp: time.Unix(0, tsNanos),
				PID:       pid,
				TID:       tid,
				Source:    events.Source(source),
				Op:        op,
				Duration:  time.Duration(durNanos),
				GPUID:     gpuID,
				Args:      [2]uint64{arg0, arg1},
				RetCode:   retCode,
			},
			SourceName: srcName,
			SourceDesc: srcDesc,
			OpName:     opName,
			OpDesc:     opDesc,
		}
		re.Event.Stack = deserializeStackIPs(stackJSON)
		result = append(result, re)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	// Reverse to chronological order.
	for i, j := 0, len(result)-1; i < j; i, j = i+1, j-1 {
		result[i], result[j] = result[j], result[i]
	}

	return result, nil
}

// OpDescriptions returns op name → description from the ops lookup table.
func (s *Store) OpDescriptions() map[string]string {
	m := make(map[string]string)
	rows, err := s.db.Query("SELECT name, description FROM ops")
	if err != nil {
		return m
	}
	defer rows.Close()
	for rows.Next() {
		var name, desc string
		if err := rows.Scan(&name, &desc); err == nil {
			m[name] = desc
		}
	}
	return m
}

// SchemaInfo returns key-value metadata from the schema_info table.
func (s *Store) SchemaInfo() map[string]string {
	m := make(map[string]string)
	rows, err := s.db.Query("SELECT key, value FROM schema_info")
	if err != nil {
		return m
	}
	defer rows.Close()
	for rows.Next() {
		var k, v string
		if err := rows.Scan(&k, &v); err == nil {
			m[k] = v
		}
	}
	return m
}

// Count returns the total number of events in the database.
func (s *Store) Count() (int64, error) {
	var count int64
	err := s.db.QueryRow("SELECT COUNT(*) FROM events").Scan(&count)
	return count, err
}

// Close closes the database connection.
func (s *Store) Close() error {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.closed {
		return nil
	}
	s.closed = true

	err := s.db.Close()

	// Chown WAL/SHM files that SQLite created during this session.
	if s.dbPath != "" && s.dbPath != ":memory:" {
		chownForSudoUser(s.dbPath)
		chownForSudoUser(s.dbPath + "-wal")
		chownForSudoUser(s.dbPath + "-shm")
	}

	return err
}

// StoredChain is a causal chain as persisted in SQLite.
// Mirrors correlate.CausalChain but uses plain types for storage independence.
type StoredChain struct {
	ID              string
	DetectedAt      time.Time
	Severity        string
	Summary         string
	RootCause       string
	Explanation     string
	Recommendations []string
	CUDAOp          string
	CUDAP99us       int64
	CUDAP50us       int64
	TailRatio       float64
	Timeline        []TimelineEntry
}

// TimelineEntry is a single event in a causal chain timeline.
type TimelineEntry struct {
	Layer    string `json:"layer"`
	Op       string `json:"op"`
	Detail   string `json:"detail"`
	DurationUS int64 `json:"dur_us,omitempty"`
}

// RecordChains upserts causal chains into the database.
// Uses INSERT OR REPLACE so repeated detections of the same chain ID update in place.
func (s *Store) RecordChains(chains []StoredChain) {
	if len(chains) == 0 {
		return
	}

	tx, err := s.db.Begin()
	if err != nil {
		return
	}

	stmt, err := tx.Prepare(`INSERT OR REPLACE INTO causal_chains
		(id, detected_at, severity, summary, root_cause, explanation, recommendations, cuda_op, cuda_p99_us, cuda_p50_us, tail_ratio, timeline)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		tx.Rollback()
		return
	}
	defer stmt.Close()

	for _, ch := range chains {
		recs := ch.Recommendations
		if recs == nil {
			recs = []string{}
		}
		recsJSON, _ := json.Marshal(recs)
		tl := ch.Timeline
		if tl == nil {
			tl = []TimelineEntry{}
		}
		tlJSON, _ := json.Marshal(tl)
		stmt.Exec(
			ch.ID,
			ch.DetectedAt.UnixNano(),
			ch.Severity,
			ch.Summary,
			ch.RootCause,
			ch.Explanation,
			string(recsJSON),
			ch.CUDAOp,
			ch.CUDAP99us,
			ch.CUDAP50us,
			ch.TailRatio,
			string(tlJSON),
		)
	}

	tx.Commit()
}

// QueryChains retrieves causal chains, optionally filtered by time range.
func (s *Store) QueryChains(since time.Duration) ([]StoredChain, error) {
	query := "SELECT id, detected_at, severity, summary, root_cause, explanation, recommendations, cuda_op, cuda_p99_us, cuda_p50_us, tail_ratio, timeline FROM causal_chains"
	var args []interface{}

	if since > 0 {
		query += " WHERE detected_at >= ?"
		args = append(args, time.Now().Add(-since).UnixNano())
	}
	query += " ORDER BY detected_at DESC"

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("querying causal_chains: %w", err)
	}
	defer rows.Close()

	var result []StoredChain
	for rows.Next() {
		var (
			ch        StoredChain
			tsNanos   int64
			recsJSON  string
			tlJSON    string
		)
		if err := rows.Scan(&ch.ID, &tsNanos, &ch.Severity, &ch.Summary, &ch.RootCause,
			&ch.Explanation, &recsJSON, &ch.CUDAOp, &ch.CUDAP99us, &ch.CUDAP50us,
			&ch.TailRatio, &tlJSON); err != nil {
			return nil, fmt.Errorf("scanning chain row: %w", err)
		}
		ch.DetectedAt = time.Unix(0, tsNanos)
		json.Unmarshal([]byte(recsJSON), &ch.Recommendations)
		json.Unmarshal([]byte(tlJSON), &ch.Timeline)
		result = append(result, ch)
	}
	return result, rows.Err()
}

// RecordSnapshot enqueues a system snapshot for async writing.
// Called once per second during --record. Written in the Run() loop
// to avoid SQLite write contention with the event batch writer.
func (s *Store) RecordSnapshot(snap SystemSnapshot) {
	select {
	case s.snapshotCh <- snap:
	default:
		// Buffer full — drop snapshot rather than block.
	}
}

// writeSnapshot inserts a single system snapshot. Called from Run() goroutine
// to serialize with event batch writes and avoid SQLite write contention.
func (s *Store) writeSnapshot(snap SystemSnapshot) {
	s.db.Exec(`INSERT INTO system_snapshots (timestamp, cpu_pct, mem_pct, mem_avail, swap_mb, load_avg)
		VALUES (?, ?, ?, ?, ?, ?)`,
		snap.Timestamp.UnixNano(),
		snap.CPUPercent,
		snap.MemUsedPct,
		snap.MemAvailMB,
		snap.SwapUsedMB,
		snap.LoadAvg1,
	)
}

// QuerySnapshots retrieves system snapshots, optionally filtered by time range.
// Use q.Since for "last N duration" or q.From/q.To for absolute ranges.
// Results are chronological (oldest first) for replay into the correlator.
func (s *Store) QuerySnapshots(q QueryParams) ([]SystemSnapshot, error) {
	query := "SELECT timestamp, cpu_pct, mem_pct, mem_avail, swap_mb, load_avg FROM system_snapshots WHERE 1=1"
	var args []interface{}

	if !q.From.IsZero() {
		query += " AND timestamp >= ?"
		args = append(args, q.From.UnixNano())
	} else if q.Since > 0 {
		query += " AND timestamp >= ?"
		args = append(args, time.Now().Add(-q.Since).UnixNano())
	}

	if !q.To.IsZero() {
		query += " AND timestamp <= ?"
		args = append(args, q.To.UnixNano())
	}

	query += " ORDER BY timestamp ASC"

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("querying system_snapshots: %w", err)
	}
	defer rows.Close()

	var result []SystemSnapshot
	for rows.Next() {
		var (
			tsNanos int64
			snap    SystemSnapshot
		)
		if err := rows.Scan(&tsNanos, &snap.CPUPercent, &snap.MemUsedPct, &snap.MemAvailMB, &snap.SwapUsedMB, &snap.LoadAvg1); err != nil {
			return nil, fmt.Errorf("scanning snapshot row: %w", err)
		}
		snap.Timestamp = time.Unix(0, tsNanos)
		result = append(result, snap)
	}
	return result, rows.Err()
}

// StartSession inserts a new session row and returns the auto-increment ID.
// Called at the beginning of 'ingero trace' to record HW/SW context.
func (s *Store) StartSession(sess Session) (int64, error) {
	result, err := s.db.Exec(`INSERT INTO sessions
		(started_at, gpu_model, gpu_driver, cpu_model, cpu_cores, mem_total,
		 kernel, os_release, cuda_ver, python_ver, ingero_ver, pid_filter, flags)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		sess.StartedAt.UnixNano(),
		sess.GPUModel, sess.GPUDriver,
		sess.CPUModel, sess.CPUCores, sess.MemTotal,
		sess.Kernel, sess.OSRelease,
		sess.CUDAVer, sess.PythonVer, sess.IngeroVer,
		sess.PIDFilter, sess.Flags,
	)
	if err != nil {
		return 0, fmt.Errorf("inserting session: %w", err)
	}
	return result.LastInsertId()
}

// StopSession updates a session's stopped_at timestamp.
// Called when 'ingero trace' exits (deferred).
func (s *Store) StopSession(id int64, stoppedAt time.Time) error {
	_, err := s.db.Exec("UPDATE sessions SET stopped_at = ? WHERE id = ?",
		stoppedAt.UnixNano(), id)
	if err != nil {
		return fmt.Errorf("stopping session %d: %w", id, err)
	}
	return nil
}

// QuerySessions retrieves sessions, optionally filtered by time range.
func (s *Store) QuerySessions(since time.Duration) ([]Session, error) {
	query := `SELECT id, started_at, stopped_at, gpu_model, gpu_driver,
		cpu_model, cpu_cores, mem_total, kernel, os_release,
		cuda_ver, python_ver, ingero_ver, pid_filter, flags
		FROM sessions`
	var args []interface{}

	if since > 0 {
		query += " WHERE started_at >= ?"
		args = append(args, time.Now().Add(-since).UnixNano())
	}
	query += " ORDER BY started_at DESC"

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("querying sessions: %w", err)
	}
	defer rows.Close()

	var result []Session
	for rows.Next() {
		var (
			sess              Session
			startNanos        int64
			stopNanos         int64
		)
		if err := rows.Scan(&sess.ID, &startNanos, &stopNanos,
			&sess.GPUModel, &sess.GPUDriver,
			&sess.CPUModel, &sess.CPUCores, &sess.MemTotal,
			&sess.Kernel, &sess.OSRelease,
			&sess.CUDAVer, &sess.PythonVer, &sess.IngeroVer,
			&sess.PIDFilter, &sess.Flags); err != nil {
			return nil, fmt.Errorf("scanning session row: %w", err)
		}
		sess.StartedAt = time.Unix(0, startNanos)
		if stopNanos > 0 {
			sess.StoppedAt = time.Unix(0, stopNanos)
		}
		result = append(result, sess)
	}
	return result, rows.Err()
}

// serializeStackIPs converts stack frames to a JSON array of hex IP strings.
// Returns empty string if no stack frames. Format: ["0x7f1234","0x7f5678"]
func serializeStackIPs(stack []events.StackFrame) string {
	if len(stack) == 0 {
		return ""
	}
	ips := make([]string, len(stack))
	for i, f := range stack {
		ips[i] = fmt.Sprintf("0x%x", f.IP)
	}
	b, _ := json.Marshal(ips)
	return string(b)
}

// deserializeStackIPs parses a JSON array of hex IP strings back into StackFrames.
func deserializeStackIPs(s string) []events.StackFrame {
	if s == "" {
		return nil
	}
	var ips []string
	if err := json.Unmarshal([]byte(s), &ips); err != nil {
		return nil
	}
	if len(ips) == 0 {
		return nil
	}
	frames := make([]events.StackFrame, len(ips))
	for i, ipStr := range ips {
		fmt.Sscanf(ipStr, "0x%x", &frames[i].IP)
	}
	return frames
}
