// Package store provides SQLite-based persistent storage for events.
//
// Events are buffered in memory and batch-flushed to SQLite for performance.
// WAL mode enables concurrent reads while the writer flushes batches.
// Size-based pruning keeps the DB under --max-db (default 10 GB).
package store

import (
	"context"
	"database/sql"
	"encoding/json"
	"fmt"
	"log"
	"math"
	"os"
	"path/filepath"
	"strconv"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	_ "modernc.org/sqlite"

	"github.com/ingero-io/ingero/pkg/events"
)

const (
	// DefaultBatchSize is the max events buffered before a flush.
	DefaultBatchSize = 1000

	// DefaultFlushInterval is the max time between flushes.
	DefaultFlushInterval = 100 * time.Millisecond

	// DefaultPruneInterval is the fallback interval for size-based pruning.
	// In practice, pruneBySize() also runs after every flushBatch(), so
	// this ticker only matters when no events arrive for an extended period.
	DefaultPruneInterval = 1 * time.Hour
)

const schema = `
CREATE TABLE IF NOT EXISTS events (
	id         TEXT PRIMARY KEY,       -- "{node}:{seq}" node-namespaced ID
	timestamp  INTEGER NOT NULL,       -- unix nanos
	pid        INTEGER NOT NULL,
	tid        INTEGER NOT NULL,
	source     INTEGER NOT NULL,
	op         INTEGER NOT NULL,
	duration   INTEGER NOT NULL,       -- nanos
	gpu_id     INTEGER NOT NULL DEFAULT 0,
	arg0       INTEGER NOT NULL DEFAULT 0,
	arg1       INTEGER NOT NULL DEFAULT 0,
	ret_code   INTEGER NOT NULL DEFAULT 0,
	stack_hash INTEGER NOT NULL DEFAULT 0, -- FK to stack_traces.hash (0 = no stack)
	node       TEXT NOT NULL DEFAULT '',
	rank       INTEGER,
	local_rank INTEGER,
	world_size INTEGER,
	comm       TEXT NOT NULL DEFAULT '' -- v0.10: process name from bpf_get_current_comm() (≤16 chars, '' = pre-v0.10 row or unknown)
);

CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(timestamp);
CREATE INDEX IF NOT EXISTS idx_events_pid ON events(pid);
`
// Note: idx_events_source_op is NOT created here — it's redundant with
// idx_events_source_op_stack (created below). SQLite uses leftmost-prefix
// matching, so (source, op, stack_hash) covers all (source, op) queries.
// Dropping it saves ~10 MB per 1M events.

// stackTracesSchema stores deduplicated stack traces. Each unique call stack
// (sequence of instruction pointers) is stored once, keyed by FNV-64 hash.
// Events reference stacks by hash instead of storing the full text inline.
//
// Why? A typical ML training loop has 50-500 unique call stacks, but each
// repeats thousands of times. On A100 at 24K events/sec with --stack:
// inline storage = ~658MB/4min; with interning = ~13MB/4min (~50x reduction).
const stackTracesSchema = `
CREATE TABLE IF NOT EXISTS stack_traces (
	hash   INTEGER PRIMARY KEY,  -- FNV-64 of raw IP bytes (no AUTOINCREMENT needed)
	ips    TEXT    NOT NULL,      -- JSON array of hex IPs: ["0x7f1234","0x7f5678"]
	frames TEXT    NOT NULL DEFAULT ''  -- resolved symbols JSON (compact): [{"s":"cudaMalloc","f":"libcudart.so"}]
);
`

// migrateAddFramesColumn adds the resolved frames column to stack_traces.
// Idempotent: fails silently if column already exists (new DBs have it from schema).
const migrateAddFramesColumn = `ALTER TABLE stack_traces ADD COLUMN frames TEXT NOT NULL DEFAULT ''`

// migrateAddSumArg0 adds the sum_arg0 column to event_aggregates.
// Used for mm_page_alloc total bytes (chain engine needs SUM(arg0) > 1GB).
// Idempotent: fails silently if column already exists (new DBs have it from schema).
const migrateAddSumArg0 = `ALTER TABLE event_aggregates ADD COLUMN sum_arg0 INTEGER NOT NULL DEFAULT 0`

// migrateAddCGroupID adds the cgroup_id column to the events table (v0.7).
// cgroup v2 inode ID from bpf_get_current_cgroup_id(). 0 = no meaningful cgroup.
// Idempotent: fails silently if column already exists.
const migrateAddCGroupID = `ALTER TABLE events ADD COLUMN cgroup_id INTEGER NOT NULL DEFAULT 0`

// migrateAddComm adds the comm column to the events table (v0.10).
// Process name from bpf_get_current_comm(), ≤16 chars. Empty string ('') for
// pre-v0.10 rows where no comm was captured. Empty is a valid runtime value
// too (some BPF contexts return empty bytes), so reads must tolerate it.
// Idempotent: fails silently if column already exists.
const migrateAddComm = `ALTER TABLE events ADD COLUMN comm TEXT NOT NULL DEFAULT ''`

// hasEventsCommColumn returns true if events.comm exists. Used to gate the
// schema_info version bump after migrateAddComm so a silently-failed ALTER
// (read-only DB, disk full) doesn't leave an inconsistent state where the
// version says 0.10 but the column doesn't exist.
func hasEventsCommColumn(db *sql.DB) bool {
	rows, err := db.Query("PRAGMA table_info(events)")
	if err != nil {
		return false
	}
	defer rows.Close()
	for rows.Next() {
		var (
			cid     int
			name    string
			ctype   string
			notnull int
			dflt    sql.NullString
			pk      int
		)
		if err := rows.Scan(&cid, &name, &ctype, &notnull, &dflt, &pk); err != nil {
			return false
		}
		if name == "comm" {
			return true
		}
	}
	return false
}

// cgroupMetadataSchema stores cgroup_id → container_id + pod metadata mappings.
// One row per cgroup — populated lazily during tracing when cgroup_id > 1.
// pod_name/namespace are populated when running in K8s (via PodCache lookup).
const cgroupMetadataSchema = `
CREATE TABLE IF NOT EXISTS cgroup_metadata (
	cgroup_id    INTEGER PRIMARY KEY,  -- bpf_get_current_cgroup_id() value
	container_id TEXT NOT NULL DEFAULT '',  -- 64-char hex container ID
	cgroup_path  TEXT NOT NULL DEFAULT '',  -- raw cgroup path from /proc/[pid]/cgroup
	pod_name     TEXT NOT NULL DEFAULT '',  -- K8s pod name (empty on bare metal)
	namespace    TEXT NOT NULL DEFAULT ''   -- K8s namespace (empty on bare metal)
);
CREATE INDEX IF NOT EXISTS idx_cgroup_metadata_container_id ON cgroup_metadata(container_id) WHERE container_id != '';
`

// migrateAddPodFields adds pod_name and namespace columns to cgroup_metadata.
// Idempotent: fails silently if columns already exist (new DBs have them from schema).
const migrateAddPodName = `ALTER TABLE cgroup_metadata ADD COLUMN pod_name TEXT NOT NULL DEFAULT ''`
const migrateAddNamespace = `ALTER TABLE cgroup_metadata ADD COLUMN namespace TEXT NOT NULL DEFAULT ''`

const chainsSchema = `
CREATE TABLE IF NOT EXISTS causal_chains (
	id              TEXT PRIMARY KEY,       -- "{node}:{descriptor}" node-namespaced ID
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
	timeline        TEXT NOT NULL DEFAULT '',-- JSON array of timeline events
	node            TEXT NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_chains_detected_at ON causal_chains(detected_at);
CREATE INDEX IF NOT EXISTS idx_chains_severity ON causal_chains(severity);
`

const snapshotsSchema = `
CREATE TABLE IF NOT EXISTS system_snapshots (
	id         INTEGER PRIMARY KEY AUTOINCREMENT,
	timestamp  INTEGER NOT NULL UNIQUE,  -- unix nanos (unique prevents duplicates)
	cpu_pct    REAL    NOT NULL,  -- CPU utilization %
	mem_pct    REAL    NOT NULL,  -- memory used %
	mem_avail  INTEGER NOT NULL,  -- available MB
	swap_mb    INTEGER NOT NULL,  -- swap used MB
	load_avg   REAL    NOT NULL   -- 1-minute load average
);
`

// snapshotsMigration upgrades pre-dedup databases (created before the UNIQUE
// constraint was added). CREATE TABLE IF NOT EXISTS is name-only — it does NOT
// apply schema changes to existing tables. This migration ensures the unique
// constraint exists by creating a unique index, and drops the old non-unique
// index that is now redundant.
const snapshotsMigration = `
DROP INDEX IF EXISTS idx_snapshots_timestamp;
CREATE UNIQUE INDEX IF NOT EXISTS idx_snapshots_ts_uniq ON system_snapshots(timestamp);
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
	flags      TEXT NOT NULL DEFAULT '',
	node       TEXT NOT NULL DEFAULT '',
	rank       INTEGER,
	local_rank INTEGER,
	world_size INTEGER
);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);
`

const processNamesSchema = `
CREATE TABLE IF NOT EXISTS process_names (
	pid     INTEGER NOT NULL,
	name    TEXT NOT NULL,
	seen_at INTEGER NOT NULL DEFAULT 0,
	PRIMARY KEY (pid)
);
`

// aggregatesSchema stores per-minute, per-op aggregates for events that were
// NOT individually stored (selective storage). This preserves accurate counts
// and latency distributions while reducing DB size significantly.
//
// Why aggregates? At 24K events/sec (A100), the DB grows to 8 GB/hour. Most
// events are bulk ops (cuLaunchKernel, sched_wakeup) with no investigation
// value. Chain-critical events (sched_switch, mm_page_alloc, sync ops,
// process lifecycle, anomalies) are stored individually. The rest are
// summarized in minute-granularity buckets.
const aggregatesSchema = `
CREATE TABLE IF NOT EXISTS event_aggregates (
	bucket    INTEGER NOT NULL,  -- minute-truncated unix nanos
	source    INTEGER NOT NULL,
	op        INTEGER NOT NULL,
	pid       INTEGER NOT NULL DEFAULT 0,
	count     INTEGER NOT NULL DEFAULT 0,
	stored    INTEGER NOT NULL DEFAULT 0,
	sum_dur   INTEGER NOT NULL DEFAULT 0,
	min_dur   INTEGER NOT NULL DEFAULT 0,
	max_dur   INTEGER NOT NULL DEFAULT 0,
	sum_arg0  INTEGER NOT NULL DEFAULT 0,
	PRIMARY KEY (bucket, source, op, pid)
);
CREATE INDEX IF NOT EXISTS idx_aggregates_bucket ON event_aggregates(bucket);
`

// cgroupSchedstatSchema stores per-cgroup off-CPU scheduling statistics for
// noisy neighbor detection. Aggregated from sched_switch events by cgroup_id.
// Written periodically during tracing — not per-event.
const cgroupSchedstatSchema = `
CREATE TABLE IF NOT EXISTS cgroup_schedstat (
	cgroup_id     INTEGER NOT NULL,
	p99_off_cpu   INTEGER NOT NULL,  -- p99 off-CPU duration (nanos)
	total_off_cpu INTEGER NOT NULL,  -- total off-CPU duration (nanos)
	event_count   INTEGER NOT NULL,
	window_start  INTEGER NOT NULL DEFAULT 0,  -- unix nanos
	window_end    INTEGER NOT NULL DEFAULT 0,  -- unix nanos
	PRIMARY KEY (cgroup_id, window_end)
);
CREATE INDEX IF NOT EXISTS idx_cgroup_schedstat_cgroup ON cgroup_schedstat(cgroup_id);
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
		{5, "IO", "Block I/O events — tracepoints (block_rq_issue/block_rq_complete)"},
		{6, "TCP", "TCP events — tracepoints (tcp_retransmit_skb)"},
		{7, "NET", "Network socket events — syscall tracepoints (sendto/recvfrom)"},
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
		{1, 7, "cudaMemcpyAsync", "Asynchronous host<->device memory copy"},
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
		{4, 6, "cuMemAllocManaged", "Unified Memory allocation via driver API"},
		// CUDA Runtime managed alloc (source=1)
		{1, 8, "cudaMallocManaged", "Unified Memory allocation via runtime API"},
		// Block I/O (source=5)
		{5, 1, "block_read", "Block device read request"},
		{5, 2, "block_write", "Block device write request"},
		{5, 3, "block_discard", "Block device discard/trim request"},
		// TCP (source=6)
		{6, 1, "tcp_retransmit", "TCP segment retransmission"},
		// Network socket (source=7)
		{7, 1, "net_send", "Socket send/sendto syscall"},
		{7, 2, "net_recv", "Socket recv/recvfrom syscall"},
		// K8s lifecycle events (source=3, synthetic host ops)
		{3, 10, "pod_restart", "K8s pod container restart detected"},
		{3, 11, "pod_eviction", "K8s pod eviction detected"},
		{3, 12, "pod_oom_kill", "K8s pod OOM kill detected"},
	}
	for _, o := range ops {
		tx.Exec("INSERT INTO ops (source_id, op_id, name, description) VALUES (?, ?, ?, ?)", o.src, o.op, o.name, o.desc)
	}

	// Schema info — units, version, helpful context
	info := []struct{ k, v string }{
		{"version", "0.8"},
		{"timestamp_unit", "nanoseconds (Unix epoch)"},
		{"duration_unit", "nanoseconds"},
		{"arg0_note", "Operation-specific: byte size for alloc/memcpy, kernel function pointer for launch"},
		{"arg1_note", "Operation-specific: memcpy direction for cudaMemcpy/cudaMemcpyAsync, unused for most ops"},
		{"ret_code_note", "CUDA return code (0 = cudaSuccess). Host events always 0."},
		{"stack_traces_note", "Deduplicated stacks: events.stack_hash → stack_traces.hash. frames column has resolved symbols. Use get_stacks MCP tool for call stack analysis."},
		{"example_query", "SELECT e.id, s.name AS source, o.name AS op, e.duration/1000 AS dur_us, e.pid FROM events e JOIN sources s ON e.source = s.id JOIN ops o ON e.source = o.source_id AND e.op = o.op_id ORDER BY e.timestamp DESC LIMIT 20"},
		{"system_snapshots_note", "System metrics sampled every 1s during recording. Replay with correlator for post-hoc causal chain analysis."},
		{"sessions_note", "One row per 'ingero trace' invocation. Correlate with events via time range."},
	}
	for _, kv := range info {
		tx.Exec("INSERT INTO schema_info (key, value) VALUES (?, ?)", kv.k, kv.v)
	}

	tx.Commit()
}

// v0.9 node identity migration — adds node/rank columns to events, sessions, causal_chains.
// Idempotent: fails silently if columns already exist (new DBs have them from schema).
const migrateAddEventsNode = `ALTER TABLE events ADD COLUMN node TEXT NOT NULL DEFAULT ''`
const migrateAddEventsRank = `ALTER TABLE events ADD COLUMN rank INTEGER`
const migrateAddEventsLocalRank = `ALTER TABLE events ADD COLUMN local_rank INTEGER`
const migrateAddEventsWorldSize = `ALTER TABLE events ADD COLUMN world_size INTEGER`
const migrateAddSessionsNode = `ALTER TABLE sessions ADD COLUMN node TEXT NOT NULL DEFAULT ''`
const migrateAddSessionsRank = `ALTER TABLE sessions ADD COLUMN rank INTEGER`
const migrateAddSessionsLocalRank = `ALTER TABLE sessions ADD COLUMN local_rank INTEGER`
const migrateAddSessionsWorldSize = `ALTER TABLE sessions ADD COLUMN world_size INTEGER`
const migrateAddChainsNode = `ALTER TABLE causal_chains ADD COLUMN node TEXT NOT NULL DEFAULT ''`

// migrateSchema adds columns that may be missing in older databases.
const migrateAddStackIPs = `ALTER TABLE events ADD COLUMN stack_ips TEXT NOT NULL DEFAULT ''`
const migrateAddStackHash = `ALTER TABLE events ADD COLUMN stack_hash INTEGER NOT NULL DEFAULT 0`

// Store provides persistent event storage backed by SQLite.
type Store struct {
	db         *sql.DB
	dbPath     string
	insertCh   chan events.Event
	snapshotCh chan SystemSnapshot
	runDone    chan struct{} // closed when Run() exits
	runActive  atomic.Bool  // true once Run() is called

	// stackCache tracks which stack hashes are already in the stack_traces
	// table, avoiding redundant INSERTs and JSON serialization. Only accessed
	// from the Run() goroutine (flushBatch), so no mutex needed.
	stackCache map[uint64]bool

	maxDBSize atomic.Int64 // 0 = no limit, >0 = target max DB+WAL+SHM in bytes

	// Node identity for multi-node correlation (v0.9).
	node    string       // set via SetNode before Run()
	eventSeq atomic.Uint64 // monotonic event ID counter

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
	Limit  int           // max results (0 = 10000, -1 = unlimited)
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
	Node      string // multi-node identity (v0.9)
	Rank      *int   // distributed training rank (nil = not set)
	LocalRank *int   // local rank (nil = not set)
	WorldSize *int   // total ranks (nil = not set)
}

// Aggregate is a per-minute summary of events that were not individually stored.
// Used by selective storage to preserve accurate counts and latency distributions.
type Aggregate struct {
	Bucket  int64  // minute-truncated unix nanos
	Source  uint8  // event source (SourceCUDA, SourceHost, SourceDriver)
	Op      uint8  // operation code within source
	PID     uint32 // process ID (0 = all PIDs aggregated)
	Count   int64  // total events in this bucket (stored + not-stored)
	Stored  int64  // how many were individually stored in the events table
	SumDur  int64  // sum of durations (nanos) — for computing mean
	MinDur  int64  // minimum duration (nanos) in this bucket
	MaxDur  int64  // maximum duration (nanos) in this bucket
	SumArg0 int64  // sum of arg0 values — for mm_page_alloc (total bytes allocated)
}

// AggregateTotals summarizes event counts across all aggregates for a time range.
// Used by explain/query/MCP to show accurate total counts even though most
// individual events were not stored.
type AggregateTotals struct {
	TotalEvents  int64            // sum of all Count fields
	StoredEvents int64            // sum of all Stored fields
	ByOp         map[string]int64 // op_name → total count
}

// DefaultDBPath returns the default database path (~/.ingero/ingero.db).
//
// Precedence:
//  1. INGERO_DB env var (explicit path, used by container deployments)
//  2. SUDO_USER home directory (so sudo trace + non-sudo explain share the same DB)
//  3. Current user's home directory (~/.ingero/ingero.db)
func DefaultDBPath() string {
	// Allow containers and automation to override the default path.
	if dbEnv := os.Getenv("INGERO_DB"); dbEnv != "" {
		return dbEnv
	}

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

	// Enable incremental auto-vacuum so pruneBySize() can reclaim disk space
	// via PRAGMA incremental_vacuum. Must be set before any tables are created
	// (i.e., before the first page is written). Only takes effect on newly
	// created DBs — existing databases retain their original auto_vacuum mode.
	db.Exec("PRAGMA auto_vacuum = INCREMENTAL")

	// Enable WAL mode for concurrent reads during writes.
	if _, err := db.Exec("PRAGMA journal_mode=WAL"); err != nil {
		db.Close()
		return nil, fmt.Errorf("setting WAL mode: %w", err)
	}

	// Set busy timeout so concurrent writers (event batch flush + aggregate
	// flush) retry instead of failing immediately with SQLITE_BUSY.
	db.Exec("PRAGMA busy_timeout = 5000")

	// Create schema.
	if _, err := db.Exec(schema); err != nil {
		db.Close()
		return nil, fmt.Errorf("creating schema: %w", err)
	}

	// Schema migrations for backward compatibility with older databases.
	// Both ALTER TABLEs are idempotent — they fail silently if the column
	// already exists. New databases get stack_hash from the schema and
	// stack_ips from this migration (unused but harmless — 0 bytes overhead).
	db.Exec(migrateAddStackHash)
	db.Exec(migrateAddStackIPs)

	// Create stack_traces table (deduplicated stack interning).
	if _, err := db.Exec(stackTracesSchema); err != nil {
		db.Close()
		return nil, fmt.Errorf("creating stack_traces table: %w", err)
	}

	// Add resolved frames column. Idempotent — no-op if column exists.
	db.Exec(migrateAddFramesColumn)

	// Migrate: if there are events with inline stack_ips, intern them into
	// the stack_traces table. No-op for new databases.
	migrateInlineStacks(db)

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
	// Upgrade: replace old stack_ips_note with stack_traces_note for pre-interning DBs.
	db.Exec("DELETE FROM schema_info WHERE key = 'stack_ips_note'")
	db.Exec("INSERT OR REPLACE INTO schema_info (key, value) VALUES ('stack_traces_note', 'Deduplicated stacks: events.stack_hash → stack_traces.hash. frames column has resolved symbols. Use get_stacks MCP tool for call stack analysis.')")

	// Composite index for get_stacks MCP tool (GROUP BY source,op,stack_hash).
	db.Exec("CREATE INDEX IF NOT EXISTS idx_events_source_op_stack ON events(source, op, stack_hash)")

	// Drop redundant idx_events_source_op — the composite idx_events_source_op_stack
	// covers all (source, op) queries via SQLite leftmost-prefix matching.
	// Old DBs created before this optimization carry the redundant index (~10MB/1M events).
	db.Exec("DROP INDEX IF EXISTS idx_events_source_op")

	// Create system_snapshots table (metrics sampled every 1s during recording).
	if _, err := db.Exec(snapshotsSchema); err != nil {
		db.Close()
		return nil, fmt.Errorf("creating system_snapshots table: %w", err)
	}
	// Migrate pre-dedup databases: add unique index, drop redundant non-unique index.
	db.Exec(snapshotsMigration)

	// Create sessions table (one row per 'ingero trace' invocation).
	if _, err := db.Exec(sessionsSchema); err != nil {
		db.Close()
		return nil, fmt.Errorf("creating sessions table: %w", err)
	}

	// Create process_names table (PID→name mapping for query enrichment).
	if _, err := db.Exec(processNamesSchema); err != nil {
		db.Close()
		return nil, fmt.Errorf("creating process_names table: %w", err)
	}

	// Create event_aggregates table (selective storage minute-buckets).
	if _, err := db.Exec(aggregatesSchema); err != nil {
		db.Close()
		return nil, fmt.Errorf("creating event_aggregates table: %w", err)
	}

	// Add sum_arg0 column for mm_page_alloc total bytes. Idempotent.
	db.Exec(migrateAddSumArg0)

	// v0.7: Add cgroup_id column to events table. Idempotent.
	db.Exec(migrateAddCGroupID)

	// Partial index on cgroup_id — only indexes non-zero values (container events).
	// Sparse: bare-metal events (cgroup_id=0) don't bloat the index.
	db.Exec("CREATE INDEX IF NOT EXISTS idx_events_cgroup ON events(cgroup_id) WHERE cgroup_id != 0")

	// v0.7: Create cgroup_metadata table (cgroup_id → container_id + pod metadata).
	if _, err := db.Exec(cgroupMetadataSchema); err != nil {
		db.Close()
		return nil, fmt.Errorf("creating cgroup_metadata table: %w", err)
	}

	// v0.7: Add pod_name/namespace columns to cgroup_metadata. Idempotent —
	// no-op for new DBs (schema already has them), adds columns for old DBs.
	db.Exec(migrateAddPodName)
	db.Exec(migrateAddNamespace)

	// v0.8: Create cgroup_schedstat table for noisy neighbor detection.
	if _, err := db.Exec(cgroupSchedstatSchema); err != nil {
		db.Close()
		return nil, fmt.Errorf("creating cgroup_schedstat table: %w", err)
	}

	// Update schema version to 0.8.
	db.Exec("INSERT OR REPLACE INTO schema_info (key, value) VALUES ('version', '0.8')")
	db.Exec("INSERT OR REPLACE INTO schema_info (key, value) VALUES ('cgroup_metadata_note', 'cgroup_id → container_id mapping. Populated during K8s tracing. JOIN with events.cgroup_id for container context.')")

	// v0.9: Add node identity and rank columns. Idempotent — no-op for new DBs.
	db.Exec(migrateAddEventsNode)
	db.Exec(migrateAddEventsRank)
	db.Exec(migrateAddEventsLocalRank)
	db.Exec(migrateAddEventsWorldSize)
	db.Exec(migrateAddSessionsNode)
	db.Exec(migrateAddSessionsRank)
	db.Exec(migrateAddSessionsLocalRank)
	db.Exec(migrateAddSessionsWorldSize)
	db.Exec(migrateAddChainsNode)

	// Update schema version to 0.9.
	db.Exec("INSERT OR REPLACE INTO schema_info (key, value) VALUES ('version', '0.9')")
	db.Exec("INSERT OR REPLACE INTO schema_info (key, value) VALUES ('node_note', 'Node identity for multi-node correlation. events.id format: {node}:{seq}. causal_chains.id format: {node}:{descriptor}.')")

	// v0.10: Add comm column to events table (PID hardening).
	// Captured kernel-side via bpf_get_current_comm() at event emission time —
	// immune to PID reuse and process exit, unlike the lazy /proc-based
	// process_names table (which is preserved as a read-side fallback).
	// Idempotent — no-op for new DBs that already have it from schema.
	//
	// Migration error gating: ALTER TABLE may silently fail on read-only DBs,
	// disk full, or other I/O errors. We bump schema_info to 0.10 ONLY after
	// confirming via PRAGMA that the column actually exists — protects against
	// inconsistent state where the version says 0.10 but the column is missing.
	db.Exec(migrateAddComm)

	if hasEventsCommColumn(db) {
		db.Exec("INSERT OR REPLACE INTO schema_info (key, value) VALUES ('version', '0.10')")
		db.Exec("INSERT OR REPLACE INTO schema_info (key, value) VALUES ('comm_note', 'events.comm captured kernel-side at event time. Empty for pre-v0.10 rows (fall back to process_names LEFT JOIN for legacy display).')")
	}
	// On read-only DBs the version stays at 0.9 — read paths must use
	// hasColumn() guards if they want to be tolerant of either schema.

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
		runDone:    make(chan struct{}),
		stackCache: make(map[uint64]bool),
	}

	// Seed eventSeq from existing events so reopening a DB doesn't
	// restart the counter at 0 and generate duplicate IDs.
	var maxSeq uint64
	row := db.QueryRow(
		"SELECT COALESCE(MAX(CAST(substr(id, instr(id, ':')+1) AS INTEGER)), 0) FROM events WHERE id LIKE ?",
		":%",
	)
	if err := row.Scan(&maxSeq); err == nil && maxSeq > 0 {
		s.eventSeq.Store(maxSeq)
	}

	return s, nil
}

// WaitDone blocks until Run() has finished its final flush and returned.
// Call this after ctx cancellation but before StopSession/Close to ensure
// all pending batch writes are committed. Safe to call if Run() was never
// started (returns immediately) — this avoids deadlocks in query/explain/MCP
// code paths that use Store without Run().
func (s *Store) WaitDone() {
	if !s.runActive.Load() {
		return
	}
	<-s.runDone
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
// Callers should use WaitDone() to wait for Run to finish before calling
// StopSession or Close, to avoid racing with the final batch flush.
func (s *Store) Run(ctx context.Context) {
	s.runActive.Store(true)
	defer close(s.runDone)

	// Pre-load stack cache from existing DB so restarted sessions don't
	// re-insert stacks that already exist from a previous trace.
	s.loadStackCache()

	flushTicker := time.NewTicker(DefaultFlushInterval)
	defer flushTicker.Stop()

	pruneTicker := time.NewTicker(DefaultPruneInterval)
	defer pruneTicker.Stop()

	// Prune on startup.
	s.prune()

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
			s.prune()

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
// Stack traces are interned: each unique stack is stored once in stack_traces,
// and events reference it by FNV-64 hash. This reduces DB size ~50x when
// --stack is enabled, since ML training loops repeat the same few call stacks
// across thousands of events.
func (s *Store) flushBatch(batch []events.Event) {
	if len(batch) == 0 {
		return
	}

	tx, err := s.db.Begin()
	if err != nil {
		return
	}

	// Prepare: event insert uses stack_hash (integer FK) instead of inline JSON.
	evtStmt, err := tx.Prepare(`INSERT INTO events (id, timestamp, pid, tid, source, op, duration, gpu_id, arg0, arg1, ret_code, stack_hash, cgroup_id, node, rank, local_rank, world_size, comm)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		tx.Rollback()
		return
	}
	defer evtStmt.Close()

	// Prepare: stack insert uses INSERT OR REPLACE so that stacks from older
	// sessions (which had empty frames) get upgraded with resolved symbols.
	// REPLACE is a DELETE+INSERT in SQLite — safe because stack_traces has no
	// foreign key constraints (events reference stack_hash by value, not FK).
	stackStmt, err := tx.Prepare(`INSERT OR REPLACE INTO stack_traces (hash, ips, frames) VALUES (?, ?, ?)`)
	if err != nil {
		tx.Rollback()
		return
	}
	defer stackStmt.Close()

	for _, evt := range batch {
		var stackHash int64
		if len(evt.Stack) > 0 {
			h := hashStack(evt.Stack)
			stackHash = int64(h)
			cached, known := s.stackCache[h]
			if !known {
				// Brand new stack — serialize and insert.
				framesJSON := serializeStackFrames(evt.Stack)
				// Skip ips when frames are resolved — frames already contain
				// the IP as the "i" field, so ips is redundant. Saves ~30% of
				// stack_traces storage. Keep ips for unresolved stacks (fallback).
				ipsJSON := ""
				if framesJSON == "" {
					ipsJSON = serializeStackIPs(evt.Stack)
				}
				stackStmt.Exec(stackHash, ipsJSON, framesJSON)
				s.stackCache[h] = framesJSON != ""
			} else if !cached {
				// Known stack without resolved frames — try to upgrade.
				framesJSON := serializeStackFrames(evt.Stack)
				if framesJSON != "" {
					// Upgrading: clear ips (now redundant with resolved frames).
					stackStmt.Exec(stackHash, "", framesJSON)
					s.stackCache[h] = true
				}
			}
		}
		evtStmt.Exec(
			s.nextEventID(),
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
			stackHash,
			evt.CGroupID,
			s.node,
			nilableInt(evt.Rank),
			nilableInt(evt.LocalRank),
			nilableInt(evt.WorldSize),
			evt.Comm,
		)
	}

	tx.Commit()

	// Check size limit after every flush. Cheap when under the limit
	// (3 stat() calls, early return). When over, prunes oldest data
	// immediately rather than waiting for the hourly prune cycle.
	s.pruneBySize()
}

// loadStackCache populates the in-memory cache of known stack hashes from the
// stack_traces table. Called once at Run() startup so that a restarted trace
// session against the same DB doesn't re-insert stacks that already exist.
//
// Stacks with resolved frames are marked as fully cached (true).
// Stacks without frames (from older sessions) are marked as needing
// upgrade (false), so INSERT OR REPLACE runs once when resolved symbols
// become available — but not repeatedly for stacks that stay unresolved.
func (s *Store) loadStackCache() {
	rows, err := s.db.Query("SELECT hash, frames FROM stack_traces")
	if err != nil {
		return
	}
	defer rows.Close()
	for rows.Next() {
		var h int64
		var frames string
		if err := rows.Scan(&h, &frames); err == nil {
			s.stackCache[uint64(h)] = frames != ""
		}
	}
}

// StoreCGroupMetadata stores a cgroup_id → container_id/path/pod mapping.
// Idempotent — uses INSERT OR REPLACE so repeated calls update the row.
// podName and namespace are empty on bare metal (populated from PodCache in K8s).
func (s *Store) StoreCGroupMetadata(cgroupID uint64, containerID, cgroupPath, podName, namespace string) {
	s.db.Exec("INSERT OR REPLACE INTO cgroup_metadata (cgroup_id, container_id, cgroup_path, pod_name, namespace) VALUES (?, ?, ?, ?, ?)",
		cgroupID, containerID, cgroupPath, podName, namespace)
}

// prune performs size-based pruning. When --max-db is set and the DB exceeds
// the limit, deletes oldest events proportionally until the file fits.
// No-op when --max-db is 0 (unlimited).
func (s *Store) prune() {
	s.pruneBySize()
}

// ParseSize parses a human-friendly size string like "10g", "500m", "1t" into bytes.
// Accepts optional trailing "b"/"B" (e.g., "10GB", "500mb"). Case-insensitive on unit.
// Returns an error for unparseable input, non-positive values, or empty strings.
//
//	ParseSize("10g")   → 10 * 1024^3
//	ParseSize("500MB") → 500 * 1024^2
//	ParseSize("1t")    → 1 * 1024^4
func ParseSize(s string) (int64, error) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, fmt.Errorf("empty size string")
	}

	// Strip optional trailing b/B (e.g., "10GB" → "10G").
	s = strings.TrimRight(s, "bB")
	if s == "" {
		return 0, fmt.Errorf("size string has no numeric part")
	}

	// Last character is the unit multiplier.
	unit := s[len(s)-1]
	var multiplier int64
	switch unit {
	case 'k', 'K':
		multiplier = 1 << 10
		s = s[:len(s)-1]
	case 'm', 'M':
		multiplier = 1 << 20
		s = s[:len(s)-1]
	case 'g', 'G':
		multiplier = 1 << 30
		s = s[:len(s)-1]
	case 't', 'T':
		multiplier = 1 << 40
		s = s[:len(s)-1]
	default:
		// No unit suffix — treat as bytes.
		multiplier = 1
	}

	n, err := strconv.ParseInt(s, 10, 64)
	if err != nil {
		return 0, fmt.Errorf("parsing size %q: %w", s, err)
	}
	if n <= 0 {
		return 0, fmt.Errorf("size must be positive, got %d", n)
	}

	// Guard against int64 overflow. Go silently wraps on overflow, so
	// e.g. "9999999t" would produce a negative value without this check.
	if n > math.MaxInt64/multiplier {
		return 0, fmt.Errorf("size %d with multiplier %d overflows int64", n, multiplier)
	}

	return n * multiplier, nil
}

// SetMaxDBSize sets the maximum DB+WAL+SHM size in bytes. 0 = no limit.
// Safe to call before or during Run(). pruneBySize() enforces this limit
// after every flush and during periodic pruning (every DefaultPruneInterval).
func (s *Store) SetMaxDBSize(bytes int64) {
	s.maxDBSize.Store(bytes)
}

// SetNode sets the node identity used for node-namespaced event and chain IDs.
// Must be called before Run(). Not safe for concurrent use with Run().
func (s *Store) SetNode(node string) {
	s.node = node
	// Seed the event ID counter from the DB to prevent collisions on restart.
	// Query the max sequence number for this node's events.
	var maxSeq uint64
	row := s.db.QueryRow(
		"SELECT COALESCE(MAX(CAST(substr(id, instr(id, ':')+1) AS INTEGER)), 0) FROM events WHERE id LIKE ?",
		node+":%",
	)
	if err := row.Scan(&maxSeq); err == nil && maxSeq > 0 {
		s.eventSeq.Store(maxSeq)
	}
}

// Node returns the current node identity.
func (s *Store) Node() string {
	return s.node
}

// nextEventID returns the next node-namespaced event ID: "{node}:{seq}".
func (s *Store) nextEventID() string {
	seq := s.eventSeq.Add(1)
	return fmt.Sprintf("%s:%d", s.node, seq)
}

// diskUsage returns the total size of the DB file + WAL + SHM in bytes.
// Returns 0 for in-memory databases or on error.
func (s *Store) diskUsage() int64 {
	if s.dbPath == ":memory:" || s.dbPath == "" {
		return 0
	}
	var total int64
	for _, suffix := range []string{"", "-wal", "-shm"} {
		info, err := os.Stat(s.dbPath + suffix)
		if err == nil {
			total += info.Size()
		}
	}
	return total
}

// pruneBySize deletes oldest events proportionally to bring the DB under maxDBSize.
// This is the sole retention mechanism — there is no time-based retention.
// The default --max-db is 10 GB, lasting a few hours under average GPU load.
//
// Algorithm: compute what fraction of the time range to keep based on
// currentSize vs target (90% of maxDBSize for headroom), then delete
// everything older than the cutoff. Loops up to 3 iterations because
// each iteration's time-proportional cut may underestimate if data is
// skewed (e.g., sparse old data, dense recent data).
//
// Key detail: in WAL mode, DELETEs write to the WAL rather than freeing
// main-DB pages. We run a PASSIVE checkpoint before each vacuum to move
// WAL frames into the main DB, making freed pages reclaimable. PASSIVE
// is used (not TRUNCATE) to avoid blocking concurrent readers.
func (s *Store) pruneBySize() {
	maxSize := s.maxDBSize.Load()
	if maxSize <= 0 {
		return
	}

	pruned := false

	for i := 0; i < 3; i++ {
		currentSize := s.diskUsage()
		if currentSize <= maxSize {
			break
		}

		// Target 90% to leave headroom for new writes between prune cycles.
		target := int64(float64(maxSize) * 0.9)

		// Compute the time range across both events and aggregates so
		// size-based pruning works even if one table is empty.
		// COALESCE handles the case where one or both tables are empty
		// (MIN/MAX return NULL for empty tables).
		var minTS, maxTS int64
		err := s.db.QueryRow(`
			SELECT COALESCE(MIN(lo), 0), COALESCE(MAX(hi), 0) FROM (
				SELECT MIN(timestamp) AS lo, MAX(timestamp) AS hi FROM events
				UNION ALL
				SELECT MIN(bucket) AS lo, MAX(bucket) AS hi FROM event_aggregates
			)`).Scan(&minTS, &maxTS)
		if err != nil || minTS == 0 || maxTS == 0 || minTS >= maxTS {
			break // no data or single-point — nothing to prune
		}

		// keepFraction: what fraction of the time range to keep.
		// E.g., if DB is 2x the limit, keep ~45% of the time range.
		keepFraction := float64(target) / float64(currentSize)
		if keepFraction >= 1.0 {
			break
		}
		cutoff := maxTS - int64(float64(maxTS-minTS)*keepFraction)

		// Delete events, aggregates, snapshots, chains, sessions, stale process names.
		s.db.Exec("DELETE FROM events WHERE timestamp < ?", cutoff)
		s.db.Exec("DELETE FROM event_aggregates WHERE bucket < ?", cutoff)
		s.db.Exec("DELETE FROM system_snapshots WHERE timestamp < ?", cutoff)
		s.db.Exec("DELETE FROM causal_chains WHERE detected_at < ?", cutoff)
		s.db.Exec("DELETE FROM sessions WHERE started_at < ?", cutoff)
		s.db.Exec("DELETE FROM process_names WHERE seen_at < ?", cutoff)

		// Clean orphaned stack traces.
		s.db.Exec(`DELETE FROM stack_traces WHERE NOT EXISTS (
			SELECT 1 FROM events WHERE events.stack_hash = stack_traces.hash LIMIT 1
		)`)

		pruned = true

		// Checkpoint WAL so deleted pages become reclaimable in the main DB.
		// PASSIVE avoids blocking concurrent readers (MCP queries, explain).
		s.db.Exec("PRAGMA wal_checkpoint(PASSIVE)")

		// Reclaim freed pages proportional to the overage. Without this,
		// the file retains its size (free pages stay allocated) and
		// diskUsage() won't see improvement on the next iteration.
		// On older DBs without incremental auto_vacuum, this is a no-op.
		overage := currentSize - maxSize
		pagesToReclaim := overage / 4096 // 1 page ≈ 4KB
		if pagesToReclaim < 1000 {
			pagesToReclaim = 1000
		}
		s.db.Exec(fmt.Sprintf("PRAGMA incremental_vacuum(%d)", pagesToReclaim))
	}

	// Rebuild stack cache after pruning — deleted stack_traces rows must
	// be evicted so new events with the same hash re-insert correctly.
	// Only pay the cost when we actually deleted something.
	if pruned {
		s.stackCache = make(map[uint64]bool)
		s.loadStackCache()
	}
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
// Stack traces are resolved via LEFT JOIN against the stack_traces interning table.
// Prefers resolved frames over raw IPs for historical investigation.
//
// v0.10: events.comm is selected directly. Pre-v0.10 rows have empty comm; the
// QueryRich path falls back to process_names for those. Plain Query() callers
// must tolerate empty Comm — see project-context.md PID hardening notes.
func (s *Store) Query(q QueryParams) ([]events.Event, error) {
	query := `SELECT e.timestamp, e.pid, e.tid, e.source, e.op, e.duration,
		e.gpu_id, e.arg0, e.arg1, e.ret_code,
		COALESCE(st.frames, ''), COALESCE(st.ips, ''),
		e.cgroup_id, e.node, e.rank, e.local_rank, e.world_size, e.comm
	FROM events e
	LEFT JOIN stack_traces st ON e.stack_hash = st.hash
	WHERE 1=1`
	var args []interface{}

	// Time range.
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

	// PID filter (single or multi).
	query, args = appendPIDFilter(query, args, q, "e.")

	// Source filter.
	if q.Source > 0 {
		query += " AND e.source = ?"
		args = append(args, q.Source)
	}

	// Op filter (only if Source is set).
	if q.Source > 0 && q.Op > 0 {
		query += " AND e.op = ?"
		args = append(args, q.Op)
	}

	// Fetch the newest events first (DESC) so the LIMIT keeps the most
	// recent data rather than the oldest.  We reverse the slice afterward
	// to return chronological order.
	query += " ORDER BY e.timestamp DESC"

	// Limit: 0 = default 10K, -1 = unlimited, >0 = explicit.
	if q.Limit >= 0 {
		limit := q.Limit
		if limit == 0 {
			limit = 10000
		}
		query += " LIMIT ?"
		args = append(args, limit)
	}

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
			framesJSON string
			ipsJSON    string
			cgroupID   uint64
			node       string
			rank       sql.NullInt64
			localRank  sql.NullInt64
			worldSize  sql.NullInt64
			comm       string
		)
		if err := rows.Scan(&tsNanos, &pid, &tid, &source, &op, &durNanos, &gpuID, &arg0, &arg1, &retCode, &framesJSON, &ipsJSON, &cgroupID, &node, &rank, &localRank, &worldSize, &comm); err != nil {
			return nil, fmt.Errorf("scanning event row: %w", err)
		}
		evt := events.Event{
			Timestamp: time.Unix(0, tsNanos),
			PID:       pid,
			TID:       tid,
			Comm:      comm,
			Source:    events.Source(source),
			Op:        op,
			Duration:  time.Duration(durNanos),
			GPUID:     gpuID,
			Args:      [2]uint64{arg0, arg1},
			RetCode:   retCode,
			CGroupID:  cgroupID,
			Node:      node,
		}
		if rank.Valid {
			v := int(rank.Int64)
			evt.Rank = &v
		}
		if localRank.Valid {
			v := int(localRank.Int64)
			evt.LocalRank = &v
		}
		if worldSize.Valid {
			v := int(worldSize.Int64)
			evt.WorldSize = &v
		}
		// Prefer resolved frames; fall back to raw IPs for old DBs.
		evt.Stack = deserializeStackFrames(framesJSON)
		if evt.Stack == nil {
			evt.Stack = deserializeStackIPs(ipsJSON)
		}
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
//
// v0.10 semantic change for ProcessName: previously sourced exclusively from
// the lazy /proc-based process_names table (set by `RecordProcessName` from
// `/proc/[pid]/comm` reads). Now resolved as
// COALESCE(NULLIF(events.comm,''), NULLIF(process_names.name,''), '')
// — kernel-captured comm at event time is preferred; process_names as
// the fallback for pre-v0.10 rows. Both wrapped in NULLIF to avoid
// empty-string short-circuiting the COALESCE chain. Callers comparing
// ProcessName against a *currently-running* /proc snapshot may see the
// historical name (the name the process had when the event was emitted)
// instead of the current name. This is the intended behavior under PID
// hardening — the embedded `Event.Comm` field carries the same value.
type RichEvent struct {
	events.Event
	SourceName  string
	SourceDesc  string
	OpName      string
	OpDesc      string
	ProcessName string // v0.10: prefers events.comm, falls back to process_names.name
}

// QueryRich retrieves events with JOIN against lookup tables for human-readable names.
// Use this for AI/MCP output where self-describing data matters.
//
// v0.10: events.comm (kernel-side captured at event time) is preferred over
// process_names.name (lazy /proc lookup). The process_names LEFT JOIN is
// preserved as a fallback for pre-v0.10 rows where events.comm is empty.
// ProcessName resolution: COALESCE(NULLIF(e.comm,''), NULLIF(pn.name,''), '').
// Both columns are NULLIF-wrapped because process_names.name is NOT NULL DEFAULT ''
// — without NULLIF on pn.name, an empty pn.name short-circuits the COALESCE.
func (s *Store) QueryRich(q QueryParams) ([]RichEvent, error) {
	query := `SELECT e.timestamp, e.pid, e.tid, e.source, e.op, e.duration,
		e.gpu_id, e.arg0, e.arg1, e.ret_code,
		COALESCE(st.frames, ''), COALESCE(st.ips, ''),
		COALESCE(s.name, 'SRC_' || e.source),
		COALESCE(s.description, ''),
		COALESCE(o.name, 'OP_' || e.op),
		COALESCE(o.description, ''),
		COALESCE(NULLIF(e.comm, ''), NULLIF(pn.name, ''), ''),
		e.cgroup_id, e.node, e.rank, e.local_rank, e.world_size, e.comm
	FROM events e
	LEFT JOIN stack_traces st ON e.stack_hash = st.hash
	LEFT JOIN sources s ON e.source = s.id
	LEFT JOIN ops o ON e.source = o.source_id AND e.op = o.op_id
	LEFT JOIN process_names pn ON e.pid = pn.pid
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

	// Limit: 0 = default 10K, -1 = unlimited, >0 = explicit.
	if q.Limit >= 0 {
		limit := q.Limit
		if limit == 0 {
			limit = 10000
		}
		query += " LIMIT ?"
		args = append(args, limit)
	}

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
			framesJSON string
			ipsJSON    string
			srcName, srcDesc, opName, opDesc string
			procName   string
			cgroupID   uint64
			node       string
			rank       sql.NullInt64
			localRank  sql.NullInt64
			worldSize  sql.NullInt64
			comm       string
		)
		if err := rows.Scan(&tsNanos, &pid, &tid, &source, &op, &durNanos,
			&gpuID, &arg0, &arg1, &retCode, &framesJSON, &ipsJSON,
			&srcName, &srcDesc, &opName, &opDesc, &procName, &cgroupID,
			&node, &rank, &localRank, &worldSize, &comm); err != nil {
			return nil, fmt.Errorf("scanning rich event: %w", err)
		}
		re := RichEvent{
			Event: events.Event{
				Timestamp: time.Unix(0, tsNanos),
				PID:       pid,
				TID:       tid,
				Comm:      comm,
				Source:    events.Source(source),
				Op:        op,
				Duration:  time.Duration(durNanos),
				GPUID:     gpuID,
				Args:      [2]uint64{arg0, arg1},
				RetCode:   retCode,
				CGroupID:  cgroupID,
				Node:      node,
			},
			SourceName:  srcName,
			SourceDesc:  srcDesc,
			OpName:      opName,
			OpDesc:      opDesc,
			ProcessName: procName, // already resolved via COALESCE(NULLIF(e.comm,''), NULLIF(pn.name,''), '')
		}
		if rank.Valid {
			v := int(rank.Int64)
			re.Event.Rank = &v
		}
		if localRank.Valid {
			v := int(localRank.Int64)
			re.Event.LocalRank = &v
		}
		if worldSize.Valid {
			v := int(worldSize.Int64)
			re.Event.WorldSize = &v
		}
		// Prefer resolved frames; fall back to raw IPs for old DBs.
		re.Event.Stack = deserializeStackFrames(framesJSON)
		if re.Event.Stack == nil {
			re.Event.Stack = deserializeStackIPs(ipsJSON)
		}
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

// DB returns the underlying *sql.DB for direct queries (e.g., MCP get_stacks).
// Callers must not Close() the returned DB — the Store owns the lifecycle.
func (s *Store) DB() *sql.DB {
	return s.db
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

// Compact reclaims wasted space at shutdown. Runs WAL checkpoint to merge
// the write-ahead log, then VACUUM if >20% of pages are free. The final
// checkpoint merges the VACUUM result into the main file. Safe to call
// after all writes are done (WaitDone + StopSession completed).
func (s *Store) Compact() error {
	if s.dbPath == "" || s.dbPath == ":memory:" {
		return nil
	}

	// Merge WAL into main DB file — always fast.
	s.db.Exec("PRAGMA wal_checkpoint(TRUNCATE)")

	// Check free page ratio.
	var pageCount, freeCount int64
	s.db.QueryRow("PRAGMA page_count").Scan(&pageCount)
	s.db.QueryRow("PRAGMA freelist_count").Scan(&freeCount)

	if pageCount == 0 || freeCount == 0 {
		return nil
	}

	freeRatio := float64(freeCount) / float64(pageCount)
	if freeRatio < 0.20 {
		return nil // less than 20% free — not worth it
	}

	log.Printf("compacting database: %d pages (%d free, %.0f%%)", pageCount, freeCount, freeRatio*100)

	// VACUUM rebuilds the entire file, reclaiming all free pages. This works
	// even when incremental_vacuum only partially reclaims (scattered pages).
	// Runs through a single pooled connection — fine for shutdown.
	if _, err := s.db.Exec("VACUUM"); err != nil {
		return err
	}

	// Merge any WAL entries created by VACUUM into the main file so the
	// on-disk size reflects the compaction.
	_, err := s.db.Exec("PRAGMA wal_checkpoint(TRUNCATE)")
	return err
}

// ExecuteReadOnly runs a read-only SQL query against the database and returns
// the column names, row data, and whether the result was truncated by maxRows.
//
// Validation layers (defense-in-depth):
//  1. First keyword must be SELECT, WITH, or EXPLAIN
//  2. Write keywords (INSERT/UPDATE/DELETE/DROP/ALTER/CREATE/ATTACH/REPLACE)
//     rejected anywhere in the query — blocks writable CTEs like
//     WITH x AS (...) DELETE FROM events RETURNING *
//  3. Multi-statement queries rejected (semicolon + non-whitespace)
//
// maxRows caps returned rows (default 1000, max 10000). Fetches maxRows+1
// internally to detect truncation accurately.
func (s *Store) ExecuteReadOnly(ctx context.Context, query string, maxRows int) ([]string, [][]any, bool, error) {
	// Default and cap row limit.
	if maxRows <= 0 {
		maxRows = 1000
	}
	if maxRows > 10000 {
		maxRows = 10000
	}

	// Validate: first keyword must be SELECT, WITH, or EXPLAIN.
	trimmed := strings.TrimSpace(query)
	if trimmed == "" {
		return nil, nil, false, fmt.Errorf("empty query")
	}
	firstWord := strings.ToUpper(strings.Fields(trimmed)[0])
	switch firstWord {
	case "SELECT", "WITH", "EXPLAIN":
		// allowed
	default:
		return nil, nil, false, fmt.Errorf("only SELECT, WITH, and EXPLAIN queries are allowed (got %s)", firstWord)
	}

	// Reject write keywords anywhere in the query. This blocks writable CTEs
	// (WITH ... DELETE/INSERT/UPDATE) which pass the first-word check.
	// SQL comments (/* */ and --) are included in the uppercase string, so
	// commented-out writes are also rejected — safe direction (false positive).
	upper := strings.ToUpper(trimmed)
	for _, kw := range []string{
		"INSERT ", "UPDATE ", "DELETE ", "DROP ", "ALTER ",
		"CREATE ", "ATTACH ", "DETACH ", "REPLACE ", "REINDEX", "VACUUM",
	} {
		if strings.Contains(upper, kw) {
			return nil, nil, false, fmt.Errorf("write operations are not allowed in read-only queries")
		}
	}

	// Reject multi-statement queries: no semicolons followed by non-whitespace.
	// Find the first semicolon and check if anything meaningful follows it.
	if idx := strings.Index(trimmed, ";"); idx >= 0 {
		rest := strings.TrimSpace(trimmed[idx+1:])
		if rest != "" {
			return nil, nil, false, fmt.Errorf("multi-statement queries are not allowed")
		}
	}

	rows, err := s.db.QueryContext(ctx, trimmed)
	if err != nil {
		return nil, nil, false, fmt.Errorf("query execution failed: %w", err)
	}
	defer rows.Close()

	cols, err := rows.Columns()
	if err != nil {
		return nil, nil, false, fmt.Errorf("reading columns: %w", err)
	}

	// Fetch maxRows+1 to detect truncation accurately. If we get exactly
	// maxRows, we don't know if there were more — the +1 row tells us.
	result := make([][]any, 0)
	for rows.Next() {
		if len(result) > maxRows {
			break
		}
		// Create scan targets — database/sql needs pointers to any.
		vals := make([]any, len(cols))
		ptrs := make([]any, len(cols))
		for i := range vals {
			ptrs[i] = &vals[i]
		}
		if err := rows.Scan(ptrs...); err != nil {
			return nil, nil, false, fmt.Errorf("scanning row: %w", err)
		}
		// Normalize []byte → string so JSON marshal doesn't base64-encode
		// TEXT column values. modernc.org/sqlite may return []byte for TEXT.
		for i, v := range vals {
			if b, ok := v.([]byte); ok {
				vals[i] = string(b)
			}
		}
		result = append(result, vals)
	}
	if err := rows.Err(); err != nil {
		return nil, nil, false, fmt.Errorf("iterating rows: %w", err)
	}

	truncated := len(result) > maxRows
	if truncated {
		result = result[:maxRows]
	}

	return cols, result, truncated, nil
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
	Node            string // multi-node identity (v0.9)
}

// TimelineEntry is a single event in a causal chain timeline.
type TimelineEntry struct {
	Layer    string `json:"layer"`
	Op       string `json:"op"`
	Detail   string `json:"detail"`
	DurationUS int64 `json:"dur_us,omitempty"`
}

// ExpireChains deletes chains older than the given duration.
// Called when the system is healthy (no new chains detected) so stale
// chains don't persist indefinitely in the dashboard.
func (s *Store) ExpireChains(maxAge time.Duration) {
	cutoff := time.Now().Add(-maxAge).UnixNano()
	s.db.Exec("DELETE FROM causal_chains WHERE detected_at < ?", cutoff)
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
		(id, detected_at, severity, summary, root_cause, explanation, recommendations, cuda_op, cuda_p99_us, cuda_p50_us, tail_ratio, timeline, node)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
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
			ch.Node,
		)
	}

	tx.Commit()
}

// QueryChains retrieves causal chains, optionally filtered by time range.
func (s *Store) QueryChains(since time.Duration) ([]StoredChain, error) {
	query := "SELECT id, detected_at, severity, summary, root_cause, explanation, recommendations, cuda_op, cuda_p99_us, cuda_p50_us, tail_ratio, timeline, node FROM causal_chains"
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
			&ch.TailRatio, &tlJSON, &ch.Node); err != nil {
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
	s.db.Exec(`INSERT OR IGNORE INTO system_snapshots (timestamp, cpu_pct, mem_pct, mem_avail, swap_mb, load_avg)
		VALUES (?, ?, ?, ?, ?, ?)`,
		snap.Timestamp.UnixNano(),
		snap.CPUPercent,
		snap.MemUsedPct,
		snap.MemAvailMB,
		snap.SwapUsedMB,
		snap.LoadAvg1,
	)
}

// CGroupSchedStat holds per-cgroup off-CPU scheduling statistics.
type CGroupSchedStat struct {
	CGroupID    uint64
	P99OffCPU   int64 // nanos
	TotalOffCPU int64 // nanos
	EventCount  int64
	WindowStart int64 // unix nanos
	WindowEnd   int64 // unix nanos
}

// RecordCGroupSchedStats writes per-cgroup scheduling statistics to SQLite.
// Called periodically (e.g., every tick) for noisy neighbor detection.
func (s *Store) RecordCGroupSchedStats(stats []CGroupSchedStat) {
	if len(stats) == 0 {
		return
	}
	tx, err := s.db.Begin()
	if err != nil {
		return
	}
	stmt, err := tx.Prepare(`INSERT OR REPLACE INTO cgroup_schedstat
		(cgroup_id, p99_off_cpu, total_off_cpu, event_count, window_start, window_end)
		VALUES (?, ?, ?, ?, ?, ?)`)
	if err != nil {
		tx.Rollback()
		return
	}
	for _, st := range stats {
		stmt.Exec(st.CGroupID, st.P99OffCPU, st.TotalOffCPU, st.EventCount, st.WindowStart, st.WindowEnd)
	}
	stmt.Close()
	tx.Commit()
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
		 kernel, os_release, cuda_ver, python_ver, ingero_ver, pid_filter, flags,
		 node, rank, local_rank, world_size)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`,
		sess.StartedAt.UnixNano(),
		sess.GPUModel, sess.GPUDriver,
		sess.CPUModel, sess.CPUCores, sess.MemTotal,
		sess.Kernel, sess.OSRelease,
		sess.CUDAVer, sess.PythonVer, sess.IngeroVer,
		sess.PIDFilter, sess.Flags,
		sess.Node, nilableInt(sess.Rank), nilableInt(sess.LocalRank), nilableInt(sess.WorldSize),
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

// RecordProcessName stores a PID→name mapping for query-time enrichment.
// Uses INSERT OR REPLACE so the most recent name wins if a PID is reused.
func (s *Store) RecordProcessName(pid uint32, name string) {
	s.db.Exec("INSERT OR REPLACE INTO process_names (pid, name, seen_at) VALUES (?, ?, ?)",
		pid, name, time.Now().UnixNano())
}

// RecordProcessNames batch-inserts PID→name mappings in a single transaction.
// Called at shutdown to persist names discovered at runtime via /proc/[pid]/comm.
// Skips empty names. Uses INSERT OR REPLACE so the most recent name wins.
func (s *Store) RecordProcessNames(names map[uint32]string) {
	if len(names) == 0 {
		return
	}

	tx, err := s.db.Begin()
	if err != nil {
		return
	}

	stmt, err := tx.Prepare("INSERT OR REPLACE INTO process_names (pid, name, seen_at) VALUES (?, ?, ?)")
	if err != nil {
		tx.Rollback()
		return
	}
	defer stmt.Close()

	now := time.Now().UnixNano()
	for pid, name := range names {
		if name == "" {
			continue
		}
		stmt.Exec(pid, name, now)
	}

	tx.Commit()
}

// RecordAggregates batch-inserts aggregate rows (one per minute-bucket per op).
// Called from the trace event loop when flushing completed minute-buckets.
//
// Each row represents one minute of a specific (source, op, pid) combination.
// The "stored" count tracks how many events were individually stored in the
// events table, so consumers can compute: discarded = count - stored.
func (s *Store) RecordAggregates(aggs []Aggregate) {
	if len(aggs) == 0 {
		return
	}

	tx, err := s.db.Begin()
	if err != nil {
		return
	}

	stmt, err := tx.Prepare(`INSERT OR REPLACE INTO event_aggregates
		(bucket, source, op, pid, count, stored, sum_dur, min_dur, max_dur, sum_arg0)
		VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)`)
	if err != nil {
		tx.Rollback()
		return
	}
	defer stmt.Close()

	for _, a := range aggs {
		stmt.Exec(a.Bucket, a.Source, a.Op, a.PID, a.Count, a.Stored, a.SumDur, a.MinDur, a.MaxDur, a.SumArg0)
	}

	tx.Commit()
}

// QueryAggregateTotals returns summarized event counts from the aggregates table
// for a given time range. This provides accurate total event counts even when
// selective storage discarded most individual events.
//
// The returned AggregateTotals.ByOp map uses human-readable op names (e.g.,
// "cudaLaunchKernel") as keys, matching the stats.OpStats.Op field.
func (s *Store) QueryAggregateTotals(q QueryParams) (AggregateTotals, error) {
	result := AggregateTotals{ByOp: make(map[string]int64)}

	query := `SELECT source, op, SUM(count), SUM(stored) FROM event_aggregates WHERE 1=1`
	var args []interface{}

	if !q.From.IsZero() {
		query += " AND bucket >= ?"
		args = append(args, q.From.UnixNano())
	} else if q.Since > 0 {
		query += " AND bucket >= ?"
		args = append(args, time.Now().Add(-q.Since).UnixNano())
	}
	if !q.To.IsZero() {
		query += " AND bucket <= ?"
		args = append(args, q.To.UnixNano())
	}
	query, args = appendPIDFilter(query, args, q, "")

	query += " GROUP BY source, op"

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return result, fmt.Errorf("querying aggregate totals: %w", err)
	}
	defer rows.Close()

	for rows.Next() {
		var source, op uint8
		var count, stored int64
		if err := rows.Scan(&source, &op, &count, &stored); err != nil {
			return result, fmt.Errorf("scanning aggregate row: %w", err)
		}
		result.TotalEvents += count
		result.StoredEvents += stored

		// Resolve op name for the ByOp map.
		evt := events.Event{Source: events.Source(source), Op: op}
		opName := evt.OpName()
		result.ByOp[opName] += count
	}
	return result, rows.Err()
}

// AggregateOpStats holds per-operation aggregate statistics from the
// event_aggregates table. Used by MCP get_trace_stats for large DBs where
// loading all individual events would time out.
type AggregateOpStats struct {
	Source uint8  // event source ID
	Op     uint8  // operation ID
	OpName string // human-readable name (e.g., "cudaMemcpy")
	Count  int64  // total events
	SumDur int64  // sum of durations (nanos)
	MinDur int64  // minimum duration (nanos)
	MaxDur int64  // maximum duration (nanos)
}

// QueryAggregatePerOp returns per-operation aggregate statistics from the
// event_aggregates table. Groups by (source, op) and returns count, sum_dur,
// min_dur, max_dur for each operation. Supports PID filtering via QueryParams.
//
// This is the "fast path" for get_trace_stats on large DBs (>500K events):
// instead of loading all events into memory for percentile calculation, it
// reads pre-computed aggregates and returns count/avg/min/max.
func (s *Store) QueryAggregatePerOp(q QueryParams) ([]AggregateOpStats, error) {
	query := `SELECT source, op, SUM(count), SUM(sum_dur), MIN(min_dur), MAX(max_dur)
		FROM event_aggregates WHERE 1=1`
	var args []interface{}

	if !q.From.IsZero() {
		query += " AND bucket >= ?"
		args = append(args, q.From.UnixNano())
	} else if q.Since > 0 {
		query += " AND bucket >= ?"
		args = append(args, time.Now().Add(-q.Since).UnixNano())
	}
	if !q.To.IsZero() {
		query += " AND bucket <= ?"
		args = append(args, q.To.UnixNano())
	}
	query, args = appendPIDFilter(query, args, q, "")

	query += " GROUP BY source, op ORDER BY SUM(count) DESC"

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("querying aggregate per-op: %w", err)
	}
	defer rows.Close()

	var result []AggregateOpStats
	for rows.Next() {
		var a AggregateOpStats
		if err := rows.Scan(&a.Source, &a.Op, &a.Count, &a.SumDur, &a.MinDur, &a.MaxDur); err != nil {
			return nil, fmt.Errorf("scanning aggregate per-op row: %w", err)
		}
		// Resolve human-readable op name.
		evt := events.Event{Source: events.Source(a.Source), Op: a.Op}
		a.OpName = evt.OpName()
		result = append(result, a)
	}
	return result, rows.Err()
}

// ProcessOpStats holds per-process, per-operation aggregate statistics.
// Used by explain --per-process for RAG pipeline contention diagnosis.
type ProcessOpStats struct {
	PID    uint32
	Source uint8
	Op     uint8
	OpName string
	Count  int64
	SumDur int64 // nanos
	MinDur int64
	MaxDur int64
}

// QueryAggregatePerProcess returns per-process, per-operation aggregate statistics.
// Groups by (pid, source, op) to show which processes use which CUDA/host ops.
func (s *Store) QueryAggregatePerProcess(q QueryParams) ([]ProcessOpStats, error) {
	query := `SELECT pid, source, op, SUM(count), SUM(sum_dur), MIN(min_dur), MAX(max_dur)
		FROM event_aggregates WHERE 1=1`
	var args []interface{}

	if !q.From.IsZero() {
		query += " AND bucket >= ?"
		args = append(args, q.From.UnixNano())
	} else if q.Since > 0 {
		query += " AND bucket >= ?"
		args = append(args, time.Now().Add(-q.Since).UnixNano())
	}
	if !q.To.IsZero() {
		query += " AND bucket <= ?"
		args = append(args, q.To.UnixNano())
	}
	query, args = appendPIDFilter(query, args, q, "")

	query += " GROUP BY pid, source, op ORDER BY pid, SUM(count) DESC"

	rows, err := s.db.Query(query, args...)
	if err != nil {
		return nil, fmt.Errorf("querying aggregate per-process: %w", err)
	}
	defer rows.Close()

	var result []ProcessOpStats
	for rows.Next() {
		var a ProcessOpStats
		if err := rows.Scan(&a.PID, &a.Source, &a.Op, &a.Count, &a.SumDur, &a.MinDur, &a.MaxDur); err != nil {
			return nil, fmt.Errorf("scanning aggregate per-process row: %w", err)
		}
		evt := events.Event{Source: events.Source(a.Source), Op: a.Op}
		a.OpName = evt.OpName()
		result = append(result, a)
	}
	return result, rows.Err()
}

// QuerySessions retrieves sessions, optionally filtered by time range.
func (s *Store) QuerySessions(since time.Duration) ([]Session, error) {
	query := `SELECT id, started_at, stopped_at, gpu_model, gpu_driver,
		cpu_model, cpu_cores, mem_total, kernel, os_release,
		cuda_ver, python_ver, ingero_ver, pid_filter, flags,
		node, rank, local_rank, world_size
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
			rank              sql.NullInt64
			localRank         sql.NullInt64
			worldSize         sql.NullInt64
		)
		if err := rows.Scan(&sess.ID, &startNanos, &stopNanos,
			&sess.GPUModel, &sess.GPUDriver,
			&sess.CPUModel, &sess.CPUCores, &sess.MemTotal,
			&sess.Kernel, &sess.OSRelease,
			&sess.CUDAVer, &sess.PythonVer, &sess.IngeroVer,
			&sess.PIDFilter, &sess.Flags,
			&sess.Node, &rank, &localRank, &worldSize); err != nil {
			return nil, fmt.Errorf("scanning session row: %w", err)
		}
		if rank.Valid {
			v := int(rank.Int64)
			sess.Rank = &v
		}
		if localRank.Valid {
			v := int(localRank.Int64)
			sess.LocalRank = &v
		}
		if worldSize.Valid {
			v := int(worldSize.Int64)
			sess.WorldSize = &v
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

// hashStackIPs delegates to the exported events.HashStackIPs. Kept as a
// package-local alias for backward compat with callers inside store.
func hashStackIPs(stack []events.StackFrame) uint64 {
	return events.HashStackIPs(stack)
}

// hashStack returns an ASLR-independent hash (HashStackSymbols) when the
// stack has at least one resolved field (native symbol/file, or Python
// file/func). Falls back to HashStackIPs for fully-unresolved stacks
// (e.g. first events before /proc/[pid]/maps is loaded). This fixes 16x
// stack_traces bloat on x86_64 caused by ASLR giving identical logical
// stacks different raw IPs per PID.
//
// The resolution check mirrors serializeStackFrames: a frame is "resolved"
// if any of SymbolName, File, PyFile, or PyFunc is non-empty.
func hashStack(stack []events.StackFrame) uint64 {
	for _, f := range stack {
		if f.SymbolName != "" || f.File != "" || f.PyFile != "" || f.PyFunc != "" {
			return events.HashStackSymbols(stack)
		}
	}
	return events.HashStackIPs(stack)
}

// compactFrame is a space-efficient JSON representation of a resolved stack frame.
// Short keys save ~60% compared to full field names when stored in SQLite.
type compactFrame struct {
	IP     string `json:"i,omitempty"`   // hex IP
	Symbol string `json:"s,omitempty"`   // resolved native symbol
	File   string `json:"f,omitempty"`   // .so basename (not full path)
	Line   int    `json:"l,omitempty"`   // source line (if DWARF)
	PyFile string `json:"pf,omitempty"`  // Python source file
	PyFunc string `json:"pfn,omitempty"` // Python function name
	PyLine int    `json:"pl,omitempty"`  // Python source line
}

// serializeStackFrames converts resolved stack frames to compact JSON for storage.
// Skips garbage frames (no symbol, no file, no Python info). Returns "" if all
// frames are garbage or the stack is empty.
func serializeStackFrames(stack []events.StackFrame) string {
	if len(stack) == 0 {
		return ""
	}
	var frames []compactFrame
	for _, f := range stack {
		// Skip garbage frames: no resolved info at all.
		if f.SymbolName == "" && f.File == "" && f.PyFile == "" && f.PyFunc == "" {
			continue
		}
		base := ""
		if f.File != "" {
			base = filepath.Base(f.File)
		}
		cf := compactFrame{
			Symbol: f.SymbolName,
			File:   base,
			Line:   f.Line,
			PyFile: f.PyFile,
			PyFunc: f.PyFunc,
			PyLine: f.PyLine,
		}
		// Only store IP if we have no symbol (partial resolution).
		if f.SymbolName == "" && f.IP != 0 {
			cf.IP = fmt.Sprintf("0x%x", f.IP)
		}
		frames = append(frames, cf)
	}
	if len(frames) == 0 {
		return ""
	}
	b, _ := json.Marshal(frames)
	return string(b)
}

// deserializeStackFrames parses compact JSON back to StackFrame slice.
// Returns nil for empty strings or parse errors (backward compat with old DBs).
func deserializeStackFrames(s string) []events.StackFrame {
	if s == "" {
		return nil
	}
	var frames []compactFrame
	if err := json.Unmarshal([]byte(s), &frames); err != nil {
		return nil
	}
	if len(frames) == 0 {
		return nil
	}
	result := make([]events.StackFrame, len(frames))
	for i, cf := range frames {
		result[i] = events.StackFrame{
			SymbolName: cf.Symbol,
			File:       cf.File,
			Line:       cf.Line,
			PyFile:     cf.PyFile,
			PyFunc:     cf.PyFunc,
			PyLine:     cf.PyLine,
		}
		if cf.IP != "" {
			fmt.Sscanf(cf.IP, "0x%x", &result[i].IP)
		}
	}
	return result
}

// migrateInlineStacks migrates events that have inline stack_ips (TEXT) but no
// stack_hash. This handles databases created before the interning optimization.
// Runs once on New() — if no events have inline stacks, it's a no-op.
func migrateInlineStacks(db *sql.DB) {
	// Check if the old stack_ips column exists. If not, nothing to migrate.
	var hasStackIPs bool
	rows, err := db.Query("PRAGMA table_info(events)")
	if err != nil {
		return
	}
	defer rows.Close()
	for rows.Next() {
		var cid int
		var name, typ string
		var notnull int
		var dflt sql.NullString
		var pk int
		if err := rows.Scan(&cid, &name, &typ, &notnull, &dflt, &pk); err != nil {
			continue
		}
		if name == "stack_ips" {
			hasStackIPs = true
		}
	}
	if !hasStackIPs {
		return
	}

	// Count events with inline stacks that need migration.
	var count int64
	db.QueryRow("SELECT COUNT(*) FROM events WHERE stack_ips != '' AND stack_hash = 0").Scan(&count)
	if count == 0 {
		return
	}

	// Migrate in batches of 10K. Loop until no more unmigrated rows remain.
	// Each iteration: read a batch of inline stacks, compute hashes, intern
	// unique stacks into stack_traces, update event rows with their hash.
	type migEntry struct {
		id      int64
		ipsJSON string
		hash    int64
	}

	for {
		migRows, err := db.Query("SELECT id, stack_ips FROM events WHERE stack_ips != '' AND stack_hash = 0 LIMIT 10000")
		if err != nil {
			return
		}

		var entries []migEntry
		for migRows.Next() {
			var id int64
			var ipsJSON string
			if err := migRows.Scan(&id, &ipsJSON); err != nil {
				continue
			}
			frames := deserializeStackIPs(ipsJSON)
			if len(frames) == 0 {
				continue
			}
			h := hashStackIPs(frames)
			entries = append(entries, migEntry{id: id, ipsJSON: ipsJSON, hash: int64(h)})
		}
		migRows.Close()

		if len(entries) == 0 {
			break // no more rows to migrate
		}

		tx, err := db.Begin()
		if err != nil {
			return
		}

		// Insert unique stacks.
		stackStmt, err := tx.Prepare("INSERT OR IGNORE INTO stack_traces (hash, ips) VALUES (?, ?)")
		if err != nil {
			tx.Rollback()
			return
		}
		inserted := make(map[int64]bool)
		for _, e := range entries {
			if !inserted[e.hash] {
				stackStmt.Exec(e.hash, e.ipsJSON)
				inserted[e.hash] = true
			}
		}
		stackStmt.Close()

		// Update events with hash.
		updStmt, err := tx.Prepare("UPDATE events SET stack_hash = ? WHERE id = ?")
		if err != nil {
			tx.Rollback()
			return
		}
		for _, e := range entries {
			updStmt.Exec(e.hash, e.id)
		}
		updStmt.Close()

		tx.Commit()
	}
}

// nilableInt converts a *int to a value suitable for SQL insertion.
// Returns nil (SQL NULL) if the pointer is nil, otherwise the int value.
func nilableInt(p *int) interface{} {
	if p == nil {
		return nil
	}
	return *p
}
