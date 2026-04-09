package store

import (
	"context"
	"database/sql"
	"fmt"
	"regexp"
	"strings"
	"testing"
	"time"

	"github.com/ingero-io/ingero/pkg/events"
)

// intPtr returns a pointer to an int value.
func intPtr(v int) *int { return &v }

func TestNodeColumnPresence(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	// Verify node column exists on all three tables.
	for _, table := range []string{"events", "sessions", "causal_chains"} {
		var found bool
		rows, err := s.db.Query(fmt.Sprintf("PRAGMA table_info(%s)", table))
		if err != nil {
			t.Fatalf("PRAGMA table_info(%s): %v", table, err)
		}
		for rows.Next() {
			var cid int
			var name, typ string
			var notNull, pk int
			var dflt sql.NullString
			rows.Scan(&cid, &name, &typ, &notNull, &dflt, &pk)
			if name == "node" {
				found = true
			}
		}
		rows.Close()
		if !found {
			t.Errorf("table %s missing 'node' column", table)
		}
	}
}

func TestNodeRankColumnsOnEvents(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	// Check that rank, local_rank, world_size columns exist on events table.
	expectedCols := map[string]bool{
		"node": false, "rank": false, "local_rank": false, "world_size": false,
	}
	rows, err := s.db.Query("PRAGMA table_info(events)")
	if err != nil {
		t.Fatalf("PRAGMA table_info(events): %v", err)
	}
	for rows.Next() {
		var cid int
		var name, typ string
		var notNull, pk int
		var dflt sql.NullString
		rows.Scan(&cid, &name, &typ, &notNull, &dflt, &pk)
		if _, ok := expectedCols[name]; ok {
			expectedCols[name] = true
		}
	}
	rows.Close()

	for col, found := range expectedCols {
		if !found {
			t.Errorf("events table missing column %q", col)
		}
	}
}

func TestNodeNamespacedEventIDs(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	s.SetNode("gpu-node-07")

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { s.Run(ctx); close(done) }()

	for i := 0; i < 5; i++ {
		s.Record(makeEvt(events.SourceCUDA, uint8(events.CUDAMalloc), time.Millisecond))
	}
	time.Sleep(300 * time.Millisecond)
	cancel()
	<-done

	// Query event IDs directly from SQLite.
	rows, err := s.db.Query("SELECT id FROM events")
	if err != nil {
		t.Fatalf("SELECT id: %v", err)
	}
	defer rows.Close()

	idPattern := regexp.MustCompile(`^gpu-node-07:\d+$`)
	count := 0
	for rows.Next() {
		var id string
		rows.Scan(&id)
		if !idPattern.MatchString(id) {
			t.Errorf("event ID %q doesn't match pattern {node}:{seq}", id)
		}
		count++
	}
	if count != 5 {
		t.Errorf("expected 5 events, got %d", count)
	}
}

func TestNodeOnEvents(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	s.SetNode("worker-3")

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { s.Run(ctx); close(done) }()

	evt := makeEvt(events.SourceCUDA, uint8(events.CUDAMalloc), time.Millisecond)
	evt.Node = "worker-3"
	evt.Rank = intPtr(2)
	evt.LocalRank = intPtr(0)
	evt.WorldSize = intPtr(4)
	s.Record(evt)

	time.Sleep(300 * time.Millisecond)
	cancel()
	<-done

	// Query back and verify fields.
	result, err := s.Query(QueryParams{Since: time.Minute})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 event, got %d", len(result))
	}
	e := result[0]
	if e.Node != "worker-3" {
		t.Errorf("node = %q, want %q", e.Node, "worker-3")
	}
	if e.Rank == nil || *e.Rank != 2 {
		t.Errorf("rank = %v, want 2", e.Rank)
	}
	if e.LocalRank == nil || *e.LocalRank != 0 {
		t.Errorf("local_rank = %v, want 0", e.LocalRank)
	}
	if e.WorldSize == nil || *e.WorldSize != 4 {
		t.Errorf("world_size = %v, want 4", e.WorldSize)
	}
}

func TestNodeOnEventsNullRank(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	s.SetNode("standalone")

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { s.Run(ctx); close(done) }()

	// Event without rank (non-distributed workload).
	evt := makeEvt(events.SourceCUDA, uint8(events.CUDAMalloc), time.Millisecond)
	evt.Node = "standalone"
	// Rank, LocalRank, WorldSize left nil.
	s.Record(evt)

	time.Sleep(300 * time.Millisecond)
	cancel()
	<-done

	result, err := s.Query(QueryParams{Since: time.Minute})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 event, got %d", len(result))
	}
	e := result[0]
	if e.Node != "standalone" {
		t.Errorf("node = %q, want %q", e.Node, "standalone")
	}
	if e.Rank != nil {
		t.Errorf("rank = %v, want nil", e.Rank)
	}
	if e.LocalRank != nil {
		t.Errorf("local_rank = %v, want nil", e.LocalRank)
	}
	if e.WorldSize != nil {
		t.Errorf("world_size = %v, want nil", e.WorldSize)
	}
}

func TestNodeNamespacedIDsNoCollision(t *testing.T) {
	// Simulate two "nodes" writing to the same DB — verify zero ID collisions.
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	// Write events as node-a via the Run pipeline.
	s.SetNode("node-a")
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { s.Run(ctx); close(done) }()

	for i := 0; i < 10; i++ {
		s.Record(makeEvt(events.SourceCUDA, uint8(events.CUDAMalloc), time.Millisecond))
	}
	time.Sleep(300 * time.Millisecond)
	cancel()
	<-done

	// Insert node-b events directly via SQL (simulating a merged DB).
	// Can't call Run() twice on the same Store — it closes runDone.
	for i := 0; i < 10; i++ {
		id := fmt.Sprintf("node-b:%d", i+1)
		s.db.Exec(`INSERT INTO events (id, timestamp, pid, tid, source, op, duration, gpu_id, arg0, arg1, ret_code, stack_hash, cgroup_id, node)
			VALUES (?, ?, 1234, 1235, 3, 1, 1000000, 0, 1000, 2000, 0, 0, 0, 'node-b')`,
			id, time.Now().UnixNano())
	}

	// Verify: 20 unique IDs, 10 with node-a prefix, 10 with node-b prefix.
	rows, err := s.db.Query("SELECT id FROM events")
	if err != nil {
		t.Fatalf("SELECT id: %v", err)
	}
	defer rows.Close()

	ids := make(map[string]bool)
	nodeACnt, nodeBCnt := 0, 0
	for rows.Next() {
		var id string
		rows.Scan(&id)
		if ids[id] {
			t.Errorf("duplicate event ID: %s", id)
		}
		ids[id] = true
		if strings.HasPrefix(id, "node-a:") {
			nodeACnt++
		} else if strings.HasPrefix(id, "node-b:") {
			nodeBCnt++
		}
	}
	if nodeACnt != 10 {
		t.Errorf("node-a events: got %d, want 10", nodeACnt)
	}
	if nodeBCnt != 10 {
		t.Errorf("node-b events: got %d, want 10", nodeBCnt)
	}
	if len(ids) != 20 {
		t.Errorf("total unique IDs: got %d, want 20", len(ids))
	}
}

func TestNodeOnSession(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	sess := Session{
		StartedAt: time.Now(),
		Node:      "gpu-node-07",
		Rank:      intPtr(3),
		LocalRank: intPtr(1),
		WorldSize: intPtr(8),
		IngeroVer: "test",
	}
	id, err := s.StartSession(sess)
	if err != nil {
		t.Fatalf("StartSession: %v", err)
	}
	if id <= 0 {
		t.Fatalf("invalid session ID: %d", id)
	}

	sessions, err := s.QuerySessions(0)
	if err != nil {
		t.Fatalf("QuerySessions: %v", err)
	}
	if len(sessions) != 1 {
		t.Fatalf("expected 1 session, got %d", len(sessions))
	}
	got := sessions[0]
	if got.Node != "gpu-node-07" {
		t.Errorf("session node = %q, want %q", got.Node, "gpu-node-07")
	}
	if got.Rank == nil || *got.Rank != 3 {
		t.Errorf("session rank = %v, want 3", got.Rank)
	}
	if got.LocalRank == nil || *got.LocalRank != 1 {
		t.Errorf("session local_rank = %v, want 1", got.LocalRank)
	}
	if got.WorldSize == nil || *got.WorldSize != 8 {
		t.Errorf("session world_size = %v, want 8", got.WorldSize)
	}
}

func TestNodeOnChains(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	chains := []StoredChain{
		{
			ID:         "gpu-node-07:tail-medium-cudaStreamSync",
			DetectedAt: time.Now(),
			Severity:   "MEDIUM",
			Summary:    "test chain",
			RootCause:  "test",
			Explanation: "test",
			Node:       "gpu-node-07",
		},
	}
	s.RecordChains(chains)

	result, err := s.QueryChains(0)
	if err != nil {
		t.Fatalf("QueryChains: %v", err)
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 chain, got %d", len(result))
	}
	if result[0].Node != "gpu-node-07" {
		t.Errorf("chain node = %q, want %q", result[0].Node, "gpu-node-07")
	}
	if result[0].ID != "gpu-node-07:tail-medium-cudaStreamSync" {
		t.Errorf("chain ID = %q, want %q", result[0].ID, "gpu-node-07:tail-medium-cudaStreamSync")
	}
}

func TestSchemaMigration(t *testing.T) {
	// Create a DB with old schema (no node columns), then open with new code.
	dir := t.TempDir()
	dbPath := dir + "/migrate_test.db"

	// Step 1: Create old-format DB without node columns.
	db, err := sql.Open("sqlite", dbPath)
	if err != nil {
		t.Fatalf("open old DB: %v", err)
	}
	db.Exec("PRAGMA journal_mode=WAL")

	// Create old events schema (INTEGER AUTOINCREMENT, no node columns).
	_, err = db.Exec(`CREATE TABLE events (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		timestamp INTEGER NOT NULL,
		pid INTEGER NOT NULL,
		tid INTEGER NOT NULL,
		source INTEGER NOT NULL,
		op INTEGER NOT NULL,
		duration INTEGER NOT NULL,
		gpu_id INTEGER NOT NULL DEFAULT 0,
		arg0 INTEGER NOT NULL DEFAULT 0,
		arg1 INTEGER NOT NULL DEFAULT 0,
		ret_code INTEGER NOT NULL DEFAULT 0,
		stack_hash INTEGER NOT NULL DEFAULT 0,
		cgroup_id INTEGER NOT NULL DEFAULT 0
	)`)
	if err != nil {
		t.Fatalf("create old events table: %v", err)
	}

	// Create old sessions schema.
	_, err = db.Exec(`CREATE TABLE sessions (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		started_at INTEGER NOT NULL,
		stopped_at INTEGER NOT NULL DEFAULT 0,
		gpu_model TEXT NOT NULL DEFAULT '',
		gpu_driver TEXT NOT NULL DEFAULT '',
		cpu_model TEXT NOT NULL DEFAULT '',
		cpu_cores INTEGER NOT NULL DEFAULT 0,
		mem_total INTEGER NOT NULL DEFAULT 0,
		kernel TEXT NOT NULL DEFAULT '',
		os_release TEXT NOT NULL DEFAULT '',
		cuda_ver TEXT NOT NULL DEFAULT '',
		python_ver TEXT NOT NULL DEFAULT '',
		ingero_ver TEXT NOT NULL DEFAULT '',
		pid_filter TEXT NOT NULL DEFAULT '',
		flags TEXT NOT NULL DEFAULT ''
	)`)
	if err != nil {
		t.Fatalf("create old sessions table: %v", err)
	}

	// Create old causal_chains schema.
	_, err = db.Exec(`CREATE TABLE causal_chains (
		id TEXT PRIMARY KEY,
		detected_at INTEGER NOT NULL,
		severity TEXT NOT NULL,
		summary TEXT NOT NULL,
		root_cause TEXT NOT NULL,
		explanation TEXT NOT NULL,
		recommendations TEXT NOT NULL DEFAULT '',
		cuda_op TEXT NOT NULL DEFAULT '',
		cuda_p99_us INTEGER NOT NULL DEFAULT 0,
		cuda_p50_us INTEGER NOT NULL DEFAULT 0,
		tail_ratio REAL NOT NULL DEFAULT 0,
		timeline TEXT NOT NULL DEFAULT ''
	)`)
	if err != nil {
		t.Fatalf("create old causal_chains table: %v", err)
	}

	// Insert test data with old schema.
	db.Exec("INSERT INTO events (timestamp, pid, tid, source, op, duration) VALUES (1000000, 123, 124, 1, 1, 5000)")
	db.Exec("INSERT INTO sessions (started_at, ingero_ver) VALUES (1000000, '0.8')")
	db.Exec("INSERT INTO causal_chains (id, detected_at, severity, summary, root_cause, explanation) VALUES ('old-chain', 1000000, 'HIGH', 'old chain', 'test', 'test')")
	db.Close()

	// Step 2: Open with new Store code — triggers migration.
	s, err := New(dbPath)
	if err != nil {
		t.Fatalf("New on old DB: %v", err)
	}
	defer s.Close()

	// Verify node columns were added.
	for _, table := range []string{"events", "sessions", "causal_chains"} {
		var found bool
		rows, err := s.db.Query(fmt.Sprintf("PRAGMA table_info(%s)", table))
		if err != nil {
			t.Fatalf("PRAGMA table_info(%s): %v", table, err)
		}
		for rows.Next() {
			var cid int
			var name, typ string
			var notNull, pk int
			var dflt sql.NullString
			rows.Scan(&cid, &name, &typ, &notNull, &dflt, &pk)
			if name == "node" {
				found = true
			}
		}
		rows.Close()
		if !found {
			t.Errorf("migration: %s still missing 'node' column", table)
		}
	}

	// Verify old data is preserved.
	var count int
	s.db.QueryRow("SELECT COUNT(*) FROM events").Scan(&count)
	if count != 1 {
		t.Errorf("migration: events count = %d, want 1", count)
	}

	s.db.QueryRow("SELECT COUNT(*) FROM sessions").Scan(&count)
	if count != 1 {
		t.Errorf("migration: sessions count = %d, want 1", count)
	}

	s.db.QueryRow("SELECT COUNT(*) FROM causal_chains").Scan(&count)
	if count != 1 {
		t.Errorf("migration: causal_chains count = %d, want 1", count)
	}

	// Verify old event has default node.
	var node string
	s.db.QueryRow("SELECT node FROM events").Scan(&node)
	if node != "" {
		t.Errorf("migration: old event node = %q, want empty string", node)
	}

	// Verify schema version is 0.10 (v0.10 added events.comm).
	var version string
	s.db.QueryRow("SELECT value FROM schema_info WHERE key = 'version'").Scan(&version)
	if version != "0.10" {
		t.Errorf("schema version = %q, want %q", version, "0.10")
	}
}

func TestSchemaVersion(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	var version string
	s.db.QueryRow("SELECT value FROM schema_info WHERE key = 'version'").Scan(&version)
	if version != "0.10" {
		t.Errorf("schema version = %q, want %q", version, "0.10")
	}
}
