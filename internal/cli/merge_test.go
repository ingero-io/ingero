package cli

import (
	"context"
	"database/sql"
	"fmt"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ingero-io/ingero/internal/store"
	"github.com/ingero-io/ingero/pkg/events"
)

// createTestDB creates a test database with events and returns its path.
func createTestDB(t *testing.T, dir, nodeName string, eventCount int) string {
	t.Helper()
	dbPath := filepath.Join(dir, nodeName+".db")
	s, err := store.New(dbPath)
	if err != nil {
		t.Fatalf("creating test DB %s: %v", dbPath, err)
	}

	s.SetNode(nodeName)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { s.Run(ctx); close(done) }()

	for i := 0; i < eventCount; i++ {
		s.Record(events.Event{
			Timestamp: time.Now(),
			PID:       uint32(1000 + i%10),
			TID:       uint32(2000 + i%10),
			Source:    events.SourceCUDA,
			Op:        uint8(events.CUDAMalloc),
			Duration:  time.Millisecond,
			Node:      nodeName,
		})
	}

	time.Sleep(300 * time.Millisecond)
	cancel()
	<-done

	// Also insert a chain.
	s.RecordChains([]store.StoredChain{
		{
			ID:          fmt.Sprintf("%s:test-chain", nodeName),
			DetectedAt:  time.Now(),
			Severity:    "HIGH",
			Summary:     fmt.Sprintf("test chain from %s", nodeName),
			RootCause:   "test",
			Explanation: "test",
			Node:        nodeName,
		},
	})

	s.Close()
	return dbPath
}

func TestMergeTwoDBs(t *testing.T) {
	dir := t.TempDir()
	db1 := createTestDB(t, dir, "node-a", 10)
	db2 := createTestDB(t, dir, "node-b", 15)
	outPath := filepath.Join(dir, "merged.db")

	// Run merge.
	mergeOutput = outPath
	mergeForceNode = ""
	err := mergeRunE(nil, []string{db1, db2})
	if err != nil {
		t.Fatalf("merge failed: %v", err)
	}

	// Open merged DB and verify.
	merged, err := store.New(outPath)
	if err != nil {
		t.Fatalf("opening merged DB: %v", err)
	}
	defer merged.Close()

	// Check event count.
	evts, err := merged.Query(store.QueryParams{Limit: -1})
	if err != nil {
		t.Fatalf("querying merged DB: %v", err)
	}
	if len(evts) != 25 {
		t.Errorf("events = %d, want 25 (10 + 15)", len(evts))
	}

	// Check chains.
	chains, err := merged.QueryChains(0)
	if err != nil {
		t.Fatalf("querying chains: %v", err)
	}
	if len(chains) != 2 {
		t.Errorf("chains = %d, want 2", len(chains))
	}
}

func TestMergeThreeDBs(t *testing.T) {
	dir := t.TempDir()
	db1 := createTestDB(t, dir, "node-a", 5)
	db2 := createTestDB(t, dir, "node-b", 5)
	db3 := createTestDB(t, dir, "node-c", 5)
	outPath := filepath.Join(dir, "merged.db")

	mergeOutput = outPath
	mergeForceNode = ""
	err := mergeRunE(nil, []string{db1, db2, db3})
	if err != nil {
		t.Fatalf("merge failed: %v", err)
	}

	merged, err := store.New(outPath)
	if err != nil {
		t.Fatalf("opening merged: %v", err)
	}
	defer merged.Close()

	evts, _ := merged.Query(store.QueryParams{Limit: -1})
	if len(evts) != 15 {
		t.Errorf("events = %d, want 15", len(evts))
	}

	chains, _ := merged.QueryChains(0)
	if len(chains) != 3 {
		t.Errorf("chains = %d, want 3", len(chains))
	}
}

func TestMergeIDUniqueness(t *testing.T) {
	dir := t.TempDir()
	db1 := createTestDB(t, dir, "node-a", 20)
	db2 := createTestDB(t, dir, "node-b", 20)
	outPath := filepath.Join(dir, "merged.db")

	mergeOutput = outPath
	mergeForceNode = ""
	mergeRunE(nil, []string{db1, db2})

	// Check for duplicate IDs.
	db, _ := sql.Open("sqlite", outPath)
	defer db.Close()

	var dupeCount int
	db.QueryRow("SELECT COUNT(*) FROM (SELECT id FROM events GROUP BY id HAVING COUNT(*) > 1)").Scan(&dupeCount)
	if dupeCount > 0 {
		t.Errorf("found %d duplicate event IDs", dupeCount)
	}
}

func TestMergeStackDeduplication(t *testing.T) {
	dir := t.TempDir()
	db1 := createTestDB(t, dir, "node-a", 5)
	db2 := createTestDB(t, dir, "node-b", 5)

	// Insert the same stack hash in both DBs.
	for _, dbPath := range []string{db1, db2} {
		db, _ := sql.Open("sqlite", dbPath)
		db.Exec("INSERT OR REPLACE INTO stack_traces (hash, ips, frames) VALUES (12345, '[\"0x1000\"]', '')")
		db.Close()
	}

	outPath := filepath.Join(dir, "merged.db")
	mergeOutput = outPath
	mergeForceNode = ""
	mergeRunE(nil, []string{db1, db2})

	// Verify only one copy of hash 12345 exists.
	db, _ := sql.Open("sqlite", outPath)
	defer db.Close()

	var count int
	db.QueryRow("SELECT COUNT(*) FROM stack_traces WHERE hash = 12345").Scan(&count)
	if count != 1 {
		t.Errorf("stack hash 12345 count = %d, want 1 (deduplicated)", count)
	}
}

func TestMergeLegacyDBError(t *testing.T) {
	dir := t.TempDir()

	// Create a legacy DB without node columns.
	legacyPath := filepath.Join(dir, "legacy.db")
	db, _ := sql.Open("sqlite", legacyPath)
	db.Exec("PRAGMA journal_mode=WAL")
	db.Exec(`CREATE TABLE events (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		timestamp INTEGER NOT NULL, pid INTEGER NOT NULL, tid INTEGER NOT NULL,
		source INTEGER NOT NULL, op INTEGER NOT NULL, duration INTEGER NOT NULL,
		gpu_id INTEGER NOT NULL DEFAULT 0, arg0 INTEGER NOT NULL DEFAULT 0,
		arg1 INTEGER NOT NULL DEFAULT 0, ret_code INTEGER NOT NULL DEFAULT 0,
		stack_hash INTEGER NOT NULL DEFAULT 0
	)`)
	db.Exec("INSERT INTO events (timestamp, pid, tid, source, op, duration) VALUES (1000, 1, 2, 1, 1, 100)")
	db.Close()

	outPath := filepath.Join(dir, "merged.db")
	mergeOutput = outPath
	mergeForceNode = ""
	err := mergeRunE(nil, []string{legacyPath})
	if err == nil {
		t.Error("expected error for legacy DB without node column")
	}
	if err != nil && !contains(err.Error(), "missing node column") {
		t.Errorf("expected 'missing node column' error, got: %v", err)
	}
}

func TestMergeForceNode(t *testing.T) {
	dir := t.TempDir()

	// Create legacy DB.
	legacyPath := filepath.Join(dir, "legacy.db")
	db, _ := sql.Open("sqlite", legacyPath)
	db.Exec("PRAGMA journal_mode=WAL")
	db.Exec(`CREATE TABLE events (
		id INTEGER PRIMARY KEY AUTOINCREMENT,
		timestamp INTEGER NOT NULL, pid INTEGER NOT NULL, tid INTEGER NOT NULL,
		source INTEGER NOT NULL, op INTEGER NOT NULL, duration INTEGER NOT NULL,
		gpu_id INTEGER NOT NULL DEFAULT 0, arg0 INTEGER NOT NULL DEFAULT 0,
		arg1 INTEGER NOT NULL DEFAULT 0, ret_code INTEGER NOT NULL DEFAULT 0,
		stack_hash INTEGER NOT NULL DEFAULT 0, cgroup_id INTEGER NOT NULL DEFAULT 0
	)`)
	db.Exec("INSERT INTO events (timestamp, pid, tid, source, op, duration) VALUES (1000, 123, 124, 1, 1, 500)")
	db.Exec("INSERT INTO events (timestamp, pid, tid, source, op, duration) VALUES (2000, 123, 124, 1, 2, 300)")
	db.Close()

	outPath := filepath.Join(dir, "merged.db")
	mergeOutput = outPath
	mergeForceNode = "legacy-node"
	err := mergeRunE(nil, []string{legacyPath})
	if err != nil {
		t.Fatalf("merge with --force-node failed: %v", err)
	}

	// Verify node was assigned.
	merged, _ := store.New(outPath)
	defer merged.Close()

	evts, _ := merged.Query(store.QueryParams{Limit: -1})
	if len(evts) != 2 {
		t.Fatalf("events = %d, want 2", len(evts))
	}
	for _, e := range evts {
		if e.Node != "legacy-node" {
			t.Errorf("event node = %q, want %q", e.Node, "legacy-node")
		}
	}
}

func TestMergeNodeAttribution(t *testing.T) {
	dir := t.TempDir()
	db1 := createTestDB(t, dir, "node-x", 5)
	db2 := createTestDB(t, dir, "node-y", 5)
	outPath := filepath.Join(dir, "merged.db")

	mergeOutput = outPath
	mergeForceNode = ""
	mergeRunE(nil, []string{db1, db2})

	// Query events per node.
	merged, _ := store.New(outPath)
	defer merged.Close()

	var nodeXCount, nodeYCount int
	merged.DB().QueryRow("SELECT COUNT(*) FROM events WHERE node = 'node-x'").Scan(&nodeXCount)
	merged.DB().QueryRow("SELECT COUNT(*) FROM events WHERE node = 'node-y'").Scan(&nodeYCount)

	if nodeXCount != 5 {
		t.Errorf("node-x events = %d, want 5", nodeXCount)
	}
	if nodeYCount != 5 {
		t.Errorf("node-y events = %d, want 5", nodeYCount)
	}
}

func TestMergeOutputCollision(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "same.db")
	os.WriteFile(dbPath, []byte{}, 0o644)

	mergeOutput = dbPath
	mergeForceNode = ""
	err := mergeRunE(nil, []string{dbPath})
	if err == nil {
		t.Error("expected error when output collides with source")
	}
}
