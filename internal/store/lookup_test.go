package store

import (
	"context"
	"database/sql"
	"os"
	"testing"
	"time"

	"github.com/ingero-io/ingero/pkg/events"
)

func TestCausalChainsRoundTrip(t *testing.T) {
	path := "/tmp/test-chains-roundtrip.db"
	os.Remove(path)
	defer os.Remove(path)

	s, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	chains := []StoredChain{
		{
			ID:              "chain-001",
			DetectedAt:      time.Now(),
			Severity:        "HIGH",
			Summary:         "cudaStreamSync p99=142ms (8.9x p50) — high CPU + 20 sched_switch",
			RootCause:       "high CPU utilization + 20 sched_switch events",
			Explanation:     "cudaStreamSync tail latency is 8.9x higher than typical.",
			Recommendations: []string{"Pin process to dedicated cores", "nice -n 19 background jobs"},
			CUDAOp:          "cudaStreamSync",
			CUDAP99us:       142000,
			CUDAP50us:       16000,
			TailRatio:       8.9,
			Timeline: []TimelineEntry{
				{Layer: "SYSTEM", Op: "cpu", Detail: "CPU 94%"},
				{Layer: "HOST", Op: "sched_switch", Detail: "20 switches (40ms off-CPU)", DurationUS: 40000},
				{Layer: "CUDA", Op: "cudaStreamSync", Detail: "p99=142ms (8.9x p50)", DurationUS: 142000},
			},
		},
		{
			ID:              "chain-002",
			DetectedAt:      time.Now(),
			Severity:        "HIGH",
			Summary:         "OOM killer triggered 1 time(s)",
			RootCause:       "host memory exhaustion",
			Explanation:     "The Linux OOM killer was invoked.",
			Recommendations: []string{"Reduce memory usage", "Add swap"},
			CUDAOp:          "all",
		},
	}
	s.RecordChains(chains)

	got, err := s.QueryChains(0)
	if err != nil {
		t.Fatalf("QueryChains: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("got %d chains, want 2", len(got))
	}

	// DESC order: chain-002 first.
	if got[0].ID != "chain-002" {
		t.Errorf("first chain ID = %q, want chain-002", got[0].ID)
	}
	if got[1].ID != "chain-001" {
		t.Errorf("second chain ID = %q, want chain-001", got[1].ID)
	}
	if len(got[1].Timeline) != 3 {
		t.Errorf("timeline entries = %d, want 3", len(got[1].Timeline))
	}
	if len(got[1].Recommendations) != 2 {
		t.Errorf("recommendations = %d, want 2", len(got[1].Recommendations))
	}
	if got[1].CUDAP99us != 142000 {
		t.Errorf("CUDAP99us = %d, want 142000", got[1].CUDAP99us)
	}

	// Verify raw SQL works.
	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("sql.Open: %v", err)
	}
	defer db.Close()

	var count int
	db.QueryRow("SELECT count(*) FROM causal_chains WHERE severity = 'HIGH'").Scan(&count)
	if count != 2 {
		t.Errorf("SQL HIGH count = %d, want 2", count)
	}
}

func TestLookupTablesCreated(t *testing.T) {
	path := "/tmp/test-lookup-tables.db"
	os.Remove(path)
	defer os.Remove(path)

	s, err := New(path)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	db, err := sql.Open("sqlite", path)
	if err != nil {
		t.Fatalf("open: %v", err)
	}
	defer db.Close()

	// Check sources table exists and has rows.
	var srcCount int
	if err := db.QueryRow("SELECT COUNT(*) FROM sources").Scan(&srcCount); err != nil {
		t.Fatalf("sources query: %v", err)
	}
	if srcCount != 4 {
		t.Errorf("sources: got %d rows, want 4", srcCount)
	}

	// Check ops table.
	var opsCount int
	if err := db.QueryRow("SELECT COUNT(*) FROM ops").Scan(&opsCount); err != nil {
		t.Fatalf("ops query: %v", err)
	}
	if opsCount != 19 {
		t.Errorf("ops: got %d rows, want 19", opsCount)
	}

	// Check schema_info.
	var infoCount int
	if err := db.QueryRow("SELECT COUNT(*) FROM schema_info").Scan(&infoCount); err != nil {
		t.Fatalf("schema_info query: %v", err)
	}
	if infoCount < 5 {
		t.Errorf("schema_info: got %d rows, want >= 5", infoCount)
	}

	// Check the JOIN query works.
	_, err = db.Query(`
		SELECT e.id, s.name, o.name, e.duration/1000, e.pid
		FROM events e
		JOIN sources s ON e.source = s.id
		JOIN ops o ON e.source = o.source_id AND e.op = o.op_id
		LIMIT 1
	`)
	if err != nil {
		t.Fatalf("JOIN query failed: %v", err)
	}
}

func TestQueryRich(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		s.Run(ctx)
		close(done)
	}()

	// Record events across different sources.
	evts := []events.Event{
		{Timestamp: time.Now(), PID: 100, TID: 100, Source: events.SourceCUDA, Op: uint8(events.CUDAMalloc), Duration: 5 * time.Microsecond},
		{Timestamp: time.Now(), PID: 100, TID: 100, Source: events.SourceCUDA, Op: uint8(events.CUDALaunchKernel), Duration: 10 * time.Microsecond},
		{Timestamp: time.Now(), PID: 100, TID: 100, Source: events.SourceHost, Op: uint8(events.HostSchedSwitch), Duration: 2 * time.Millisecond},
		{Timestamp: time.Now(), PID: 100, TID: 100, Source: events.SourceDriver, Op: uint8(events.DriverLaunchKernel), Duration: 8 * time.Microsecond},
	}
	for _, e := range evts {
		s.Record(e)
	}
	time.Sleep(300 * time.Millisecond)

	// Query with JOIN.
	rich, err := s.QueryRich(QueryParams{Since: 1 * time.Minute})
	if err != nil {
		t.Fatalf("QueryRich: %v", err)
	}
	if len(rich) != 4 {
		t.Fatalf("got %d events, want 4", len(rich))
	}

	// Verify enriched names from lookup tables.
	found := map[string]bool{}
	for _, re := range rich {
		found[re.OpName] = true
		if re.SourceName == "" {
			t.Errorf("empty SourceName for op %d", re.Op)
		}
		if re.OpDesc == "" {
			t.Errorf("empty OpDesc for %s", re.OpName)
		}
	}
	for _, want := range []string{"cudaMalloc", "cudaLaunchKernel", "sched_switch", "cuLaunchKernel"} {
		if !found[want] {
			t.Errorf("missing op %q in QueryRich results", want)
		}
	}

	// Verify source names.
	for _, re := range rich {
		switch re.Event.Source {
		case events.SourceCUDA:
			if re.SourceName != "CUDA" {
				t.Errorf("CUDA source name = %q, want CUDA", re.SourceName)
			}
		case events.SourceHost:
			if re.SourceName != "HOST" {
				t.Errorf("HOST source name = %q, want HOST", re.SourceName)
			}
		case events.SourceDriver:
			if re.SourceName != "DRIVER" {
				t.Errorf("DRIVER source name = %q, want DRIVER", re.SourceName)
			}
		}
	}

	cancel()
	<-done
}

func TestOpDescriptions(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	descs := s.OpDescriptions()
	if len(descs) != 19 {
		t.Errorf("OpDescriptions: got %d, want 19", len(descs))
	}
	if descs["cudaMalloc"] != "GPU memory allocation" {
		t.Errorf("cudaMalloc desc = %q", descs["cudaMalloc"])
	}
	if descs["cuLaunchKernel"] == "" {
		t.Error("cuLaunchKernel description missing")
	}
}

func TestSchemaInfo(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	info := s.SchemaInfo()
	if info["version"] != "0.7" {
		t.Errorf("version = %q, want 0.7", info["version"])
	}
	if info["timestamp_unit"] == "" {
		t.Error("timestamp_unit missing")
	}
	if info["example_query"] == "" {
		t.Error("example_query missing")
	}
}

func TestChainsNilSliceJSON(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	// Chain with nil Recommendations and nil Timeline — should marshal as []
	chains := []StoredChain{{
		ID:         "chain-nil",
		DetectedAt: time.Now(),
		Severity:   "LOW",
		Summary:    "test nil slices",
		RootCause:  "test",
		Explanation: "test",
	}}
	s.RecordChains(chains)

	got, err := s.QueryChains(0)
	if err != nil {
		t.Fatalf("QueryChains: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d chains, want 1", len(got))
	}
	// nil slices should round-trip as empty (not nil) due to [] JSON.
	if got[0].Recommendations == nil {
		t.Error("Recommendations should be [] not nil after round-trip")
	}
	if got[0].Timeline == nil {
		t.Error("Timeline should be [] not nil after round-trip")
	}
}
