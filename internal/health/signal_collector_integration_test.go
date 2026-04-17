//go:build integration

package health

import (
	"context"
	"io"
	"path/filepath"
	"testing"
	"time"

	"github.com/ingero-io/ingero/internal/store"
	"github.com/ingero-io/ingero/pkg/events"
)

// TestSQLiteCollector_EndToEnd_5sFlushPath exercises the full signal
// pipeline without the event_aggregates_5s shortcut the unit tests take:
//
//	store.New  ->  RecordAggregates5s  ->  NewSQLiteCollector  ->  Collect
//
// The unit tests seed the DB with a direct `sql.Exec`. This integration
// test uses the public store API so a regression in RecordAggregates5s,
// table creation, or QueryAggregatePerOp5s would surface here even when
// signal_collector_test.go still passes.
func TestSQLiteCollector_EndToEnd_5sFlushPath(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "integration.db")

	s, err := store.New(dbPath)
	if err != nil {
		t.Fatalf("store.New: %v", err)
	}
	defer s.Close()

	// Simulate ~90 s of a healthy workload by writing 18 consecutive 5 s
	// buckets of cudaLaunchKernel aggregates. Each bucket represents 50
	// kernel launches, for a steady rate of 10 kernels/sec.
	now := time.Now()
	const buckets = 18
	const perBucket = 50
	var aggs []store.Aggregate
	for i := 0; i < buckets; i++ {
		bucket := now.Add(-time.Duration(buckets-1-i) * 5 * time.Second).Truncate(5 * time.Second).UnixNano()
		aggs = append(aggs, store.Aggregate{
			Bucket: bucket,
			Source: uint8(events.SourceCUDA),
			Op:     uint8(events.CUDALaunchKernel),
			Count:  perBucket,
			SumDur: int64((200 * time.Millisecond).Nanoseconds() * perBucket),
			MinDur: int64((100 * time.Millisecond).Nanoseconds()),
			MaxDur: int64((400 * time.Millisecond).Nanoseconds()),
		})
	}
	s.RecordAggregates5s(aggs)

	col, err := NewSQLiteCollector(SQLiteCollectorConfig{
		DBPath:  dbPath,
		Window:  60 * time.Second,
		NumGPUs: 1,
		Log:     discardLogger(),
	})
	if err != nil {
		t.Fatalf("NewSQLiteCollector: %v", err)
	}
	defer col.(io.Closer).Close()

	obs, launches, err := col.Collect(context.Background(), now)
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}

	// 60-s window, 12 buckets * 50 = 600 kernel launches -> 10/s throughput.
	if launches != 600 {
		t.Errorf("launches=%d, want 600", launches)
	}
	if obs.Throughput < 9 || obs.Throughput > 11 {
		t.Errorf("throughput=%v, want ~10", obs.Throughput)
	}
	if obs.Compute <= 0 || obs.Compute > 1 {
		t.Errorf("compute=%v, want in (0, 1]", obs.Compute)
	}
	// Memory + CPU come from live /proc; just enforce bounds.
	if obs.Memory < 0 || obs.Memory > 1 {
		t.Errorf("memory=%v out of [0,1]", obs.Memory)
	}
	if obs.CPU < 0 || obs.CPU > 1 {
		t.Errorf("cpu=%v out of [0,1]", obs.CPU)
	}
}

// TestSQLiteCollector_EndToEnd_1mFallback verifies that when the window is
// longer than a minute the collector reads the 1m table, even when the 5s
// table is fresh. Catches regressions in the subMinuteThreshold logic.
func TestSQLiteCollector_EndToEnd_1mFallback(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "integration-1m.db")

	s, err := store.New(dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer s.Close()

	now := time.Now()
	// Write ONLY to the 1m table. If the collector accidentally reads the
	// 5s table for a >60s window, throughput will be 0.
	minuteBucket := now.Truncate(time.Minute).UnixNano()
	s.RecordAggregates([]store.Aggregate{{
		Bucket: minuteBucket,
		Source: uint8(events.SourceCUDA),
		Op:     uint8(events.CUDALaunchKernel),
		Count:  1200,
		SumDur: int64((100 * time.Millisecond).Nanoseconds() * 1200),
		MinDur: 1, MaxDur: 10,
	}})

	col, err := NewSQLiteCollector(SQLiteCollectorConfig{
		DBPath:  dbPath,
		Window:  120 * time.Second, // > subMinuteThreshold -> 1m table
		NumGPUs: 1,
		Log:     discardLogger(),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer col.(io.Closer).Close()

	obs, launches, err := col.Collect(context.Background(), now)
	if err != nil {
		t.Fatal(err)
	}
	if launches != 1200 {
		t.Errorf("launches=%d, want 1200 (from 1m table)", launches)
	}
	wantThroughput := 1200.0 / 120.0
	if obs.Throughput < wantThroughput-0.1 || obs.Throughput > wantThroughput+0.1 {
		t.Errorf("throughput=%v, want ~%v", obs.Throughput, wantThroughput)
	}
}
