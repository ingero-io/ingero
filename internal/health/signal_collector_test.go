package health

import (
	"context"
	"database/sql"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ingero-io/ingero/pkg/events"

	_ "modernc.org/sqlite"
)

// TestSQLiteCollector_MissingDB verifies a clear error is returned when the
// DB path doesn't exist (the common first-run operator mistake).
func TestSQLiteCollector_MissingDB(t *testing.T) {
	_, err := NewSQLiteCollector(SQLiteCollectorConfig{
		DBPath: filepath.Join(t.TempDir(), "nonexistent.db"),
		Log:    discardLogger(),
	})
	if err == nil {
		t.Fatal("expected error for missing DB, got nil")
	}
}

// TestSQLiteCollector_EmptyDB verifies that an empty (brand-new) DB yields
// zero throughput/compute without crashing. Memory/CPU come from sysinfo
// so they're non-zero.
func TestSQLiteCollector_EmptyDB(t *testing.T) {
	dbPath := seedTestDB(t, nil)

	col, err := NewSQLiteCollector(SQLiteCollectorConfig{
		DBPath: dbPath,
		Window: 60 * time.Second,
		Log:    discardLogger(),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer col.(io.Closer).Close()

	obs, launches, err := col.Collect(context.Background(), time.Now())
	if err != nil {
		t.Fatalf("Collect: %v", err)
	}
	if obs.Throughput != 0 {
		t.Errorf("empty DB should yield throughput=0, got %v", obs.Throughput)
	}
	if obs.Compute != 0 {
		t.Errorf("empty DB should yield compute=0, got %v", obs.Compute)
	}
	if launches != 0 {
		t.Errorf("empty DB should yield launches=0, got %d", launches)
	}
	// Memory/CPU come from live /proc — just assert bounds.
	if obs.Memory < 0 || obs.Memory > 1 {
		t.Errorf("memory out of bounds: %v", obs.Memory)
	}
	if obs.CPU < 0 || obs.CPU > 1 {
		t.Errorf("cpu out of bounds: %v", obs.CPU)
	}
}

// TestSQLiteCollector_Throughput verifies Throughput = kernel_launches / window.
// Seeds 600 cudaLaunchKernel events into the current minute bucket and
// asserts throughput equals 600 / 60 = 10 kernels/sec.
func TestSQLiteCollector_Throughput(t *testing.T) {
	now := time.Now()
	bucket := now.Truncate(time.Minute)
	rows := []aggRow{
		{
			bucket: bucket,
			source: uint8(events.SourceCUDA),
			op:     uint8(events.CUDALaunchKernel),
			count:  600,
			sumDur: 0, // only throughput matters here
			minDur: 100,
			maxDur: 500,
		},
	}
	dbPath := seedTestDB(t, rows)

	col, err := NewSQLiteCollector(SQLiteCollectorConfig{
		DBPath: dbPath,
		Window: 60 * time.Second,
		Log:    discardLogger(),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer col.(io.Closer).Close()

	obs, launches, err := col.Collect(context.Background(), now)
	if err != nil {
		t.Fatal(err)
	}
	if obs.Throughput != 10 {
		t.Errorf("Throughput = %v, want 10", obs.Throughput)
	}
	if launches != 600 {
		t.Errorf("launches = %d, want 600", launches)
	}
}

// TestSQLiteCollector_ComputeClamp verifies Compute is clamped to [0, 1]
// even when sum_dur > window * numGPUs (would otherwise yield >1).
func TestSQLiteCollector_ComputeClamp(t *testing.T) {
	now := time.Now()
	bucket := now.Truncate(time.Minute)
	rows := []aggRow{
		{
			bucket: bucket,
			source: uint8(events.SourceCUDA),
			op:     uint8(events.CUDALaunchKernel),
			count:  10,
			sumDur: int64((120 * time.Second).Nanoseconds()), // 2x window
			minDur: 1,
			maxDur: 1,
		},
	}
	dbPath := seedTestDB(t, rows)

	col, err := NewSQLiteCollector(SQLiteCollectorConfig{
		DBPath:  dbPath,
		Window:  60 * time.Second,
		NumGPUs: 1,
		Log:     discardLogger(),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer col.(io.Closer).Close()

	obs, _, err := col.Collect(context.Background(), now)
	if err != nil {
		t.Fatal(err)
	}
	if obs.Compute != 1 {
		t.Errorf("Compute = %v, want 1 (clamped)", obs.Compute)
	}
}

// TestSQLiteCollector_BoundsAlways verifies all four signals stay within
// their documented bounds across a range of pathological inputs.
func TestSQLiteCollector_BoundsAlways(t *testing.T) {
	now := time.Now()
	bucket := now.Truncate(time.Minute)
	// Negative sum_dur (shouldn't happen but guard against it).
	rows := []aggRow{
		{bucket: bucket, source: uint8(events.SourceCUDA), op: uint8(events.CUDALaunchKernel), count: 1, sumDur: -1000},
	}
	dbPath := seedTestDB(t, rows)

	col, err := NewSQLiteCollector(SQLiteCollectorConfig{
		DBPath: dbPath,
		Window: 60 * time.Second,
		Log:    discardLogger(),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer col.(io.Closer).Close()

	obs, _, err := col.Collect(context.Background(), now)
	if err != nil {
		t.Fatal(err)
	}
	if obs.Compute < 0 || obs.Compute > 1 {
		t.Errorf("Compute out of [0,1]: %v", obs.Compute)
	}
	if obs.Memory < 0 || obs.Memory > 1 {
		t.Errorf("Memory out of [0,1]: %v", obs.Memory)
	}
	if obs.CPU < 0 || obs.CPU > 1 {
		t.Errorf("CPU out of [0,1]: %v", obs.CPU)
	}
}

// aggRow is a minimal row description for seedTestDB.
type aggRow struct {
	bucket time.Time
	source uint8
	op     uint8
	count  int64
	sumDur int64
	minDur int64
	maxDur int64
}

// seedTestDB creates a SQLite file with the subset of schema that the
// collector queries (event_aggregates and event_aggregates_5s). Each input
// row is inserted into BOTH tables with its bucket re-truncated to the
// correct granularity, so tests that use any Window value see consistent
// aggregates. The file is cleaned up when the test ends.
//
// NOTE: this deliberately does NOT use store.New() because that runs the
// full schema migrations (including CREATE TABLE events) which take ~300ms
// and aren't needed here.
func seedTestDB(t *testing.T, rows []aggRow) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "trace.db")

	db, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatal(err)
	}
	for _, table := range []string{"event_aggregates", "event_aggregates_5s"} {
		if _, err := db.Exec(`CREATE TABLE ` + table + ` (
			bucket INTEGER NOT NULL,
			source INTEGER NOT NULL,
			op INTEGER NOT NULL,
			count INTEGER NOT NULL DEFAULT 0,
			sum_dur INTEGER NOT NULL DEFAULT 0,
			min_dur INTEGER NOT NULL DEFAULT 0,
			max_dur INTEGER NOT NULL DEFAULT 0,
			sum_arg0 INTEGER NOT NULL DEFAULT 0,
			PRIMARY KEY (bucket, source, op)
		)`); err != nil {
			t.Fatal(err)
		}
	}
	for _, r := range rows {
		minuteBucket := r.bucket.Truncate(time.Minute).UnixNano()
		fiveSecBucket := r.bucket.Truncate(5 * time.Second).UnixNano()
		if _, err := db.Exec(
			"INSERT INTO event_aggregates (bucket, source, op, count, sum_dur, min_dur, max_dur) VALUES (?, ?, ?, ?, ?, ?, ?)",
			minuteBucket, r.source, r.op, r.count, r.sumDur, r.minDur, r.maxDur,
		); err != nil {
			t.Fatal(err)
		}
		if _, err := db.Exec(
			"INSERT INTO event_aggregates_5s (bucket, source, op, count, sum_dur, min_dur, max_dur) VALUES (?, ?, ?, ?, ?, ?, ?)",
			fiveSecBucket, r.source, r.op, r.count, r.sumDur, r.minDur, r.maxDur,
		); err != nil {
			t.Fatal(err)
		}
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	// Ensure file is flushed before NewReadOnly opens it.
	_ = os.Chtimes(path, time.Now(), time.Now())
	return path
}

// TestSQLiteCollector_WindowSelectsTable asserts that Collect reads from
// event_aggregates_5s when Window <= 60s and from event_aggregates when
// Window > 60s. Seeds each table with DIFFERENT counts so the throughput
// unambiguously identifies which table was queried.
func TestSQLiteCollector_WindowSelectsTable(t *testing.T) {
	now := time.Now()
	dir := t.TempDir()
	path := filepath.Join(dir, "trace.db")

	db, err := sql.Open("sqlite", "file:"+path)
	if err != nil {
		t.Fatal(err)
	}
	for _, table := range []string{"event_aggregates", "event_aggregates_5s"} {
		if _, err := db.Exec(`CREATE TABLE ` + table + ` (
			bucket INTEGER NOT NULL,
			source INTEGER NOT NULL,
			op INTEGER NOT NULL,
			count INTEGER NOT NULL DEFAULT 0,
			sum_dur INTEGER NOT NULL DEFAULT 0,
			min_dur INTEGER NOT NULL DEFAULT 0,
			max_dur INTEGER NOT NULL DEFAULT 0,
			sum_arg0 INTEGER NOT NULL DEFAULT 0,
			PRIMARY KEY (bucket, source, op)
		)`); err != nil {
			t.Fatal(err)
		}
	}
	// Distinct counts: 1m table gets 100, 5s table gets 600.
	minuteBucket := now.Truncate(time.Minute).UnixNano()
	fiveSecBucket := now.Truncate(5 * time.Second).UnixNano()
	src := uint8(events.SourceCUDA)
	op := uint8(events.CUDALaunchKernel)
	if _, err := db.Exec(
		"INSERT INTO event_aggregates (bucket, source, op, count) VALUES (?, ?, ?, 100)",
		minuteBucket, src, op); err != nil {
		t.Fatal(err)
	}
	if _, err := db.Exec(
		"INSERT INTO event_aggregates_5s (bucket, source, op, count) VALUES (?, ?, ?, 600)",
		fiveSecBucket, src, op); err != nil {
		t.Fatal(err)
	}
	if err := db.Close(); err != nil {
		t.Fatal(err)
	}
	_ = os.Chtimes(path, time.Now(), time.Now())

	// Window = 30s: sub-minute, should read 5s table -> 600 / 30 = 20.
	col5, err := NewSQLiteCollector(SQLiteCollectorConfig{
		DBPath: path,
		Window: 30 * time.Second,
		Log:    discardLogger(),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer col5.(io.Closer).Close()
	obs5, launches5, err := col5.Collect(context.Background(), now)
	if err != nil {
		t.Fatal(err)
	}
	if obs5.Throughput != 20 {
		t.Errorf("window=30s: throughput=%v, want 20 (from 5s table)", obs5.Throughput)
	}
	if launches5 != 600 {
		t.Errorf("window=30s: launches=%d, want 600", launches5)
	}

	// Window = 120s: long, should read 1m table -> 100 / 120 ≈ 0.833.
	col1m, err := NewSQLiteCollector(SQLiteCollectorConfig{
		DBPath: path,
		Window: 120 * time.Second,
		Log:    discardLogger(),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer col1m.(io.Closer).Close()
	obs1m, launches1m, err := col1m.Collect(context.Background(), now)
	if err != nil {
		t.Fatal(err)
	}
	wantThroughput := 100.0 / 120.0
	if obs1m.Throughput < wantThroughput-0.001 || obs1m.Throughput > wantThroughput+0.001 {
		t.Errorf("window=120s: throughput=%v, want ~%v (from 1m table)", obs1m.Throughput, wantThroughput)
	}
	if launches1m != 100 {
		t.Errorf("window=120s: launches=%d, want 100", launches1m)
	}
}

