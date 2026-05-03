package health

import (
	"context"
	"database/sql"
	"errors"
	"fmt"
	"io"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ingero-io/ingero/pkg/contract"
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

// stubResolver returns a resolver function that maps configured PIDs
// to fixed cgroup paths and an error for any unknown PID.
func stubResolver(paths map[uint32]string) func(uint32) (string, error) {
	return func(pid uint32) (string, error) {
		if p, ok := paths[pid]; ok {
			return p, nil
		}
		return "", errors.New("no path for pid")
	}
}

// seedPerCGroupDB writes per-pid aggregates plus memcpy events into a
// real store DB using the public Record / RecordAggregates5s API. The
// per-cgroup tests need both the 5s aggregate table populated AND
// individual events in the events table (memcpy direction lives there
// because aggregates strip arg1).
func seedPerCGroupDB(t *testing.T, dbPath string, perPID []aggRowPID, memcpyEvts []memcpyEvtSeed) {
	t.Helper()
	// Use the package-internal store here so the schema migrations run.
	// Tests in this file already cover the happy path via store.New.
	dir := filepath.Dir(dbPath)
	if err := os.MkdirAll(dir, 0o755); err != nil {
		t.Fatal(err)
	}

	db, err := sql.Open("sqlite", "file:"+dbPath)
	if err != nil {
		t.Fatal(err)
	}
	defer db.Close()
	for _, table := range []string{"event_aggregates", "event_aggregates_5s"} {
		if _, err := db.Exec(`CREATE TABLE ` + table + ` (
			bucket INTEGER NOT NULL,
			source INTEGER NOT NULL,
			op INTEGER NOT NULL,
			pid INTEGER NOT NULL DEFAULT 0,
			count INTEGER NOT NULL DEFAULT 0,
			stored INTEGER NOT NULL DEFAULT 0,
			sum_dur INTEGER NOT NULL DEFAULT 0,
			min_dur INTEGER NOT NULL DEFAULT 0,
			max_dur INTEGER NOT NULL DEFAULT 0,
			sum_arg0 INTEGER NOT NULL DEFAULT 0,
			PRIMARY KEY (bucket, source, op, pid)
		)`); err != nil {
			t.Fatal(err)
		}
	}
	if _, err := db.Exec(`CREATE TABLE events (
		id TEXT PRIMARY KEY,
		timestamp INTEGER NOT NULL,
		pid INTEGER NOT NULL,
		tid INTEGER NOT NULL,
		source INTEGER NOT NULL,
		op INTEGER NOT NULL,
		duration INTEGER NOT NULL,
		gpu_id INTEGER NOT NULL DEFAULT 0,
		arg0 INTEGER NOT NULL DEFAULT 0,
		arg1 INTEGER NOT NULL DEFAULT 0,
		ret_code INTEGER NOT NULL DEFAULT 0
	)`); err != nil {
		t.Fatal(err)
	}

	for _, r := range perPID {
		bucket := r.bucket.Truncate(5 * time.Second).UnixNano()
		if _, err := db.Exec(
			"INSERT INTO event_aggregates_5s (bucket, source, op, pid, count, sum_dur) VALUES (?, ?, ?, ?, ?, ?)",
			bucket, r.source, r.op, r.pid, r.count, r.sumDur,
		); err != nil {
			t.Fatal(err)
		}
	}
	for i, m := range memcpyEvts {
		op := uint8(events.CUDAMemcpy)
		if m.async {
			op = uint8(events.CUDAMemcpyAsync)
		}
		id := fmt.Sprintf(":%d", i+1)
		if _, err := db.Exec(
			"INSERT INTO events (id, timestamp, pid, tid, source, op, duration, arg0, arg1) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
			id, m.ts.UnixNano(), m.pid, m.pid, uint8(events.SourceCUDA), op, 1000, m.bytes, uint64(m.dir),
		); err != nil {
			t.Fatal(err)
		}
	}
	_ = os.Chtimes(dbPath, time.Now(), time.Now())
}

type aggRowPID struct {
	bucket time.Time
	source uint8
	op     uint8
	pid    uint32
	count  int64
	sumDur int64
}

type memcpyEvtSeed struct {
	ts    time.Time
	pid   uint32
	dir   uint8
	bytes uint64
	async bool
}

// TestCollectPerCGroup_TwoCGroupsTwoSeries verifies that two PIDs in
// distinct cgroups produce two output entries with distinct hashes,
// and the kernel-launch + memcpy aggregates land in their respective
// hashes' buckets.
func TestCollectPerCGroup_TwoCGroupsTwoSeries(t *testing.T) {
	now := time.Now()
	dbPath := filepath.Join(t.TempDir(), "trace.db")
	seedPerCGroupDB(t, dbPath, []aggRowPID{
		{bucket: now, source: uint8(events.SourceCUDA), op: uint8(events.CUDALaunchKernel), pid: 100, count: 50},
		{bucket: now, source: uint8(events.SourceCUDA), op: uint8(events.CUDALaunchKernel), pid: 200, count: 30},
	}, []memcpyEvtSeed{
		{ts: now.Add(-1 * time.Second), pid: 100, dir: 1, bytes: 1024},
		{ts: now.Add(-1 * time.Second), pid: 200, dir: 2, bytes: 2048},
	})

	col, err := NewSQLiteCollector(SQLiteCollectorConfig{
		DBPath: dbPath,
		Window: 30 * time.Second,
		Log:    discardLogger(),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer col.(io.Closer).Close()

	sc := col.(*sqliteCollector)
	sc.cache.resolveFn = stubResolver(map[uint32]string{
		100: "/kubepods.slice/pod-A",
		200: "/kubepods.slice/pod-B",
	})

	out, err := sc.CollectPerCGroup(context.Background(), now)
	if err != nil {
		t.Fatalf("CollectPerCGroup: %v", err)
	}

	hashA := hashCGroupPath("/kubepods.slice/pod-A")
	hashB := hashCGroupPath("/kubepods.slice/pod-B")
	idx := indexByHash(out)
	if got := len(out); got != 2 {
		t.Fatalf("got %d cgroup buckets, want 2: %+v", got, out)
	}
	if a := idx[hashA]; a == nil || a.KernelLaunchCount != 50 || a.MemcpyBytesByDir[contract.MemcpyDirectionH2D] != 1024 {
		t.Errorf("pod-A bucket wrong: %+v", a)
	}
	if b := idx[hashB]; b == nil || b.KernelLaunchCount != 30 || b.MemcpyBytesByDir[contract.MemcpyDirectionD2H] != 2048 {
		t.Errorf("pod-B bucket wrong: %+v", b)
	}
}

// TestCollectPerCGroup_SameCGroupSumsMerged verifies that two PIDs
// resolving to the same cgroup path collapse onto a single output
// entry with summed counters.
func TestCollectPerCGroup_SameCGroupSumsMerged(t *testing.T) {
	now := time.Now()
	dbPath := filepath.Join(t.TempDir(), "trace.db")
	seedPerCGroupDB(t, dbPath, []aggRowPID{
		{bucket: now, source: uint8(events.SourceCUDA), op: uint8(events.CUDALaunchKernel), pid: 100, count: 50},
		{bucket: now, source: uint8(events.SourceCUDA), op: uint8(events.CUDALaunchKernel), pid: 101, count: 25},
		{bucket: now, source: uint8(events.SourceHost), op: uint8(events.HostSchedSwitch), pid: 100, count: 4, sumDur: 4_000_000},
	}, nil)

	col, err := NewSQLiteCollector(SQLiteCollectorConfig{
		DBPath: dbPath,
		Window: 30 * time.Second,
		Log:    discardLogger(),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer col.(io.Closer).Close()

	sc := col.(*sqliteCollector)
	sc.cache.resolveFn = stubResolver(map[uint32]string{
		100: "/kubepods.slice/pod-A",
		101: "/kubepods.slice/pod-A",
	})

	out, err := sc.CollectPerCGroup(context.Background(), now)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 {
		t.Fatalf("got %d buckets, want 1 (same cgroup): %+v", len(out), out)
	}
	got := out[0]
	if got.KernelLaunchCount != 75 {
		t.Errorf("KernelLaunchCount = %d, want 75", got.KernelLaunchCount)
	}
	if got.CPUStallNanos != 4_000_000 {
		t.Errorf("CPUStallNanos = %d, want 4_000_000", got.CPUStallNanos)
	}
}

// TestCollectPerCGroup_MemcpyThreeDirections asserts the direction
// dimension is preserved in the per-cgroup MemcpyBytesByDir map.
func TestCollectPerCGroup_MemcpyThreeDirections(t *testing.T) {
	now := time.Now()
	dbPath := filepath.Join(t.TempDir(), "trace.db")
	seedPerCGroupDB(t, dbPath, nil, []memcpyEvtSeed{
		{ts: now.Add(-1 * time.Second), pid: 100, dir: 1, bytes: 1024},      // h2d
		{ts: now.Add(-1 * time.Second), pid: 100, dir: 2, bytes: 2048},      // d2h
		{ts: now.Add(-1 * time.Second), pid: 100, dir: 3, bytes: 4096, async: true}, // d2d
	})

	col, err := NewSQLiteCollector(SQLiteCollectorConfig{
		DBPath: dbPath,
		Window: 30 * time.Second,
		Log:    discardLogger(),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer col.(io.Closer).Close()

	sc := col.(*sqliteCollector)
	sc.cache.resolveFn = stubResolver(map[uint32]string{100: "/kubepods.slice/pod-A"})

	out, err := sc.CollectPerCGroup(context.Background(), now)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 {
		t.Fatalf("got %d buckets, want 1: %+v", len(out), out)
	}
	dirMap := out[0].MemcpyBytesByDir
	if dirMap[contract.MemcpyDirectionH2D] != 1024 ||
		dirMap[contract.MemcpyDirectionD2H] != 2048 ||
		dirMap[contract.MemcpyDirectionD2D] != 4096 {
		t.Errorf("direction map = %+v, want h2d=1024 d2h=2048 d2d=4096", dirMap)
	}
}

// TestCollectPerCGroup_UnattributablePID verifies that a PID whose
// cgroup resolution fails (or returns "") still produces an output
// row with empty hash. Empty-hash entries reconcile node-wide totals
// against per-cgroup sums and must never panic the loop.
func TestCollectPerCGroup_UnattributablePID(t *testing.T) {
	now := time.Now()
	dbPath := filepath.Join(t.TempDir(), "trace.db")
	seedPerCGroupDB(t, dbPath, []aggRowPID{
		{bucket: now, source: uint8(events.SourceCUDA), op: uint8(events.CUDALaunchKernel), pid: 7, count: 11},
	}, nil)

	col, err := NewSQLiteCollector(SQLiteCollectorConfig{
		DBPath: dbPath,
		Window: 30 * time.Second,
		Log:    discardLogger(),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer col.(io.Closer).Close()

	sc := col.(*sqliteCollector)
	sc.cache.resolveFn = func(pid uint32) (string, error) {
		return "", errors.New("kernel thread")
	}

	out, err := sc.CollectPerCGroup(context.Background(), now)
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 1 || out[0].CgroupPathHash != "" {
		t.Fatalf("expected single empty-hash bucket, got %+v", out)
	}
	if out[0].KernelLaunchCount != 11 {
		t.Errorf("KernelLaunchCount = %d, want 11", out[0].KernelLaunchCount)
	}
}

// TestCollectPerCGroup_EmptyWindow returns no buckets when there are
// no per-pid aggregates and no memcpy events in the window.
func TestCollectPerCGroup_EmptyWindow(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "trace.db")
	seedPerCGroupDB(t, dbPath, nil, nil)

	col, err := NewSQLiteCollector(SQLiteCollectorConfig{
		DBPath: dbPath,
		Window: 30 * time.Second,
		Log:    discardLogger(),
	})
	if err != nil {
		t.Fatal(err)
	}
	defer col.(io.Closer).Close()

	out, err := col.(*sqliteCollector).CollectPerCGroup(context.Background(), time.Now())
	if err != nil {
		t.Fatal(err)
	}
	if len(out) != 0 {
		t.Errorf("empty window should yield 0 buckets, got %d: %+v", len(out), out)
	}
}

func indexByHash(stats []PerCGroupStats) map[string]*PerCGroupStats {
	idx := make(map[string]*PerCGroupStats, len(stats))
	for i := range stats {
		idx[stats[i].CgroupPathHash] = &stats[i]
	}
	return idx
}

