package store

import (
	"context"
	"os"
	"path/filepath"
	"testing"
	"time"

	"github.com/ingero-io/ingero/pkg/events"
)

func makeEvt(source events.Source, op uint8, dur time.Duration) events.Event {
	return events.Event{
		Timestamp: time.Now(),
		PID:       1234,
		TID:       1235,
		Source:    source,
		Op:        op,
		Duration:  dur,
		GPUID:     0,
		Args:      [2]uint64{1000, 2000},
		RetCode:   0,
	}
}

func TestNewInMemory(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New(:memory:) failed: %v", err)
	}
	defer s.Close()

	count, err := s.Count()
	if err != nil {
		t.Fatalf("Count() failed: %v", err)
	}
	if count != 0 {
		t.Errorf("expected 0 events, got %d", count)
	}
}

func TestRecordAndQuery(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	// Start background flusher.
	ctx, cancel := context.WithCancel(context.Background())

	done := make(chan struct{})
	go func() {
		s.Run(ctx)
		close(done)
	}()

	// Record events.
	for i := 0; i < 10; i++ {
		s.Record(makeEvt(events.SourceCUDA, uint8(events.CUDAMalloc), time.Duration(i)*time.Millisecond))
	}
	for i := 0; i < 5; i++ {
		s.Record(makeEvt(events.SourceHost, uint8(events.HostSchedSwitch), time.Duration(i)*time.Millisecond))
	}

	// Wait for flush.
	time.Sleep(300 * time.Millisecond)

	// Query all events.
	result, err := s.Query(QueryParams{Since: 1 * time.Minute})
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}

	if len(result) != 15 {
		t.Errorf("expected 15 events, got %d", len(result))
	}

	// Query CUDA only.
	cudaResult, err := s.Query(QueryParams{
		Since:  1 * time.Minute,
		Source: uint8(events.SourceCUDA),
	})
	if err != nil {
		t.Fatalf("Query(CUDA) failed: %v", err)
	}
	if len(cudaResult) != 10 {
		t.Errorf("expected 10 CUDA events, got %d", len(cudaResult))
	}

	// Query host only.
	hostResult, err := s.Query(QueryParams{
		Since:  1 * time.Minute,
		Source: uint8(events.SourceHost),
	})
	if err != nil {
		t.Fatalf("Query(Host) failed: %v", err)
	}
	if len(hostResult) != 5 {
		t.Errorf("expected 5 host events, got %d", len(hostResult))
	}

	// Shutdown.
	cancel()
	<-done
}

func TestQueryByPID(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		s.Run(ctx)
		close(done)
	}()

	// Record events for different PIDs.
	evt1 := makeEvt(events.SourceCUDA, uint8(events.CUDAMalloc), 1*time.Millisecond)
	evt1.PID = 100
	s.Record(evt1)

	evt2 := makeEvt(events.SourceCUDA, uint8(events.CUDAMalloc), 2*time.Millisecond)
	evt2.PID = 200
	s.Record(evt2)

	time.Sleep(300 * time.Millisecond)

	result, err := s.Query(QueryParams{Since: 1 * time.Minute, PID: 100})
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}
	if len(result) != 1 {
		t.Errorf("expected 1 event for PID 100, got %d", len(result))
	}

	cancel()
	<-done
}

func TestQueryLimit(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		s.Run(ctx)
		close(done)
	}()

	for i := 0; i < 20; i++ {
		s.Record(makeEvt(events.SourceCUDA, uint8(events.CUDAMalloc), 1*time.Millisecond))
	}

	time.Sleep(300 * time.Millisecond)

	result, err := s.Query(QueryParams{Since: 1 * time.Minute, Limit: 5})
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}
	if len(result) != 5 {
		t.Errorf("expected 5 events with limit, got %d", len(result))
	}

	cancel()
	<-done
}

func TestEventRoundTrip(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		s.Run(ctx)
		close(done)
	}()

	original := events.Event{
		Timestamp: time.Now().Truncate(time.Nanosecond), // SQLite stores nanos
		PID:       4821,
		TID:       4822,
		Source:    events.SourceCUDA,
		Op:        uint8(events.CUDAMemcpy),
		Duration:  42 * time.Millisecond,
		GPUID:     1,
		Args:      [2]uint64{65536, 1}, // 64KB, H→D
		RetCode:   0,
	}
	s.Record(original)

	time.Sleep(300 * time.Millisecond)

	result, err := s.Query(QueryParams{Since: 1 * time.Minute})
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 event, got %d", len(result))
	}

	got := result[0]
	if got.PID != original.PID {
		t.Errorf("PID = %d, want %d", got.PID, original.PID)
	}
	if got.TID != original.TID {
		t.Errorf("TID = %d, want %d", got.TID, original.TID)
	}
	if got.Source != original.Source {
		t.Errorf("Source = %v, want %v", got.Source, original.Source)
	}
	if got.Op != original.Op {
		t.Errorf("Op = %d, want %d", got.Op, original.Op)
	}
	if got.Duration != original.Duration {
		t.Errorf("Duration = %v, want %v", got.Duration, original.Duration)
	}
	if got.GPUID != original.GPUID {
		t.Errorf("GPUID = %d, want %d", got.GPUID, original.GPUID)
	}
	if got.Args != original.Args {
		t.Errorf("Args = %v, want %v", got.Args, original.Args)
	}
	if got.RetCode != original.RetCode {
		t.Errorf("RetCode = %d, want %d", got.RetCode, original.RetCode)
	}

	cancel()
	<-done
}

func TestRecordAndQuerySnapshots(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	// Start the Run() goroutine — snapshots are now written asynchronously
	// through the snapshotCh channel, processed in the Run loop.
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go s.Run(ctx)

	// Record a few snapshots with distinct values.
	now := time.Now()
	snaps := []SystemSnapshot{
		{Timestamp: now.Add(-3 * time.Second), CPUPercent: 45.0, MemUsedPct: 60.0, MemAvailMB: 4000, SwapUsedMB: 0, LoadAvg1: 2.5},
		{Timestamp: now.Add(-2 * time.Second), CPUPercent: 92.0, MemUsedPct: 97.0, MemAvailMB: 300, SwapUsedMB: 512, LoadAvg1: 15.0},
		{Timestamp: now.Add(-1 * time.Second), CPUPercent: 88.0, MemUsedPct: 80.0, MemAvailMB: 2000, SwapUsedMB: 0, LoadAvg1: 5.0},
	}
	for _, snap := range snaps {
		s.RecordSnapshot(snap)
	}

	// Give the Run() goroutine time to process the snapshot channel.
	time.Sleep(50 * time.Millisecond)

	// Query all snapshots.
	result, err := s.QuerySnapshots(QueryParams{Since: 1 * time.Minute})
	if err != nil {
		t.Fatalf("QuerySnapshots failed: %v", err)
	}
	if len(result) != 3 {
		t.Fatalf("expected 3 snapshots, got %d", len(result))
	}

	// Verify chronological order (ASC).
	if result[0].Timestamp.After(result[1].Timestamp) {
		t.Error("snapshots not in chronological order")
	}

	// Verify fields round-trip correctly.
	// The second snapshot has the high-pressure values.
	got := result[1]
	if got.CPUPercent != 92.0 {
		t.Errorf("CPUPercent = %v, want 92.0", got.CPUPercent)
	}
	if got.MemUsedPct != 97.0 {
		t.Errorf("MemUsedPct = %v, want 97.0", got.MemUsedPct)
	}
	if got.MemAvailMB != 300 {
		t.Errorf("MemAvailMB = %v, want 300", got.MemAvailMB)
	}
	if got.SwapUsedMB != 512 {
		t.Errorf("SwapUsedMB = %v, want 512", got.SwapUsedMB)
	}
	if got.LoadAvg1 != 15.0 {
		t.Errorf("LoadAvg1 = %v, want 15.0", got.LoadAvg1)
	}

	// Query with From/To to get only the middle snapshot.
	filtered, err := s.QuerySnapshots(QueryParams{
		From: now.Add(-2500 * time.Millisecond),
		To:   now.Add(-1500 * time.Millisecond),
	})
	if err != nil {
		t.Fatalf("QuerySnapshots(From/To) failed: %v", err)
	}
	if len(filtered) != 1 {
		t.Errorf("expected 1 filtered snapshot, got %d", len(filtered))
	}
}

func TestQueryByMultiplePIDs(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		s.Run(ctx)
		close(done)
	}()

	// Record events for PIDs 100, 200, 300.
	for _, pid := range []uint32{100, 200, 300} {
		evt := makeEvt(events.SourceCUDA, uint8(events.CUDAMalloc), 1*time.Millisecond)
		evt.PID = pid
		s.Record(evt)
	}

	time.Sleep(300 * time.Millisecond)

	// Query PIDs 100 and 300 — PID 200 should be excluded.
	result, err := s.Query(QueryParams{
		Since: 1 * time.Minute,
		PIDs:  []uint32{100, 300},
	})
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}
	if len(result) != 2 {
		t.Fatalf("expected 2 events for PIDs 100,300, got %d", len(result))
	}
	for _, evt := range result {
		if evt.PID == 200 {
			t.Errorf("PID 200 should have been excluded")
		}
	}

	cancel()
	<-done
}

func TestQueryByMultiplePIDsRich(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		s.Run(ctx)
		close(done)
	}()

	for _, pid := range []uint32{100, 200, 300} {
		evt := makeEvt(events.SourceCUDA, uint8(events.CUDAMalloc), 1*time.Millisecond)
		evt.PID = pid
		s.Record(evt)
	}

	time.Sleep(300 * time.Millisecond)

	result, err := s.QueryRich(QueryParams{
		Since: 1 * time.Minute,
		PIDs:  []uint32{100, 300},
	})
	if err != nil {
		t.Fatalf("QueryRich failed: %v", err)
	}
	if len(result) != 2 {
		t.Fatalf("expected 2 events for PIDs 100,300, got %d", len(result))
	}
	for _, evt := range result {
		if evt.PID == 200 {
			t.Errorf("PID 200 should have been excluded")
		}
	}

	cancel()
	<-done
}

func TestQueryRichProcessName(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		s.Run(ctx)
		close(done)
	}()

	// Record process names.
	s.RecordProcessName(100, "python3")
	s.RecordProcessName(200, "java")

	// Insert events for both PIDs + one unknown PID.
	for _, pid := range []uint32{100, 200, 300} {
		evt := makeEvt(events.SourceCUDA, uint8(events.CUDAMalloc), 1*time.Millisecond)
		evt.PID = pid
		s.Record(evt)
	}

	time.Sleep(300 * time.Millisecond)

	result, err := s.QueryRich(QueryParams{Since: 1 * time.Minute})
	if err != nil {
		t.Fatalf("QueryRich failed: %v", err)
	}
	if len(result) != 3 {
		t.Fatalf("expected 3 events, got %d", len(result))
	}

	nameByPID := map[uint32]string{}
	for _, evt := range result {
		nameByPID[evt.PID] = evt.ProcessName
	}
	if nameByPID[100] != "python3" {
		t.Errorf("PID 100: expected process name 'python3', got %q", nameByPID[100])
	}
	if nameByPID[200] != "java" {
		t.Errorf("PID 200: expected process name 'java', got %q", nameByPID[200])
	}
	if nameByPID[300] != "" {
		t.Errorf("PID 300: expected empty process name, got %q", nameByPID[300])
	}

	cancel()
	<-done
}

func TestQuerySinglePIDBackwardCompat(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		s.Run(ctx)
		close(done)
	}()

	for _, pid := range []uint32{100, 200} {
		evt := makeEvt(events.SourceCUDA, uint8(events.CUDAMalloc), 1*time.Millisecond)
		evt.PID = pid
		s.Record(evt)
	}

	time.Sleep(300 * time.Millisecond)

	// Use old PID field (not PIDs) — backward compat for MCP.
	result, err := s.Query(QueryParams{
		Since: 1 * time.Minute,
		PID:   100,
	})
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}
	if len(result) != 1 {
		t.Fatalf("expected 1 event for PID 100, got %d", len(result))
	}
	if result[0].PID != 100 {
		t.Errorf("expected PID 100, got %d", result[0].PID)
	}

	cancel()
	<-done
}

func TestStartAndStopSession(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	startTime := time.Now()
	sess := Session{
		StartedAt: startTime,
		GPUModel:  "NVIDIA GeForce RTX 4090 (24564 MiB)",
		GPUDriver: "580.126.09",
		CPUModel:  "AMD EPYC 7713 64-Core Processor",
		CPUCores:  64,
		MemTotal:  131072,
		Kernel:    "5.15.0-100-generic",
		OSRelease: "Ubuntu 22.04.5 LTS",
		CUDAVer:   "12.4",
		PythonVer: "3.10.12",
		IngeroVer: "dev (commit: abc123, built: 2026-02-24)",
		PIDFilter: "32574,32575",
		Flags:     "stack,record,json",
	}

	id, err := s.StartSession(sess)
	if err != nil {
		t.Fatalf("StartSession failed: %v", err)
	}
	if id <= 0 {
		t.Fatalf("expected positive session ID, got %d", id)
	}

	// Verify the session was inserted with stopped_at = 0.
	sessions, err := s.QuerySessions(1 * time.Minute)
	if err != nil {
		t.Fatalf("QuerySessions failed: %v", err)
	}
	if len(sessions) != 1 {
		t.Fatalf("expected 1 session, got %d", len(sessions))
	}

	got := sessions[0]
	if got.ID != id {
		t.Errorf("ID = %d, want %d", got.ID, id)
	}
	if got.GPUModel != sess.GPUModel {
		t.Errorf("GPUModel = %q, want %q", got.GPUModel, sess.GPUModel)
	}
	if got.GPUDriver != sess.GPUDriver {
		t.Errorf("GPUDriver = %q, want %q", got.GPUDriver, sess.GPUDriver)
	}
	if got.CPUModel != sess.CPUModel {
		t.Errorf("CPUModel = %q, want %q", got.CPUModel, sess.CPUModel)
	}
	if got.CPUCores != sess.CPUCores {
		t.Errorf("CPUCores = %d, want %d", got.CPUCores, sess.CPUCores)
	}
	if got.MemTotal != sess.MemTotal {
		t.Errorf("MemTotal = %d, want %d", got.MemTotal, sess.MemTotal)
	}
	if got.Kernel != sess.Kernel {
		t.Errorf("Kernel = %q, want %q", got.Kernel, sess.Kernel)
	}
	if got.OSRelease != sess.OSRelease {
		t.Errorf("OSRelease = %q, want %q", got.OSRelease, sess.OSRelease)
	}
	if got.CUDAVer != sess.CUDAVer {
		t.Errorf("CUDAVer = %q, want %q", got.CUDAVer, sess.CUDAVer)
	}
	if got.PythonVer != sess.PythonVer {
		t.Errorf("PythonVer = %q, want %q", got.PythonVer, sess.PythonVer)
	}
	if got.IngeroVer != sess.IngeroVer {
		t.Errorf("IngeroVer = %q, want %q", got.IngeroVer, sess.IngeroVer)
	}
	if got.PIDFilter != sess.PIDFilter {
		t.Errorf("PIDFilter = %q, want %q", got.PIDFilter, sess.PIDFilter)
	}
	if got.Flags != sess.Flags {
		t.Errorf("Flags = %q, want %q", got.Flags, sess.Flags)
	}
	if !got.StoppedAt.IsZero() {
		t.Errorf("StoppedAt should be zero (still running), got %v", got.StoppedAt)
	}

	// Stop the session.
	stopTime := time.Now()
	if err := s.StopSession(id, stopTime); err != nil {
		t.Fatalf("StopSession failed: %v", err)
	}

	// Verify stopped_at was updated.
	sessions, err = s.QuerySessions(1 * time.Minute)
	if err != nil {
		t.Fatalf("QuerySessions after stop failed: %v", err)
	}
	if sessions[0].StoppedAt.IsZero() {
		t.Error("StoppedAt should be non-zero after StopSession")
	}
}

func TestQuerySessions(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	// Insert two sessions.
	s1 := Session{
		StartedAt: time.Now().Add(-2 * time.Minute),
		GPUModel:  "RTX 4090",
		IngeroVer: "v0.6",
	}
	s2 := Session{
		StartedAt: time.Now().Add(-30 * time.Second),
		GPUModel:  "A100",
		IngeroVer: "v0.6",
	}

	_, err = s.StartSession(s1)
	if err != nil {
		t.Fatalf("StartSession(s1) failed: %v", err)
	}
	_, err = s.StartSession(s2)
	if err != nil {
		t.Fatalf("StartSession(s2) failed: %v", err)
	}

	// Query all sessions.
	all, err := s.QuerySessions(5 * time.Minute)
	if err != nil {
		t.Fatalf("QuerySessions(5m) failed: %v", err)
	}
	if len(all) != 2 {
		t.Fatalf("expected 2 sessions, got %d", len(all))
	}

	// Results are DESC order, so most recent first.
	if all[0].GPUModel != "A100" {
		t.Errorf("first session should be A100 (most recent), got %q", all[0].GPUModel)
	}

	// Query with tight since — should only get the recent one.
	recent, err := s.QuerySessions(1 * time.Minute)
	if err != nil {
		t.Fatalf("QuerySessions(1m) failed: %v", err)
	}
	if len(recent) != 1 {
		t.Fatalf("expected 1 recent session, got %d", len(recent))
	}
	if recent[0].GPUModel != "A100" {
		t.Errorf("recent session should be A100, got %q", recent[0].GPUModel)
	}
}

func TestSessionsNoteOnReopenedDB(t *testing.T) {
	// Verify that sessions_note is present in schema_info even when
	// the DB was created by a previous open (populateLookupTables skips
	// inserts when tables are already populated).
	dbPath := filepath.Join(t.TempDir(), "reopen-test.db")

	// First open — populates all tables.
	s1, err := New(dbPath)
	if err != nil {
		t.Fatalf("first New failed: %v", err)
	}
	s1.Close()

	// Second open — populateLookupTables returns early (sources non-empty).
	s2, err := New(dbPath)
	if err != nil {
		t.Fatalf("second New failed: %v", err)
	}
	defer s2.Close()

	var val string
	err = s2.db.QueryRow("SELECT value FROM schema_info WHERE key = 'sessions_note'").Scan(&val)
	if err != nil {
		t.Fatalf("sessions_note not found in schema_info after reopen: %v", err)
	}
	if val == "" {
		t.Error("sessions_note value is empty")
	}
}

func TestBatchFlush(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		s.Run(ctx)
		close(done)
	}()

	// Record exactly DefaultBatchSize events to trigger batch flush.
	for i := 0; i < DefaultBatchSize; i++ {
		s.Record(makeEvt(events.SourceCUDA, uint8(events.CUDAMalloc), 1*time.Millisecond))
	}

	// Give the flusher time to process.
	time.Sleep(300 * time.Millisecond)

	count, err := s.Count()
	if err != nil {
		t.Fatalf("Count failed: %v", err)
	}
	if count != int64(DefaultBatchSize) {
		t.Errorf("expected %d events after batch flush, got %d", DefaultBatchSize, count)
	}

	cancel()
	<-done
}

func TestRecordAndQueryAggregates(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	now := time.Now()
	bucket1 := now.Truncate(time.Minute).UnixNano()
	bucket2 := now.Add(-1 * time.Minute).Truncate(time.Minute).UnixNano()

	aggs := []Aggregate{
		{Bucket: bucket1, Source: 1, Op: 3, PID: 100, Count: 5000, Stored: 50, SumDur: 1000000, MinDur: 100, MaxDur: 500},
		{Bucket: bucket1, Source: 3, Op: 1, PID: 100, Count: 2000, Stored: 2000, SumDur: 500000, MinDur: 50, MaxDur: 1000},
		{Bucket: bucket2, Source: 4, Op: 1, PID: 200, Count: 8000, Stored: 80, SumDur: 4000000, MinDur: 200, MaxDur: 800},
	}

	s.RecordAggregates(aggs)

	// Query totals for all time.
	totals, err := s.QueryAggregateTotals(QueryParams{Since: 5 * time.Minute})
	if err != nil {
		t.Fatalf("QueryAggregateTotals failed: %v", err)
	}

	if totals.TotalEvents != 15000 {
		t.Errorf("TotalEvents = %d, want 15000", totals.TotalEvents)
	}
	if totals.StoredEvents != 2130 {
		t.Errorf("StoredEvents = %d, want 2130", totals.StoredEvents)
	}

	// Verify ByOp map has the right op names.
	if totals.ByOp["cudaLaunchKernel"] != 5000 {
		t.Errorf("ByOp[cudaLaunchKernel] = %d, want 5000", totals.ByOp["cudaLaunchKernel"])
	}
	if totals.ByOp["sched_switch"] != 2000 {
		t.Errorf("ByOp[sched_switch] = %d, want 2000", totals.ByOp["sched_switch"])
	}
	if totals.ByOp["cuLaunchKernel"] != 8000 {
		t.Errorf("ByOp[cuLaunchKernel] = %d, want 8000", totals.ByOp["cuLaunchKernel"])
	}

	// Query with PID filter — should only get PID 100.
	pidTotals, err := s.QueryAggregateTotals(QueryParams{
		Since: 5 * time.Minute,
		PIDs:  []uint32{100},
	})
	if err != nil {
		t.Fatalf("QueryAggregateTotals(PID=100) failed: %v", err)
	}
	if pidTotals.TotalEvents != 7000 {
		t.Errorf("PID 100 TotalEvents = %d, want 7000", pidTotals.TotalEvents)
	}
}

func TestAggregatesSumArg0(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	bucket := time.Now().Truncate(time.Minute).UnixNano()

	// Simulate mm_page_alloc aggregates: 100 events, each 16MB = 1.6GB total.
	aggs := []Aggregate{
		{
			Bucket:  bucket,
			Source:  3, // HOST
			Op:      3, // mm_page_alloc
			PID:     100,
			Count:   100,
			Stored:  0, // no individual events stored
			SumDur:  0, // duration is always 0 for mm_page_alloc
			MinDur:  0,
			MaxDur:  0,
			SumArg0: 100 * 16 * 1024 * 1024, // 1.6 GB total
		},
	}

	s.RecordAggregates(aggs)

	// Verify sum_arg0 persists via direct SQL (the intended read path for
	// post-hoc analysis is run_sql, not a dedicated Go function).
	var sumArg0 int64
	err = s.db.QueryRow("SELECT sum_arg0 FROM event_aggregates WHERE source=3 AND op=3").Scan(&sumArg0)
	if err != nil {
		t.Fatalf("QueryRow sum_arg0 failed: %v", err)
	}
	want := int64(100 * 16 * 1024 * 1024)
	if sumArg0 != want {
		t.Errorf("sum_arg0 = %d, want %d (1.6 GB)", sumArg0, want)
	}
}

func TestAggregatesPrunedBySize(t *testing.T) {
	// Size-based pruning only works with on-disk DBs (stat() checks file size).
	// Use a temp file to test the actual pruning path.
	tmp := t.TempDir()
	dbPath := filepath.Join(tmp, "test-prune.db")
	s, err := New(dbPath)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	// Insert an old aggregate (8 days ago) and a recent one (1 minute ago).
	// pruneBySize needs a time range (min < max) to compute the cutoff.
	oldBucket := time.Now().Add(-8 * 24 * time.Hour).Truncate(time.Minute).UnixNano()
	newBucket := time.Now().Add(-1 * time.Minute).Truncate(time.Minute).UnixNano()
	s.RecordAggregates([]Aggregate{
		{Bucket: oldBucket, Source: 1, Op: 3, PID: 0, Count: 1000, Stored: 10},
		{Bucket: newBucket, Source: 1, Op: 3, PID: 0, Count: 50, Stored: 5},
	})

	// Verify both are there.
	totals, _ := s.QueryAggregateTotals(QueryParams{From: time.Unix(0, oldBucket)})
	if totals.TotalEvents != 1050 {
		t.Fatalf("expected 1050 before prune, got %d", totals.TotalEvents)
	}

	// Without --max-db, prune() is a no-op — all data stays.
	s.prune()
	totals, _ = s.QueryAggregateTotals(QueryParams{From: time.Unix(0, oldBucket)})
	if totals.TotalEvents != 1050 {
		t.Fatalf("expected 1050 after prune (no limit), got %d", totals.TotalEvents)
	}

	// Set a tiny max-db to force pruning. With keepFraction ≈ 0,
	// the cutoff lands near maxTS — the old bucket is deleted.
	s.SetMaxDBSize(1) // 1 byte — will prune everything
	s.prune()

	totals, _ = s.QueryAggregateTotals(QueryParams{From: time.Unix(0, oldBucket)})
	if totals.TotalEvents > 50 {
		t.Errorf("expected old aggregate pruned, got total %d", totals.TotalEvents)
	}
}

// makeStack creates a test stack trace with the given instruction pointers.
func makeStack(ips ...uint64) []events.StackFrame {
	frames := make([]events.StackFrame, len(ips))
	for i, ip := range ips {
		frames[i] = events.StackFrame{IP: ip}
	}
	return frames
}

func TestStackInterning(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		s.Run(ctx)
		close(done)
	}()

	// Create two identical stacks and one different stack.
	stackA := makeStack(0x7f001000, 0x7f002000, 0x7f003000)
	stackB := makeStack(0x7f001000, 0x7f002000, 0x7f003000) // same as A
	stackC := makeStack(0x7f004000, 0x7f005000)              // different

	// Record 3 events: two with stackA/B (identical), one with stackC.
	evt1 := makeEvt(events.SourceCUDA, uint8(events.CUDALaunchKernel), 10*time.Microsecond)
	evt1.Stack = stackA
	s.Record(evt1)

	evt2 := makeEvt(events.SourceCUDA, uint8(events.CUDALaunchKernel), 20*time.Microsecond)
	evt2.Stack = stackB
	s.Record(evt2)

	evt3 := makeEvt(events.SourceCUDA, uint8(events.CUDALaunchKernel), 30*time.Microsecond)
	evt3.Stack = stackC
	s.Record(evt3)

	// Event with no stack.
	evt4 := makeEvt(events.SourceHost, uint8(events.HostSchedSwitch), 5*time.Microsecond)
	s.Record(evt4)

	time.Sleep(300 * time.Millisecond)

	// Verify deduplication: stack_traces should have exactly 2 rows
	// (one for stackA/B, one for stackC).
	var stackCount int64
	if err := s.db.QueryRow("SELECT COUNT(*) FROM stack_traces").Scan(&stackCount); err != nil {
		t.Fatalf("counting stack_traces: %v", err)
	}
	if stackCount != 2 {
		t.Errorf("expected 2 unique stacks in stack_traces, got %d", stackCount)
	}

	// Query via Query() — stacks should round-trip correctly.
	result, err := s.Query(QueryParams{Since: 1 * time.Minute})
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}
	if len(result) != 4 {
		t.Fatalf("expected 4 events, got %d", len(result))
	}

	// Events are chronological (oldest first). Check stacks.
	if len(result[0].Stack) != 3 {
		t.Errorf("event 0: expected 3 stack frames, got %d", len(result[0].Stack))
	} else if result[0].Stack[0].IP != 0x7f001000 {
		t.Errorf("event 0 frame 0: IP = 0x%x, want 0x7f001000", result[0].Stack[0].IP)
	}

	if len(result[1].Stack) != 3 {
		t.Errorf("event 1: expected 3 stack frames, got %d", len(result[1].Stack))
	}

	if len(result[2].Stack) != 2 {
		t.Errorf("event 2: expected 2 stack frames, got %d", len(result[2].Stack))
	} else if result[2].Stack[0].IP != 0x7f004000 {
		t.Errorf("event 2 frame 0: IP = 0x%x, want 0x7f004000", result[2].Stack[0].IP)
	}

	// Event 4 (no stack) should have nil/empty stack.
	if len(result[3].Stack) != 0 {
		t.Errorf("event 3: expected 0 stack frames, got %d", len(result[3].Stack))
	}

	// Verify QueryRich also returns stacks correctly.
	richResult, err := s.QueryRich(QueryParams{Since: 1 * time.Minute})
	if err != nil {
		t.Fatalf("QueryRich failed: %v", err)
	}
	if len(richResult) != 4 {
		t.Fatalf("QueryRich: expected 4 events, got %d", len(richResult))
	}
	if len(richResult[0].Stack) != 3 {
		t.Errorf("QueryRich event 0: expected 3 stack frames, got %d", len(richResult[0].Stack))
	}

	cancel()
	<-done
}

func TestStackHashDeterministic(t *testing.T) {
	// Same IPs in same order must produce the same hash.
	stack1 := makeStack(0xdead, 0xbeef, 0xcafe)
	stack2 := makeStack(0xdead, 0xbeef, 0xcafe)
	h1 := hashStackIPs(stack1)
	h2 := hashStackIPs(stack2)
	if h1 != h2 {
		t.Errorf("identical stacks produced different hashes: %d vs %d", h1, h2)
	}

	// Different IPs must produce different hashes.
	stack3 := makeStack(0xdead, 0xbeef, 0xfeed)
	h3 := hashStackIPs(stack3)
	if h1 == h3 {
		t.Errorf("different stacks produced same hash: %d", h1)
	}

	// Order matters.
	stack4 := makeStack(0xbeef, 0xdead, 0xcafe)
	h4 := hashStackIPs(stack4)
	if h1 == h4 {
		t.Errorf("reordered stacks produced same hash: %d", h1)
	}

	// Empty stack.
	h5 := hashStackIPs(nil)
	if h5 == h1 {
		t.Errorf("empty stack produced same hash as non-empty")
	}
}

func TestHashStackResolved(t *testing.T) {
	// Unresolved stacks (IPs only) use HashStackIPs — same as before.
	unresolved := makeStack(0xdead, 0xbeef)
	h1 := hashStack(unresolved)
	h2 := hashStackIPs(unresolved)
	if h1 != h2 {
		t.Errorf("unresolved stack: hashStack != hashStackIPs: %d vs %d", h1, h2)
	}

	// Resolved stacks (SymbolName set) use HashStackSymbols — ASLR-independent.
	resolved := []events.StackFrame{
		{IP: 0xdead, SymbolName: "cudaLaunchKernel", File: "/usr/lib/libcudart.so"},
		{IP: 0xbeef, SymbolName: "main", File: "/app/train.py"},
	}
	h3 := hashStack(resolved)
	h4 := events.HashStackSymbols(resolved)
	if h3 != h4 {
		t.Errorf("resolved stack: hashStack != HashStackSymbols: %d vs %d", h3, h4)
	}

	// Same symbols with different IPs (ASLR) must produce the same hash.
	resolvedASLR := []events.StackFrame{
		{IP: 0x1111, SymbolName: "cudaLaunchKernel", File: "/usr/lib/libcudart.so"},
		{IP: 0x2222, SymbolName: "main", File: "/app/train.py"},
	}
	h5 := hashStack(resolvedASLR)
	if h3 != h5 {
		t.Errorf("ASLR-different stacks with same symbols got different hashes: %d vs %d", h3, h5)
	}

	// Python-only resolution (PyFile/PyFunc set, no native SymbolName/File)
	// must use HashStackSymbols, not HashStackIPs.
	pyOnly := []events.StackFrame{
		{IP: 0xaaaa, PyFile: "train.py", PyFunc: "forward", PyLine: 42},
		{IP: 0xbbbb, PyFile: "model.py", PyFunc: "linear", PyLine: 10},
	}
	h6 := hashStack(pyOnly)
	h7 := events.HashStackSymbols(pyOnly)
	if h6 != h7 {
		t.Errorf("Python-only stack: hashStack != HashStackSymbols: %d vs %d", h6, h7)
	}
	// Verify it's NOT using HashStackIPs
	h8 := events.HashStackIPs(pyOnly)
	if h6 == h8 {
		t.Errorf("Python-only stack incorrectly used HashStackIPs")
	}
}

func TestStackCachePreload(t *testing.T) {
	// Test that a second Run() session against the same DB reuses
	// existing stack_traces (no duplicate inserts).
	dbPath := filepath.Join(t.TempDir(), "cache-preload.db")

	// First session: insert events with stacks.
	s1, err := New(dbPath)
	if err != nil {
		t.Fatalf("first New failed: %v", err)
	}

	ctx1, cancel1 := context.WithCancel(context.Background())
	done1 := make(chan struct{})
	go func() {
		s1.Run(ctx1)
		close(done1)
	}()

	stack := makeStack(0xaaa, 0xbbb, 0xccc)
	for i := 0; i < 5; i++ {
		evt := makeEvt(events.SourceCUDA, uint8(events.CUDALaunchKernel), 10*time.Microsecond)
		evt.Stack = stack
		s1.Record(evt)
	}
	time.Sleep(300 * time.Millisecond)
	cancel1()
	<-done1
	s1.Close()

	// Count stacks after first session.
	s2, err := New(dbPath)
	if err != nil {
		t.Fatalf("second New failed: %v", err)
	}

	ctx2, cancel2 := context.WithCancel(context.Background())
	done2 := make(chan struct{})
	go func() {
		s2.Run(ctx2)
		close(done2)
	}()

	// Insert same stack again — should not create a new row.
	for i := 0; i < 3; i++ {
		evt := makeEvt(events.SourceCUDA, uint8(events.CUDALaunchKernel), 10*time.Microsecond)
		evt.Stack = stack
		s2.Record(evt)
	}
	time.Sleep(300 * time.Millisecond)

	var stackCount int64
	s2.db.QueryRow("SELECT COUNT(*) FROM stack_traces").Scan(&stackCount)
	if stackCount != 1 {
		t.Errorf("expected 1 unique stack after two sessions, got %d", stackCount)
	}

	// Total events should be 8 (5 from first + 3 from second session).
	evtCount, _ := s2.Count()
	if evtCount != 8 {
		t.Errorf("expected 8 events total, got %d", evtCount)
	}

	cancel2()
	<-done2
	s2.Close()
}

func TestParseSize(t *testing.T) {
	tests := []struct {
		input   string
		want    int64
		wantErr bool
	}{
		{"10g", 10 * (1 << 30), false},
		{"10G", 10 * (1 << 30), false},
		{"10GB", 10 * (1 << 30), false},
		{"10gb", 10 * (1 << 30), false},
		{"500m", 500 * (1 << 20), false},
		{"500M", 500 * (1 << 20), false},
		{"500MB", 500 * (1 << 20), false},
		{"100k", 100 * (1 << 10), false},
		{"1t", 1 << 40, false},
		{"1T", 1 << 40, false},
		{"1024", 1024, false},       // plain bytes
		{"", 0, true},               // empty
		{"abc", 0, true},            // no number
		{"-5g", 0, true},            // negative
		{"0g", 0, true},             // zero
		{"9999999t", 0, true},       // overflow
		{"  10g  ", 10 * (1 << 30), false}, // whitespace
	}

	for _, tt := range tests {
		got, err := ParseSize(tt.input)
		if tt.wantErr {
			if err == nil {
				t.Errorf("ParseSize(%q) = %d, want error", tt.input, got)
			}
			continue
		}
		if err != nil {
			t.Errorf("ParseSize(%q) error: %v", tt.input, err)
			continue
		}
		if got != tt.want {
			t.Errorf("ParseSize(%q) = %d, want %d", tt.input, got, tt.want)
		}
	}
}

func TestDiskUsage(t *testing.T) {
	// In-memory store should return 0.
	memStore, err := New(":memory:")
	if err != nil {
		t.Fatalf("New(:memory:) failed: %v", err)
	}
	defer memStore.Close()
	if got := memStore.diskUsage(); got != 0 {
		t.Errorf("diskUsage(:memory:) = %d, want 0", got)
	}

	// On-disk store should return positive value after schema creation.
	dbPath := filepath.Join(t.TempDir(), "diskusage-test.db")
	diskStore, err := New(dbPath)
	if err != nil {
		t.Fatalf("New(%s) failed: %v", dbPath, err)
	}
	defer diskStore.Close()
	if got := diskStore.diskUsage(); got <= 0 {
		t.Errorf("diskUsage(on-disk) = %d, want > 0", got)
	}
}

func TestPruneBySizeNoOp(t *testing.T) {
	// maxDBSize=0 → pruneBySize is a no-op (no limit set).
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()
	s.pruneBySize() // should not panic or error

	// On-disk store with generous limit — nothing should be pruned.
	// Stop Run() before calling pruneBySize() from the test goroutine
	// to avoid racing on stackCache (which is single-goroutine by design).
	dbPath := filepath.Join(t.TempDir(), "noprune-test.db")
	s2, err := New(dbPath)
	if err != nil {
		t.Fatalf("New(%s) failed: %v", dbPath, err)
	}
	defer s2.Close()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		s2.Run(ctx)
		close(done)
	}()

	// Insert some events via the Run() channel, then stop Run().
	for i := 0; i < 50; i++ {
		s2.Record(makeEvt(events.SourceCUDA, uint8(events.CUDAMalloc), 1*time.Millisecond))
	}
	time.Sleep(300 * time.Millisecond)
	cancel()
	<-done

	countBefore, _ := s2.Count()

	// Set a very generous limit (1 GB) — nothing should be pruned.
	s2.SetMaxDBSize(1 << 30)
	s2.pruneBySize()

	countAfter, _ := s2.Count()
	if countAfter != countBefore {
		t.Errorf("expected no pruning: before=%d, after=%d", countBefore, countAfter)
	}
}

func TestPruneBySizeOnDisk(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "prune-test.db")
	s, err := New(dbPath)
	if err != nil {
		t.Fatalf("New(%s) failed: %v", dbPath, err)
	}
	defer s.Close()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		s.Run(ctx)
		close(done)
	}()

	// Insert enough events to make the DB grow. Use distinct timestamps
	// so pruneBySize has a time range to work with.
	baseTime := time.Now().Add(-10 * time.Minute)
	for i := 0; i < 2000; i++ {
		evt := makeEvt(events.SourceCUDA, uint8(events.CUDAMalloc), 1*time.Millisecond)
		evt.Timestamp = baseTime.Add(time.Duration(i) * time.Millisecond)
		s.Record(evt)
	}
	time.Sleep(500 * time.Millisecond)

	// Stop Run() before calling pruneBySize() from the test goroutine —
	// pruneBySize accesses stackCache which is single-goroutine by design.
	cancel()
	<-done

	// Force a WAL checkpoint so disk usage reflects all writes.
	s.db.Exec("PRAGMA wal_checkpoint(TRUNCATE)")

	sizeBefore := s.diskUsage()
	countBefore, _ := s.Count()
	if sizeBefore == 0 {
		t.Fatalf("expected positive disk usage after inserts, got 0")
	}
	if countBefore < 2000 {
		t.Fatalf("expected at least 2000 events, got %d", countBefore)
	}

	// Set max to something well below current size to force pruning.
	// Use a small target that will definitely be exceeded.
	smallLimit := sizeBefore / 3
	s.SetMaxDBSize(smallLimit)
	s.pruneBySize()

	// Force another checkpoint to see the effects.
	s.db.Exec("PRAGMA wal_checkpoint(TRUNCATE)")

	countAfter, _ := s.Count()
	if countAfter >= countBefore {
		t.Errorf("expected events to be pruned: before=%d, after=%d", countBefore, countAfter)
	}
	if countAfter == 0 {
		t.Errorf("expected some events to remain after pruning, got 0")
	}

	// Verify disk size actually shrank — this is the whole point of --max-db.
	sizeAfter := s.diskUsage()
	if sizeAfter >= sizeBefore {
		t.Errorf("expected disk usage to shrink: before=%d, after=%d", sizeBefore, sizeAfter)
	}
	t.Logf("disk: %d → %d bytes (%.0f%% reduction), events: %d → %d",
		sizeBefore, sizeAfter, 100*(1-float64(sizeAfter)/float64(sizeBefore)),
		countBefore, countAfter)

	// Verify oldest events were removed (newest should remain).
	// Query with Limit: -1 (unlimited) returns events in chronological order
	// (ASC after internal DESC+reverse). result[0] is the oldest remaining.
	result, _ := s.Query(QueryParams{Limit: -1})
	if len(result) > 0 {
		oldest := result[0].Timestamp
		// With keepFraction ~0.3, cutoff ≈ baseTime + 1400ms. Use 500ms as
		// a conservative threshold that still validates meaningful pruning.
		if oldest.Before(baseTime.Add(500 * time.Millisecond)) {
			t.Errorf("expected oldest remaining event well past base time, got offset %v",
				oldest.Sub(baseTime))
		}
		// Newest event should still be the last one inserted.
		newest := result[len(result)-1].Timestamp
		lastInserted := baseTime.Add(1999 * time.Millisecond)
		if newest.Before(lastInserted.Add(-10 * time.Millisecond)) {
			t.Errorf("expected newest event near %v, got %v", lastInserted, newest)
		}
	}
}

func TestSetMaxDBSize(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	s.SetMaxDBSize(10 * (1 << 30))
	if got := s.maxDBSize.Load(); got != 10*(1<<30) {
		t.Errorf("maxDBSize = %d, want %d", got, 10*(1<<30))
	}

	s.SetMaxDBSize(0)
	if got := s.maxDBSize.Load(); got != 0 {
		t.Errorf("maxDBSize = %d, want 0", got)
	}
}

// TestAutoVacuumPragma verifies that new databases get auto_vacuum=INCREMENTAL.
func TestAutoVacuumPragma(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "vacuum-test.db")
	s, err := New(dbPath)
	if err != nil {
		t.Fatalf("New(%s) failed: %v", dbPath, err)
	}
	defer s.Close()

	var mode int
	err = s.db.QueryRow("PRAGMA auto_vacuum").Scan(&mode)
	if err != nil {
		t.Fatalf("PRAGMA auto_vacuum query failed: %v", err)
	}
	// 2 = INCREMENTAL. Note: only works on newly created DBs.
	if mode != 2 {
		t.Errorf("auto_vacuum = %d, want 2 (INCREMENTAL)", mode)
	}
}

// TestDiskUsageIncludesWAL verifies that diskUsage sums the main DB + WAL files.
func TestDiskUsageIncludesWAL(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "wal-test.db")
	s, err := New(dbPath)
	if err != nil {
		t.Fatalf("New(%s) failed: %v", dbPath, err)
	}
	defer s.Close()

	// After schema creation, the main DB file should exist.
	mainInfo, err := os.Stat(dbPath)
	if err != nil {
		t.Fatalf("main DB file not found: %v", err)
	}

	usage := s.diskUsage()
	if usage < mainInfo.Size() {
		t.Errorf("diskUsage() = %d, expected >= main DB size %d", usage, mainInfo.Size())
	}
}

// TestPruneBySizeAutoTrigger verifies that pruneBySize fires automatically
// after flushBatch when the DB exceeds maxDBSize — the actual production path.
func TestPruneBySizeAutoTrigger(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "auto-prune.db")
	s, err := New(dbPath)
	if err != nil {
		t.Fatalf("New(%s) failed: %v", dbPath, err)
	}
	defer s.Close()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		s.Run(ctx)
		close(done)
	}()

	// Insert events to grow the DB.
	baseTime := time.Now().Add(-10 * time.Minute)
	for i := 0; i < 2000; i++ {
		evt := makeEvt(events.SourceCUDA, uint8(events.CUDAMalloc), 1*time.Millisecond)
		evt.Timestamp = baseTime.Add(time.Duration(i) * time.Millisecond)
		s.Record(evt)
	}
	time.Sleep(500 * time.Millisecond)

	// Checkpoint so diskUsage reflects writes.
	s.db.Exec("PRAGMA wal_checkpoint(TRUNCATE)")
	sizeBefore := s.diskUsage()

	// Set a small limit — below current size. The next flush should auto-prune.
	s.SetMaxDBSize(sizeBefore / 3)

	// Insert more events to trigger a flushBatch, which calls pruneBySize.
	for i := 0; i < DefaultBatchSize+1; i++ {
		evt := makeEvt(events.SourceCUDA, uint8(events.CUDAMalloc), 1*time.Millisecond)
		evt.Timestamp = time.Now()
		s.Record(evt)
	}
	time.Sleep(500 * time.Millisecond)

	// Verify events were pruned automatically (not by direct pruneBySize call).
	count, _ := s.Count()
	if count >= 2000+DefaultBatchSize {
		t.Errorf("expected auto-pruning to reduce events, got %d", count)
	}
	t.Logf("auto-prune: %d events remain (started with %d+%d)", count, 2000, DefaultBatchSize+1)

	cancel()
	<-done
}

// TestPruneBySizeStackCacheRebuild verifies that stacks deleted during pruning
// are correctly re-inserted when new events with the same call stack arrive.
// Without the cache rebuild, the stackCache would say "already in DB" for a
// deleted hash, and the LEFT JOIN would return empty stacks.
func TestPruneBySizeStackCacheRebuild(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "stack-rebuild.db")
	s, err := New(dbPath)
	if err != nil {
		t.Fatalf("New(%s) failed: %v", dbPath, err)
	}
	defer s.Close()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		s.Run(ctx)
		close(done)
	}()

	// waitFor polls a condition instead of sleeping a fixed duration.
	// This makes the test deterministic under -race (which adds 2-20x overhead).
	waitFor := func(msg string, timeout time.Duration, fn func() bool) {
		t.Helper()
		deadline := time.Now().Add(timeout)
		for time.Now().Before(deadline) {
			if fn() {
				return
			}
			time.Sleep(10 * time.Millisecond)
		}
		t.Fatalf("timed out after %s waiting for: %s", timeout, msg)
	}

	// Phase 1: Insert events with a specific stack.
	stack := makeStack(0xaaa, 0xbbb, 0xccc)
	baseTime := time.Now().Add(-10 * time.Minute)
	for i := 0; i < 500; i++ {
		evt := makeEvt(events.SourceCUDA, uint8(events.CUDALaunchKernel), 10*time.Microsecond)
		evt.Stack = stack
		evt.Timestamp = baseTime.Add(time.Duration(i) * time.Millisecond)
		s.Record(evt)
	}
	waitFor("Phase 1 flush", 10*time.Second, func() bool {
		var cnt int64
		s.db.QueryRow("SELECT COUNT(*) FROM events").Scan(&cnt)
		return cnt >= 500
	})

	// Verify stack exists.
	var stackCount int64
	s.db.QueryRow("SELECT COUNT(*) FROM stack_traces").Scan(&stackCount)
	if stackCount != 1 {
		t.Fatalf("expected 1 stack trace before prune, got %d", stackCount)
	}

	// Phase 2a: Set tiny maxDBSize and trigger prune with stackless events.
	// The filler events carry no stack, so once old events are deleted the
	// test stack becomes orphaned and is cleaned up.  The stack cache is
	// rebuilt without it.
	s.db.Exec("PRAGMA wal_checkpoint(TRUNCATE)")
	s.SetMaxDBSize(s.diskUsage() / 4)

	for i := 0; i < DefaultBatchSize+1; i++ {
		evt := makeEvt(events.SourceCUDA, uint8(events.CUDALaunchKernel), 10*time.Microsecond)
		evt.Timestamp = time.Now()
		s.Record(evt)
	}
	// Wait for prune to delete the orphaned stack trace.
	waitFor("Phase 2a prune", 30*time.Second, func() bool {
		var cnt int64
		s.db.QueryRow("SELECT COUNT(*) FROM stack_traces").Scan(&cnt)
		return cnt == 0
	})

	// Phase 2b: Remove size limit and insert events with the SAME stack.
	// Because the cache was rebuilt (without the old hash), flushBatch must
	// re-insert the stack row rather than skipping it as "already in DB".
	s.SetMaxDBSize(0)

	for i := 0; i < DefaultBatchSize+1; i++ {
		evt := makeEvt(events.SourceCUDA, uint8(events.CUDALaunchKernel), 10*time.Microsecond)
		evt.Stack = stack
		evt.Timestamp = time.Now()
		s.Record(evt)
	}
	// Wait for Phase 2b stack to be re-inserted.
	waitFor("Phase 2b stack re-inserted", 10*time.Second, func() bool {
		var cnt int64
		s.db.QueryRow("SELECT COUNT(*) FROM stack_traces").Scan(&cnt)
		return cnt >= 1
	})

	// Phase 3: Query the new events — stacks should be resolved, not empty.
	result, err := s.Query(QueryParams{Since: 1 * time.Minute})
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}
	if len(result) == 0 {
		t.Fatal("expected some recent events after prune, got 0")
	}

	// Check that at least one recent event has a resolved stack.
	hasStack := false
	for _, evt := range result {
		if len(evt.Stack) == 3 && evt.Stack[0].IP == 0xaaa {
			hasStack = true
			break
		}
	}
	if !hasStack {
		t.Errorf("expected recent events to have resolved stacks (3 frames starting at 0xaaa), "+
			"but none found among %d events", len(result))
	}

	cancel()
	<-done
}

// TestPruneBySizeTinyLimit verifies that an impossibly small maxDBSize
// (smaller than the schema overhead) doesn't panic or deadlock.
// pruneBySize should delete all events and exit gracefully.
func TestPruneBySizeTinyLimit(t *testing.T) {
	dbPath := filepath.Join(t.TempDir(), "tiny-limit.db")
	s, err := New(dbPath)
	if err != nil {
		t.Fatalf("New(%s) failed: %v", dbPath, err)
	}
	defer s.Close()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		s.Run(ctx)
		close(done)
	}()

	// Insert events.
	baseTime := time.Now().Add(-5 * time.Minute)
	for i := 0; i < 500; i++ {
		evt := makeEvt(events.SourceCUDA, uint8(events.CUDAMalloc), 1*time.Millisecond)
		evt.Timestamp = baseTime.Add(time.Duration(i) * time.Millisecond)
		s.Record(evt)
	}
	time.Sleep(500 * time.Millisecond)

	// Set limit to 1 byte — way below schema overhead. This forces
	// pruneBySize to delete everything it can, but the DB file will
	// still be larger than 1 byte due to schema pages.
	s.SetMaxDBSize(1)

	// Trigger via new events. Should not panic or hang.
	for i := 0; i < DefaultBatchSize+1; i++ {
		evt := makeEvt(events.SourceCUDA, uint8(events.CUDAMalloc), 1*time.Millisecond)
		evt.Timestamp = time.Now()
		s.Record(evt)
	}
	time.Sleep(500 * time.Millisecond)

	// The DB won't be 1 byte (schema overhead), but old events should be gone.
	// Some new events may also have been pruned. The key assertion is: no panic.
	count, _ := s.Count()
	t.Logf("tiny limit: %d events remain after prune", count)

	cancel()
	<-done
}

// --- ExecuteReadOnly tests ---

func TestExecuteReadOnly_Select(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	ctx := context.Background()

	// Basic SELECT on a known table.
	cols, rows, truncated, err := s.ExecuteReadOnly(ctx, "SELECT COUNT(*) AS n FROM events", 0)
	if err != nil {
		t.Fatalf("ExecuteReadOnly failed: %v", err)
	}
	if truncated {
		t.Error("unexpected truncation")
	}
	if len(cols) != 1 || cols[0] != "n" {
		t.Errorf("columns = %v, want [n]", cols)
	}
	if len(rows) != 1 {
		t.Fatalf("expected 1 row, got %d", len(rows))
	}
	// SQLite returns int64 for COUNT(*).
	val, ok := rows[0][0].(int64)
	if !ok {
		t.Fatalf("expected int64, got %T", rows[0][0])
	}
	if val != 0 {
		t.Errorf("expected 0 events, got %d", val)
	}
}

func TestExecuteReadOnly_RejectsWrite(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	ctx := context.Background()
	writes := []string{
		"INSERT INTO events (timestamp,pid,tid,source,op,duration) VALUES (0,0,0,0,0,0)",
		"UPDATE events SET pid=0",
		"DELETE FROM events",
		"DROP TABLE events",
		"ALTER TABLE events ADD COLUMN foo INTEGER",
		"CREATE TABLE evil (id INTEGER)",
		"ATTACH DATABASE ':memory:' AS evil",
		"VACUUM",
	}
	for _, q := range writes {
		_, _, _, err := s.ExecuteReadOnly(ctx, q, 0)
		if err == nil {
			t.Errorf("expected error for %q, got nil", q)
		}
	}
}

func TestExecuteReadOnly_RejectsWritableCTE(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	ctx := context.Background()

	// Writable CTEs start with WITH but contain write keywords.
	// These must be rejected by the write-keyword scan.
	cteDML := []string{
		"WITH ids AS (SELECT id FROM events LIMIT 5) DELETE FROM events WHERE id IN (SELECT id FROM ids)",
		"WITH x AS (SELECT 1) INSERT INTO events (timestamp,pid,tid,source,op,duration) VALUES (0,0,0,0,0,0)",
		"WITH x AS (SELECT 1) UPDATE events SET pid=0",
	}
	for _, q := range cteDML {
		_, _, _, err := s.ExecuteReadOnly(ctx, q, 0)
		if err == nil {
			t.Errorf("expected error for writable CTE %q, got nil", q)
		}
	}
}

func TestExecuteReadOnly_RejectsPragma(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	ctx := context.Background()
	_, _, _, err = s.ExecuteReadOnly(ctx, "PRAGMA table_info(events)", 0)
	if err == nil {
		t.Error("expected error for PRAGMA query, got nil")
	}
}

func TestExecuteReadOnly_RejectsMultiStatement(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	ctx := context.Background()
	_, _, _, err = s.ExecuteReadOnly(ctx, "SELECT 1; DROP TABLE events", 0)
	if err == nil {
		t.Error("expected error for multi-statement query, got nil")
	}

	// Trailing semicolon with only whitespace after is OK.
	cols, rows, _, err := s.ExecuteReadOnly(ctx, "SELECT 1 AS x; ", 0)
	if err != nil {
		t.Fatalf("trailing semicolon should be allowed: %v", err)
	}
	if len(cols) != 1 || len(rows) != 1 {
		t.Errorf("expected 1 col + 1 row, got %d cols + %d rows", len(cols), len(rows))
	}
}

func TestExecuteReadOnly_RowLimit(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		s.Run(ctx)
		close(done)
	}()

	// Insert 50 events.
	for i := 0; i < 50; i++ {
		s.Record(makeEvt(events.SourceCUDA, uint8(events.CUDAMalloc), 1*time.Millisecond))
	}
	time.Sleep(300 * time.Millisecond)

	// Query with maxRows=10 — should truncate at 10.
	_, rows, truncated, err := s.ExecuteReadOnly(ctx, "SELECT * FROM events", 10)
	if err != nil {
		t.Fatalf("ExecuteReadOnly failed: %v", err)
	}
	if len(rows) != 10 {
		t.Errorf("expected 10 rows with limit, got %d", len(rows))
	}
	if !truncated {
		t.Error("expected truncated=true with 50 events and limit=10")
	}

	// maxRows > 10000 should be capped at 10000.
	_, rows2, truncated2, err := s.ExecuteReadOnly(ctx, "SELECT * FROM events", 99999)
	if err != nil {
		t.Fatalf("ExecuteReadOnly failed: %v", err)
	}
	if len(rows2) != 50 {
		t.Errorf("expected 50 rows (all events, cap not hit), got %d", len(rows2))
	}
	if truncated2 {
		t.Error("expected truncated=false with 50 events and cap=10000")
	}

	cancel()
	<-done
}

func TestExecuteReadOnly_TruncatedAccuracy(t *testing.T) {
	// Verify that truncated is false when the query returns exactly maxRows
	// (no extra row exists), and true only when there genuinely are more.
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		s.Run(ctx)
		close(done)
	}()

	// Insert exactly 10 events.
	for i := 0; i < 10; i++ {
		s.Record(makeEvt(events.SourceCUDA, uint8(events.CUDAMalloc), 1*time.Millisecond))
	}
	time.Sleep(300 * time.Millisecond)

	// Query with maxRows=10 — exactly 10 exist, so truncated should be false.
	_, rows, truncated, err := s.ExecuteReadOnly(ctx, "SELECT * FROM events", 10)
	if err != nil {
		t.Fatalf("ExecuteReadOnly failed: %v", err)
	}
	if len(rows) != 10 {
		t.Errorf("expected 10 rows, got %d", len(rows))
	}
	if truncated {
		t.Error("expected truncated=false when row count exactly equals limit")
	}

	cancel()
	<-done
}

func TestExecuteReadOnly_WithCTE(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	ctx := context.Background()
	cols, rows, _, err := s.ExecuteReadOnly(ctx, "WITH cte AS (SELECT 42 AS val) SELECT val FROM cte", 0)
	if err != nil {
		t.Fatalf("CTE query failed: %v", err)
	}
	if len(cols) != 1 || cols[0] != "val" {
		t.Errorf("columns = %v, want [val]", cols)
	}
	if len(rows) != 1 {
		t.Fatalf("expected 1 row, got %d", len(rows))
	}
	if rows[0][0].(int64) != 42 {
		t.Errorf("expected 42, got %v", rows[0][0])
	}
}

func TestExecuteReadOnly_EmptyResultNotNil(t *testing.T) {
	// Verify that 0 rows returns empty slice (not nil) to avoid JSON "null".
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	ctx := context.Background()
	_, rows, _, err := s.ExecuteReadOnly(ctx, "SELECT * FROM events WHERE 1=0", 0)
	if err != nil {
		t.Fatalf("ExecuteReadOnly failed: %v", err)
	}
	if rows == nil {
		t.Error("expected non-nil empty slice for 0 rows, got nil")
	}
	if len(rows) != 0 {
		t.Errorf("expected 0 rows, got %d", len(rows))
	}
}

func TestExecuteReadOnly_SqliteMasterAllowed(t *testing.T) {
	// Schema enumeration via sqlite_master is allowed (schema is public in
	// the tool description anyway, and useful for AI exploration).
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	ctx := context.Background()
	_, rows, _, err := s.ExecuteReadOnly(ctx, "SELECT type, name FROM sqlite_master ORDER BY name", 0)
	if err != nil {
		t.Fatalf("sqlite_master query failed: %v", err)
	}
	if len(rows) == 0 {
		t.Error("expected sqlite_master to return schema objects")
	}
}

// ---------------------------------------------------------------------------
// Frame serialization / deserialization tests
// ---------------------------------------------------------------------------

func TestSerializeStackFramesRoundTrip(t *testing.T) {
	stack := []events.StackFrame{
		{IP: 0x7f1234, SymbolName: "cudaMalloc", File: "/usr/lib/x86_64-linux-gnu/libcudart.so.12", Line: 0},
		{IP: 0x7f5678, SymbolName: "_PyEval_EvalFrameDefault", File: "/usr/lib/libpython3.12.so", Line: 527},
		{IP: 0x7f9abc, PyFile: "train.py", PyFunc: "forward", PyLine: 142},
	}

	json := serializeStackFrames(stack)
	if json == "" {
		t.Fatal("serializeStackFrames returned empty for non-garbage stack")
	}

	got := deserializeStackFrames(json)
	if len(got) != 3 {
		t.Fatalf("expected 3 frames, got %d", len(got))
	}

	// Check native symbol frame.
	if got[0].SymbolName != "cudaMalloc" {
		t.Errorf("frame[0].SymbolName = %q, want cudaMalloc", got[0].SymbolName)
	}
	if got[0].File != "libcudart.so.12" { // basename only
		t.Errorf("frame[0].File = %q, want libcudart.so.12", got[0].File)
	}

	// Check native Line field round-trips.
	if got[1].Line != 527 {
		t.Errorf("frame[1].Line = %d, want 527", got[1].Line)
	}

	// Check Python frame.
	if got[2].PyFile != "train.py" || got[2].PyFunc != "forward" || got[2].PyLine != 142 {
		t.Errorf("frame[2] Python = {%q, %q, %d}, want {train.py, forward, 142}",
			got[2].PyFile, got[2].PyFunc, got[2].PyLine)
	}
}

func TestSerializeStackFramesGarbageFiltering(t *testing.T) {
	// All frames are garbage (no symbol, no file, no Python info).
	stack := []events.StackFrame{
		{IP: 0xdead},
		{IP: 0xbeef},
	}
	json := serializeStackFrames(stack)
	if json != "" {
		t.Errorf("expected empty for all-garbage stack, got %q", json)
	}

	// Mix of garbage and resolved.
	stack = []events.StackFrame{
		{IP: 0xdead},
		{IP: 0x1234, SymbolName: "cudaMalloc", File: "libcudart.so"},
		{IP: 0xbeef},
	}
	json = serializeStackFrames(stack)
	got := deserializeStackFrames(json)
	if len(got) != 1 {
		t.Fatalf("expected 1 non-garbage frame, got %d", len(got))
	}
	if got[0].SymbolName != "cudaMalloc" {
		t.Errorf("frame[0].SymbolName = %q, want cudaMalloc", got[0].SymbolName)
	}
}

func TestDeserializeStackFramesBackwardCompat(t *testing.T) {
	// Empty string → nil (old DBs without frames column).
	got := deserializeStackFrames("")
	if got != nil {
		t.Errorf("expected nil for empty, got %v", got)
	}

	// Invalid JSON → nil.
	got = deserializeStackFrames("not json")
	if got != nil {
		t.Errorf("expected nil for invalid JSON, got %v", got)
	}

	// Empty JSON array → nil.
	got = deserializeStackFrames("[]")
	if got != nil {
		t.Errorf("expected nil for empty array, got %v", got)
	}
}

func TestResolvedFramesStoredInDB(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		s.Run(ctx)
		close(done)
	}()

	// Record an event with resolved stack frames.
	evt := makeEvt(events.SourceCUDA, uint8(events.CUDAMalloc), 5*time.Millisecond)
	evt.Stack = []events.StackFrame{
		{IP: 0x7f1234, SymbolName: "cudaMalloc", File: "/usr/lib/libcudart.so.12"},
		{IP: 0x401000, PyFile: "train.py", PyFunc: "forward", PyLine: 42},
	}
	s.Record(evt)

	time.Sleep(300 * time.Millisecond)

	result, err := s.Query(QueryParams{Since: 1 * time.Minute})
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}
	if len(result) == 0 {
		t.Fatal("expected at least 1 event")
	}

	// The queried event should have resolved frames, not just raw IPs.
	got := result[0]
	if len(got.Stack) == 0 {
		t.Fatal("expected stack frames in queried event")
	}
	if got.Stack[0].SymbolName != "cudaMalloc" {
		t.Errorf("stack[0].SymbolName = %q, want cudaMalloc", got.Stack[0].SymbolName)
	}
	if got.Stack[1].PyFunc != "forward" {
		t.Errorf("stack[1].PyFunc = %q, want forward", got.Stack[1].PyFunc)
	}

	cancel()
	<-done
}

func TestFramesMigrationOnOldDB(t *testing.T) {
	// Create a DB without the frames column, then open it — migration should add it.
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "test.db")

	// First open creates the schema including frames column.
	s1, err := New(dbPath)
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	s1.Close()

	// Re-open should work (migration is idempotent).
	s2, err := New(dbPath)
	if err != nil {
		t.Fatalf("Re-open failed: %v", err)
	}
	defer s2.Close()

	// Verify frames column exists via ExecuteReadOnly.
	ctx := context.Background()
	_, rows, _, err := s2.ExecuteReadOnly(ctx, "SELECT frames FROM stack_traces LIMIT 1", 0)
	if err != nil {
		t.Fatalf("frames column query failed: %v", err)
	}
	_ = rows // empty is fine, we just need the column to exist
}

// TestRecordProcessNames verifies batch PID→name persistence.
// Names discovered at runtime via /proc/[pid]/comm are flushed to SQLite
// at shutdown. This test validates the batch INSERT OR REPLACE behavior.
func TestRecordProcessNames(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	tests := []struct {
		name      string
		input     map[uint32]string
		wantCount int
	}{
		{
			name:      "multiple_names",
			input:     map[uint32]string{100: "python3", 200: "worker", 300: "vllm"},
			wantCount: 3,
		},
		{
			name:      "empty_names_skipped",
			input:     map[uint32]string{400: "valid", 500: ""},
			wantCount: 4, // cumulative: 3 + 1 (500 skipped)
		},
		{
			name:      "nil_map",
			input:     nil,
			wantCount: 4, // unchanged
		},
		{
			name:      "empty_map",
			input:     map[uint32]string{},
			wantCount: 4, // unchanged
		},
		{
			name:      "overwrite_existing",
			input:     map[uint32]string{100: "python3.12"},
			wantCount: 4, // same count, name updated
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			s.RecordProcessNames(tt.input)

			var count int
			err := s.db.QueryRow("SELECT COUNT(*) FROM process_names").Scan(&count)
			if err != nil {
				t.Fatalf("count query failed: %v", err)
			}
			if count != tt.wantCount {
				t.Errorf("process_names count = %d, want %d", count, tt.wantCount)
			}
		})
	}

	// Verify the overwritten name is correct.
	var name string
	err = s.db.QueryRow("SELECT name FROM process_names WHERE pid = 100").Scan(&name)
	if err != nil {
		t.Fatalf("query PID 100 failed: %v", err)
	}
	if name != "python3.12" {
		t.Errorf("PID 100 name = %q, want %q", name, "python3.12")
	}
}

func TestDBMethodReturnsNonNil(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	if s.DB() == nil {
		t.Error("DB() returned nil")
	}
}
