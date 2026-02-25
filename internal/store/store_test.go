package store

import (
	"context"
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
