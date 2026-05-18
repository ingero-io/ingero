package store

import (
	"context"
	"path/filepath"
	"testing"
	"time"

	"github.com/ingero-io/ingero/pkg/annotate"
	"github.com/ingero-io/ingero/pkg/events"
)

// newRunCtx is a cancellable context for driving Store.Run in tests.
func newRunCtx() (context.Context, context.CancelFunc) {
	return context.WithCancel(context.Background())
}

// waitForEvents polls the live DB until it holds at least n events or a
// deadline elapses. Used so a test can be sure the async flush loop has
// drained a batch to disk before it queries.
func waitForEvents(t *testing.T, s *Store, n int) {
	t.Helper()
	deadline := time.Now().Add(5 * time.Second)
	for time.Now().Before(deadline) {
		evts, err := s.Query(QueryParams{Since: 3 * time.Hour, Limit: -1})
		if err == nil && len(evts) >= n {
			return
		}
		time.Sleep(10 * time.Millisecond)
	}
	t.Fatalf("timed out waiting for %d events to flush", n)
}

// recordLifecycle writes a process_exec or process_exit event so the
// incarnation index has boundaries to work with.
func recordLifecycle(t *testing.T, s *Store, pid uint32, op events.HostOp, tsNs int64) {
	t.Helper()
	s.Record(events.Event{
		Timestamp: time.Unix(0, tsNs),
		PID:       pid,
		Source:    events.SourceHost,
		Op:        uint8(op),
	})
}

// TestAnnotateEvents_IncarnationJoin asserts the incarnation-aware join
// over a live store: events and annotations in the same incarnation
// interval match, and a reused PID does not cross-attribute.
func TestAnnotateEvents_IncarnationJoin(t *testing.T) {
	dir := t.TempDir()
	s, err := New(filepath.Join(dir, "ingero.db"))
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	ctx, cancel := newRunCtx()
	defer cancel()
	go s.Run(ctx)
	s.WaitStarted()

	base := time.Now().Add(-time.Hour).UnixNano()
	// Incarnation 1 of PID 500: [base, base+1000].
	recordLifecycle(t, s, 500, events.HostProcessExec, base)
	recordLifecycle(t, s, 500, events.HostProcessExit, base+1000)
	// Incarnation 2 of the SAME PID 500: [base+2000, base+3000].
	recordLifecycle(t, s, 500, events.HostProcessExec, base+2000)
	recordLifecycle(t, s, 500, events.HostProcessExit, base+3000)
	// A cudaMalloc event in incarnation 1 and one in incarnation 2.
	s.Record(events.Event{Timestamp: time.Unix(0, base+500), PID: 500, Source: events.SourceCUDA, Op: uint8(events.CUDAMalloc)})
	s.Record(events.Event{Timestamp: time.Unix(0, base+2500), PID: 500, Source: events.SourceCUDA, Op: uint8(events.CUDAMalloc)})

	cancel()
	s.WaitDone()

	// An annotation scoped to PID 500 with a timestamp inside
	// incarnation 1. Its start_time is left 0; the join resolves the
	// incarnation by the annotation timestamp falling in interval 1.
	if err := s.RecordAnnotation(annotate.Annotation{
		TimestampNs: base + 100,
		Labels:      map[string]string{"step": "incarnation1"},
		Process:     annotate.ProcessIncarnation{PID: 500},
	}); err != nil {
		t.Fatalf("RecordAnnotation: %v", err)
	}

	evts, err := s.Query(QueryParams{Since: 2 * time.Hour, Limit: -1})
	if err != nil {
		t.Fatalf("Query: %v", err)
	}
	metas := make([]EventWithMeta, len(evts))
	for i, e := range evts {
		metas[i] = EventWithMeta{
			TimestampNs: e.Timestamp.UnixNano(),
			PID:         e.PID,
			Source:      uint8(e.Source),
			Op:          e.Op,
		}
	}
	anns, err := s.AnnotateEvents(metas, base-1000, time.Now().UnixNano())
	if err != nil {
		t.Fatalf("AnnotateEvents: %v", err)
	}

	var inc1Labeled, inc2Labeled bool
	for i, e := range evts {
		if e.Source != events.SourceCUDA {
			continue
		}
		ts := e.Timestamp.UnixNano()
		if ts == base+500 && anns[i].Labels["step"] == "incarnation1" {
			inc1Labeled = true
		}
		if ts == base+2500 && len(anns[i].Labels) > 0 {
			inc2Labeled = true
		}
	}
	if !inc1Labeled {
		t.Error("incarnation-1 event should carry the annotation label")
	}
	if inc2Labeled {
		t.Error("incarnation-2 event (reused PID, different interval) must NOT carry the label")
	}
}

// TestAnnotateEvents_RolloverBoundary writes an annotation, forces a
// rollover, writes an event in the fresh DB, and asserts the per-file
// join still resolves within each file (the event after rollover and
// the annotation before it live in different files, so a cross-file
// join correctly does NOT happen - each file is self-contained).
func TestAnnotateEvents_RolloverBoundary(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "ingero.db")
	s, err := New(dbPath)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	base := time.Now().Add(-time.Hour).UnixNano()

	// File 1: an exec, an event, an annotation - all in one incarnation.
	// Run stays active across the rollover, which is the production path.
	ctx, cancel := newRunCtx()
	defer cancel()
	go s.Run(ctx)
	s.WaitStarted()
	recordLifecycle(t, s, 700, events.HostProcessExec, base)
	s.Record(events.Event{Timestamp: time.Unix(0, base+100), PID: 700, Source: events.SourceCUDA, Op: uint8(events.CUDAMalloc)})
	// Give the flush loop a moment to drain the batch to disk.
	waitForEvents(t, s, 2)
	if err := s.RecordAnnotation(annotate.Annotation{
		TimestampNs: base + 50,
		Labels:      map[string]string{"phase": "file1"},
		Process:     annotate.ProcessIncarnation{PID: 700},
	}); err != nil {
		t.Fatalf("RecordAnnotation: %v", err)
	}

	// Join file 1 in place: the event should pick up the file-1 label.
	evts1, _ := s.Query(QueryParams{Since: 2 * time.Hour, Limit: -1})
	metas1 := make([]EventWithMeta, len(evts1))
	for i, e := range evts1 {
		metas1[i] = EventWithMeta{TimestampNs: e.Timestamp.UnixNano(), PID: e.PID,
			Source: uint8(e.Source), Op: e.Op}
	}
	anns1, err := s.AnnotateEvents(metas1, base-1000, time.Now().UnixNano())
	if err != nil {
		t.Fatalf("AnnotateEvents file1: %v", err)
	}
	file1Labeled := false
	for i, e := range evts1 {
		if e.Source == events.SourceCUDA && anns1[i].Labels["phase"] == "file1" {
			file1Labeled = true
		}
	}
	if !file1Labeled {
		t.Error("file-1 event should carry the file-1 annotation before rollover")
	}

	// Force a rollover; file 1 becomes a frozen sibling.
	if err := s.RolloverNow("test", &RolloverConfig{MaxSize: 1, KeepFiles: 6}); err != nil {
		t.Fatalf("RolloverNow: %v", err)
	}

	// File 2 (fresh): a new event, no annotation. Run is still active.
	recordLifecycle(t, s, 700, events.HostProcessExec, base+10000)
	s.Record(events.Event{Timestamp: time.Unix(0, base+10100), PID: 700, Source: events.SourceCUDA, Op: uint8(events.CUDAMalloc)})
	waitForEvents(t, s, 2)

	// The live (file-2) join has no annotations; the file-2 event must
	// NOT inherit the file-1 annotation - they were never the same file.
	evts2, _ := s.Query(QueryParams{Since: 2 * time.Hour, Limit: -1})
	metas2 := make([]EventWithMeta, len(evts2))
	for i, e := range evts2 {
		metas2[i] = EventWithMeta{TimestampNs: e.Timestamp.UnixNano(), PID: e.PID,
			Source: uint8(e.Source), Op: e.Op}
	}
	anns2, err := s.AnnotateEvents(metas2, base-1000, time.Now().UnixNano())
	if err != nil {
		t.Fatalf("AnnotateEvents file2: %v", err)
	}
	for i := range evts2 {
		if len(anns2[i].Labels) > 0 {
			t.Error("file-2 event must not inherit a file-1 annotation")
		}
	}

	// And the rolled file-1 sibling, read-only, still resolves its own join.
	rolled, err := ListRolledFiles(dbPath)
	if err != nil || len(rolled) == 0 {
		t.Fatalf("expected a rolled sibling, got %v err=%v", rolled, err)
	}
	rs, err := NewReadOnly(rolled[0])
	if err != nil {
		t.Fatalf("NewReadOnly: %v", err)
	}
	defer rs.Close()
	rEvts, _ := rs.Query(QueryParams{Since: 2 * time.Hour, Limit: -1})
	rMetas := make([]EventWithMeta, len(rEvts))
	for i, e := range rEvts {
		rMetas[i] = EventWithMeta{TimestampNs: e.Timestamp.UnixNano(), PID: e.PID,
			Source: uint8(e.Source), Op: e.Op}
	}
	rAnns, err := rs.AnnotateEvents(rMetas, base-1000, time.Now().UnixNano())
	if err != nil {
		t.Fatalf("AnnotateEvents rolled: %v", err)
	}
	rolledLabeled := false
	for i, e := range rEvts {
		if e.Source == events.SourceCUDA && rAnns[i].Labels["phase"] == "file1" {
			rolledLabeled = true
		}
	}
	if !rolledLabeled {
		t.Error("rolled file-1 event should still carry its own file-1 annotation")
	}
}

// TestAnnotateEvents_LabelValueNoInjection asserts a label value
// containing SQL metacharacters is stored and read back verbatim - the
// annotation path uses parameterized SQL only, so the value cannot be
// interpreted as SQL.
func TestAnnotateEvents_LabelValueNoInjection(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	evil := `'; DROP TABLE annotations; --`
	if err := s.RecordAnnotation(annotate.Annotation{
		TimestampNs: time.Now().UnixNano(),
		Labels:      map[string]string{"task_id": evil},
	}); err != nil {
		t.Fatalf("RecordAnnotation: %v", err)
	}
	got, err := s.QueryAnnotations(AnnotationQuery{Since: time.Hour})
	if err != nil {
		t.Fatalf("QueryAnnotations: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d annotations, want 1 (table still intact)", len(got))
	}
	if got[0].Labels["task_id"] != evil {
		t.Errorf("label value = %q, want verbatim %q", got[0].Labels["task_id"], evil)
	}
	// The table still exists and is queryable - the injection was inert.
	if _, err := s.QueryAnnotations(AnnotationQuery{Since: time.Hour}); err != nil {
		t.Errorf("annotations table should still be intact: %v", err)
	}
}

// TestAnnotateEvents_SequentialSameKey asserts that for a same-key
// collision the most recent annotation at-or-before the event wins.
// Two sequential step annotations (step=1@T1, step=2@T2) must leave an
// event at T3 carrying step=2, not the oldest value.
func TestAnnotateEvents_SequentialSameKey(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	base := time.Now().Add(-time.Hour).UnixNano()
	t1 := base + 100
	t2 := base + 200
	t3 := base + 300

	// Two unscoped instant annotations advancing the step counter.
	if err := s.RecordAnnotation(annotate.Annotation{
		TimestampNs: t1, Labels: map[string]string{"step": "1"},
	}); err != nil {
		t.Fatalf("RecordAnnotation step=1: %v", err)
	}
	if err := s.RecordAnnotation(annotate.Annotation{
		TimestampNs: t2, Labels: map[string]string{"step": "2"},
	}); err != nil {
		t.Fatalf("RecordAnnotation step=2: %v", err)
	}

	// An event at T3 - after both annotations.
	metas := []EventWithMeta{{TimestampNs: t3, PID: 0}}
	anns, err := s.AnnotateEvents(metas, base-1000, time.Now().UnixNano())
	if err != nil {
		t.Fatalf("AnnotateEvents: %v", err)
	}
	if got := anns[0].Labels["step"]; got != "2" {
		t.Errorf("event at T3 has step=%q, want the most recent value 2", got)
	}
	if anns[0].MatchedAnnotations != 2 {
		t.Errorf("MatchedAnnotations = %d, want 2 (both instant annotations apply)", anns[0].MatchedAnnotations)
	}
}

// TestBuildIncarnationIndex covers the exec/exit interval construction
// including an open-ended (no exit) incarnation.
func TestBuildIncarnationIndex(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	ctx, cancel := newRunCtx()
	go s.Run(ctx)
	s.WaitStarted()
	base := time.Now().Add(-time.Hour).UnixNano()
	recordLifecycle(t, s, 1, events.HostProcessExec, base)
	recordLifecycle(t, s, 1, events.HostProcessExit, base+100)
	recordLifecycle(t, s, 1, events.HostProcessExec, base+200) // no exit: still running
	cancel()
	s.WaitDone()

	idx, err := s.buildIncarnationIndex(base-1, time.Now().UnixNano())
	if err != nil {
		t.Fatalf("buildIncarnationIndex: %v", err)
	}
	if got := idx.incarnationAt(1, base+50); got != 0 {
		t.Errorf("ts in interval 0: got index %d, want 0", got)
	}
	if got := idx.incarnationAt(1, base+150); got != -1 {
		t.Errorf("ts in the gap between incarnations: got %d, want -1", got)
	}
	if got := idx.incarnationAt(1, base+10_000_000); got != 1 {
		t.Errorf("ts in the open-ended interval 1: got %d, want 1", got)
	}
}
