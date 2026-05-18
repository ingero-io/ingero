package store

import (
	"path/filepath"
	"testing"
	"time"

	"github.com/ingero-io/ingero/pkg/annotate"
)

// TestAnnotation_StoreReadRoundTrip stores annotations and reads them back.
func TestAnnotation_StoreReadRoundTrip(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	now := time.Now().UnixNano()
	a := annotate.Annotation{
		TimestampNs: now,
		Labels:      map[string]string{"step": "42", "epoch": "3"},
		Process:     annotate.ProcessIncarnation{PID: 1234, StartTime: 999888},
		Provenance:  annotate.Provenance{PeerUID: 1000, PeerGID: 1000, PeerPID: 4321},
	}
	if err := s.RecordAnnotation(a); err != nil {
		t.Fatalf("RecordAnnotation: %v", err)
	}

	got, err := s.QueryAnnotations(AnnotationQuery{Since: time.Hour})
	if err != nil {
		t.Fatalf("QueryAnnotations: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d annotations, want 1", len(got))
	}
	g := got[0]
	if g.TimestampNs != now {
		t.Errorf("timestamp = %d, want %d", g.TimestampNs, now)
	}
	if g.Labels["step"] != "42" || g.Labels["epoch"] != "3" {
		t.Errorf("labels = %v", g.Labels)
	}
	if g.Process.PID != 1234 || g.Process.StartTime != 999888 {
		t.Errorf("incarnation = %v", g.Process)
	}
	if g.Provenance.PeerUID != 1000 || g.Provenance.PeerPID != 4321 {
		t.Errorf("provenance = %v", g.Provenance)
	}
}

// TestAnnotation_QueryByPID asserts the PID filter returns the scoped
// annotation plus unscoped (trace-wide) annotations, and excludes other
// PIDs.
func TestAnnotation_QueryByPID(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	now := time.Now().UnixNano()
	anns := []annotate.Annotation{
		{TimestampNs: now, Labels: map[string]string{"a": "1"}, Process: annotate.ProcessIncarnation{PID: 100, StartTime: 1}},
		{TimestampNs: now, Labels: map[string]string{"b": "2"}, Process: annotate.ProcessIncarnation{PID: 200, StartTime: 2}},
		{TimestampNs: now, Labels: map[string]string{"c": "3"}}, // unscoped
	}
	for _, a := range anns {
		if err := s.RecordAnnotation(a); err != nil {
			t.Fatalf("RecordAnnotation: %v", err)
		}
	}

	got, err := s.QueryAnnotations(AnnotationQuery{Since: time.Hour, PID: 100})
	if err != nil {
		t.Fatalf("QueryAnnotations: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("PID-filtered query returned %d, want 2 (pid 100 + unscoped)", len(got))
	}
	for _, g := range got {
		if g.Process.PID == 200 {
			t.Error("pid 200 annotation leaked into a pid-100 query")
		}
	}
}

// TestAnnotation_Span round-trips a span annotation.
func TestAnnotation_Span(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	now := time.Now().UnixNano()
	a := annotate.Annotation{
		TimestampNs: now,
		Labels:      map[string]string{"phase": "warmup"},
		SpanStartNs: now,
		SpanEndNs:   now + int64(time.Minute),
	}
	if err := s.RecordAnnotation(a); err != nil {
		t.Fatalf("RecordAnnotation: %v", err)
	}
	got, err := s.QueryAnnotations(AnnotationQuery{Since: time.Hour})
	if err != nil {
		t.Fatalf("QueryAnnotations: %v", err)
	}
	if len(got) != 1 || !got[0].IsSpan() {
		t.Fatalf("expected one span annotation, got %d", len(got))
	}
	if got[0].SpanEndNs-got[0].SpanStartNs != int64(time.Minute) {
		t.Errorf("span width = %d", got[0].SpanEndNs-got[0].SpanStartNs)
	}
}

// TestAnnotation_TableOnRolledDB asserts a rolled-over fresh DB carries
// the annotations table, so an annotation written after a rollover lands.
func TestAnnotation_TableOnRolledDB(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "ingero.db")
	s, err := New(dbPath)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	// Force a rollover; the fresh DB must be created with the table.
	if err := s.RolloverNow("test", &RolloverConfig{MaxSize: 1, KeepFiles: 6}); err != nil {
		t.Fatalf("RolloverNow: %v", err)
	}

	a := annotate.Annotation{
		TimestampNs: time.Now().UnixNano(),
		Labels:      map[string]string{"step": "1"},
	}
	if err := s.RecordAnnotation(a); err != nil {
		t.Fatalf("RecordAnnotation after rollover failed (table missing?): %v", err)
	}
	got, err := s.QueryAnnotations(AnnotationQuery{Since: time.Hour})
	if err != nil {
		t.Fatalf("QueryAnnotations: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("got %d annotations on post-rollover DB, want 1", len(got))
	}
}

// TestAnnotation_PruneBound asserts the size-prune deletes old
// annotation rows so an externally-writable table cannot grow unbounded.
func TestAnnotation_PruneBound(t *testing.T) {
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "ingero.db")
	s, err := New(dbPath)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	defer s.Close()

	base := time.Now().Add(-2 * time.Hour).UnixNano()
	// Spread 400 annotations across two hours.
	for i := 0; i < 400; i++ {
		a := annotate.Annotation{
			TimestampNs: base + int64(i)*int64(18*time.Second),
			Labels:      map[string]string{"i": "x"},
		}
		if err := s.RecordAnnotation(a); err != nil {
			t.Fatalf("RecordAnnotation: %v", err)
		}
	}
	before, _ := s.QueryAnnotations(AnnotationQuery{Since: 3 * time.Hour, Limit: -1})

	// Set a tiny max size and prune. The cutoff is computed from the
	// annotations time range (no events present), so old rows go.
	s.SetMaxDBSize(1) // 1 byte: everything is "over"
	s.pruneBySize()

	after, _ := s.QueryAnnotations(AnnotationQuery{Since: 3 * time.Hour, Limit: -1})
	if len(after) >= len(before) {
		t.Errorf("prune did not delete annotation rows: before=%d after=%d",
			len(before), len(after))
	}
}

func TestJoinAnnotations_IncarnationAndWindow(t *testing.T) {
	now := int64(1_000_000_000)
	evts := []EventWithMeta{
		{TimestampNs: now + 10, PID: 100, StartTime: 1},
		{TimestampNs: now + 10, PID: 100, StartTime: 2}, // reused PID, new incarnation
		{TimestampNs: now - 10, PID: 100, StartTime: 1}, // before the annotation
	}
	anns := []annotate.Annotation{
		{
			TimestampNs: now,
			Labels:      map[string]string{"step": "5"},
			Process:     annotate.ProcessIncarnation{PID: 100, StartTime: 1},
		},
	}
	out := JoinAnnotations(evts, anns)
	if out[0].Labels["step"] != "5" {
		t.Error("event 0 (same incarnation, after annotation) should carry the label")
	}
	if out[1].MatchedAnnotations != 0 {
		t.Error("event 1 (reused PID, different start_time) must NOT match - PID reuse cross-attribution")
	}
	if out[2].MatchedAnnotations != 0 {
		t.Error("event 2 (before the annotation instant) must NOT match")
	}
}

func TestJoinAnnotations_Span(t *testing.T) {
	anns := []annotate.Annotation{
		{
			Labels:      map[string]string{"phase": "train"},
			SpanStartNs: 100,
			SpanEndNs:   200,
		},
	}
	evts := []EventWithMeta{
		{TimestampNs: 50},  // before span
		{TimestampNs: 150}, // inside span
		{TimestampNs: 250}, // after span
	}
	out := JoinAnnotations(evts, anns)
	if out[0].MatchedAnnotations != 0 || out[2].MatchedAnnotations != 0 {
		t.Error("events outside the span must not match")
	}
	if out[1].Labels["phase"] != "train" {
		t.Error("event inside the span must carry the phase label")
	}
}

func TestJoinAnnotations_Unscoped(t *testing.T) {
	anns := []annotate.Annotation{
		{TimestampNs: 0, Labels: map[string]string{"run_id": "abc"}}, // unscoped, instant at 0
	}
	evts := []EventWithMeta{
		{TimestampNs: 10, PID: 7, StartTime: 1},
		{TimestampNs: 20, PID: 9, StartTime: 2},
	}
	out := JoinAnnotations(evts, anns)
	for i, ae := range out {
		if ae.Labels["run_id"] != "abc" {
			t.Errorf("event %d should carry the unscoped label", i)
		}
	}
}

func TestJoinAnnotations_UnresolvedIncarnation(t *testing.T) {
	// start_time 0 means the agent could not resolve the incarnation;
	// the annotation falls back to PID-only matching.
	anns := []annotate.Annotation{
		{TimestampNs: 0, Labels: map[string]string{"task_id": "t1"},
			Process: annotate.ProcessIncarnation{PID: 55, StartTime: 0}},
	}
	evts := []EventWithMeta{
		{TimestampNs: 10, PID: 55, StartTime: 12345},
		{TimestampNs: 10, PID: 66, StartTime: 12345},
	}
	out := JoinAnnotations(evts, anns)
	if out[0].Labels["task_id"] != "t1" {
		t.Error("PID-only fallback should match the same PID")
	}
	if out[1].MatchedAnnotations != 0 {
		t.Error("a different PID must not match even with unresolved incarnation")
	}
}
