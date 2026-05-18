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

// TestAnnotationAppliesViaIncarnation_Conservative covers the
// degradation rule: the join only matches when both sides resolve to
// the same incarnation interval, or when neither resolves. It must NOT
// bare-PID match when exactly one side resolves.
func TestAnnotationAppliesViaIncarnation_Conservative(t *testing.T) {
	a := annotate.Annotation{
		TimestampNs: 100,
		Labels:      map[string]string{"step": "1"},
		Process:     annotate.ProcessIncarnation{PID: 42},
	}
	e := EventWithMeta{TimestampNs: 200, PID: 42}

	cases := []struct {
		name        string
		annIv, evIv int
		want        bool
	}{
		{"both unresolved (pre-trace process)", -1, -1, true},
		{"both resolved, same interval", 0, 0, true},
		{"both resolved, different interval", 0, 1, false},
		{"annotation resolved, event unresolved", 0, -1, false},
		{"event resolved, annotation unresolved", -1, 0, false},
	}
	for _, tc := range cases {
		if got := annotationAppliesViaIncarnation(a, e, tc.annIv, tc.evIv); got != tc.want {
			t.Errorf("%s: got %v, want %v", tc.name, got, tc.want)
		}
	}
}
