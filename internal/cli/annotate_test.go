package cli

import (
	"os"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"

	intannotate "github.com/ingero-io/ingero/internal/annotate"
	"github.com/ingero-io/ingero/pkg/annotate"
	"github.com/ingero-io/ingero/pkg/contract"
)

func TestValidateAnnotationLine(t *testing.T) {
	good := []string{
		`{"labels":{"step":"1"}}`,
		`{"labels":{"step":"1","epoch":"2"},"pid":4321}`,
		`{"labels":{"phase":"warmup"},"span_start":100,"span_end":200}`,
	}
	for _, g := range good {
		if err := validateAnnotationLine([]byte(g)); err != nil {
			t.Errorf("valid line %q rejected: %v", g, err)
		}
	}
	bad := []string{
		`not json`,
		`{"labels":{}}`,
		`{"labels":{"bad key":"v"}}`,
		`{"pid":1}`, // no labels
		`{"labels":{"k":"v"},"span_start":9,"span_end":2}`,
	}
	for _, b := range bad {
		if err := validateAnnotationLine([]byte(b)); err == nil {
			t.Errorf("invalid line %q accepted", b)
		}
	}
}

func TestValidateAnnotationLine_Oversized(t *testing.T) {
	huge := `{"labels":{"k":"` + strings.Repeat("x", contract.AnnotationMaxLineBytes) + `"}}`
	if err := validateAnnotationLine([]byte(huge)); err == nil {
		t.Error("oversized line should be rejected")
	}
}

// resetAnnotateFlags clears the package-level flag vars between tests.
func resetAnnotateFlags() {
	annotateFromFile = ""
	annotateLabels = nil
	annotatePID = 0
}

func TestAnnotateCollectLines_FromLabels(t *testing.T) {
	resetAnnotateFlags()
	defer resetAnnotateFlags()
	annotateLabels = []string{"step=42", "epoch=3"}
	annotatePID = 4821

	lines, err := annotateCollectLines()
	if err != nil {
		t.Fatalf("annotateCollectLines: %v", err)
	}
	if len(lines) != 1 {
		t.Fatalf("got %d lines, want 1", len(lines))
	}
	if !strings.Contains(lines[0], `"step":"42"`) ||
		!strings.Contains(lines[0], `"pid":4821`) {
		t.Errorf("label line missing expected content: %s", lines[0])
	}
}

func TestAnnotateCollectLines_BadLabel(t *testing.T) {
	resetAnnotateFlags()
	defer resetAnnotateFlags()
	annotateLabels = []string{"noequalsign"}
	if _, err := annotateCollectLines(); err == nil {
		t.Error("expected an error for a --label without '='")
	}
}

func TestAnnotateCollectLines_FromFile(t *testing.T) {
	resetAnnotateFlags()
	defer resetAnnotateFlags()

	dir := t.TempDir()
	path := filepath.Join(dir, "markers.ndjson")
	content := `{"labels":{"step":"1"}}
{"labels":{"step":"2"}}

{"labels":{"step":"3"}}
`
	if err := os.WriteFile(path, []byte(content), 0o600); err != nil {
		t.Fatalf("write file: %v", err)
	}
	annotateFromFile = path

	lines, err := annotateCollectLines()
	if err != nil {
		t.Fatalf("annotateCollectLines: %v", err)
	}
	if len(lines) != 3 {
		t.Fatalf("got %d lines, want 3 (blank line skipped)", len(lines))
	}
}

func TestAnnotateCollectLines_FromFile_BadLine(t *testing.T) {
	resetAnnotateFlags()
	defer resetAnnotateFlags()

	dir := t.TempDir()
	path := filepath.Join(dir, "bad.ndjson")
	if err := os.WriteFile(path, []byte(`{"labels":{"k":"v"}}`+"\nnot json\n"), 0o600); err != nil {
		t.Fatalf("write file: %v", err)
	}
	annotateFromFile = path
	if _, err := annotateCollectLines(); err == nil {
		t.Error("expected an error for a malformed NDJSON line")
	}
}

func TestAnnotateCollectLines_FromStdin(t *testing.T) {
	resetAnnotateFlags()
	defer resetAnnotateFlags()

	r, w, err := os.Pipe()
	if err != nil {
		t.Fatalf("pipe: %v", err)
	}
	orig := os.Stdin
	os.Stdin = r
	defer func() { os.Stdin = orig }()

	go func() {
		w.WriteString(`{"labels":{"step":"99"}}` + "\n")
		w.Close()
	}()

	lines, err := annotateCollectLines()
	if err != nil {
		t.Fatalf("annotateCollectLines: %v", err)
	}
	if len(lines) != 1 || !strings.Contains(lines[0], `"step":"99"`) {
		t.Errorf("stdin lines = %v", lines)
	}
}

// memSink collects annotations for the end-to-end test.
type memSink struct {
	mu   sync.Mutex
	rows []annotate.Annotation
}

func (m *memSink) RecordAnnotation(a annotate.Annotation) error {
	m.mu.Lock()
	m.rows = append(m.rows, a)
	m.mu.Unlock()
	return nil
}

func (m *memSink) count() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.rows)
}

// TestAnnotate_EndToEnd runs annotateRunE against a real ingest server.
func TestAnnotate_EndToEnd(t *testing.T) {
	resetAnnotateFlags()
	defer resetAnnotateFlags()

	sink := &memSink{}
	srv := intannotate.NewServer(sink)
	if err := srv.Start(); err != nil {
		t.Skipf("ingest server start failed (shared socket path in use?): %v", err)
	}
	defer srv.Close()

	// Point the client at the server's actual socket. The client
	// normally dials the fixed agent-owned path; the test seam keeps
	// this hermetic.
	annotateDialPath = func() string { return srv.SocketPath() }
	defer func() { annotateDialPath = intannotate.SocketPath }()
	annotateLabels = []string{"step=7"}

	if err := annotateRunE(annotateCmd, nil); err != nil {
		t.Fatalf("annotateRunE: %v", err)
	}

	deadline := time.Now().Add(3 * time.Second)
	for time.Now().Before(deadline) && sink.count() == 0 {
		time.Sleep(5 * time.Millisecond)
	}
	if sink.count() != 1 {
		t.Fatalf("server received %d annotations, want 1", sink.count())
	}
}

func TestAnnotate_NoSource(t *testing.T) {
	resetAnnotateFlags()
	defer resetAnnotateFlags()

	// Empty stdin, no flags - annotateRunE should report "no annotations".
	r, w, _ := os.Pipe()
	orig := os.Stdin
	os.Stdin = r
	w.Close()
	defer func() { os.Stdin = orig }()

	err := annotateRunE(annotateCmd, nil)
	if err == nil || !strings.Contains(err.Error(), "no annotations") {
		t.Errorf("expected a 'no annotations' error, got %v", err)
	}
}

func TestAnnotate_SocketUnreachable(t *testing.T) {
	resetAnnotateFlags()
	defer resetAnnotateFlags()

	annotateLabels = []string{"step=1"}
	// Point at a missing per-test socket so the dial fails deterministically.
	annotateDialPath = func() string { return filepath.Join(t.TempDir(), "nope.sock") }
	defer func() { annotateDialPath = intannotate.SocketPath }()
	err := annotateRunE(annotateCmd, nil)
	if err == nil {
		t.Error("expected an error connecting to a missing socket")
	}
}

// TestExplainAnnotationsFlag_RolloverNote asserts the explain
// --annotations flag help documents that it covers the live DB only
// and points at 'query --include-rolled --annotations' for the
// rollover-spanning path. explain joins only the live DB.
func TestExplainAnnotationsFlag_RolloverNote(t *testing.T) {
	f := explainCmd.Flags().Lookup("annotations")
	if f == nil {
		t.Fatal("explain has no --annotations flag")
	}
	if !strings.Contains(f.Usage, "live DB only") {
		t.Errorf("--annotations help should state it covers the live DB only: %q", f.Usage)
	}
	if !strings.Contains(f.Usage, "--include-rolled") {
		t.Errorf("--annotations help should point at query --include-rolled: %q", f.Usage)
	}
}
