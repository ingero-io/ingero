package store

import (
	"context"
	"os"
	"path/filepath"
	"sort"
	"testing"
	"time"

	"github.com/ingero-io/ingero/pkg/events"
)

// newTestStore opens a Store on a temp DB path, returns it plus the
// path so tests can inspect the on-disk file directly. Closes the
// Store at test end.
func newTestStore(t *testing.T) (*Store, string) {
	t.Helper()
	dir := t.TempDir()
	dbPath := filepath.Join(dir, "ingero.db")
	s, err := New(dbPath)
	if err != nil {
		t.Fatalf("New: %v", err)
	}
	t.Cleanup(func() { s.Close() })
	return s, dbPath
}

func TestBuildRolledPath(t *testing.T) {
	got := buildRolledPath("/var/lib/ingero/ingero.db", time.Date(2026, 5, 9, 14, 30, 55, 0, time.UTC))
	want := "/var/lib/ingero/ingero.20260509T143055Z.db"
	// On Windows the separator differs; comparing with ToSlash makes the
	// test portable. The actual rollover code uses filepath.Join which
	// is platform-correct.
	if filepath.ToSlash(got) != want {
		t.Errorf("buildRolledPath = %q, want %q", got, want)
	}
}

func TestListRolledFiles(t *testing.T) {
	dir := t.TempDir()
	livePath := filepath.Join(dir, "ingero.db")

	// Create the live file plus three rolled siblings (out of order
	// chronologically) and a hand-placed sibling that doesn't match
	// the rollover-timestamp pattern (must be skipped).
	mustTouch := func(p string) {
		t.Helper()
		f, err := os.Create(p)
		if err != nil {
			t.Fatal(err)
		}
		f.Close()
	}
	mustTouch(livePath)
	mustTouch(filepath.Join(dir, "ingero.20260509T140000Z.db"))
	mustTouch(filepath.Join(dir, "ingero.20260509T120000Z.db"))
	mustTouch(filepath.Join(dir, "ingero.20260509T130000Z.db"))
	// Hand-placed sibling that should NOT count as a rolled file.
	mustTouch(filepath.Join(dir, "ingero.notes.db"))
	// Sibling without the pattern - also skipped.
	mustTouch(filepath.Join(dir, "ingero.backup.db"))

	got, err := ListRolledFiles(livePath)
	if err != nil {
		t.Fatalf("ListRolledFiles: %v", err)
	}
	want := []string{
		filepath.Join(dir, "ingero.20260509T120000Z.db"),
		filepath.Join(dir, "ingero.20260509T130000Z.db"),
		filepath.Join(dir, "ingero.20260509T140000Z.db"),
	}
	if len(got) != len(want) {
		t.Fatalf("ListRolledFiles: got %d, want %d (got=%v)", len(got), len(want), got)
	}
	for i, w := range want {
		if got[i] != w {
			t.Errorf("ListRolledFiles[%d] = %q, want %q", i, got[i], w)
		}
	}
}

func TestListRolledFiles_NoRolled(t *testing.T) {
	dir := t.TempDir()
	livePath := filepath.Join(dir, "ingero.db")
	f, err := os.Create(livePath)
	if err != nil {
		t.Fatal(err)
	}
	f.Close()
	got, err := ListRolledFiles(livePath)
	if err != nil {
		t.Fatalf("ListRolledFiles: %v", err)
	}
	if len(got) != 0 {
		t.Errorf("ListRolledFiles on lone file = %v, want empty", got)
	}
}

func TestLooksLikeRolloverTimestamp(t *testing.T) {
	cases := []struct {
		s    string
		want bool
	}{
		{"20260509T143055Z", true},
		{"20260509T143055", false},  // missing Z
		{"20260509T143055X", false}, // wrong terminator
		{"2026050T1430555Z", false}, // misplaced T
		{"", false},
		{"abc", false},
		{"20260509T14305aZ", false}, // non-digit
	}
	for _, tc := range cases {
		if got := looksLikeRolloverTimestamp(tc.s); got != tc.want {
			t.Errorf("looksLikeRolloverTimestamp(%q) = %v, want %v", tc.s, got, tc.want)
		}
	}
}

func TestRollover_DisabledWhenNoConfig(t *testing.T) {
	s, _ := newTestStore(t)
	if err := s.MaybeRollover(); err != nil {
		t.Errorf("MaybeRollover with no config returned err: %v", err)
	}
}

func TestRollover_TriggersAtSizeThreshold(t *testing.T) {
	s, dbPath := newTestStore(t)

	// Configure rollover with a tiny threshold and drive rollover
	// directly (no Run goroutine concurrent with the swap). In
	// production the flush hooks inside Run() invoke MaybeRollover()
	// from the same goroutine that owns flushBatch, so external
	// concurrency is not a concern. The TestRollover_*Recordable
	// case below covers the post-rollover write path.
	s.SetRolloverConfig(RolloverConfig{
		MaxSize:   1,
		KeepFiles: 6,
	})

	// Sanity: the freshly-opened DB has a non-zero footprint
	// (page header + lookup tables + schema_info).
	beforeUsage := s.diskUsage()
	if beforeUsage == 0 {
		t.Fatalf("fresh DB usage is 0; nothing to roll")
	}

	if err := s.MaybeRollover(); err != nil {
		t.Fatalf("MaybeRollover: %v", err)
	}
	if got := s.RolloverStats().Count; got != 1 {
		t.Errorf("rollover count = %d, want 1", got)
	}

	// A rolled sibling exists at the timestamped path.
	dir := filepath.Dir(dbPath)
	matches, err := filepath.Glob(filepath.Join(dir, "ingero.*.db"))
	if err != nil {
		t.Fatalf("glob: %v", err)
	}
	rolled := 0
	for _, m := range matches {
		if m != dbPath {
			rolled++
		}
	}
	if rolled != 1 {
		t.Errorf("rolled file count = %d, want 1", rolled)
	}
}

func TestRollover_RetentionSweepKeepsN(t *testing.T) {
	s, dbPath := newTestStore(t)
	s.SetRolloverConfig(RolloverConfig{MaxSize: 1, KeepFiles: 2})

	// Force several rollovers in succession. Sleep 1.1s between rollovers
	// so the timestamp-based filenames don't collide (resolution = 1
	// second). Driven directly to avoid the Run-goroutine race window.
	for i := 0; i < 5; i++ {
		if err := s.MaybeRollover(); err != nil {
			t.Fatalf("MaybeRollover #%d: %v", i, err)
		}
		time.Sleep(1100 * time.Millisecond)
	}

	dir := filepath.Dir(dbPath)
	matches, err := filepath.Glob(filepath.Join(dir, "ingero.*.db"))
	if err != nil {
		t.Fatalf("glob: %v", err)
	}
	var rolled []string
	for _, m := range matches {
		if m != dbPath {
			rolled = append(rolled, m)
		}
	}
	sort.Strings(rolled)
	if len(rolled) != 2 {
		t.Errorf("rolled count after sweep = %d, want 2 (KeepFiles)\nfiles: %v", len(rolled), rolled)
	}
}

func TestRollover_FreshDBStillRecordable(t *testing.T) {
	s, _ := newTestStore(t)

	// Roll BEFORE starting Run so the swap happens with no concurrent
	// writer. After the rollover we start Run and verify that events
	// flow into the freshly-created DB.
	s.SetRolloverConfig(RolloverConfig{MaxSize: 1, KeepFiles: 6})
	if err := s.MaybeRollover(); err != nil {
		t.Fatalf("MaybeRollover: %v", err)
	}
	if got := s.RolloverStats().Count; got != 1 {
		t.Fatalf("rollover count = %d, want 1", got)
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	go s.Run(ctx)

	// Drop the size threshold so subsequent flushes don't keep
	// rolling — the freshly opened DB starts under 1 byte but
	// MaybeRollover would chase a 1-byte cap forever.
	s.SetRolloverConfig(RolloverConfig{MaxSize: 1 << 30, KeepFiles: 6})

	// Record events on the fresh DB.
	for i := 0; i < 20; i++ {
		s.Record(events.Event{
			Timestamp: time.Now(),
			PID:       uint32(3000 + i),
			Source:    events.SourceCUDA,
			Op:        1,
			Duration:  time.Microsecond,
		})
	}
	time.Sleep(300 * time.Millisecond)
	cancel()
	s.WaitDone()

	// Query the live DB to confirm post-rollover events landed.
	q := QueryParams{
		From:   time.Now().Add(-time.Hour),
		To:     time.Now().Add(time.Hour),
		Source: uint8(events.SourceCUDA),
	}
	results, err := s.Query(q)
	if err != nil {
		t.Fatalf("Query post-rollover: %v", err)
	}
	if len(results) == 0 {
		t.Error("post-rollover Query returned 0 events; expected the 20 we just recorded")
	}
}

func TestRollover_InMemoryNotSupported(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New(:memory:): %v", err)
	}
	defer s.Close()
	s.SetRolloverConfig(RolloverConfig{MaxSize: 1, KeepFiles: 6})
	if err := s.RolloverNow("test", nil); err == nil {
		t.Error("RolloverNow on :memory: should error")
	}
}

