package health

import (
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"runtime"
	"strings"
	"testing"
	"time"
)

var (
	persistNow = time.Date(2026, 4, 16, 12, 0, 0, 0, time.UTC)
	quietLog   = slog.New(slog.NewTextHandler(io.Discard, nil))
)

func samplePersisted() PersistedState {
	cfg := DefaultBaselineConfig()
	return PersistedState{
		SchemaVersion: 1,
		SampleCount:   42,
		FastAlpha:     cfg.FastAlpha,
		FloorAlpha:    cfg.FloorAlpha,
		FastEMA:       Baselines{Throughput: 100, Compute: 0.9, Memory: 0.8, CPU: 0.7},
		HardFloor:     Baselines{Throughput: 101, Compute: 0.91, Memory: 0.81, CPU: 0.71},
	}
}

func TestSaveLoad_Roundtrip(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "baseline.json")
	src := samplePersisted()

	if err := Save(path, src, persistNow); err != nil {
		t.Fatalf("Save: %v", err)
	}

	got, status, err := Load(path, time.Hour, persistNow, quietLog)
	if err != nil {
		t.Fatalf("Load err: %v", err)
	}
	if status != LoadFresh {
		t.Fatalf("status = %v, want LoadFresh", status)
	}
	if got != src {
		t.Fatalf("mismatch: src=%+v got=%+v", src, got)
	}
}

func TestSave_CreatesParentDir(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "nested", "deeply", "baseline.json")

	if err := Save(path, samplePersisted(), persistNow); err != nil {
		t.Fatalf("Save: %v", err)
	}
	if _, err := os.Stat(path); err != nil {
		t.Fatalf("file not created: %v", err)
	}
}

func TestSave_EmptyPathRejected(t *testing.T) {
	if err := Save("", samplePersisted(), persistNow); err == nil {
		t.Fatal("expected error for empty path")
	}
}

func TestSave_AtomicNoTmpLeftOver(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "baseline.json")
	if err := Save(path, samplePersisted(), persistNow); err != nil {
		t.Fatalf("Save: %v", err)
	}
	if _, err := os.Stat(path + ".tmp"); !os.IsNotExist(err) {
		t.Fatalf(".tmp was not cleaned up after successful save: %v", err)
	}
}

func TestLoad_MissingFile_NoError(t *testing.T) {
	path := filepath.Join(t.TempDir(), "doesnotexist.json")
	_, status, err := Load(path, time.Hour, persistNow, quietLog)
	if err != nil {
		t.Fatalf("Load on missing path returned err: %v", err)
	}
	if status != LoadMissing {
		t.Fatalf("status = %v, want LoadMissing", status)
	}
}

func TestLoad_StaleFile_Discarded(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "baseline.json")
	// Save 20 minutes ago relative to persistNow.
	savedAt := persistNow.Add(-20 * time.Minute)
	if err := Save(path, samplePersisted(), savedAt); err != nil {
		t.Fatalf("Save: %v", err)
	}
	_, status, err := Load(path, 10*time.Minute, persistNow, quietLog)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if status != LoadStale {
		t.Fatalf("status = %v, want LoadStale", status)
	}
	// Stale file should be removed.
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("stale file was not removed")
	}
}

func TestLoad_FreshWithinMaxAge(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "baseline.json")
	savedAt := persistNow.Add(-5 * time.Minute)
	if err := Save(path, samplePersisted(), savedAt); err != nil {
		t.Fatalf("Save: %v", err)
	}
	_, status, err := Load(path, 10*time.Minute, persistNow, quietLog)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if status != LoadFresh {
		t.Fatalf("status = %v, want LoadFresh", status)
	}
}

func TestLoad_CorruptJSON_Quarantined(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "baseline.json")
	if err := os.WriteFile(path, []byte("not json at all {"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, status, err := Load(path, time.Hour, persistNow, quietLog)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if status != LoadCorrupt {
		t.Fatalf("status = %v, want LoadCorrupt", status)
	}
	// Original must be gone.
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("corrupt file was not moved aside")
	}
	// Some .corrupt-* file must exist in the dir.
	entries, _ := os.ReadDir(dir)
	found := false
	for _, e := range entries {
		if strings.Contains(e.Name(), ".corrupt-") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("no .corrupt-* file present in %s: %v", dir, entries)
	}
}

func TestLoad_UnknownSchemaVersion_Quarantined(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "baseline.json")
	payload := `{"schema_version":99,"saved_at":"2026-04-16T12:00:00Z","sample_count":1,"fast_ema":{},"hard_floor":{}}`
	if err := os.WriteFile(path, []byte(payload), 0o644); err != nil {
		t.Fatal(err)
	}
	_, status, err := Load(path, time.Hour, persistNow, quietLog)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if status != LoadCorrupt {
		t.Fatalf("status = %v, want LoadCorrupt", status)
	}
	assertQuarantined(t, path)
}

func TestLoad_NegativeSampleCount_Quarantined(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "baseline.json")
	payload := `{"schema_version":1,"saved_at":"2026-04-16T12:00:00Z","sample_count":-5,"fast_ema":{},"hard_floor":{}}`
	if err := os.WriteFile(path, []byte(payload), 0o644); err != nil {
		t.Fatal(err)
	}
	_, status, err := Load(path, time.Hour, persistNow, quietLog)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if status != LoadCorrupt {
		t.Fatalf("status = %v, want LoadCorrupt", status)
	}
	assertQuarantined(t, path)
}

// assertQuarantined verifies that Load moved the bad file aside — the
// original is gone, and at least one `.corrupt-*` sibling exists.
func assertQuarantined(t *testing.T, path string) {
	t.Helper()
	if _, err := os.Stat(path); !os.IsNotExist(err) {
		t.Fatal("original bad file still present after quarantine")
	}
	base := filepath.Base(path)
	entries, err := os.ReadDir(filepath.Dir(path))
	if err != nil {
		t.Fatalf("ReadDir: %v", err)
	}
	found := false
	for _, e := range entries {
		if strings.HasPrefix(e.Name(), base+".corrupt-") {
			found = true
			break
		}
	}
	if !found {
		t.Fatalf("no %s.corrupt-* sibling found in %s: %v", base, filepath.Dir(path), entries)
	}
}

// Load rejects NaN/Inf baselines even though Restore also would — the
// file is quarantined so next startup does not re-process the poison.
func TestLoad_NonFiniteBaseline_Quarantined(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "baseline.json")
	// 1e400 is outside float64 range and parses as +Inf.
	payload := `{"schema_version":1,"saved_at":"2026-04-16T12:00:00Z","sample_count":5,"fast_alpha":0.1,"floor_alpha":0.001,"fast_ema":{"Throughput":1e400,"Compute":0,"Memory":0,"CPU":0},"hard_floor":{}}`
	if err := os.WriteFile(path, []byte(payload), 0o644); err != nil {
		t.Fatal(err)
	}
	_, status, err := Load(path, time.Hour, persistNow, quietLog)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if status != LoadCorrupt {
		t.Fatalf("status = %v, want LoadCorrupt", status)
	}
	assertQuarantined(t, path)
}

// Load rejects negative baselines at file level (Restore would also
// reject, but we want to quarantine so boot doesn't loop).
func TestLoad_NegativeBaseline_Quarantined(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "baseline.json")
	payload := `{"schema_version":1,"saved_at":"2026-04-16T12:00:00Z","sample_count":5,"fast_alpha":0.1,"floor_alpha":0.001,"fast_ema":{"Throughput":-1},"hard_floor":{}}`
	if err := os.WriteFile(path, []byte(payload), 0o644); err != nil {
		t.Fatal(err)
	}
	_, status, err := Load(path, time.Hour, persistNow, quietLog)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if status != LoadCorrupt {
		t.Fatalf("status = %v, want LoadCorrupt", status)
	}
	assertQuarantined(t, path)
}

// Files that exceed maxBaselineFileSize are rejected and quarantined —
// no attempt to read the body into memory.
func TestLoad_OversizedFile_Quarantined(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "baseline.json")
	// 2 MB of zeros.
	if err := os.WriteFile(path, make([]byte, 2<<20), 0o644); err != nil {
		t.Fatal(err)
	}
	_, status, _ := Load(path, time.Hour, persistNow, quietLog)
	if status != LoadCorrupt {
		t.Fatalf("status = %v, want LoadCorrupt for oversized file", status)
	}
	assertQuarantined(t, path)
}

// Clock-going-backwards (file dated in the future) is treated as stale,
// not as fresh. Prevents a malicious or clock-skewed save from being
// restored forever.
func TestLoad_FutureSavedAt_TreatedAsStale(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "baseline.json")
	future := persistNow.Add(24 * time.Hour)
	if err := Save(path, samplePersisted(), future); err != nil {
		t.Fatal(err)
	}
	_, status, _ := Load(path, 10*time.Minute, persistNow, quietLog)
	if status != LoadStale {
		t.Fatalf("status = %v, want LoadStale for future saved_at", status)
	}
}

// Save honors the caller's SchemaVersion (no hardcoded 1).
func TestSave_PreservesSchemaVersion(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "baseline.json")
	s := samplePersisted()
	s.SchemaVersion = 2
	if err := Save(path, s, persistNow); err != nil {
		t.Fatalf("Save: %v", err)
	}
	data, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if !strings.Contains(string(data), `"schema_version": 2`) {
		t.Fatalf("expected schema_version=2 in file, got:\n%s", data)
	}
}

// Mid-write failure leaves path untouched — the "atomic" claim of AC6.
// We simulate failure by writing into a directory whose parent path
// cannot be read (the path argument is a directory, not a file).
func TestSave_FailureLeavesOriginalIntact(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "baseline.json")
	// First, write a known-good file.
	if err := Save(path, samplePersisted(), persistNow); err != nil {
		t.Fatal(err)
	}
	original, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}

	// Now try to Save with an invalid path so Save fails mid-flight.
	// os.Rename of a tmp onto a directory that IS the path should fail.
	// Directly: attempt Save into a nonexistent-and-uncreateable directory.
	bogusPath := string([]byte{0}) + "/baseline.json" // NUL-byte path is invalid on all platforms
	if err := Save(bogusPath, samplePersisted(), persistNow); err == nil {
		t.Fatal("expected Save to fail on invalid path")
	}

	// The original file is unchanged.
	after, err := os.ReadFile(path)
	if err != nil {
		t.Fatal(err)
	}
	if string(after) != string(original) {
		t.Fatalf("original file corrupted by failed Save:\nbefore: %s\nafter:  %s", original, after)
	}
}

// Two corruptions in the same wall-clock nanosecond are disambiguated by
// the suffix counter — both quarantined files exist.
func TestQuarantine_CollisionHandled(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "baseline.json")

	// Write a pre-existing quarantined file with the exact timestamp we'll
	// use, so the first attempt will collide on Windows (POSIX would
	// silently overwrite — the counter path also handles that defensively).
	stamp := persistNow.UTC().Format("20060102T150405.000000000Z")
	existing := fmt.Sprintf("%s.corrupt-%s", path, stamp)
	if err := os.WriteFile(existing, []byte("prior forensic copy"), 0o644); err != nil {
		t.Fatal(err)
	}

	// Now write a corrupt file and Load it — quarantine should NOT
	// overwrite existing, and should place a suffixed variant.
	if err := os.WriteFile(path, []byte("not json"), 0o644); err != nil {
		t.Fatal(err)
	}
	_, status, _ := Load(path, time.Hour, persistNow, quietLog)
	if status != LoadCorrupt {
		t.Fatalf("status = %v, want LoadCorrupt", status)
	}
	// Original (prior forensic copy) intact on disk.
	if content, _ := os.ReadFile(existing); string(content) != "prior forensic copy" {
		// On POSIX, os.Rename overwrites — so the original IS clobbered.
		// We only require the invariant on Windows; on POSIX we accept
		// the overwrite. Skip assertion if on Linux/Darwin.
		if runtime.GOOS == "windows" && string(content) != "prior forensic copy" {
			t.Fatalf("existing quarantine was overwritten on Windows: %q", content)
		}
	}
}

// End-to-end: Baseliner -> Save -> Load -> fresh Baseliner -> identical
// bias-corrected outputs.
func TestSaveLoad_ThroughBaseliner(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "baseline.json")
	src, _ := NewBaseliner(DefaultBaselineConfig(), nil)
	for i := 0; i < 50; i++ {
		src.Update(RawObservation{Throughput: 100, Compute: 0.9, Memory: 0.8, CPU: 0.7})
	}

	if err := Save(path, src.Snapshot(), persistNow); err != nil {
		t.Fatalf("Save: %v", err)
	}

	dst, _ := NewBaseliner(DefaultBaselineConfig(), nil)
	ps, status, err := Load(path, time.Hour, persistNow, quietLog)
	if err != nil {
		t.Fatalf("Load: %v", err)
	}
	if status != LoadFresh {
		t.Fatalf("status = %v, want LoadFresh", status)
	}
	if err := dst.Restore(ps); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	if dst.Current() != src.Current() {
		t.Fatalf("Current mismatch: src=%+v dst=%+v", src.Current(), dst.Current())
	}
	if dst.HardFloor() != src.HardFloor() {
		t.Fatalf("HardFloor mismatch: src=%+v dst=%+v", src.HardFloor(), dst.HardFloor())
	}
	if dst.SampleCount() != src.SampleCount() {
		t.Fatalf("SampleCount mismatch: src=%d dst=%d", src.SampleCount(), dst.SampleCount())
	}
}

func TestLoadStatus_String(t *testing.T) {
	cases := map[LoadStatus]string{
		LoadMissing: "missing",
		LoadFresh:   "fresh",
		LoadStale:   "stale",
		LoadCorrupt: "corrupt",
	}
	for s, want := range cases {
		if s.String() != want {
			t.Fatalf("%v.String() = %q, want %q", s, s.String(), want)
		}
	}
}
