package health

import (
	"fmt"
	"io"
	"log/slog"
	"os"
	"path/filepath"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// Adversarial: symlink attack on baseline file. If daemon runs as root and
// attacker has write access to the baseline directory, they can replace
// the file with a symlink. Tests that Load() handles this safely.
func TestAdv_SymlinkToArbitraryFile(t *testing.T) {
	dir := t.TempDir()
	baselinePath := filepath.Join(dir, "baseline.json")
	targetPath := filepath.Join(dir, "target.txt")

	if err := os.WriteFile(targetPath, []byte("this is an attacker secret"), 0644); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(targetPath, baselinePath); err != nil {
		t.Skip("symlink not supported:", err)
	}

	_, status, err := Load(baselinePath, DefaultStaleAge, time.Now(), slog.New(slog.NewTextHandler(io.Discard, nil)))
	t.Logf("Load status=%v err=%v", status, err)

	if _, err := os.Lstat(baselinePath); err != nil {
		t.Logf("baselinePath after: %v", err)
	}
	if _, err := os.Lstat(targetPath); err != nil {
		t.Logf("target after: %v (target deleted? would be BAD)", err)
	} else {
		t.Log("target file survived, good")
	}

	entries, _ := os.ReadDir(dir)
	for _, e := range entries {
		t.Logf("  remaining: %s", e.Name())
	}
}

// Adversarial: symlink to sensitive file. os.Rename on a symlink renames
// the link, not the target. Verify that.
func TestAdv_SymlinkToSensitiveTarget(t *testing.T) {
	dir := t.TempDir()
	baselinePath := filepath.Join(dir, "baseline.json")
	secretPath := filepath.Join(dir, "secret-that-must-survive.txt")
	secretContent := []byte("SECRET_PASSWORD=do-not-leak")
	if err := os.WriteFile(secretPath, secretContent, 0600); err != nil {
		t.Fatal(err)
	}
	if err := os.Symlink(secretPath, baselinePath); err != nil {
		t.Skip("no symlink:", err)
	}
	_, _, _ = Load(baselinePath, DefaultStaleAge, time.Now(), slog.New(slog.NewTextHandler(io.Discard, nil)))

	got, err := os.ReadFile(secretPath)
	if err != nil {
		t.Fatalf("SECRET FILE WAS DELETED OR MOVED: %v", err)
	}
	if string(got) != string(secretContent) {
		t.Fatalf("SECRET FILE WAS MODIFIED: got %q", got)
	}
}

// Adversarial: concurrent Save() contention. Two goroutines save
// different PersistedStates simultaneously. Resulting file must parse.
func TestAdv_ConcurrentSave(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "baseline.json")

	stateA := PersistedState{SchemaVersion: 1, SampleCount: 100, FastAlpha: 0.1, FloorAlpha: 0.001}
	stateA.FastEMA.Throughput = 1000.0
	stateA.HardFloor.Throughput = 900.0

	stateB := PersistedState{SchemaVersion: 1, SampleCount: 200, FastAlpha: 0.1, FloorAlpha: 0.001}
	stateB.FastEMA.Throughput = 2000.0
	stateB.HardFloor.Throughput = 1800.0

	var wg sync.WaitGroup
	errors := make(chan error, 200)
	for i := 0; i < 100; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			if err := Save(path, stateA, time.Now()); err != nil {
				errors <- err
			}
		}()
		go func() {
			defer wg.Done()
			if err := Save(path, stateB, time.Now()); err != nil {
				errors <- err
			}
		}()
	}
	wg.Wait()
	close(errors)
	errCount := 0
	for e := range errors {
		t.Logf("save err: %v", e)
		errCount++
	}
	t.Logf("save errors: %d / 200", errCount)

	loaded, status, err := Load(path, DefaultStaleAge, time.Now(), slog.New(slog.NewTextHandler(io.Discard, nil)))
	t.Logf("Final load status=%v err=%v sampleCount=%d throughput=%v", status, err, loaded.SampleCount, loaded.FastEMA.Throughput)
	if status != LoadFresh {
		t.Fatalf("concurrent Save produced non-parseable result: status=%v err=%v", status, err)
	}
	if loaded.SampleCount != 100 && loaded.SampleCount != 200 {
		t.Fatalf("concurrent Save torn read: SampleCount=%d (expected 100 or 200)", loaded.SampleCount)
	}

	entries, _ := os.ReadDir(dir)
	tmpCount := 0
	for _, e := range entries {
		name := e.Name()
		if len(name) >= 4 && name[len(name)-4:] == ".tmp" {
			tmpCount++
		}
	}
	t.Logf("tmp leftover: %d files; total entries: %d", tmpCount, len(entries))
}

// Adversarial: concurrent Save() and Load(). Ensures no torn reads.
func TestAdv_ConcurrentSaveLoad(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "baseline.json")

	state := PersistedState{SchemaVersion: 1, SampleCount: 100, FastAlpha: 0.1, FloorAlpha: 0.001}
	state.FastEMA.Throughput = 1000.0
	state.HardFloor.Throughput = 900.0

	if err := Save(path, state, time.Now()); err != nil {
		t.Fatal(err)
	}

	var stopFlag atomic.Bool
	var savesDone, loadsDone atomic.Int64
	var tornReads atomic.Int64

	var wg sync.WaitGroup
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for !stopFlag.Load() {
				_ = Save(path, state, time.Now())
				savesDone.Add(1)
			}
		}()
	}
	for i := 0; i < 4; i++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for !stopFlag.Load() {
				s, status, _ := Load(path, DefaultStaleAge, time.Now(), slog.New(slog.NewTextHandler(io.Discard, nil)))
				if status == LoadCorrupt {
					tornReads.Add(1)
				} else if status == LoadFresh && s.SampleCount != 100 {
					t.Errorf("torn read: SampleCount=%d", s.SampleCount)
				}
				loadsDone.Add(1)
			}
		}()
	}
	time.Sleep(2 * time.Second)
	stopFlag.Store(true)
	wg.Wait()
	t.Logf("saves=%d loads=%d torn/corrupt=%d", savesDone.Load(), loadsDone.Load(), tornReads.Load())
	if tornReads.Load() > 0 {
		t.Errorf("FINDING: %d torn/corrupt reads during concurrent Save/Load", tornReads.Load())
	}
}

// Adversarial: quarantine TOCTOU race.
func TestAdv_QuarantineRace(t *testing.T) {
	dir := t.TempDir()
	var quarantineErrors int64

	for trial := 0; trial < 50; trial++ {
		path := filepath.Join(dir, fmt.Sprintf("baseline-%d.json", trial))
		if err := os.WriteFile(path, []byte("{not valid json"), 0644); err != nil {
			t.Fatal(err)
		}
		var wg sync.WaitGroup
		for i := 0; i < 8; i++ {
			wg.Add(1)
			go func() {
				defer wg.Done()
				_, _, err := Load(path, DefaultStaleAge, time.Now(), slog.New(slog.NewTextHandler(io.Discard, nil)))
				if err != nil {
					atomic.AddInt64(&quarantineErrors, 1)
				}
			}()
		}
		wg.Wait()
	}
	t.Logf("quarantine errors across races: %d / 400 attempts", quarantineErrors)

	entries, _ := os.ReadDir(dir)
	corruptCount := 0
	for _, e := range entries {
		n := e.Name()
		for i := 0; i+8 <= len(n); i++ {
			if n[i:i+8] == "corrupt-" {
				corruptCount++
				break
			}
		}
	}
	t.Logf("quarantined files: %d (expected ~50 if no race, one per trial)", corruptCount)
}

// Adversarial: /dev/zero symlink - don't hang.
func TestAdv_DeviceFileAsBaseline(t *testing.T) {
	if _, err := os.Stat("/dev/zero"); err != nil {
		t.Skip("no /dev/zero")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "baseline.json")
	if err := os.Symlink("/dev/zero", path); err != nil {
		t.Skip("no symlink:", err)
	}
	done := make(chan struct{})
	var status LoadStatus
	var loadErr error
	go func() {
		_, status, loadErr = Load(path, DefaultStaleAge, time.Now(), slog.New(slog.NewTextHandler(io.Discard, nil)))
		close(done)
	}()
	select {
	case <-done:
		t.Logf("Load on /dev/zero symlink returned status=%v err=%v", status, loadErr)
	case <-time.After(30 * time.Second):
		t.Fatal("FINDING: Load on /dev/zero symlink HUNG for 30 seconds - likely read-unbounded")
	}
}

// Adversarial: permission-denied file.
func TestAdv_PermissionDenied(t *testing.T) {
	if os.Getuid() == 0 {
		t.Skip("running as root: chmod 0 does not deny root")
	}
	dir := t.TempDir()
	path := filepath.Join(dir, "baseline.json")
	if err := os.WriteFile(path, []byte("{}"), 0000); err != nil {
		t.Fatal(err)
	}
	defer os.Chmod(path, 0644)
	_, status, err := Load(path, DefaultStaleAge, time.Now(), slog.New(slog.NewTextHandler(io.Discard, nil)))
	t.Logf("chmod-000 status=%v err=%v", status, err)
	if status == LoadCorrupt {
		t.Error("FINDING: permission-denied file was QUARANTINED (destructive) instead of returning LoadUnreadable")
	}
	if _, err := os.Stat(path); err != nil {
		t.Error("FINDING: permission-denied file was MOVED or DELETED during Load")
	}
}

// Adversarial: quarantine-to-escape. Load is called with a path-traversal
// path. If Load follows the path into a parent dir, it may quarantine
// or remove files outside the baseline directory.
func TestAdv_PathTraversal(t *testing.T) {
	dir := t.TempDir()
	outsideDir := filepath.Dir(dir)
	outside := filepath.Join(outsideDir, fmt.Sprintf("adv-outside-%d", os.Getpid()))
	if err := os.WriteFile(outside, []byte("{not-valid-json"), 0644); err != nil {
		t.Fatal(err)
	}
	defer os.Remove(outside)

	traversal := filepath.Join(dir, "..", filepath.Base(outside))
	_, status, err := Load(traversal, DefaultStaleAge, time.Now(), slog.New(slog.NewTextHandler(io.Discard, nil)))
	t.Logf("traversal path=%s status=%v err=%v", traversal, status, err)
	// Known INFORMATIONAL: operator-controlled path with .. allows Load to
	// quarantine files outside the intended directory. Not attacker-exploitable
	// (config is operator-set) but a foot-gun. Logged, not failed.
	if _, err := os.Stat(outside); err != nil {
		t.Logf("INFO: Load with path-traversal quarantined file outside configured dir: %v", err)
	}
}
