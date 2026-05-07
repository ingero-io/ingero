//go:build integration_bpf

// Package ncclprobe — integration tests that require root + CAP_BPF
// and a real libnccl-bearing ELF on disk. Built only with the
// `integration_bpf` tag so unit-test CI does not try to load BPF.
//
// Run with:
//
//	sudo go test -tags integration_bpf -run TestIntegration ./internal/ebpf/ncclprobe/...
//
// Or via the lambda-e2e-harness CI workflow (which provisions a
// privileged GPU VM).
package ncclprobe

import (
	"os"
	"path/filepath"
	"runtime"
	"sort"
	"sync"
	"testing"
)

func canRunBPF(t *testing.T) {
	if os.Geteuid() != 0 {
		t.Skipf("integration_bpf tests require root (euid=%d)", os.Geteuid())
	}
	if runtime.GOOS != "linux" {
		t.Skipf("integration_bpf tests are linux-only (GOOS=%s)", runtime.GOOS)
	}
}

func findRealLibNCCL(t *testing.T) string {
	candidates := []string{
		"/usr/lib/x86_64-linux-gnu/libnccl.so.2",
		"/usr/lib/aarch64-linux-gnu/libnccl.so.2",
		"/opt/nccl/lib/libnccl.so.2",
		"/usr/local/nccl/lib/libnccl.so.2",
	}
	for _, p := range candidates {
		if _, err := os.Stat(p); err == nil {
			return p
		}
	}
	t.Skipf("no systemwide libnccl.so.2 in expected locations; install or skip")
	return ""
}

// v0.15 F1 (gap #1): AttachAt against a real libnccl ELF wires
// uprobes successfully + appears in AttachedPaths.
func TestIntegration_AttachAt_Success(t *testing.T) {
	canRunBPF(t)
	libPath := findRealLibNCCL(t)
	tr := New("")
	if err := tr.Prepare(nil); err != nil {
		t.Fatalf("Prepare: %v", err)
	}
	defer tr.Close()
	if err := tr.AttachAt(libPath); err != nil {
		t.Fatalf("AttachAt(%q): %v", libPath, err)
	}
	paths := tr.AttachedPaths()
	if len(paths) != 1 || paths[0] != libPath {
		t.Errorf("AttachedPaths=%v, want [%q]", paths, libPath)
	}
	if tr.AttachedProbeCount() == 0 {
		t.Errorf("AttachedProbeCount=0, want >0")
	}
}

// v0.15 F1 (gap #1): AttachAt with the same path twice is a no-op
// on the second call (idempotency at the BPF level, not just the
// in-memory map check).
func TestIntegration_AttachAt_Idempotent(t *testing.T) {
	canRunBPF(t)
	libPath := findRealLibNCCL(t)
	tr := New("")
	if err := tr.Prepare(nil); err != nil {
		t.Fatalf("Prepare: %v", err)
	}
	defer tr.Close()
	if err := tr.AttachAt(libPath); err != nil {
		t.Fatalf("AttachAt 1st: %v", err)
	}
	syms1 := tr.AttachedProbeCount()
	if err := tr.AttachAt(libPath); err != nil {
		t.Fatalf("AttachAt 2nd (idempotent): %v", err)
	}
	syms2 := tr.AttachedProbeCount()
	if syms1 != syms2 {
		t.Errorf("idempotent AttachAt grew probe count: %d -> %d", syms1, syms2)
	}
	if got := len(tr.AttachedPaths()); got != 1 {
		t.Errorf("AttachedPaths len=%d, want 1", got)
	}
}

// v0.15 F1 (gap #1): AttachAt against two different real ELFs is
// additive: probe count grows, AttachedPaths grows.
func TestIntegration_AttachAt_TwoPaths(t *testing.T) {
	canRunBPF(t)
	libPath := findRealLibNCCL(t)
	// Second "ELF" is a copy in /tmp masqueraded under /opt/python
	// to bypass the user-writable denylist. Real-world this would be
	// a venv-installed libnccl. Test only.
	dir := t.TempDir()
	other := filepath.Join(dir, "libnccl.so.2")
	src, err := os.ReadFile(libPath)
	if err != nil {
		t.Fatalf("read source libnccl: %v", err)
	}
	if err := os.WriteFile(other, src, 0o644); err != nil {
		t.Fatalf("write copy: %v", err)
	}
	// isSafeLibPath denies /tmp; this test deliberately bypasses
	// that gate by calling AttachAt with the path directly. (Real
	// runtime would route through FindLibNCCL's allowlist.)
	tr := New("")
	if err := tr.Prepare(nil); err != nil {
		t.Fatalf("Prepare: %v", err)
	}
	defer tr.Close()
	if err := tr.AttachAt(libPath); err != nil {
		t.Fatalf("AttachAt 1st: %v", err)
	}
	if err := tr.AttachAt(other); err != nil {
		t.Fatalf("AttachAt 2nd: %v", err)
	}
	got := tr.AttachedPaths()
	sort.Strings(got)
	if len(got) != 2 {
		t.Errorf("AttachedPaths=%v, want 2 paths", got)
	}
}

// v0.15 F1 (gap #1): true concurrent AttachAt calls (not the
// helper-driven variant). Each goroutine attaches a different ELF
// path. Race detector + correct count proves the BPF-write path is
// thread-safe.
func TestIntegration_AttachAt_TrueConcurrentWrites(t *testing.T) {
	canRunBPF(t)
	libPath := findRealLibNCCL(t)
	src, err := os.ReadFile(libPath)
	if err != nil {
		t.Fatalf("read libnccl: %v", err)
	}
	const N = 8
	dir := t.TempDir()
	paths := make([]string, N)
	for i := 0; i < N; i++ {
		paths[i] = filepath.Join(dir, "libnccl_"+itoa(i)+".so.2")
		if err := os.WriteFile(paths[i], src, 0o644); err != nil {
			t.Fatalf("write copy %d: %v", i, err)
		}
	}
	tr := New("")
	if err := tr.Prepare(nil); err != nil {
		t.Fatalf("Prepare: %v", err)
	}
	defer tr.Close()

	var wg sync.WaitGroup
	errs := make(chan error, N)
	for i := 0; i < N; i++ {
		wg.Add(1)
		go func(p string) {
			defer wg.Done()
			if err := tr.AttachAt(p); err != nil {
				errs <- err
			}
		}(paths[i])
	}
	wg.Wait()
	close(errs)
	for err := range errs {
		t.Errorf("concurrent AttachAt err: %v", err)
	}
	if got := len(tr.AttachedPaths()); got != N {
		t.Errorf("AttachedPaths len=%d, want %d", got, N)
	}
}
