package ncclprobe

import (
	"context"
	"errors"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestVersionFromBasename(t *testing.T) {
	cases := []struct {
		in   string
		want string
	}{
		{"libnccl.so", ""},
		{"libnccl.so.2", "2"},
		{"libnccl.so.2.21", "2.21"},
		{"libnccl.so.2.21.5", "2.21.5"},
		{"/usr/lib/x86_64-linux-gnu/libnccl.so.2.18.3", "2.18.3"},
		{"libnccl_static.so.2.20.5", "2.20.5"},
		{"libnccl.so.2.21.5.0", "2.21.5.0"},
		{"libtorch_cuda.so", ""}, // pytorch shim - no NCCL version here
		{"random.so.5", ""},
	}
	for _, c := range cases {
		t.Run(c.in, func(t *testing.T) {
			if got := versionFromBasename(c.in); got != c.want {
				t.Errorf("versionFromBasename(%q) = %q, want %q", c.in, got, c.want)
			}
		})
	}
}

// TestLibNCCLVersionUnknownPath confirms the resolver returns "unknown"
// (not a panic) on a non-existent path. The metric label must always
// be a non-empty string so the OTLP exporter does not need to special
// -case the absence.
func TestLibNCCLVersionUnknownPath(t *testing.T) {
	if got := libNCCLVersion("/no/such/file/libnccl.so"); got != "unknown" {
		// Special case: if the basename happens to encode a version
		// (it won't here), the function legitimately returns it.
		t.Errorf("libNCCLVersion(missing path) = %q, want unknown", got)
	}
}

// TestScannerEmitsBatch wires a fake PIDLister + findLibForPID +
// versionFor and asserts the Sink receives one record per matched PID.
func TestScannerEmitsBatch(t *testing.T) {
	pidLister := func() ([]uint32, error) {
		return []uint32{100, 200, 300}, nil
	}
	findLib := func(pid int) string {
		switch pid {
		case 100:
			return "/usr/lib/x86_64-linux-gnu/libnccl.so.2.21.5"
		case 200:
			return "/opt/conda/lib/libnccl.so.2.18.3"
		case 300:
			return "" // not loaded
		}
		return ""
	}
	versionFor := func(path string) string {
		switch path {
		case "/usr/lib/x86_64-linux-gnu/libnccl.so.2.21.5":
			return "2.21.5"
		case "/opt/conda/lib/libnccl.so.2.18.3":
			return "2.18.3"
		}
		return "unknown"
	}

	var got []NCCLProcess
	var mu sync.Mutex
	sink := func(p []NCCLProcess) {
		mu.Lock()
		got = append([]NCCLProcess(nil), p...)
		mu.Unlock()
	}

	s := NewScanner(pidLister, sink, time.Hour) // we won't tick; call scanOnce directly
	s.findLibForPID = findLib
	s.versionFor = versionFor
	s.scanOnce(context.Background())

	mu.Lock()
	defer mu.Unlock()
	if len(got) != 2 {
		t.Fatalf("expected 2 records, got %d (%+v)", len(got), got)
	}
	if got[0].PID != 100 || got[0].LibVersion != "2.21.5" {
		t.Errorf("rec[0] = %+v", got[0])
	}
	if got[1].PID != 200 || got[1].LibVersion != "2.18.3" {
		t.Errorf("rec[1] = %+v", got[1])
	}
}

// TestScannerListerError surfaces lister errors and increments the
// error counter without forwarding to the sink.
func TestScannerListerError(t *testing.T) {
	listerErr := errors.New("readdir /proc: ENOENT")
	pidLister := func() ([]uint32, error) {
		return nil, listerErr
	}
	var sinkCalled atomic.Bool
	sink := func([]NCCLProcess) {
		sinkCalled.Store(true)
	}
	s := NewScanner(pidLister, sink, time.Hour)
	s.scanOnce(context.Background())
	_, err := s.LastResult()
	if !errors.Is(err, listerErr) {
		t.Fatalf("LastResult err = %v, want %v", err, listerErr)
	}
	scans, errs := s.Stats()
	if scans != 0 || errs != 1 {
		t.Errorf("Stats = (%d,%d), want (0,1)", scans, errs)
	}
	// Sink IS still called on a lister error path? Per the impl, sink
	// is only called on success. Verify that.
	if sinkCalled.Load() {
		t.Errorf("sink should not be called when lister fails")
	}
}

// TestScannerCtxCancel exits Run promptly when context is cancelled.
func TestScannerCtxCancel(t *testing.T) {
	pidLister := func() ([]uint32, error) {
		return []uint32{}, nil
	}
	s := NewScanner(pidLister, nil, 10*time.Millisecond)
	s.findLibForPID = func(int) string { return "" }
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		s.Run(ctx)
		close(done)
	}()
	cancel()
	select {
	case <-done:
	case <-time.After(2 * time.Second):
		t.Fatal("Run did not exit on cancel")
	}
}

// TestScannerNoIntervalNoTicker confirms zero/negative interval makes
// Run a no-op (the agent uses 0 to disable the feature).
func TestScannerNoIntervalNoTicker(t *testing.T) {
	s := NewScanner(func() ([]uint32, error) { return nil, nil }, nil, 0)
	done := make(chan struct{})
	go func() {
		s.Run(context.Background())
		close(done)
	}()
	select {
	case <-done:
	case <-time.After(500 * time.Millisecond):
		t.Fatal("Run with interval=0 should return immediately")
	}
}

// TestProcPIDListerLocalhost is a smoke test on /proc - every Unix
// host has /proc, and our own PID will always be in it. Skips on
// non-Linux test runners.
func TestProcPIDListerLocalhost(t *testing.T) {
	pids, err := ProcPIDLister()()
	if err != nil {
		t.Skipf("/proc unavailable: %v", err)
	}
	if len(pids) == 0 {
		t.Fatal("ProcPIDLister returned no PIDs")
	}
}
