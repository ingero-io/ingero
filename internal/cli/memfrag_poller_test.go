package cli

import (
	"context"
	"log/slog"
	"testing"
	"time"

	"github.com/ingero-io/ingero/internal/nvml"
)

func TestMemFragDrainBeforeFirstScan(t *testing.T) {
	resetMemFragState()
	defer resetMemFragState()
	gpus, procs := drainMemFragBuf()
	if gpus != nil || procs != nil {
		t.Errorf("drain before scan = (%v, %v), want (nil, nil)", gpus, procs)
	}
}

func TestPollMemFragOnceFakeRunner(t *testing.T) {
	resetMemFragState()
	defer resetMemFragState()

	memRun := func(context.Context) ([]byte, error) {
		return []byte("GPU-aaaa, 4096, 12288, 16384\nGPU-bbbb, 8192, 8192, 16384\n"), nil
	}
	appsRun := func(context.Context) ([]byte, error) {
		return []byte("GPU-aaaa, 12345, 1024\nGPU-bbbb, 12346, 2048\n"), nil
	}
	pollMemFragOnce(context.Background(), memRun, appsRun, slog.Default())

	gpus, procs := drainMemFragBuf()
	if len(gpus) != 2 {
		t.Fatalf("len(gpus) = %d, want 2", len(gpus))
	}
	if len(procs) != 2 {
		t.Fatalf("len(procs) = %d, want 2", len(procs))
	}
	// Both GPUs are fully accounted -> fragmentation 0.
	for _, g := range gpus {
		if g.FragmentationEstimate != 0 {
			t.Errorf("GPU %s frag = %v, want 0 (fully accounted)", g.UUID, g.FragmentationEstimate)
		}
		if g.TotalBytes != 16384*1024*1024 {
			t.Errorf("GPU %s total = %d", g.UUID, g.TotalBytes)
		}
	}
}

func TestPollMemFragSecondTickReplacesProcMap(t *testing.T) {
	resetMemFragState()
	defer resetMemFragState()

	mem := func(context.Context) ([]byte, error) {
		return []byte("GPU-aaaa, 4096, 12288, 16384\n"), nil
	}
	apps1 := func(context.Context) ([]byte, error) {
		return []byte("GPU-aaaa, 12345, 1024\nGPU-aaaa, 12346, 2048\n"), nil
	}
	apps2 := func(context.Context) ([]byte, error) {
		// only PID 12347 now (12345/12346 exited)
		return []byte("GPU-aaaa, 12347, 4096\n"), nil
	}
	pollMemFragOnce(context.Background(), mem, apps1, slog.Default())
	_, procs := drainMemFragBuf()
	if len(procs) != 2 {
		t.Fatalf("after tick 1: len(procs) = %d, want 2", len(procs))
	}
	pollMemFragOnce(context.Background(), mem, apps2, slog.Default())
	_, procs = drainMemFragBuf()
	if len(procs) != 1 {
		t.Fatalf("after tick 2: len(procs) = %d, want 1 (replace semantics)", len(procs))
	}
	if procs[0].PID != 12347 {
		t.Errorf("expected PID 12347, got %+v", procs[0])
	}
}

func TestStartMemFragPollerNilRunner(t *testing.T) {
	resetMemFragState()
	defer resetMemFragState()
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	// nil memRun -> poller is no-op; no goroutine spawned. Sanity:
	// drain still returns nil since no scan happened.
	startMemFragPoller(ctx, 10*time.Millisecond, nil, nil, slog.Default())
	gpus, procs := drainMemFragBuf()
	if gpus != nil || procs != nil {
		t.Errorf("nil runner should leave buffers untouched, got (%v, %v)", gpus, procs)
	}
}

// Ensure the sentinel check uses interval, not just the runner.
func TestStartMemFragPollerZeroInterval(t *testing.T) {
	resetMemFragState()
	defer resetMemFragState()
	memRun := func(context.Context) ([]byte, error) {
		t.Fatal("runner must not be invoked when interval=0")
		return nil, nil
	}
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	startMemFragPoller(ctx, 0, nvml.Runner(memRun), nil, slog.Default())
	time.Sleep(20 * time.Millisecond)
}
