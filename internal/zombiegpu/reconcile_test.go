package zombiegpu

import (
	"context"
	"errors"
	"testing"

	"github.com/ingero-io/ingero/internal/nvml"
)

// fakeRunner satisfies nvml.Runner for tests; the actual bytes are
// ignored because the test reconciler overrides GetComputeApps.
func fakeRunner(_ context.Context) ([]byte, error) { return nil, nil }

func newTestReconciler(rows []nvml.ComputeAppReading, alive map[uint32]bool) *Reconciler {
	r := New()
	r.SetGetAppsForTest(func(_ context.Context, _ nvml.Runner) ([]nvml.ComputeAppReading, error) {
		return rows, nil
	})
	r.SetPidLivenessProbe(func(pid uint32) bool {
		v, ok := alive[pid]
		if !ok {
			return false
		}
		return v
	})
	return r
}

func TestTick_LivePidNotEmitted(t *testing.T) {
	rows := []nvml.ComputeAppReading{
		{UUID: "GPU-A", PID: 1000, UsedBytes: 1 << 30},
	}
	r := newTestReconciler(rows, map[uint32]bool{1000: true})
	got, err := r.Tick(context.Background(), fakeRunner)
	if err != nil {
		t.Fatalf("Tick: %v", err)
	}
	if len(got) != 0 {
		t.Fatalf("live PID must not emit; got %+v", got)
	}
}

func TestTick_ZombiePidEmits(t *testing.T) {
	rows := []nvml.ComputeAppReading{
		{UUID: "GPU-A", PID: 1000, UsedBytes: 1 << 30}, // 1 GiB orphan
	}
	r := newTestReconciler(rows, map[uint32]bool{1000: false})
	got, err := r.Tick(context.Background(), fakeRunner)
	if err != nil {
		t.Fatalf("Tick: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("zombie PID must emit one allocation; got %+v", got)
	}
	if got[0].PID != 1000 {
		t.Errorf("unexpected pid: %v", got[0].PID)
	}
	if got[0].GPUUUID != "GPU-A" {
		t.Errorf("unexpected uuid: %v", got[0].GPUUUID)
	}
	if got[0].AllocatedBytes != 1<<30 {
		t.Errorf("unexpected bytes: %v", got[0].AllocatedBytes)
	}
}

func TestTick_ZombieEmittedOnlyOncePerEpisode(t *testing.T) {
	rows := []nvml.ComputeAppReading{
		{UUID: "GPU-A", PID: 1000, UsedBytes: 1 << 30},
	}
	r := newTestReconciler(rows, map[uint32]bool{1000: false})
	if got, _ := r.Tick(context.Background(), fakeRunner); len(got) != 1 {
		t.Fatalf("first tick must emit: %+v", got)
	}
	// Same zombie still in the readings; must NOT re-emit until cleared.
	if got, _ := r.Tick(context.Background(), fakeRunner); len(got) != 0 {
		t.Fatalf("re-emit must be suppressed while PID still in readings; got %+v", got)
	}
	if r.EmittedCount() != 1 {
		t.Fatalf("emitted set should track the in-flight episode; got %d", r.EmittedCount())
	}
}

func TestTick_ZombieRearmsAfterDriverDropsRow(t *testing.T) {
	// Episode 1: zombie appears, emits.
	rows := []nvml.ComputeAppReading{
		{UUID: "GPU-A", PID: 1000, UsedBytes: 1 << 30},
	}
	r := newTestReconciler(rows, map[uint32]bool{1000: false})
	if got, _ := r.Tick(context.Background(), fakeRunner); len(got) != 1 {
		t.Fatalf("episode 1 must emit: %+v", got)
	}
	// pod_drain reclaimed the VRAM; driver no longer reports the PID.
	r.SetGetAppsForTest(func(_ context.Context, _ nvml.Runner) ([]nvml.ComputeAppReading, error) {
		return []nvml.ComputeAppReading{}, nil
	})
	if got, _ := r.Tick(context.Background(), fakeRunner); len(got) != 0 {
		t.Fatalf("empty reading must yield no emissions: %+v", got)
	}
	if r.EmittedCount() != 0 {
		t.Fatalf("emitted set must clear after driver drops the row; got %d", r.EmittedCount())
	}
	// Episode 2: same PID re-appears as a zombie (different incident).
	r.SetGetAppsForTest(func(_ context.Context, _ nvml.Runner) ([]nvml.ComputeAppReading, error) {
		return rows, nil
	})
	if got, _ := r.Tick(context.Background(), fakeRunner); len(got) != 1 {
		t.Fatalf("episode 2 must re-emit after rearm: %+v", got)
	}
}

func TestTick_MixedLiveAndZombiePids(t *testing.T) {
	rows := []nvml.ComputeAppReading{
		{UUID: "GPU-A", PID: 1000, UsedBytes: 100},
		{UUID: "GPU-A", PID: 2000, UsedBytes: 200},
		{UUID: "GPU-B", PID: 3000, UsedBytes: 300},
	}
	alive := map[uint32]bool{1000: true, 2000: false, 3000: false}
	r := newTestReconciler(rows, alive)
	got, err := r.Tick(context.Background(), fakeRunner)
	if err != nil {
		t.Fatalf("Tick: %v", err)
	}
	if len(got) != 2 {
		t.Fatalf("expected 2 zombies (2000, 3000); got %+v", got)
	}
	seen := map[uint32]bool{}
	for _, a := range got {
		seen[a.PID] = true
	}
	if !seen[2000] || !seen[3000] {
		t.Fatalf("expected zombies for pids 2000 and 3000; got %+v", got)
	}
	if seen[1000] {
		t.Fatalf("live pid 1000 must not be in zombie list")
	}
}

func TestTick_NilRunnerDegradesSilently(t *testing.T) {
	r := New()
	got, err := r.Tick(context.Background(), nil)
	if err != nil {
		t.Fatalf("nil runner must not error: %v", err)
	}
	if got != nil {
		t.Fatalf("nil runner must yield nil emissions: %+v", got)
	}
}

func TestTick_RunnerErrorBubblesUp(t *testing.T) {
	r := New()
	r.SetGetAppsForTest(func(_ context.Context, _ nvml.Runner) ([]nvml.ComputeAppReading, error) {
		return nil, errors.New("nvidia-smi exec failed")
	})
	if _, err := r.Tick(context.Background(), fakeRunner); err == nil {
		t.Fatalf("expected runner error to bubble up")
	}
}

func TestTick_DropsPid0Rows(t *testing.T) {
	// PID 0 should never appear, but defend against malformed nvidia-smi output.
	rows := []nvml.ComputeAppReading{
		{UUID: "GPU-A", PID: 0, UsedBytes: 100},
	}
	r := newTestReconciler(rows, map[uint32]bool{})
	got, _ := r.Tick(context.Background(), fakeRunner)
	if len(got) != 0 {
		t.Fatalf("pid 0 row must be filtered: %+v", got)
	}
}
