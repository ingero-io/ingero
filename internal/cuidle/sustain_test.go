package cuidle

import (
	"testing"
	"time"
)

var baseTime = time.Date(2026, 5, 17, 12, 0, 0, 0, time.UTC)

// newTestTracker uses 2s idle + 5s suppression for fast assertions,
// and an alive-probe that always returns true so tests don't depend
// on real PID state.
func newTestTracker() *Tracker {
	t := NewWithThresholds(2*time.Second, 5*time.Second)
	t.SetPidLivenessProbe(func(uint32) bool { return true })
	return t
}

func TestObserve_FreshLaunchNoHang(t *testing.T) {
	tr := newTestTracker()
	tr.Observe(1234, 0, baseTime)
	// Sweep 1 second later — idle is 1s, below the 2s threshold.
	if got := tr.Sweep(baseTime.Add(1 * time.Second)); len(got) != 0 {
		t.Fatalf("fresh launch must not emit: %+v", got)
	}
}

func TestObserve_IdleAboveThresholdEmits(t *testing.T) {
	tr := newTestTracker()
	tr.Observe(1234, 2, baseTime)
	got := tr.Sweep(baseTime.Add(3 * time.Second))
	if len(got) != 1 {
		t.Fatalf("idle 3s > threshold 2s must emit; got %+v", got)
	}
	if got[0].PID != 1234 {
		t.Errorf("unexpected pid: %v", got[0].PID)
	}
	if got[0].GPUID != 2 {
		t.Errorf("gpu id must round-trip: got %v want 2", got[0].GPUID)
	}
	if got[0].IdleMs < 3000 {
		t.Errorf("idle_ms must reflect elapsed: got %v", got[0].IdleMs)
	}
}

func TestObserve_FreshLaunchResetsEpisode(t *testing.T) {
	tr := newTestTracker()
	tr.Observe(1234, 0, baseTime)
	first := tr.Sweep(baseTime.Add(3 * time.Second))
	if len(first) != 1 {
		t.Fatalf("first emission expected: %+v", first)
	}
	// A new launch arrives — the workload is no longer hung.
	tr.Observe(1234, 0, baseTime.Add(4*time.Second))
	// A subsequent hang past the threshold must emit again, because
	// the fresh launch cleared the emitted flag.
	got := tr.Sweep(baseTime.Add(7 * time.Second))
	if len(got) != 1 {
		t.Fatalf("post-recovery hang must re-emit: %+v", got)
	}
}

func TestSweep_SameEpisodeDoesNotReEmit(t *testing.T) {
	tr := newTestTracker()
	tr.Observe(1234, 0, baseTime)
	first := tr.Sweep(baseTime.Add(3 * time.Second))
	if len(first) != 1 {
		t.Fatalf("first sweep must emit: %+v", first)
	}
	// Same episode, no new launches, sweeping again must NOT re-emit
	// until the suppression window elapses.
	if got := tr.Sweep(baseTime.Add(4 * time.Second)); len(got) != 0 {
		t.Fatalf("same episode re-emit must be suppressed; got %+v", got)
	}
	if got := tr.Sweep(baseTime.Add(7 * time.Second)); len(got) != 0 {
		t.Fatalf("same episode re-emit must be suppressed; got %+v", got)
	}
}

func TestSweep_RearmsAfterSuppressionWindow(t *testing.T) {
	tr := newTestTracker()
	tr.Observe(1234, 0, baseTime)
	first := tr.Sweep(baseTime.Add(3 * time.Second))
	if len(first) != 1 {
		t.Fatalf("first sweep must emit: %+v", first)
	}
	// After the suppression window (>= 5s past emittedAt),
	// continuing to be hung allows a re-emit.
	rearm := tr.Sweep(baseTime.Add(9 * time.Second))
	if len(rearm) != 0 {
		t.Fatalf("first post-suppression sweep clears the flag but should not emit; got %+v", rearm)
	}
	// Now flag is cleared; next sweep finds idle still above threshold and emits.
	got := tr.Sweep(baseTime.Add(10 * time.Second))
	if len(got) != 1 {
		t.Fatalf("post-suppression sweep must emit a fresh hang: %+v", got)
	}
}

func TestSweep_DeadPidDroppedNotEmitted(t *testing.T) {
	tr := newTestTracker()
	// Liveness probe says PID is gone.
	tr.SetPidLivenessProbe(func(uint32) bool { return false })
	tr.Observe(1234, 0, baseTime)
	got := tr.Sweep(baseTime.Add(3 * time.Second))
	if len(got) != 0 {
		t.Fatalf("dead PID must not emit hang: %+v", got)
	}
	if tr.TrackedPIDs() != 0 {
		t.Fatalf("dead PID must be dropped from tracker; got %d", tr.TrackedPIDs())
	}
}

func TestObserve_IgnoresPid0(t *testing.T) {
	tr := newTestTracker()
	tr.Observe(0, 0, baseTime)
	if tr.TrackedPIDs() != 0 {
		t.Fatalf("pid 0 must be ignored, got %d", tr.TrackedPIDs())
	}
}

func TestForget_DropsPidState(t *testing.T) {
	tr := newTestTracker()
	tr.Observe(99, 0, baseTime)
	tr.Observe(100, 0, baseTime)
	if tr.TrackedPIDs() != 2 {
		t.Fatalf("expected 2 tracked, got %d", tr.TrackedPIDs())
	}
	tr.Forget(99)
	if tr.TrackedPIDs() != 1 {
		t.Fatalf("after Forget expected 1 tracked, got %d", tr.TrackedPIDs())
	}
}
