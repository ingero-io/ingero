package ncclhang

import (
	"testing"
	"time"
)

var baseTime = time.Date(2026, 5, 17, 12, 0, 0, 0, time.UTC)

// newTestTracker uses 2s idle / 2s min-observation / 5s suppression
// for deterministic short assertions; alive-probe always returns true.
func newTestTracker() *Tracker {
	t := NewWithThresholds(2*time.Second, 2*time.Second, 5*time.Second)
	t.SetPidLivenessProbe(func(uint32) bool { return true })
	return t
}

func TestObserve_FreshEventNoHang(t *testing.T) {
	tr := newTestTracker()
	tr.Observe(1234, 0xdead, baseTime)
	if got := tr.Sweep(baseTime.Add(1 * time.Second)); len(got) != 0 {
		t.Fatalf("fresh event must not emit; got %+v", got)
	}
}

func TestObserve_IdleAboveThresholdAndObservedEmits(t *testing.T) {
	tr := newTestTracker()
	tr.Observe(1234, 0xdead, baseTime)
	// Sweep at +3s: idle 3s > 2s threshold; observed 3s >= 2s min-observation.
	got := tr.Sweep(baseTime.Add(3 * time.Second))
	if len(got) != 1 {
		t.Fatalf("idle-and-observed PID must emit hang: %+v", got)
	}
	if got[0].PID != 1234 || got[0].CommIDHash != 0xdead {
		t.Errorf("payload incorrect: %+v", got[0])
	}
	if got[0].IdleMs < 3000 {
		t.Errorf("idle_ms must reflect elapsed: %v", got[0].IdleMs)
	}
}

func TestObserve_IdleButNotObservedLongEnoughDoesNotEmit(t *testing.T) {
	tr := newTestTracker()
	// First observation at +0; sweep at +0.5s. Idle is 0.5s (below
	// threshold); observed is 0.5s (below min-observation). The
	// classic "we just started watching" guard.
	tr.Observe(1234, 0xdead, baseTime)
	if got := tr.Sweep(baseTime.Add(500 * time.Millisecond)); len(got) != 0 {
		t.Fatalf("fresh PID must not emit before min-observation: %+v", got)
	}
}

func TestObserve_FreshEventResetsEpisode(t *testing.T) {
	tr := newTestTracker()
	tr.Observe(1234, 0xdead, baseTime)
	first := tr.Sweep(baseTime.Add(3 * time.Second))
	if len(first) != 1 {
		t.Fatalf("first emission expected: %+v", first)
	}
	tr.Observe(1234, 0xdead, baseTime.Add(4*time.Second))
	got := tr.Sweep(baseTime.Add(7 * time.Second))
	if len(got) != 1 {
		t.Fatalf("post-recovery re-hang must emit: %+v", got)
	}
}

func TestSweep_SameEpisodeDoesNotReEmit(t *testing.T) {
	tr := newTestTracker()
	tr.Observe(1234, 0xdead, baseTime)
	first := tr.Sweep(baseTime.Add(3 * time.Second))
	if len(first) != 1 {
		t.Fatalf("first sweep must emit: %+v", first)
	}
	if got := tr.Sweep(baseTime.Add(4 * time.Second)); len(got) != 0 {
		t.Fatalf("same episode re-emit must be suppressed: %+v", got)
	}
}

func TestSweep_RearmsAfterSuppressionWindow(t *testing.T) {
	tr := newTestTracker()
	tr.Observe(1234, 0xdead, baseTime)
	first := tr.Sweep(baseTime.Add(3 * time.Second))
	if len(first) != 1 {
		t.Fatalf("first sweep must emit: %+v", first)
	}
	// First post-suppression sweep clears the emitted flag.
	if got := tr.Sweep(baseTime.Add(9 * time.Second)); len(got) != 0 {
		t.Fatalf("post-suppression sweep clears flag but should not emit: %+v", got)
	}
	// Next sweep emits a fresh hang (still idle).
	got := tr.Sweep(baseTime.Add(10 * time.Second))
	if len(got) != 1 {
		t.Fatalf("post-suppression sweep must emit fresh: %+v", got)
	}
}

func TestSweep_DeadPidDroppedNotEmitted(t *testing.T) {
	tr := newTestTracker()
	tr.SetPidLivenessProbe(func(uint32) bool { return false })
	tr.Observe(1234, 0xdead, baseTime)
	got := tr.Sweep(baseTime.Add(3 * time.Second))
	if len(got) != 0 {
		t.Fatalf("dead PID must not emit: %+v", got)
	}
	if tr.TrackedPIDs() != 0 {
		t.Fatalf("dead PID must be dropped: %d", tr.TrackedPIDs())
	}
}

func TestObserve_IgnoresPid0(t *testing.T) {
	tr := newTestTracker()
	tr.Observe(0, 0xdead, baseTime)
	if tr.TrackedPIDs() != 0 {
		t.Fatalf("pid 0 must be ignored: %d", tr.TrackedPIDs())
	}
}

func TestObserve_CommIDHashUpdatedButZeroPreservesPrevious(t *testing.T) {
	tr := newTestTracker()
	tr.Observe(1234, 0xfeed, baseTime)
	// Subsequent observation with commIDHash=0 (e.g. event before
	// ncclCommInitRank fully completed) must not wipe the existing
	// hash — operators rely on it for log context.
	tr.Observe(1234, 0, baseTime.Add(500*time.Millisecond))
	got := tr.Sweep(baseTime.Add(3 * time.Second))
	if len(got) != 1 || got[0].CommIDHash != 0xfeed {
		t.Fatalf("commIDHash should be preserved: %+v", got)
	}
}

func TestForget_DropsPidState(t *testing.T) {
	tr := newTestTracker()
	tr.Observe(99, 0xdead, baseTime)
	tr.Observe(100, 0xbeef, baseTime)
	if tr.TrackedPIDs() != 2 {
		t.Fatalf("expected 2; got %d", tr.TrackedPIDs())
	}
	tr.Forget(99)
	if tr.TrackedPIDs() != 1 {
		t.Fatalf("after Forget expected 1; got %d", tr.TrackedPIDs())
	}
}
