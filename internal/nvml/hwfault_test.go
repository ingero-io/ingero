package nvml

import "testing"

// Thermal=false short-circuits emission and resets the run; nothing to
// observe until the bucket flips true.
func TestThermalSustainTracker_NoEmitWhenThermalFalse(t *testing.T) {
	tr := NewThermalSustainTracker(2)
	for range 5 {
		got := tr.Observe("GPU-A", ThrottleBuckets{Power: true}, 0, 0)
		if got.Kind != "" {
			t.Fatalf("unexpected emission on Thermal=false: %+v", got)
		}
	}
}

// One Thermal=true poll is not enough at sustainPolls=2 — the operator
// floor is two consecutive observations.
func TestThermalSustainTracker_NoEmitBelowSustain(t *testing.T) {
	tr := NewThermalSustainTracker(2)
	got := tr.Observe("GPU-A", ThrottleBuckets{Thermal: true}, 0x80, 3)
	if got.Kind != "" {
		t.Fatalf("emission below sustain: %+v", got)
	}
}

// Crossing the sustain threshold emits exactly once, then re-emits are
// suppressed while the run continues. The emission carries the GPU id,
// the throttle bitmask, and a critical severity.
func TestThermalSustainTracker_EmitsAtSustainThenSuppresses(t *testing.T) {
	tr := NewThermalSustainTracker(2)
	tr.Observe("GPU-A", ThrottleBuckets{Thermal: true}, 0x80, 7) // consecutive=1

	got := tr.Observe("GPU-A", ThrottleBuckets{Thermal: true}, 0x80, 7) // consecutive=2 -> EMIT
	if got.Kind != FaultKindThermalThrottle {
		t.Fatalf("Kind=%q want %q", got.Kind, FaultKindThermalThrottle)
	}
	if got.Severity != HardwareFaultCritical {
		t.Fatalf("Severity=%q want %q", got.Severity, HardwareFaultCritical)
	}
	if got.GPUID != 7 {
		t.Fatalf("GPUID=%d want 7", got.GPUID)
	}
	if got.ThrottleReasons != 0x80 {
		t.Fatalf("ThrottleReasons=0x%x want 0x80", got.ThrottleReasons)
	}
	if got.Timestamp.IsZero() {
		t.Fatal("Timestamp not stamped")
	}

	// Further Thermal=true polls must not re-emit.
	for range 5 {
		again := tr.Observe("GPU-A", ThrottleBuckets{Thermal: true}, 0x80, 7)
		if again.Kind != "" {
			t.Fatalf("re-emit after suppression: %+v", again)
		}
	}
}

// Clearing Thermal=false resets state so the next sustained run emits
// again. Without this, a flapping workload would emit only once across
// a recurring problem.
func TestThermalSustainTracker_ReEmitsAfterClearAndResustain(t *testing.T) {
	tr := NewThermalSustainTracker(2)
	tr.Observe("GPU-A", ThrottleBuckets{Thermal: true}, 0, 0)
	first := tr.Observe("GPU-A", ThrottleBuckets{Thermal: true}, 0, 0)
	if first.Kind == "" {
		t.Fatal("first emit did not fire")
	}

	tr.Observe("GPU-A", ThrottleBuckets{}, 0, 0) // clear

	tr.Observe("GPU-A", ThrottleBuckets{Thermal: true}, 0, 0)
	second := tr.Observe("GPU-A", ThrottleBuckets{Thermal: true}, 0, 0)
	if second.Kind == "" {
		t.Fatal("post-clear re-sustain did not emit")
	}
}

// Per-GPU state is independent: a thermal run on GPU-A must not gate
// emission for GPU-B.
func TestThermalSustainTracker_PerGPUIndependence(t *testing.T) {
	tr := NewThermalSustainTracker(2)
	tr.Observe("GPU-A", ThrottleBuckets{Thermal: true}, 0, 0)
	got := tr.Observe("GPU-B", ThrottleBuckets{Thermal: true}, 0, 0)
	if got.Kind != "" {
		t.Fatalf("GPU-B emitted while only seen once: %+v", got)
	}
	tr.Observe("GPU-A", ThrottleBuckets{Thermal: true}, 0, 0) // GPU-A emits

	gotB := tr.Observe("GPU-B", ThrottleBuckets{Thermal: true}, 0, 0)
	if gotB.Kind == "" {
		t.Fatal("GPU-B did not emit at its own sustain")
	}
}

// sustainPolls<1 is clamped to 1 (test-only "any observation emits"
// mode). NewThermalSustainTracker(0) must not divide by zero or panic.
func TestThermalSustainTracker_SustainPollsClampedAtLeastOne(t *testing.T) {
	tr := NewThermalSustainTracker(0)
	got := tr.Observe("GPU-A", ThrottleBuckets{Thermal: true}, 0, 0)
	if got.Kind == "" {
		t.Fatal("sustainPolls=0 should clamp to 1 and emit on first observation")
	}
}
