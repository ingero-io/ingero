package health

import (
	"io"
	"log/slog"
	"testing"
	"time"
)

// quietSM returns a state machine whose logger is discarded — transitions
// are expected, not noisy assertions.
func quietSM(t *testing.T, cfg StateConfig) StateMachine {
	t.Helper()
	sm, err := NewStateMachine(cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatalf("NewStateMachine: %v", err)
	}
	return sm
}

func goodObs(count int) Observation {
	return Observation{KernelLaunchCount: count, EventReadOK: true, Timestamp: testTS}
}

func badObs() Observation {
	return Observation{KernelLaunchCount: 0, EventReadOK: false, Timestamp: testTS}
}

func TestDefaultStateConfig_Valid(t *testing.T) {
	if err := DefaultStateConfig().Validate(); err != nil {
		t.Fatalf("default invalid: %v", err)
	}
}

func TestStateConfig_Validate(t *testing.T) {
	tests := []struct {
		name    string
		cfg     StateConfig
		wantErr bool
	}{
		{"defaults", DefaultStateConfig(), false},
		{"idle_intervals_zero", StateConfig{IdleIntervals: 0, WarmupSamples: 30, StaleReadFailures: 3}, true},
		{"idle_intervals_overflow", StateConfig{IdleIntervals: MaxIdleIntervals + 1, WarmupSamples: 30, StaleReadFailures: 3}, true},
		{"warmup_negative", StateConfig{IdleIntervals: 3, WarmupSamples: -1, StaleReadFailures: 3}, true},
		{"stale_zero", StateConfig{IdleIntervals: 3, WarmupSamples: 30, StaleReadFailures: 0}, true},
		{"recent_negative", StateConfig{IdleIntervals: 3, WarmupSamples: 30, StaleReadFailures: 3, RecentWindow: -1}, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.cfg.Validate()
			if tc.wantErr && err == nil {
				t.Fatal("expected error")
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("unexpected: %v", err)
			}
		})
	}
}

// AC2: Startup is CALIBRATING; transitions to ACTIVE after warmup samples.
func TestStartup_CalibratingThenActive(t *testing.T) {
	cfg := StateConfig{IdleIntervals: 3, WarmupSamples: 5, StaleReadFailures: 3}
	sm := quietSM(t, cfg)
	if sm.Current() != StateCalibrating {
		t.Fatalf("initial = %v, want CALIBRATING", sm.Current())
	}
	for i := 0; i < 4; i++ {
		if _, _, _, changed := sm.TransitionIfNeeded(goodObs(10)); changed {
			t.Fatalf("unexpected transition at sample %d", i+1)
		}
	}
	prev, next, reason, changed := sm.TransitionIfNeeded(goodObs(10))
	if !changed || prev != StateCalibrating || next != StateActive {
		t.Fatalf("expected CALIBRATING->ACTIVE, got prev=%v next=%v changed=%v reason=%q",
			prev, next, changed, reason)
	}
}

// AC3: ACTIVE with zero launches for idle_intervals -> IDLE.
func TestActive_ToIdle(t *testing.T) {
	cfg := StateConfig{IdleIntervals: 3, WarmupSamples: 0, StaleReadFailures: 3}
	sm := quietSM(t, cfg)
	// Skip warmup: one tick with warmup=0 still requires a transition cycle,
	// so push one good obs to move to ACTIVE.
	sm.TransitionIfNeeded(goodObs(10))
	if sm.Current() != StateActive {
		t.Fatalf("expected ACTIVE after zero-warmup, got %v", sm.Current())
	}
	// Two zero-launch observations: still ACTIVE.
	sm.TransitionIfNeeded(goodObs(0))
	sm.TransitionIfNeeded(goodObs(0))
	if sm.Current() != StateActive {
		t.Fatalf("too-eager IDLE transition: %v", sm.Current())
	}
	// Third zero: transition to IDLE.
	_, next, _, changed := sm.TransitionIfNeeded(goodObs(0))
	if !changed || next != StateIdle {
		t.Fatalf("expected IDLE at 3rd zero-launch, got next=%v changed=%v", next, changed)
	}
}

// AC4: IDLE with a kernel launch -> CALIBRATING, not directly ACTIVE.
func TestIdle_RoutesThroughCalibrating(t *testing.T) {
	cfg := StateConfig{IdleIntervals: 1, WarmupSamples: 5, StaleReadFailures: 3}
	sm := quietSM(t, cfg)
	// Calibrate, then go idle.
	for i := 0; i < 5; i++ {
		sm.TransitionIfNeeded(goodObs(10))
	}
	// Now ACTIVE — the 5th tick should have transitioned.
	if sm.Current() != StateActive {
		t.Fatalf("expected ACTIVE after warmup, got %v", sm.Current())
	}
	sm.TransitionIfNeeded(goodObs(0))
	if sm.Current() != StateIdle {
		t.Fatalf("expected IDLE, got %v", sm.Current())
	}
	// Kernel launch from IDLE should land in CALIBRATING.
	_, next, _, changed := sm.TransitionIfNeeded(goodObs(5))
	if !changed || next != StateCalibrating {
		t.Fatalf("expected IDLE->CALIBRATING, got next=%v changed=%v", next, changed)
	}
}

// AC4 continued: from CALIBRATING (after IDLE wake) re-warm for warmup
// samples before ACTIVE. The waking observation itself counts toward
// warmup (same as the startup path), so `WarmupSamples=3` means total
// three samples including the one that triggered the IDLE->CALIBRATING
// transition.
func TestCalibrating_RewarmsFromIdle(t *testing.T) {
	cfg := StateConfig{IdleIntervals: 1, WarmupSamples: 3, StaleReadFailures: 3}
	sm := quietSM(t, cfg)
	for i := 0; i < 3; i++ {
		sm.TransitionIfNeeded(goodObs(10))
	}
	sm.TransitionIfNeeded(goodObs(0)) // -> IDLE
	sm.TransitionIfNeeded(goodObs(5)) // -> CALIBRATING (tick 1)
	if sm.Current() != StateCalibrating {
		t.Fatalf("setup precondition failed: %v", sm.Current())
	}
	// WarmupSamples=3 and tick 1 already counted. Need 2 more.
	sm.TransitionIfNeeded(goodObs(10)) // tick 2
	if sm.Current() != StateCalibrating {
		t.Fatalf("rewarming cut short at tick 2: %v", sm.Current())
	}
	sm.TransitionIfNeeded(goodObs(10)) // tick 3 -> ACTIVE
	if sm.Current() != StateActive {
		t.Fatalf("expected ACTIVE after 3 samples in CALIBRATING, got %v", sm.Current())
	}
}

// AC5: event read failures trip STALE from any state.
func TestEventReadFailure_ToStale(t *testing.T) {
	cfg := StateConfig{IdleIntervals: 1, WarmupSamples: 0, StaleReadFailures: 3}
	sm := quietSM(t, cfg)
	sm.TransitionIfNeeded(goodObs(10)) // -> ACTIVE
	sm.TransitionIfNeeded(badObs())
	sm.TransitionIfNeeded(badObs())
	if sm.Current() == StateStale {
		t.Fatal("STALE too early (before 3rd failure)")
	}
	_, next, _, changed := sm.TransitionIfNeeded(badObs())
	if !changed || next != StateStale {
		t.Fatalf("expected STALE at 3rd failure, got next=%v changed=%v", next, changed)
	}
}

// AC5 recovery: STALE + good observation -> CALIBRATING.
func TestStaleRecovery_ViaCalibrating(t *testing.T) {
	cfg := StateConfig{IdleIntervals: 3, WarmupSamples: 3, StaleReadFailures: 1}
	sm := quietSM(t, cfg)
	sm.TransitionIfNeeded(badObs())
	if sm.Current() != StateStale {
		t.Fatalf("setup failed: %v", sm.Current())
	}
	_, next, _, changed := sm.TransitionIfNeeded(goodObs(5))
	if !changed || next != StateCalibrating {
		t.Fatalf("expected CALIBRATING after STALE recovery, got next=%v changed=%v", next, changed)
	}
}

// Consecutive failure counter resets on a good read.
func TestFailureCounter_ResetsOnGoodRead(t *testing.T) {
	cfg := StateConfig{IdleIntervals: 3, WarmupSamples: 0, StaleReadFailures: 3}
	sm := quietSM(t, cfg)
	sm.TransitionIfNeeded(goodObs(10))
	sm.TransitionIfNeeded(badObs())
	sm.TransitionIfNeeded(badObs())
	sm.TransitionIfNeeded(goodObs(10))
	// One more bad should NOT trip STALE because counter reset.
	sm.TransitionIfNeeded(badObs())
	if sm.Current() == StateStale {
		t.Fatal("STALE triggered despite counter reset")
	}
}

// Short idle gap (fewer than idle_intervals zeros) does not trip IDLE.
func TestShortIdleDoesNotTrigger(t *testing.T) {
	cfg := StateConfig{IdleIntervals: 3, WarmupSamples: 0, StaleReadFailures: 3}
	sm := quietSM(t, cfg)
	sm.TransitionIfNeeded(goodObs(10))
	sm.TransitionIfNeeded(goodObs(0))
	sm.TransitionIfNeeded(goodObs(0))
	sm.TransitionIfNeeded(goodObs(10)) // resets counter
	sm.TransitionIfNeeded(goodObs(0))
	sm.TransitionIfNeeded(goodObs(0))
	if sm.Current() != StateActive {
		t.Fatalf("should still be ACTIVE after bounded zero-gap: %v", sm.Current())
	}
}

// Repeated identical observations do not cause repeat transitions.
func TestTransitions_IdempotentOnceInState(t *testing.T) {
	cfg := StateConfig{IdleIntervals: 2, WarmupSamples: 0, StaleReadFailures: 3}
	sm := quietSM(t, cfg)
	sm.TransitionIfNeeded(goodObs(10)) // -> ACTIVE
	sm.TransitionIfNeeded(goodObs(0))
	sm.TransitionIfNeeded(goodObs(0)) // -> IDLE
	for i := 0; i < 5; i++ {
		_, _, _, changed := sm.TransitionIfNeeded(goodObs(0))
		if changed {
			t.Fatalf("unexpected transition on repeat idle obs #%d", i)
		}
	}
	if sm.Current() != StateIdle {
		t.Fatalf("should still be IDLE, got %v", sm.Current())
	}
}

// AC8: KernelLaunchesSince sums launches in the retention window.
func TestKernelLaunchesSince(t *testing.T) {
	cfg := StateConfig{IdleIntervals: 2, WarmupSamples: 0, StaleReadFailures: 3, RecentWindow: 10}
	sm := quietSM(t, cfg)
	base := time.Date(2026, 4, 16, 10, 0, 0, 0, time.UTC)
	for i := 0; i < 5; i++ {
		sm.TransitionIfNeeded(Observation{
			KernelLaunchCount: i + 1,
			EventReadOK:       true,
			Timestamp:         base.Add(time.Duration(i) * time.Second),
		})
	}
	// All 5: 1+2+3+4+5 = 15.
	if got := sm.KernelLaunchesSince(base); got != 15 {
		t.Fatalf("total launches = %d, want 15", got)
	}
	// Since t=base+3s: 4+5 = 9.
	if got := sm.KernelLaunchesSince(base.Add(3 * time.Second)); got != 9 {
		t.Fatalf("since +3s = %d, want 9", got)
	}
	// Future time: 0.
	if got := sm.KernelLaunchesSince(base.Add(time.Hour)); got != 0 {
		t.Fatalf("since +1h = %d, want 0", got)
	}
}

// Retention is bounded by RecentWindow.
func TestKernelLaunchesSince_RetentionBounded(t *testing.T) {
	cfg := StateConfig{IdleIntervals: 2, WarmupSamples: 0, StaleReadFailures: 3, RecentWindow: 3}
	sm := quietSM(t, cfg)
	base := testTS
	for i := 0; i < 10; i++ {
		sm.TransitionIfNeeded(Observation{
			KernelLaunchCount: 1,
			EventReadOK:       true,
			Timestamp:         base.Add(time.Duration(i) * time.Second),
		})
	}
	// Only last 3 retained.
	if got := sm.KernelLaunchesSince(base); got != 3 {
		t.Fatalf("retained count = %d, want 3", got)
	}
}

// Default RecentWindow derives from IdleIntervals*4.
func TestDefaultRecentWindow_Derived(t *testing.T) {
	cfg := StateConfig{IdleIntervals: 5, WarmupSamples: 0, StaleReadFailures: 3}
	sm := quietSM(t, cfg)
	base := testTS
	for i := 0; i < 100; i++ {
		sm.TransitionIfNeeded(Observation{
			KernelLaunchCount: 1,
			EventReadOK:       true,
			Timestamp:         base.Add(time.Duration(i) * time.Second),
		})
	}
	// IdleIntervals(5) * 4 = 20 retained.
	if got := sm.KernelLaunchesSince(base); got != 20 {
		t.Fatalf("derived retention = %d, want 20", got)
	}
}

// NewStateMachineFromRestore starts in ACTIVE — used after a successful
// baseline restore (Story 2.4 AC3).
func TestNewStateMachineFromRestore_StartsActive(t *testing.T) {
	sm, err := NewStateMachineFromRestore(DefaultStateConfig(), nil)
	if err != nil {
		t.Fatalf("NewStateMachineFromRestore: %v", err)
	}
	if sm.Current() != StateActive {
		t.Fatalf("state after restore = %v, want ACTIVE", sm.Current())
	}
}

// Negative KernelLaunchCount must be coerced to 0 at the boundary so
// malformed collectors can't produce "kernel activity" from garbage.
func TestNormalize_NegativeKernelCount(t *testing.T) {
	cfg := StateConfig{IdleIntervals: 2, WarmupSamples: 0, StaleReadFailures: 3}
	sm := quietSM(t, cfg)
	// Promote to ACTIVE via first good sample.
	sm.TransitionIfNeeded(goodObs(10))
	// Now feed negative counts — should count as zero-launch.
	sm.TransitionIfNeeded(Observation{KernelLaunchCount: -5, EventReadOK: true, Timestamp: testTS})
	sm.TransitionIfNeeded(Observation{KernelLaunchCount: -1, EventReadOK: true, Timestamp: testTS})
	if sm.Current() != StateIdle {
		t.Fatalf("negative counts should act as zero-launch: state = %v", sm.Current())
	}
	// KernelLaunchesSince must never return a negative total.
	if got := sm.KernelLaunchesSince(testTS.Add(-time.Hour)); got < 0 {
		t.Fatalf("total launches = %d, must not be negative", got)
	}
}

// Far-future timestamps are treated as read failures — malicious or
// clock-skewed input should not influence idle detection.
func TestNormalize_FutureTimestampIsReadFailure(t *testing.T) {
	cfg := StateConfig{IdleIntervals: 3, WarmupSamples: 0, StaleReadFailures: 2}
	sm := quietSM(t, cfg)
	sm.TransitionIfNeeded(goodObs(10)) // -> ACTIVE
	// Feed 2 observations whose timestamps are far in the future.
	future := time.Now().Add(time.Hour)
	sm.TransitionIfNeeded(Observation{KernelLaunchCount: 10, EventReadOK: true, Timestamp: future})
	sm.TransitionIfNeeded(Observation{KernelLaunchCount: 10, EventReadOK: true, Timestamp: future})
	if sm.Current() != StateStale {
		t.Fatalf("future-dated observations should trigger STALE after 2: got %v", sm.Current())
	}
}

// Timestamps within the FutureTolerance window are accepted normally.
func TestNormalize_NearFutureTimestampAccepted(t *testing.T) {
	cfg := StateConfig{IdleIntervals: 3, WarmupSamples: 0, StaleReadFailures: 2}
	sm := quietSM(t, cfg)
	slightlyAhead := time.Now().Add(30 * time.Second) // within 1-minute tolerance
	sm.TransitionIfNeeded(Observation{KernelLaunchCount: 10, EventReadOK: true, Timestamp: slightlyAhead})
	if sm.Current() != StateActive {
		t.Fatalf("near-future timestamp should be accepted, state = %v", sm.Current())
	}
}

// Concurrent TransitionIfNeeded + KernelLaunchesSince + Current must not
// race. A torn read would corrupt the retention slice.
func TestStateMachine_ConcurrentAccess(t *testing.T) {
	cfg := StateConfig{IdleIntervals: 3, WarmupSamples: 0, StaleReadFailures: 10, RecentWindow: 50}
	sm := quietSM(t, cfg)
	done := make(chan struct{})
	// Writer
	go func() {
		base := time.Now()
		for i := 0; i < 500; i++ {
			sm.TransitionIfNeeded(Observation{
				KernelLaunchCount: i % 3,
				EventReadOK:       true,
				Timestamp:         base.Add(time.Duration(i) * time.Millisecond),
			})
		}
		close(done)
	}()
	readDone := make(chan struct{}, 4)
	base := time.Now()
	for r := 0; r < 4; r++ {
		go func() {
			for i := 0; i < 500; i++ {
				_ = sm.Current()
				_ = sm.KernelLaunchesSince(base)
			}
			readDone <- struct{}{}
		}()
	}
	<-done
	for r := 0; r < 4; r++ {
		<-readDone
	}
}

// consecutiveFails must not overflow int on very long outages. Saturates
// at a safe bound so the STALE transition remains correct forever.
func TestStaleCounter_Saturates(t *testing.T) {
	cfg := StateConfig{IdleIntervals: 1, WarmupSamples: 0, StaleReadFailures: 3}
	sm := quietSM(t, cfg)
	// Force STALE first.
	for i := 0; i < 3; i++ {
		sm.TransitionIfNeeded(badObs())
	}
	if sm.Current() != StateStale {
		t.Fatalf("setup: state = %v, want STALE", sm.Current())
	}
	// Now drive a LOT of failures — the counter must not overflow.
	for i := 0; i < 10000; i++ {
		sm.TransitionIfNeeded(badObs())
	}
	// A single good observation should still recover via CALIBRATING.
	_, next, _, _ := sm.TransitionIfNeeded(goodObs(1))
	if next != StateCalibrating {
		t.Fatalf("post-saturation recovery: next = %v, want CALIBRATING", next)
	}
}

// STALE -> CALIBRATING recovery counts the triggering good observation
// toward warmup. Consistent with startup path.
func TestStaleRecovery_CountsTriggeringSample(t *testing.T) {
	cfg := StateConfig{IdleIntervals: 3, WarmupSamples: 3, StaleReadFailures: 1}
	sm := quietSM(t, cfg)
	sm.TransitionIfNeeded(badObs()) // -> STALE
	if sm.Current() != StateStale {
		t.Fatal("setup failed")
	}
	sm.TransitionIfNeeded(goodObs(1)) // -> CALIBRATING (counts as sample 1 of 3)
	sm.TransitionIfNeeded(goodObs(1)) // sample 2
	if sm.Current() != StateCalibrating {
		t.Fatalf("should still be CALIBRATING at sample 2: %v", sm.Current())
	}
	sm.TransitionIfNeeded(goodObs(1)) // sample 3 -> ACTIVE
	if sm.Current() != StateActive {
		t.Fatalf("expected ACTIVE after 3 samples, got %v", sm.Current())
	}
}
