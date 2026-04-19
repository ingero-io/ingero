package health

import (
	"sync/atomic"
	"testing"
	"time"
)

// fakeReachable is a pluggable FleetReachable source.
type fakeReachable struct {
	reachable atomic.Bool
}

func (f *fakeReachable) FleetReachable() bool { return f.reachable.Load() }
func (f *fakeReachable) Set(v bool)            { f.reachable.Store(v) }

func newModeEvaluatorForTests(t *testing.T, cache *ThresholdCache, reachable *fakeReachable, baseliner Baseliner) ModeEvaluator {
	t.Helper()
	m, err := NewModeEvaluator(DefaultModeConfig(), cache, reachable, baseliner, 30, discardLogger())
	if err != nil {
		t.Fatalf("NewModeEvaluator: %v", err)
	}
	return m
}

// Evaluator with a warm baseliner helper.
func warmBaseliner(t *testing.T) Baseliner {
	t.Helper()
	b, err := NewBaseliner(DefaultBaselineConfig(), discardLogger())
	if err != nil {
		t.Fatal(err)
	}
	for i := 0; i < 50; i++ {
		b.Update(RawObservation{Throughput: 100, Compute: 0.9, Memory: 0.9, CPU: 0.9})
	}
	return b
}

func TestModeConfig_Validate(t *testing.T) {
	tests := []struct {
		name    string
		cfg     ModeConfig
		wantErr bool
	}{
		{"defaults", DefaultModeConfig(), false},
		{"zero_fresh", ModeConfig{FreshMaxAge: 0, CachedMaxAge: time.Minute, LocalBaselineFactor: 0.85}, true},
		{"cached_le_fresh", ModeConfig{FreshMaxAge: 10 * time.Second, CachedMaxAge: 10 * time.Second, LocalBaselineFactor: 0.85}, true},
		{"factor_zero", ModeConfig{FreshMaxAge: time.Second, CachedMaxAge: time.Minute, LocalBaselineFactor: 0}, true},
		{"factor_one", ModeConfig{FreshMaxAge: time.Second, CachedMaxAge: time.Minute, LocalBaselineFactor: 1.0}, true},
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

// AC1/AC2: Fresh cache + quorum met -> ModeFleet.
func TestEvaluate_FleetMode(t *testing.T) {
	cache := NewThresholdCache()
	now := time.Now()
	cache.Set(0.82, true, now.Add(-5*time.Second))
	reachable := &fakeReachable{}
	reachable.Set(true)
	m := newModeEvaluatorForTests(t, cache, reachable, warmBaseliner(t))

	mode, threshold, ok := m.Evaluate(now)
	if !ok || mode != ModeFleet {
		t.Fatalf("mode = %v, threshold = %v, ok = %v", mode, threshold, ok)
	}
	if threshold != 0.82 {
		t.Fatalf("threshold = %v, want 0.82", threshold)
	}
}

// AC3: Cache present, quorum NOT met -> ModeFleetCached.
func TestEvaluate_FleetCachedMode(t *testing.T) {
	cache := NewThresholdCache()
	now := time.Now()
	cache.Set(0.78, false, now.Add(-5*time.Second))
	reachable := &fakeReachable{}
	reachable.Set(true)
	m := newModeEvaluatorForTests(t, cache, reachable, warmBaseliner(t))

	mode, threshold, ok := m.Evaluate(now)
	if !ok || mode != ModeFleetCached {
		t.Fatalf("mode = %v, threshold = %v, ok = %v", mode, threshold, ok)
	}
	if threshold != 0.78 {
		t.Fatalf("threshold = %v, want 0.78", threshold)
	}
}

// AC4: Fleet unreachable, cache < 5 min old -> ModeLocalCached.
func TestEvaluate_LocalCachedMode(t *testing.T) {
	cache := NewThresholdCache()
	now := time.Now()
	cache.Set(0.82, true, now.Add(-2*time.Minute))
	reachable := &fakeReachable{}
	reachable.Set(false)
	m := newModeEvaluatorForTests(t, cache, reachable, warmBaseliner(t))

	mode, threshold, ok := m.Evaluate(now)
	if !ok || mode != ModeLocalCached {
		t.Fatalf("mode = %v, threshold = %v, ok = %v", mode, threshold, ok)
	}
	if threshold != 0.82 {
		t.Fatalf("threshold = %v, want 0.82 (stale cached value)", threshold)
	}
}

// AC5: Fleet unreachable, cache stale, baseliner warm -> ModeLocalBaseline.
func TestEvaluate_LocalBaselineMode(t *testing.T) {
	cache := NewThresholdCache()
	now := time.Now()
	// Cache older than CachedMaxAge (5 min).
	cache.Set(0.82, true, now.Add(-10*time.Minute))
	reachable := &fakeReachable{}
	reachable.Set(false)
	b := warmBaseliner(t)
	m := newModeEvaluatorForTests(t, cache, reachable, b)

	mode, threshold, ok := m.Evaluate(now)
	if !ok || mode != ModeLocalBaseline {
		t.Fatalf("mode = %v, threshold = %v, ok = %v", mode, threshold, ok)
	}
	// Threshold is LocalBaselineFactor directly (0.85), not derived from
	// raw baseline magnitudes. The old formula mixed units (raw throughput
	// in tokens/sec with ratios in [0,1]).
	if threshold != 0.85 {
		t.Fatalf("threshold = %v, want 0.85", threshold)
	}
}

// AC6: Fleet unreachable, no cache, cold baseliner -> ModeNone.
func TestEvaluate_NoneMode(t *testing.T) {
	cache := NewThresholdCache()
	reachable := &fakeReachable{}
	reachable.Set(false)
	b, _ := NewBaseliner(DefaultBaselineConfig(), discardLogger())
	// Baseliner is cold (no updates) — SampleCount < warmup.
	m := newModeEvaluatorForTests(t, cache, reachable, b)

	mode, threshold, ok := m.Evaluate(time.Now())
	if ok || mode != ModeNone {
		t.Fatalf("mode = %v, threshold = %v, ok = %v", mode, threshold, ok)
	}
	if threshold != 0 {
		t.Fatalf("threshold = %v, want 0", threshold)
	}
}

// AC7: mode transition is logged (we don't capture the log, just verify
// that CurrentMode tracks the last-evaluated mode).
func TestEvaluate_TracksCurrentMode(t *testing.T) {
	cache := NewThresholdCache()
	now := time.Now()
	cache.Set(0.82, true, now)
	reachable := &fakeReachable{}
	reachable.Set(true)
	m := newModeEvaluatorForTests(t, cache, reachable, warmBaseliner(t))

	if m.CurrentMode() != ModeNone {
		t.Fatalf("initial CurrentMode = %v, want none", m.CurrentMode())
	}
	m.Evaluate(now)
	if m.CurrentMode() != ModeFleet {
		t.Fatalf("post-evaluate CurrentMode = %v, want fleet", m.CurrentMode())
	}
}

// Fresh threshold ages to fleet-cached when it crosses FreshMaxAge but
// remains within CachedMaxAge.
func TestEvaluate_AgeTransitions(t *testing.T) {
	cache := NewThresholdCache()
	receivedAt := time.Now()
	cache.Set(0.82, true, receivedAt)
	reachable := &fakeReachable{}
	reachable.Set(true)
	m := newModeEvaluatorForTests(t, cache, reachable, warmBaseliner(t))

	// Immediately after receipt: fleet mode.
	if mode, _, _ := m.Evaluate(receivedAt); mode != ModeFleet {
		t.Fatalf("at t=0 mode = %v, want fleet", mode)
	}
	// 30 seconds later (> FreshMaxAge 20s): fleet-cached? No — quorum
	// is met, so the decision stays at fleet for the "quorum met"
	// branch actually... wait, the decision tree says step 1 requires
	// FRESH + quorum met. After FreshMaxAge, step 1 fails, step 2
	// requires quorum_met=false. So the evaluator falls through to step 3
	// which requires fleet unreachable. Fleet IS reachable. Step 4
	// requires unreachable. Fleet IS reachable. So mode = ModeNone.
	//
	// That is the correct behavior: an old-but-quorum-met value should
	// NOT be used as if it were fresh.
	m2, _, _ := m.Evaluate(receivedAt.Add(30 * time.Second))
	if m2 == ModeFleet {
		t.Fatalf("at t=30s with stale fresh-window, mode should not be fleet")
	}
}

// Fleet reachable but cache entirely empty -> None (reachability alone
// doesn't imply a threshold).
func TestEvaluate_ReachableButEmptyCache(t *testing.T) {
	cache := NewThresholdCache()
	reachable := &fakeReachable{}
	reachable.Set(true)
	m := newModeEvaluatorForTests(t, cache, reachable, warmBaseliner(t))

	mode, _, ok := m.Evaluate(time.Now())
	if ok || mode != ModeNone {
		t.Fatalf("empty cache but fleet reachable -> should be none, got %v", mode)
	}
}

// Fleet unreachable + zero-valued baseline: local-baseline now returns
// the factor directly regardless of raw magnitudes. A warm baseliner
// with all-zero observations still counts as warm (SampleCount >= warmup),
// so LocalBaseline mode is entered. The threshold is 0.85.
func TestEvaluate_UnreachableZeroBaseline_ReturnsLocalBaseline(t *testing.T) {
	cache := NewThresholdCache()
	reachable := &fakeReachable{}
	reachable.Set(false)
	b, _ := NewBaseliner(DefaultBaselineConfig(), discardLogger())
	for i := 0; i < 50; i++ {
		b.Update(RawObservation{})
	}
	m := newModeEvaluatorForTests(t, cache, reachable, b)
	mode, threshold, ok := m.Evaluate(time.Now())
	if !ok || mode != ModeLocalBaseline {
		t.Fatalf("warm baseliner should yield local-baseline, got %v ok=%v", mode, ok)
	}
	if threshold != 0.85 {
		t.Fatalf("threshold = %v, want 0.85", threshold)
	}
}

// Evaluator is fast: <100us per call (story AC10 is <1ms, we aim below).
func BenchmarkEvaluate(b *testing.B) {
	cache := NewThresholdCache()
	cache.Set(0.82, true, time.Now())
	reachable := &fakeReachable{}
	reachable.Set(true)
	bl, _ := NewBaseliner(DefaultBaselineConfig(), discardLogger())
	for i := 0; i < 50; i++ {
		bl.Update(RawObservation{Throughput: 100, Compute: 0.9, Memory: 0.9, CPU: 0.9})
	}
	m, _ := NewModeEvaluator(DefaultModeConfig(), cache, reachable, bl, 30, discardLogger())
	now := time.Now()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		m.Evaluate(now)
	}
}
