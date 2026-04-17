//go:build linux

package health

import (
	"io"
	"log/slog"
	"math"
	"testing"
	"time"
)

// TestAdversarial_StateMachine — hit the state machine with weird inputs.
func TestAdversarial_StateMachine(t *testing.T) {
	cfg := StateConfig{
		IdleIntervals:     3,
		WarmupSamples:     3,
		StaleReadFailures: 3,
	}
	q := slog.New(slog.NewTextHandler(io.Discard, nil))
	sm, _ := NewStateMachine(cfg, q)

	t.Run("negative-kernel-count-coerced", func(t *testing.T) {
		obs := Observation{KernelLaunchCount: -9999999, EventReadOK: true, Timestamp: time.Now()}
		sm.TransitionIfNeeded(obs)
		// Should not crash; negative coerced to 0.
	})
	t.Run("maxint-kernel-count", func(t *testing.T) {
		obs := Observation{KernelLaunchCount: math.MaxInt, EventReadOK: true, Timestamp: time.Now()}
		sm.TransitionIfNeeded(obs)
	})
	t.Run("past-timestamp", func(t *testing.T) {
		obs := Observation{KernelLaunchCount: 1, EventReadOK: true, Timestamp: time.Time{}} // zero time
		sm.TransitionIfNeeded(obs)
	})
	t.Run("far-future-timestamp", func(t *testing.T) {
		obs := Observation{KernelLaunchCount: 1, EventReadOK: true, Timestamp: time.Now().Add(100 * 365 * 24 * time.Hour)}
		sm.TransitionIfNeeded(obs)
		// Future > FutureTolerance; should coerce to EventReadOK=false.
	})
}

// TestAdversarial_Baseliner — feed extreme values and verify no NaN/Inf escape.
func TestAdversarial_Baseliner(t *testing.T) {
	bl, _ := NewBaseliner(DefaultBaselineConfig(), slog.New(slog.NewTextHandler(io.Discard, nil)))

	// Flood with gigantic values
	for i := 0; i < 100; i++ {
		bl.Update(RawObservation{Throughput: math.MaxFloat64 / 1e10, Compute: 0.5, Memory: 0.5, CPU: 0.5})
	}
	cur := bl.Current()
	if math.IsNaN(cur.Throughput) || math.IsInf(cur.Throughput, 0) {
		t.Errorf("NaN/Inf leaked into Current.Throughput: %v", cur.Throughput)
	}

	// Signals should return finite output
	sig := bl.Signals(RawObservation{Throughput: math.MaxFloat64, Compute: math.MaxFloat64, Memory: math.MaxFloat64, CPU: math.MaxFloat64})
	if math.IsNaN(sig.Throughput) || math.IsInf(sig.Throughput, 0) {
		t.Errorf("NaN/Inf in Signals.Throughput: %v", sig.Throughput)
	}

	// Negative-inf input
	bl.Update(RawObservation{Throughput: math.Inf(-1), Compute: math.NaN(), Memory: math.Inf(1), CPU: -1})
	cur = bl.Current()
	if math.IsNaN(cur.Throughput) || math.IsInf(cur.Throughput, 0) {
		t.Errorf("Inf input poisoned baseline: %v", cur.Throughput)
	}
}

// TestAdversarial_ClassifierHysteresis — can we flap the classifier?
func TestAdversarial_ClassifierHysteresis(t *testing.T) {
	c, _ := NewClassifier(ClassifierConfig{Hysteresis: 0.02})
	now := time.Now()
	var flaps int
	prev := false
	for i := 0; i < 100; i++ {
		score := 0.80
		if i%2 == 0 {
			score = 0.81
		} else {
			score = 0.79
		}
		is, _ := c.Classify(score, 0.80, now.Add(time.Duration(i)*time.Second))
		if is != prev && i > 0 {
			flaps++
		}
		prev = is
	}
	t.Logf("flaps over 100 ticks oscillating around threshold: %d", flaps)
	if flaps > 2 {
		t.Errorf("hysteresis failed to prevent flapping: %d transitions", flaps)
	}

	// NaN score
	is, _ := c.Classify(math.NaN(), 0.5, now)
	t.Logf("NaN score -> is straggler: %v", is)
	// Expected: NaN < 0.5 is false, so no change. Stuck-is-fine since prev was some state.

	// Inf score
	is, _ = c.Classify(math.Inf(1), 0.5, now)
	t.Logf("+Inf score -> is straggler: %v", is)
	is, _ = c.Classify(math.Inf(-1), 0.5, now)
	t.Logf("-Inf score -> is straggler: %v", is)
}

// TestAdversarial_ModeEvaluator_ZeroBaseline — local-baseline fallback math.
func TestAdversarial_ModeEvaluator_ZeroBaseline(t *testing.T) {
	cache := NewThresholdCache()
	bl, _ := NewBaseliner(DefaultBaselineConfig(), slog.New(slog.NewTextHandler(io.Discard, nil)))
	// Warm with tiny values
	for i := 0; i < 50; i++ {
		bl.Update(RawObservation{Throughput: 0, Compute: 0, Memory: 0, CPU: 0})
	}

	em := &reachableStub{reachable: false}
	me, _ := NewModeEvaluator(DefaultModeConfig(), cache, em, bl, 30, slog.New(slog.NewTextHandler(io.Discard, nil)))

	mode, th, ok := me.Evaluate(time.Now())
	t.Logf("zero-baseline + unreachable: mode=%s threshold=%v ok=%v", mode, th, ok)
	// With baseline mean = 0, we should NOT produce a threshold.
}

type reachableStub struct{ reachable bool }

func (r *reachableStub) FleetReachable() bool { return r.reachable }
