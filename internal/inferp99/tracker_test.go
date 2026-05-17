package inferp99

import (
	"testing"
	"time"
)

// testConfig keeps wall-clock-independent tests short: 5-sample warmup,
// 2-tick sustain, generous window + cap so the time/cap mechanisms
// don't interfere with sustain-machine cases.
func testConfig() Config {
	return Config{
		BreachRatio:    1.5,
		ClearRatio:     1.1,
		SustainTicks:   2,
		WarmupSamples:  5,
		WindowDuration: time.Hour,
		MaxSamples:     1000,
		RearmDuration:  time.Minute,
	}
}

// Below warmup the tracker has no baseline and must not emit, even if
// the samples it has seen would cross the configured ratio when
// compared against themselves.
func TestTracker_NoEmitBeforeWarmup(t *testing.T) {
	tr := NewTracker(testConfig())
	now := time.Unix(0, 0)
	tr.Observe(100, now)
	tr.Observe(100, now)
	tr.Observe(100, now)
	if _, ok := tr.CheckAt(now); ok {
		t.Fatal("emitted before warmup samples reached the threshold")
	}
}

// The first CheckAt after warmup samples have landed freezes the
// baseline_p99 and returns no emit. Subsequent CheckAt calls compare
// against the frozen value.
func TestTracker_BaselineFreezesOnFirstCheck(t *testing.T) {
	tr := NewTracker(testConfig())
	now := time.Unix(0, 0)
	for i := 0; i < 10; i++ {
		tr.Observe(100, now)
	}
	if _, ok := tr.CheckAt(now); ok {
		t.Fatal("baseline-freeze CheckAt unexpectedly emitted")
	}
	got := tr.BaselineP99Ns()
	if got != 100 {
		t.Fatalf("baseline_p99=%d want 100", got)
	}

	// Even if all future samples are identical (ratio == 1.0), the
	// baseline must not move and no emit must fire.
	for i := 0; i < 20; i++ {
		tr.Observe(100, now)
	}
	if _, ok := tr.CheckAt(now); ok {
		t.Fatal("emitted on identical-distribution post-baseline")
	}
	if tr.BaselineP99Ns() != 100 {
		t.Fatal("baseline drifted after freeze")
	}
}

// The full happy path: baseline freezes, ratio crosses BreachRatio,
// sustain ticks count up, emission fires at exactly SustainTicks,
// suppression then prevents re-emit during the rearm window.
func TestTracker_EmitsAtSustainThenSuppresses(t *testing.T) {
	tr := NewTracker(testConfig())
	now := time.Unix(0, 0)

	// Baseline at 100ns.
	for i := 0; i < 10; i++ {
		tr.Observe(100, now)
	}
	tr.CheckAt(now) // baseline freeze

	// Push the rolling p99 up to 200ns (2x ratio, clearly above 1.5x).
	advance := func(d time.Duration) {
		now = now.Add(d)
	}
	advance(time.Second)
	for i := 0; i < 10; i++ {
		tr.Observe(200, now)
	}

	// First CheckAt above ratio: consecutive=1, no emit (SustainTicks=2).
	if _, ok := tr.CheckAt(now); ok {
		t.Fatal("emitted at sustain=1 with SustainTicks=2")
	}

	// Second CheckAt above ratio: emit.
	advance(time.Second)
	tr.Observe(200, now) // keep window above ratio
	b, ok := tr.CheckAt(now)
	if !ok {
		t.Fatal("did not emit at sustain threshold")
	}
	if b.BaselineP99Ns != 100 {
		t.Errorf("BaselineP99Ns=%d want 100", b.BaselineP99Ns)
	}
	if b.Ratio < 1.5 {
		t.Errorf("Ratio=%f want >= 1.5", b.Ratio)
	}
	if !b.At.Equal(now) {
		t.Errorf("At=%v want %v", b.At, now)
	}

	// Immediately re-check at the same conditions: suppressed.
	if _, ok := tr.CheckAt(now); ok {
		t.Fatal("re-emitted inside rearm window")
	}

	// Advance past the rearm window. Sustain must restart from zero,
	// not fire immediately on the first call past suppression.
	advance(time.Minute + time.Second) // past RearmDuration
	tr.Observe(200, now)
	if _, ok := tr.CheckAt(now); ok {
		t.Fatal("emitted on first sustain tick past suppression (need full SustainTicks again)")
	}
	advance(time.Second)
	tr.Observe(200, now)
	if _, ok := tr.CheckAt(now); !ok {
		t.Fatal("did not re-emit at full sustain past suppression")
	}
}

// Hysteresis: a ratio between ClearRatio and BreachRatio holds state
// (no emit, no sustain bump, no sustain reset). The signal needs to
// cross either boundary to change state.
func TestTracker_HysteresisBandDoesNotChangeState(t *testing.T) {
	tr := NewTracker(testConfig())
	now := time.Unix(0, 0)
	for i := 0; i < 10; i++ {
		tr.Observe(100, now)
	}
	tr.CheckAt(now) // baseline=100

	// Push to ratio=1.3 (above clear=1.1, below breach=1.5).
	now = now.Add(time.Second)
	for i := 0; i < 10; i++ {
		tr.Observe(130, now)
	}
	for i := 0; i < 5; i++ {
		if _, ok := tr.CheckAt(now); ok {
			t.Fatal("emitted inside hysteresis band")
		}
		now = now.Add(time.Second)
	}
}

// Below ClearRatio the sustain counter must reset so a future episode
// counts from zero, not from a partially-accumulated previous run.
//
// Uses a small MaxSamples cap (20) so each phase's bulk-observe
// evicts the prior values from the window via FIFO, leaving the
// current p99 dominated by the just-observed shape. Without the
// cap, prior 200s linger in the top-1% tail and the "drop to 50"
// phase wouldn't actually clear the sustain (p99 stays at 200).
func TestTracker_BelowClearResetsSustain(t *testing.T) {
	cfg := testConfig()
	cfg.MaxSamples = 20
	tr := NewTracker(cfg)
	now := time.Unix(0, 0)

	// Baseline at 100.
	for i := 0; i < 20; i++ {
		tr.Observe(100, now)
	}
	tr.CheckAt(now)

	// Push above breach. Cap evicts the 100s as we observe 20 200s.
	now = now.Add(time.Second)
	for i := 0; i < 20; i++ {
		tr.Observe(200, now)
	}
	tr.CheckAt(now) // sustain=1

	// Drop below clear. Cap evicts the 200s as we observe 20 50s.
	now = now.Add(time.Second)
	for i := 0; i < 20; i++ {
		tr.Observe(50, now)
	}
	if _, ok := tr.CheckAt(now); ok {
		t.Fatal("emitted with ratio below clear")
	}

	// Push back above breach. Cap evicts the 50s. Need TWO consecutive
	// sustain ticks before emit (sustain counter was reset).
	now = now.Add(time.Second)
	for i := 0; i < 20; i++ {
		tr.Observe(300, now)
	}
	if _, ok := tr.CheckAt(now); ok {
		t.Fatal("emitted on first post-reset sustain tick")
	}
	now = now.Add(time.Second)
	tr.Observe(300, now)
	if _, ok := tr.CheckAt(now); !ok {
		t.Fatal("did not emit on second post-reset sustain tick")
	}
}

// Time-bounded window: samples older than WindowDuration relative to
// the last Observe / CheckAt are pruned. A workload that breached an
// hour ago but has been calm since must NOT emit.
func TestTracker_WindowPrunesOldSamples(t *testing.T) {
	cfg := testConfig()
	cfg.WindowDuration = 10 * time.Second
	tr := NewTracker(cfg)

	now := time.Unix(0, 0)
	// Old breach samples that should age out of the window.
	for i := 0; i < 100; i++ {
		tr.Observe(500, now)
	}

	// Jump forward past the window.
	now = now.Add(time.Minute)
	// New healthy samples (warmupSamples=5).
	for i := 0; i < 10; i++ {
		tr.Observe(100, now)
	}

	// CheckAt should see ONLY the 10 healthy samples (old 500s pruned).
	tr.CheckAt(now) // baseline freezes at 100
	if got := tr.BaselineP99Ns(); got != 100 {
		t.Fatalf("baseline=%d want 100 (old samples should have been pruned)", got)
	}
}

// MaxSamples cap protects against unbounded memory on very-high-
// throughput workloads. Once the cap is hit, older samples are
// dropped FIFO and the tracker keeps responding to recent state.
func TestTracker_MaxSamplesCapDropsOldFIFO(t *testing.T) {
	cfg := testConfig()
	cfg.MaxSamples = 20
	cfg.WindowDuration = time.Hour // disable time-based pruning
	tr := NewTracker(cfg)

	now := time.Unix(0, 0)
	for i := 0; i < 100; i++ {
		tr.Observe(float64(i+1), now)
	}
	if got := tr.Samples(); got != 20 {
		t.Fatalf("retained=%d want 20 (cap)", got)
	}
	// The oldest 80 should have been dropped; the kept window should
	// be the last 20 values (81..100). p99 of that = 100.
	tr.CheckAt(now) // baseline freezes at 100
	if got := tr.BaselineP99Ns(); got != 100 {
		t.Fatalf("baseline=%d want 100 (last 20 samples)", got)
	}
}

// Non-positive step durations are ignored. The hot path filter
// catches malformed input from broken event sources.
func TestTracker_NonPositiveSamplesIgnored(t *testing.T) {
	tr := NewTracker(testConfig())
	now := time.Unix(0, 0)
	tr.Observe(0, now)
	tr.Observe(-1, now)
	if tr.Samples() != 0 {
		t.Fatalf("non-positive samples retained: %d", tr.Samples())
	}
}

// Two trackers must maintain independent state. A breach on tracker A
// must not affect tracker B's sustain counter or suppression window.
func TestTracker_PerTrackerIndependence(t *testing.T) {
	a := NewTracker(testConfig())
	b := NewTracker(testConfig())
	now := time.Unix(0, 0)
	for i := 0; i < 10; i++ {
		a.Observe(100, now)
		b.Observe(100, now)
	}
	a.CheckAt(now)
	b.CheckAt(now)

	// Push A above breach across two ticks; emit.
	now = now.Add(time.Second)
	for i := 0; i < 10; i++ {
		a.Observe(200, now)
	}
	a.CheckAt(now) // A sustain=1
	now = now.Add(time.Second)
	a.Observe(200, now)
	if _, ok := a.CheckAt(now); !ok {
		t.Fatal("A did not emit")
	}

	// B never breached. It must not emit, even after A has fired.
	if _, ok := b.CheckAt(now); ok {
		t.Fatal("B emitted while never breaching")
	}
}

// computeP99 boundary cases: empty input -> 0; single sample ->
// returns that sample.
func TestComputeP99_BoundaryCases(t *testing.T) {
	if got := computeP99(nil); got != 0 {
		t.Errorf("empty input p99=%f want 0", got)
	}
	if got := computeP99([]float64{42}); got != 42 {
		t.Errorf("single-sample p99=%f want 42", got)
	}
	// Nearest-rank on 100 samples: index = floor(0.99*100)-1 = 98.
	// Sorted 1..100 -> index 98 = value 99.
	vs := make([]float64, 100)
	for i := range vs {
		vs[i] = float64(i + 1)
	}
	if got := computeP99(vs); got != 99 {
		t.Errorf("p99 of 1..100 = %f want 99", got)
	}
}
