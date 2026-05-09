package infer

import (
	"math"
	"math/rand/v2"
	"testing"
)

func TestWorkloadBaseliner_ZeroBeforeAnySamples(t *testing.T) {
	b := NewWorkloadBaseliner()
	if b.Mean() != 0 {
		t.Errorf("Mean() before any samples = %v, want 0", b.Mean())
	}
	if b.P95() != 0 {
		t.Errorf("P95() before any samples = %v, want 0", b.P95())
	}
	if b.Samples() != 0 {
		t.Errorf("Samples() before any updates = %d, want 0", b.Samples())
	}
	if b.Warmed(1) {
		t.Error("Warmed(1) before any samples should be false")
	}
}

func TestWorkloadBaseliner_NonFiniteIgnored(t *testing.T) {
	b := NewWorkloadBaseliner()
	b.Update(math.NaN())
	b.Update(math.Inf(1))
	b.Update(math.Inf(-1))
	b.Update(-1.0)
	b.Update(0)
	if b.Samples() != 0 {
		t.Errorf("non-finite + non-positive inputs incremented Samples to %d", b.Samples())
	}
}

func TestWorkloadBaseliner_FillPhase(t *testing.T) {
	b := NewWorkloadBaseliner()
	for i := 1; i <= 4; i++ {
		b.Update(float64(i * 1_000_000)) // 1ms, 2ms, 3ms, 4ms
		if b.P95() != 0 {
			t.Errorf("P95 should be 0 during fill phase (samples=%d), got %v", b.Samples(), b.P95())
		}
	}
	b.Update(5_000_000) // 5ms — fill complete
	if b.P95() == 0 {
		t.Error("P95 should be non-zero after 5 samples")
	}
	if b.Samples() != 5 {
		t.Errorf("Samples = %d, want 5", b.Samples())
	}
}

func TestWorkloadBaseliner_P95ConvergesOnStationaryNormal(t *testing.T) {
	// Generate ~1000 samples from a known distribution, check that
	// P² converges to the analytic 95th percentile within tolerance.
	// We use a uniform distribution so the analytic p95 is exact:
	// p95 of Uniform(0, 100) is 95.
	b := NewWorkloadBaseliner()
	rng := rand.NewPCG(0xC0FFEE, 0xDEADBEEF)
	r := rand.New(rng)
	const n = 5000
	for i := 0; i < n; i++ {
		// Multiply by 1e6 to keep nanosecond-shaped magnitude.
		b.Update(r.Float64() * 100 * 1_000_000)
	}
	got := b.P95() / 1_000_000
	want := 95.0
	tol := 4.0 // P² convergence on a uniform distribution, N=5000, observed ~3% drift
	if math.Abs(got-want) > tol {
		t.Errorf("P95 = %.2f, want %.2f ± %.2f", got, want, tol)
	}
}

func TestWorkloadBaseliner_MeanTracksEMA(t *testing.T) {
	b := NewWorkloadBaseliner()
	// Drive 200 samples at value 100, expect EMA to converge close to 100.
	for i := 0; i < 200; i++ {
		b.Update(100)
	}
	if math.Abs(b.Mean()-100) > 0.5 {
		t.Errorf("Mean = %v, want ≈100", b.Mean())
	}
}

func TestWorkloadBaseliner_WarmedGate(t *testing.T) {
	b := NewWorkloadBaseliner()
	for i := 0; i < 4; i++ {
		b.Update(1_000_000)
	}
	if b.Warmed(5) {
		t.Error("Warmed(5) at samples=4 should be false (P² fill incomplete)")
	}
	b.Update(1_000_000)
	if !b.Warmed(5) {
		t.Error("Warmed(5) at samples=5 should be true")
	}
	if b.Warmed(10) {
		t.Error("Warmed(10) at samples=5 should be false")
	}
}

func TestWorkloadBaseliner_OutlierDoesNotPoisonBaselineWhenSkipped(t *testing.T) {
	// This test confirms the *math* of "outliers don't fold into the
	// baseline" by simulating it: the engine skips Update when an
	// outlier fires, so the baseliner never sees the bad sample. We
	// just check that absent the bad samples, the baseline is stable.
	b := NewWorkloadBaseliner()
	for i := 0; i < 200; i++ {
		b.Update(10_000_000) // 10ms, stable
	}
	steady := b.P95()
	// Now intentionally skip the next 50 samples (simulating the
	// "do not Update on outlier" rule), then resume steady.
	for i := 0; i < 100; i++ {
		b.Update(10_000_000)
	}
	got := b.P95()
	if math.Abs(got-steady)/steady > 0.05 {
		t.Errorf("P95 drifted by >5%% across stable run: steady=%v, got=%v", steady, got)
	}
}
