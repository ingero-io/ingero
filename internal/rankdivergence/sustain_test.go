package rankdivergence

import (
	"math"
	"testing"
	"time"
)

var baseTime = time.Date(2026, 5, 17, 12, 0, 0, 0, time.UTC)

// newTestTracker uses windowSize=8, sigma=4.0, sustainedSweeps=2 for
// deterministic short assertions.
func newTestTracker() *Tracker {
	return NewWithThresholds(8, 4.0, 2, 1*time.Second)
}

// feedCohort observes `samples` per rank from `ranks` with normal
// duration `normal`; rank `divergentRank` (if >= 0) gets `divergent`
// duration instead.
func feedCohort(tr *Tracker, comm uint64, pid uint32, ranks int, samples int, normal uint64, divergentRank int, divergent uint64) {
	for s := 0; s < samples; s++ {
		for r := 0; r < ranks; r++ {
			d := normal
			if r == divergentRank {
				d = divergent
			}
			tr.Observe(pid, comm, uint32(r), uint32(ranks), d)
		}
	}
}

func TestObserve_HealthyCohortNoDivergence(t *testing.T) {
	tr := newTestTracker()
	// 4 ranks, 8 samples each, all 10ms ± nothing.
	feedCohort(tr, 0xfeed, 1000, 4, 8, 10_000_000, -1, 0)
	got := tr.Compute(baseTime)
	if len(got) != 0 {
		t.Fatalf("healthy cohort must not emit: %+v", got)
	}
}

func TestObserve_DivergentRankEmitsAfterSustain(t *testing.T) {
	tr := newTestTracker()
	// 4 ranks; rank 2 is 10x slower (clear MAD outlier).
	feedCohort(tr, 0xfeed, 1000, 4, 8, 10_000_000, 2, 100_000_000)
	// Tick 1: rank 2 flagged; sweep count = 1, below sustainedSweeps.
	if got := tr.Compute(baseTime); len(got) != 0 {
		t.Fatalf("first tick should record but not emit: %+v", got)
	}
	// Tick 2: rank 2 still flagged; sweep count = 2, threshold met -> emit.
	got := tr.Compute(baseTime.Add(1 * time.Second))
	if len(got) != 1 {
		t.Fatalf("second tick must emit: %+v", got)
	}
	if got[0].Rank != 2 {
		t.Errorf("wrong rank flagged: %v", got[0].Rank)
	}
	if got[0].PID != 1000 {
		t.Errorf("pid round-trip failed: %v", got[0].PID)
	}
	if got[0].DriftSigma <= 4.0 {
		t.Errorf("drift_sigma must exceed threshold: %v", got[0].DriftSigma)
	}
}

func TestObserve_SameDivergenceDoesNotReEmit(t *testing.T) {
	tr := newTestTracker()
	feedCohort(tr, 0xfeed, 1000, 4, 8, 10_000_000, 2, 100_000_000)
	tr.Compute(baseTime)
	first := tr.Compute(baseTime.Add(1 * time.Second))
	if len(first) != 1 {
		t.Fatalf("first emission expected: %+v", first)
	}
	// Same data; second emission must be suppressed.
	if got := tr.Compute(baseTime.Add(2 * time.Second)); len(got) != 0 {
		t.Fatalf("same episode re-emit must be suppressed: %+v", got)
	}
}

func TestObserve_DivergenceClearsAndRearms(t *testing.T) {
	tr := newTestTracker()
	feedCohort(tr, 0xfeed, 1000, 4, 8, 10_000_000, 2, 100_000_000)
	tr.Compute(baseTime)
	first := tr.Compute(baseTime.Add(1 * time.Second))
	if len(first) != 1 {
		t.Fatalf("first emission expected: %+v", first)
	}
	// Cohort returns to healthy (rank 2 normal). Feed 8 healthy
	// samples per rank to overwrite the window.
	feedCohort(tr, 0xfeed, 1000, 4, 8, 10_000_000, -1, 0)
	if got := tr.Compute(baseTime.Add(2 * time.Second)); len(got) != 0 {
		t.Fatalf("healthy data must not emit: %+v", got)
	}
	// Re-diverge. Sustained ticks must rearm and emit again.
	feedCohort(tr, 0xfeed, 1000, 4, 8, 10_000_000, 2, 100_000_000)
	tr.Compute(baseTime.Add(3 * time.Second))
	got := tr.Compute(baseTime.Add(4 * time.Second))
	if len(got) != 1 {
		t.Fatalf("re-diverged episode must emit: %+v", got)
	}
}

func TestCompute_InsufficientSamplesSkipped(t *testing.T) {
	tr := newTestTracker()
	// Only 1 sample per rank — below minPerRankSamples (windowSize/4 = 2).
	for r := 0; r < 4; r++ {
		tr.Observe(1000, 0xfeed, uint32(r), 4, 10_000_000)
	}
	if got := tr.Compute(baseTime); len(got) != 0 {
		t.Fatalf("insufficient samples must skip cohort: %+v", got)
	}
}

func TestCompute_InsufficientRanksSkipped(t *testing.T) {
	tr := newTestTracker()
	// Only 2 ranks: MAD across 2 points is degenerate; skip.
	feedCohort(tr, 0xfeed, 1000, 2, 8, 10_000_000, -1, 0)
	if got := tr.Compute(baseTime); len(got) != 0 {
		t.Fatalf("<3 ranks must skip cohort: %+v", got)
	}
}

func TestObserve_IgnoresPidOrCommZero(t *testing.T) {
	tr := newTestTracker()
	tr.Observe(0, 0xfeed, 0, 4, 10_000_000)
	tr.Observe(1000, 0, 0, 4, 10_000_000)
	if tr.TrackedComms() != 0 {
		t.Fatalf("pid==0 or commIDHash==0 must be dropped: %d", tr.TrackedComms())
	}
}

func TestForget_DropsCommState(t *testing.T) {
	tr := newTestTracker()
	feedCohort(tr, 0xfeed, 1000, 4, 8, 10_000_000, -1, 0)
	feedCohort(tr, 0xbeef, 1001, 4, 8, 10_000_000, -1, 0)
	if tr.TrackedComms() != 2 {
		t.Fatalf("expected 2 comms; got %d", tr.TrackedComms())
	}
	tr.Forget(0xfeed)
	if tr.TrackedComms() != 1 {
		t.Fatalf("after Forget expected 1 comm; got %d", tr.TrackedComms())
	}
}

func TestMedianAndMAD(t *testing.T) {
	// median: [1, 3, 5] -> 3
	m := medianFloat64([]float64{1, 3, 5})
	if m != 3 {
		t.Errorf("median odd: got %v want 3", m)
	}
	// median: [1, 2, 3, 4] -> 2.5
	m = medianFloat64([]float64{1, 2, 3, 4})
	if math.Abs(m-2.5) > 1e-9 {
		t.Errorf("median even: got %v want 2.5", m)
	}
	// MAD: values [1, 2, 3, 4, 5], center=3, devs=[2,1,0,1,2] -> median=1 -> MAD=1.4826
	mad := medianAbsoluteDeviation([]float64{1, 2, 3, 4, 5}, 3)
	if math.Abs(mad-1.4826) > 1e-3 {
		t.Errorf("MAD: got %v want ~1.4826", mad)
	}
}
