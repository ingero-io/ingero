package sampling

import (
	"sync"
	"testing"
	"time"
)

// fakeClock is an injectable monotonic clock for deterministic time-based
// tests. Advance() moves it forward; Now() reads it.
type fakeClock struct {
	mu sync.Mutex
	t  time.Time
}

func newFakeClock(start time.Time) *fakeClock {
	return &fakeClock{t: start}
}

func (c *fakeClock) Now() time.Time {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.t
}

func (c *fakeClock) Advance(d time.Duration) {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.t = c.t.Add(d)
}

func TestShouldEmit_ModeBypass(t *testing.T) {
	cases := []struct {
		name string
		mode string
	}{
		{"training mode bypass", "training"},
		{"unknown mode bypass", "unknown"},
		{"empty mode bypass", ""},
		{"misspelled mode bypass", "Inference"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			// rate=0 would normally drop everything, but bypass overrides.
			s := New(tc.mode, 0.0, 30*time.Second)
			for i := 0; i < 1000; i++ {
				if !s.ShouldEmit() {
					t.Fatalf("non-inference mode %q dropped event at iteration %d (expected always emit)", tc.mode, i)
				}
			}
			// SetDegraded must be a no-op in bypass; subsequent ShouldEmit
			// calls still return true.
			s.SetDegraded(true)
			if !s.ShouldEmit() {
				t.Fatalf("non-inference mode %q dropped after SetDegraded(true)", tc.mode)
			}
			s.SetDegraded(false)
			if !s.ShouldEmit() {
				t.Fatalf("non-inference mode %q dropped after SetDegraded(false)", tc.mode)
			}
		})
	}
}

func TestShouldEmit_InferenceHealthyRateZero(t *testing.T) {
	clk := newFakeClock(time.Unix(0, 0))
	s := newWithClock("inference", 0.0, 30*time.Second, clk.Now)
	for i := 0; i < 1000; i++ {
		if s.ShouldEmit() {
			t.Fatalf("rate=0 admitted event at iteration %d (expected never)", i)
		}
	}
}

func TestShouldEmit_InferenceHealthyRateOne(t *testing.T) {
	clk := newFakeClock(time.Unix(0, 0))
	s := newWithClock("inference", 1.0, 30*time.Second, clk.Now)
	for i := 0; i < 1000; i++ {
		if !s.ShouldEmit() {
			t.Fatalf("rate=1 dropped event at iteration %d (expected always)", i)
		}
	}
}

func TestShouldEmit_InferenceDegradedAlwaysEmits(t *testing.T) {
	clk := newFakeClock(time.Unix(0, 0))
	// rate=0 is the strongest test: only the degraded override should produce
	// admissions.
	s := newWithClock("inference", 0.0, 30*time.Second, clk.Now)
	s.SetDegraded(true)
	for i := 0; i < 1000; i++ {
		if !s.ShouldEmit() {
			t.Fatalf("degraded state dropped event at iteration %d", i)
		}
	}
}

func TestShouldEmit_InferenceCooldownActive(t *testing.T) {
	clk := newFakeClock(time.Unix(0, 0))
	s := newWithClock("inference", 0.0, 30*time.Second, clk.Now)

	s.SetDegraded(true)
	s.SetDegraded(false) // -> cooldown, ends at +30s

	// 29s in: still in cooldown, must always emit.
	clk.Advance(29 * time.Second)
	for i := 0; i < 1000; i++ {
		if !s.ShouldEmit() {
			t.Fatalf("cooldown dropped event at iteration %d (cooldown should be 100%%)", i)
		}
	}
}

func TestShouldEmit_InferenceCooldownExpired(t *testing.T) {
	clk := newFakeClock(time.Unix(0, 0))
	s := newWithClock("inference", 0.0, 30*time.Second, clk.Now)

	s.SetDegraded(true)
	s.SetDegraded(false)

	// Past cooldown end: should revert to healthy and rate=0 takes over.
	clk.Advance(31 * time.Second)
	for i := 0; i < 1000; i++ {
		if s.ShouldEmit() {
			t.Fatalf("post-cooldown admitted event at iteration %d (expected revert to rate=0)", i)
		}
	}
}

func TestSetDegraded_NoOpFalseToFalse(t *testing.T) {
	clk := newFakeClock(time.Unix(0, 0))
	s := newWithClock("inference", 0.0, 30*time.Second, clk.Now)

	// Healthy -> healthy via SetDegraded(false) repeatedly. Internal state
	// must remain healthy and cooldownEnd must stay zero (no transition).
	for i := 0; i < 5; i++ {
		s.SetDegraded(false)
	}

	s.mu.Lock()
	if s.state != stateHealthy {
		s.mu.Unlock()
		t.Fatalf("state moved off healthy after SetDegraded(false) loop: got %v", s.state)
	}
	if !s.cooldownEnd.IsZero() {
		s.mu.Unlock()
		t.Fatalf("cooldownEnd was set without a degraded->healthy edge: %v", s.cooldownEnd)
	}
	s.mu.Unlock()

	// And it still drops at rate=0.
	if s.ShouldEmit() {
		t.Fatalf("rate=0 healthy admitted after SetDegraded(false) loop")
	}
}

func TestSetDegraded_NoOpTrueToTrue(t *testing.T) {
	clk := newFakeClock(time.Unix(0, 0))
	s := newWithClock("inference", 0.0, 30*time.Second, clk.Now)

	s.SetDegraded(true)
	// Repeated true: must remain in degraded with no transition side-effects.
	for i := 0; i < 5; i++ {
		s.SetDegraded(true)
	}

	s.mu.Lock()
	if s.state != stateDegraded {
		s.mu.Unlock()
		t.Fatalf("state moved off degraded after SetDegraded(true) loop: got %v", s.state)
	}
	s.mu.Unlock()
}

func TestShouldEmit_HealthyAdmitRateApproximatesConfigured(t *testing.T) {
	clk := newFakeClock(time.Unix(0, 0))
	const trials = 10000
	const rate = 0.01
	s := newWithClock("inference", rate, 30*time.Second, clk.Now)

	admitted := 0
	for i := 0; i < trials; i++ {
		if s.ShouldEmit() {
			admitted++
		}
	}

	// Loose bounds: expected mean is 100, stddev ~10. Bounds [50,200] are
	// ~5 stddev wide — designed to never flake under math/rand/v2 auto-seeding.
	if admitted < 50 || admitted > 200 {
		t.Fatalf("healthy admit count %d outside [50,200] for rate=%v over %d trials", admitted, rate, trials)
	}
}

func TestShouldEmit_FullCycleRecoversToHealthy(t *testing.T) {
	clk := newFakeClock(time.Unix(0, 0))
	s := newWithClock("inference", 1.0, 5*time.Second, clk.Now)

	if !s.ShouldEmit() {
		t.Fatalf("initial healthy with rate=1 dropped")
	}

	s.SetDegraded(true)
	if !s.ShouldEmit() {
		t.Fatalf("degraded dropped")
	}

	s.SetDegraded(false)
	clk.Advance(2 * time.Second) // mid-cooldown
	if !s.ShouldEmit() {
		t.Fatalf("mid-cooldown dropped (rate should be 100%%)")
	}

	clk.Advance(10 * time.Second) // past cooldown end
	// Now back to healthy with rate=1.0 — should still emit, but via the
	// healthy-rate path, not cooldown override.
	if !s.ShouldEmit() {
		t.Fatalf("post-cooldown healthy at rate=1 dropped")
	}
	s.mu.Lock()
	if s.state != stateHealthy {
		s.mu.Unlock()
		t.Fatalf("expected stateHealthy after cooldown expiry, got %v", s.state)
	}
	s.mu.Unlock()
}
