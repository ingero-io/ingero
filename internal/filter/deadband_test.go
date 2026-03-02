package filter

import (
	"sync"
	"testing"
	"time"
)

func TestShouldEmit_DisabledByDefault(t *testing.T) {
	cfg := Config{DeadbandPct: 0}
	if !cfg.Disabled() {
		t.Fatal("expected Disabled()=true when DeadbandPct=0")
	}
	f := cfg.NewSnapshotFilter()
	if f != nil {
		t.Fatal("expected nil filter when disabled")
	}
	// Nil filter should always emit.
	for i := 0; i < 5; i++ {
		if !f.ShouldEmit(50, 60, 1000, 0, 1.0) {
			t.Fatalf("nil filter ShouldEmit returned false on call %d", i)
		}
	}
}

func TestShouldEmit_FirstCallAlwaysEmits(t *testing.T) {
	f := makeFilter(5.0, 0)
	if !f.ShouldEmit(50, 60, 1000, 0, 1.0) {
		t.Fatal("first call must always emit")
	}
}

func TestShouldEmit_WithinDeadbandSuppressed(t *testing.T) {
	f := makeFilter(5.0, 0)

	// First call: emit baseline.
	f.ShouldEmit(50, 60, 1000, 0, 2.0)

	// Small changes within 5% deadband — should be suppressed.
	// CPU 50 → 51 = 2% change (threshold = 50 * 5% = 2.5).
	// Mem 60 → 61 = 1.67% change (threshold = 60 * 5% = 3.0).
	if f.ShouldEmit(51, 61, 1000, 0, 2.0) {
		t.Fatal("small change within deadband should be suppressed")
	}
}

func TestShouldEmit_BeyondDeadbandEmits(t *testing.T) {
	f := makeFilter(5.0, 0)

	// Baseline.
	f.ShouldEmit(50, 60, 1000, 0, 2.0)

	// CPU 50 → 55 = 10% change (threshold = 50 * 5% = 2.5). Exceeds.
	if !f.ShouldEmit(55, 60, 1000, 0, 2.0) {
		t.Fatal("change beyond deadband should emit")
	}
}

func TestShouldEmit_AnyMetricTriggersEmit(t *testing.T) {
	f := makeFilter(5.0, 0)

	// Baseline.
	f.ShouldEmit(50, 60, 1000, 0, 2.0)

	// Only swap changes: 0 → 100. Other metrics unchanged.
	// Swap base = max(|0|, 1.0) = 1.0; threshold = 1.0 * 5% = 0.05. 100 > 0.05.
	if !f.ShouldEmit(50, 60, 1000, 100, 2.0) {
		t.Fatal("single metric exceeding deadband should trigger emit")
	}
}

func TestShouldEmit_HeartbeatForcesEmit(t *testing.T) {
	now := time.Now()
	f := makeFilterWithClock(5.0, 30*time.Second, &now)

	// Baseline at t=0.
	f.ShouldEmit(50, 60, 1000, 0, 2.0)

	// t=10s: small change, no heartbeat yet.
	now = now.Add(10 * time.Second)
	if f.ShouldEmit(50, 60, 1000, 0, 2.0) {
		t.Fatal("should suppress within deadband before heartbeat")
	}

	// t=30s: heartbeat fires even though values unchanged.
	now = now.Add(20 * time.Second)
	if !f.ShouldEmit(50, 60, 1000, 0, 2.0) {
		t.Fatal("heartbeat should force emit after interval")
	}

	// t=31s: suppressed again (heartbeat just reset).
	now = now.Add(1 * time.Second)
	if f.ShouldEmit(50, 60, 1000, 0, 2.0) {
		t.Fatal("should suppress right after heartbeat emit")
	}
}

func TestShouldEmit_ZeroBaseMetric(t *testing.T) {
	f := makeFilter(5.0, 0)

	// Baseline with zero swap.
	f.ShouldEmit(50, 60, 1000, 0, 0)

	// Swap 0 → 1: base = max(|0|, 1.0) = 1.0; threshold = 0.05; |1-0| = 1 > 0.05.
	if !f.ShouldEmit(50, 60, 1000, 1, 0) {
		t.Fatal("zero-to-nonzero swap transition should emit")
	}

	// Load 0 → 0.1: base = max(|0|, 1.0) = 1.0; threshold = 0.05; |0.1| = 0.1 > 0.05.
	f2 := makeFilter(5.0, 0)
	f2.ShouldEmit(50, 60, 1000, 0, 0)
	if !f2.ShouldEmit(50, 60, 1000, 0, 0.1) {
		t.Fatal("zero-to-nonzero load transition should emit")
	}
}

func TestShouldEmit_NilFilter(t *testing.T) {
	var f *SnapshotFilter
	if !f.ShouldEmit(99, 99, 1, 500, 50) {
		t.Fatal("nil filter must always return true")
	}
}

func TestShouldEmit_ConcurrentAccess(t *testing.T) {
	f := makeFilter(1.0, 0)
	f.ShouldEmit(50, 60, 1000, 0, 2.0) // baseline

	var wg sync.WaitGroup
	for i := 0; i < 100; i++ {
		wg.Add(1)
		go func(i int) {
			defer wg.Done()
			// Vary CPU to sometimes exceed deadband.
			cpu := 50.0 + float64(i%10)
			f.ShouldEmit(cpu, 60, 1000, 0, 2.0)
		}(i)
	}
	wg.Wait()
	// No race detector failure = pass.
}

func TestExceedsDeadband_EdgeCases(t *testing.T) {
	tests := []struct {
		name   string
		old    float64
		new    float64
		pct    float64
		expect bool
	}{
		{"identical values", 50, 50, 5, false},
		{"exactly at threshold", 100, 105, 5, false}, // |5| > 5.0? no, == not >
		{"just above threshold", 100, 105.01, 5, true},
		{"negative to positive", -5, 5, 5, true},           // |10| > 5*5%=0.25 → true
		{"zero old small change", 0, 0.04, 5, false},       // base=1.0; threshold=0.05; |0.04|=0.04 ≤ 0.05
		{"zero old at threshold", 0, 0.05, 5, false},       // |0.05| not > 0.05
		{"zero old above threshold", 0, 0.06, 5, true},     // |0.06| > 0.05
		{"large old tiny pct", 10000, 10001, 0.001, true},  // base=10000; threshold=0.1; |1|>0.1
		{"large old small change", 10000, 10000.05, 0.001, false}, // |0.05| ≤ 0.1
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := exceedsDeadband(tt.old, tt.new, tt.pct)
			if got != tt.expect {
				t.Errorf("exceedsDeadband(%v, %v, %v) = %v, want %v",
					tt.old, tt.new, tt.pct, got, tt.expect)
			}
		})
	}
}

func TestConfig_Disabled(t *testing.T) {
	tests := []struct {
		pct      float64
		disabled bool
	}{
		{0, true},
		{-1, true},
		{0.001, false},
		{5, false},
	}
	for _, tt := range tests {
		cfg := Config{DeadbandPct: tt.pct}
		if cfg.Disabled() != tt.disabled {
			t.Errorf("Config{DeadbandPct: %v}.Disabled() = %v, want %v",
				tt.pct, cfg.Disabled(), tt.disabled)
		}
	}
}

// --- helpers ---

func makeFilter(pct float64, heartbeat time.Duration) *SnapshotFilter {
	return &SnapshotFilter{
		deadbandPct: pct,
		heartbeat:   heartbeat,
		nowFn:       time.Now,
	}
}

func makeFilterWithClock(pct float64, heartbeat time.Duration, now *time.Time) *SnapshotFilter {
	return &SnapshotFilter{
		deadbandPct: pct,
		heartbeat:   heartbeat,
		nowFn:       func() time.Time { return *now },
	}
}
