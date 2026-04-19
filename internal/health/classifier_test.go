package health

import (
	"testing"
	"time"
)

func TestClassifierConfig_Validate(t *testing.T) {
	tests := []struct {
		name    string
		cfg     ClassifierConfig
		wantErr bool
	}{
		{"default", DefaultClassifierConfig(), false},
		{"zero_hysteresis", ClassifierConfig{Hysteresis: 0}, false},
		{"negative", ClassifierConfig{Hysteresis: -0.01}, true},
		{"too_large", ClassifierConfig{Hysteresis: 0.5}, true},
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

// Below threshold -> straggler; above threshold+hysteresis -> recover.
// Verifies the basic state machine.
func TestClassify_BasicStragglerAndRecover(t *testing.T) {
	c, _ := NewClassifier(DefaultClassifierConfig())
	now := time.Now()

	is, changed := c.Classify(0.62, 0.80, now)
	if !is || !changed {
		t.Fatalf("drop below threshold should trigger straggler+changed, got is=%v changed=%v", is, changed)
	}

	// Hysteresis = 0.02 so recovery needs score >= 0.82.
	is, changed = c.Classify(0.81, 0.80, now.Add(time.Second))
	if !is || changed {
		t.Fatalf("0.81 >= threshold but < threshold+hysteresis -> stay straggler, got is=%v changed=%v", is, changed)
	}

	is, changed = c.Classify(0.82, 0.80, now.Add(2*time.Second))
	if is || !changed {
		t.Fatalf("0.82 >= threshold+hysteresis -> recover+changed, got is=%v changed=%v", is, changed)
	}
}

// Hysteresis band prevents flap across the threshold.
func TestClassify_HysteresisPreventsFlap(t *testing.T) {
	c, _ := NewClassifier(DefaultClassifierConfig())
	now := time.Now()

	// Become straggler.
	c.Classify(0.75, 0.80, now)

	// Score oscillates around 0.80 (but within hysteresis band).
	scores := []float64{0.79, 0.81, 0.79, 0.81, 0.79}
	for i, s := range scores {
		is, changed := c.Classify(s, 0.80, now.Add(time.Duration(i)*time.Second))
		if !is {
			t.Fatalf("tick %d score=%v: expected still straggler, got healthy", i, s)
		}
		if changed {
			t.Fatalf("tick %d score=%v: expected no transition, got changed=true", i, s)
		}
	}

	// Finally cross threshold+hysteresis to recover.
	is, changed := c.Classify(0.83, 0.80, now.Add(10*time.Second))
	if is || !changed {
		t.Fatalf("should recover at 0.83 >= 0.82, got is=%v changed=%v", is, changed)
	}
}

// Staying healthy produces no transition events.
func TestClassify_StayHealthyNoChange(t *testing.T) {
	c, _ := NewClassifier(DefaultClassifierConfig())
	now := time.Now()
	for i := 0; i < 10; i++ {
		is, changed := c.Classify(0.9, 0.8, now.Add(time.Duration(i)*time.Second))
		if is {
			t.Fatal("unexpected straggler")
		}
		if changed {
			t.Fatalf("unexpected transition on tick %d", i)
		}
	}
}

// A score exactly at the threshold is NOT a straggler (AC: `score <
// threshold`, strict).
func TestClassify_ExactBoundary_NotStraggler(t *testing.T) {
	c, _ := NewClassifier(DefaultClassifierConfig())
	is, changed := c.Classify(0.80, 0.80, time.Now())
	if is {
		t.Fatal("score == threshold should NOT be straggler (strict <)")
	}
	if changed {
		t.Fatal("should not change on first healthy classification")
	}
}

func TestClassify_LastClassification(t *testing.T) {
	c, _ := NewClassifier(DefaultClassifierConfig())
	at := time.Date(2026, 4, 16, 12, 0, 0, 0, time.UTC)

	got := c.LastClassification()
	if got.IsStraggler || !got.ChangedAt.IsZero() {
		t.Fatalf("initial LastClassification = %+v, want zero", got)
	}

	c.Classify(0.50, 0.80, at)
	got = c.LastClassification()
	if !got.IsStraggler || !got.ChangedAt.Equal(at) {
		t.Fatalf("after straggler LastClassification = %+v, want {true, %v}", got, at)
	}
}

// DominantSignal picks the most-degraded signal by normalized drop.
func TestDominantSignal(t *testing.T) {
	baseline := Baselines{Throughput: 100, Compute: 1.0, Memory: 1.0, CPU: 1.0}

	cases := []struct {
		name    string
		current Baselines
		want    string
	}{
		{
			"throughput_dominant",
			Baselines{Throughput: 40, Compute: 0.9, Memory: 0.9, CPU: 0.9},
			"throughput",
		},
		{
			"compute_dominant",
			Baselines{Throughput: 95, Compute: 0.3, Memory: 0.9, CPU: 0.9},
			"compute",
		},
		{
			"memory_dominant",
			Baselines{Throughput: 95, Compute: 0.9, Memory: 0.2, CPU: 0.9},
			"memory",
		},
		{
			"cpu_dominant",
			Baselines{Throughput: 95, Compute: 0.9, Memory: 0.9, CPU: 0.1},
			"cpu",
		},
		{
			"all_healthy_unknown",
			Baselines{Throughput: 100, Compute: 1.0, Memory: 1.0, CPU: 1.0},
			"unknown",
		},
		{
			"all_above_baseline_unknown",
			Baselines{Throughput: 110, Compute: 1.1, Memory: 1.1, CPU: 1.1},
			"unknown",
		},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			got := DominantSignal(tc.current, baseline)
			if got != tc.want {
				t.Errorf("got %q, want %q", got, tc.want)
			}
		})
	}
}

// NormalizedDrop gives zero when baseline is zero (avoid div-by-zero).
func TestNormalizedDrop_ZeroBaseline(t *testing.T) {
	if got := normalizedDrop(0, 0.5); got != 0 {
		t.Fatalf("normalizedDrop(0, 0.5) = %v, want 0", got)
	}
	if got := normalizedDrop(-1, 0.5); got != 0 {
		t.Fatalf("normalizedDrop(-1, 0.5) = %v, want 0", got)
	}
}

// Benchmark: AC8 is <1ms per classify call. We target <100ns.
func BenchmarkClassify(b *testing.B) {
	c, _ := NewClassifier(DefaultClassifierConfig())
	now := time.Now()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		c.Classify(0.7, 0.8, now)
	}
}
