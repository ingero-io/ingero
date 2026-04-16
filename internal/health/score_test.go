package health

import (
	"math"
	"testing"
	"time"
)

var testTS = time.Date(2026, 4, 16, 10, 0, 0, 0, time.UTC)

func TestDefaultConfig_WeightsSumToOne(t *testing.T) {
	c := DefaultConfig()
	w := c.Weights
	sum := w.Throughput + w.Compute + w.Memory + w.CPU
	if math.Abs(sum-1.0) > 1e-9 {
		t.Fatalf("default weights sum = %v, want 1.0", sum)
	}
}

func TestConfig_Validate(t *testing.T) {
	tests := []struct {
		name    string
		cfg     Config
		wantErr bool
	}{
		{"defaults_ok", DefaultConfig(), false},
		{
			"weights_sum_wrong",
			Config{
				Weights: Weights{0.5, 0.5, 0.5, 0.5},
				Floor:   Floor{DegradationZone: 0.35, PenaltyFloor: 0.25},
			},
			true,
		},
		{
			"negative_weight",
			Config{
				Weights: Weights{-0.1, 0.35, 0.35, 0.40},
				Floor:   Floor{DegradationZone: 0.35, PenaltyFloor: 0.25},
			},
			true,
		},
		{
			"floor_inverted",
			Config{
				Weights: DefaultConfig().Weights,
				Floor:   Floor{DegradationZone: 0.25, PenaltyFloor: 0.35},
			},
			true,
		},
		{
			"penalty_floor_negative",
			Config{
				Weights: DefaultConfig().Weights,
				Floor:   Floor{DegradationZone: 0.35, PenaltyFloor: -0.01},
			},
			true,
		},
		{
			"degradation_zone_over_one",
			Config{
				Weights: DefaultConfig().Weights,
				Floor:   Floor{DegradationZone: 1.5, PenaltyFloor: 0.25},
			},
			true,
		},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.cfg.Validate()
			if tc.wantErr && err == nil {
				t.Fatal("expected error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestCompute_AllHealthy_NoPenalty(t *testing.T) {
	cfg := DefaultConfig()
	sig := Signals{0.9, 0.9, 0.9, 0.9}
	got, anoms := Compute(testTS, sig, "training", cfg)
	if len(anoms) != 0 {
		t.Fatalf("unexpected anomalies: %+v", anoms)
	}
	// All signals are well above DegradationZone so penalty = 1.0.
	// Score = 1.0 * (0.40+0.25+0.20+0.15)*0.9 = 0.9.
	if math.Abs(got.Value-0.9) > 1e-9 {
		t.Fatalf("Value = %v, want 0.9", got.Value)
	}
	if got.WorkloadType != "training" {
		t.Fatalf("WorkloadType = %q", got.WorkloadType)
	}
	if !got.Timestamp.Equal(testTS) {
		t.Fatalf("Timestamp = %v, want %v", got.Timestamp, testTS)
	}
}

func TestCompute_OneDegradedAboveZone_NoPenalty(t *testing.T) {
	cfg := DefaultConfig()
	// min signal 0.40 > 0.35 so penalty = 1.0.
	sig := Signals{0.40, 0.90, 0.90, 0.90}
	got, _ := Compute(testTS, sig, "", cfg)
	raw := 0.40*0.40 + 0.25*0.90 + 0.20*0.90 + 0.15*0.90
	if math.Abs(got.Value-raw) > 1e-9 {
		t.Fatalf("Value = %v, want %v (no penalty)", got.Value, raw)
	}
}

func TestCompute_BoundaryAt035_NoPenalty(t *testing.T) {
	cfg := DefaultConfig()
	sig := Signals{0.35, 0.80, 0.80, 0.80}
	got, _ := Compute(testTS, sig, "", cfg)
	raw := 0.40*0.35 + 0.25*0.80 + 0.20*0.80 + 0.15*0.80
	if math.Abs(got.Value-raw) > 1e-9 {
		t.Fatalf("at boundary min=0.35 penalty should be 1.0, got Value=%v want %v", got.Value, raw)
	}
}

func TestCompute_BoundaryAt025_MaxPenalty(t *testing.T) {
	cfg := DefaultConfig()
	sig := Signals{0.25, 0.80, 0.80, 0.80}
	got, _ := Compute(testTS, sig, "", cfg)
	raw := 0.40*0.25 + 0.25*0.80 + 0.20*0.80 + 0.15*0.80
	want := raw * 0.5
	if math.Abs(got.Value-want) > 1e-9 {
		t.Fatalf("at boundary min=0.25 penalty should be 0.5, got Value=%v want %v", got.Value, want)
	}
}

func TestCompute_PartialPenaltyLinearRamp(t *testing.T) {
	cfg := DefaultConfig()
	// min = 0.30 -> midpoint of [0.25, 0.35] -> progress = 0.5 -> penalty = 0.75.
	sig := Signals{0.30, 0.80, 0.80, 0.80}
	got, _ := Compute(testTS, sig, "", cfg)
	raw := 0.40*0.30 + 0.25*0.80 + 0.20*0.80 + 0.15*0.80
	want := raw * 0.75
	if math.Abs(got.Value-want) > 1e-9 {
		t.Fatalf("Value = %v, want %v (penalty 0.75)", got.Value, want)
	}
}

func TestCompute_BelowFloor_CapsAtHalfPenalty(t *testing.T) {
	cfg := DefaultConfig()
	sig := Signals{0.10, 0.80, 0.80, 0.80}
	got, _ := Compute(testTS, sig, "", cfg)
	raw := 0.40*0.10 + 0.25*0.80 + 0.20*0.80 + 0.15*0.80
	want := raw * 0.5
	if math.Abs(got.Value-want) > 1e-9 {
		t.Fatalf("below floor penalty must cap at 0.5, got Value=%v want %v", got.Value, want)
	}
}

func TestCompute_NaNCoercedAndReported(t *testing.T) {
	cfg := DefaultConfig()
	sig := Signals{math.NaN(), 0.90, 0.90, 0.90}
	got, anoms := Compute(testTS, sig, "", cfg)
	if got.Throughput != 0 {
		t.Fatalf("NaN throughput should be coerced to 0, got %v", got.Throughput)
	}
	if len(anoms) != 1 {
		t.Fatalf("expected 1 anomaly, got %d: %+v", len(anoms), anoms)
	}
	if anoms[0].Signal != "throughput" || anoms[0].Reason != "NaN" {
		t.Fatalf("wrong anomaly: %+v", anoms[0])
	}
}

func TestCompute_InfCoercedAndReported(t *testing.T) {
	cfg := DefaultConfig()
	sig := Signals{0.90, 0.90, math.Inf(1), 0.90}
	got, anoms := Compute(testTS, sig, "", cfg)
	if got.Memory != 0 {
		t.Fatalf("Inf memory should be coerced to 0, got %v", got.Memory)
	}
	if len(anoms) != 1 || anoms[0].Signal != "memory" || anoms[0].Reason != "Inf" {
		t.Fatalf("wrong anomaly: %+v", anoms)
	}
}

func TestCompute_NegativeInfCoerced(t *testing.T) {
	cfg := DefaultConfig()
	sig := Signals{0.90, math.Inf(-1), 0.90, 0.90}
	got, anoms := Compute(testTS, sig, "", cfg)
	if got.Compute != 0 {
		t.Fatalf("-Inf compute should be coerced to 0, got %v", got.Compute)
	}
	if len(anoms) != 1 || anoms[0].Signal != "compute" {
		t.Fatalf("wrong anomaly: %+v", anoms)
	}
}

func TestCompute_MultipleAnomalies(t *testing.T) {
	cfg := DefaultConfig()
	sig := Signals{math.NaN(), math.Inf(1), 0.90, math.NaN()}
	_, anoms := Compute(testTS, sig, "", cfg)
	if len(anoms) != 3 {
		t.Fatalf("expected 3 anomalies, got %d: %+v", len(anoms), anoms)
	}
}

func TestCompute_ClampsAbove1(t *testing.T) {
	cfg := DefaultConfig()
	sig := Signals{2.0, 2.0, 2.0, 2.0}
	got, _ := Compute(testTS, sig, "", cfg)
	if got.Value != 1.0 {
		t.Fatalf("Value should clamp to 1.0, got %v", got.Value)
	}
	if got.Throughput != 1.0 {
		t.Fatalf("Throughput should clamp to 1.0, got %v", got.Throughput)
	}
}

func TestCompute_ClampsBelow0(t *testing.T) {
	cfg := DefaultConfig()
	sig := Signals{-0.5, -0.5, -0.5, -0.5}
	got, _ := Compute(testTS, sig, "", cfg)
	if got.Value != 0 {
		t.Fatalf("Value should clamp to 0, got %v", got.Value)
	}
	if got.CPU != 0 {
		t.Fatalf("CPU should clamp to 0, got %v", got.CPU)
	}
}

func TestCompute_PerSignalValuesPreserved(t *testing.T) {
	cfg := DefaultConfig()
	sig := Signals{0.11, 0.22, 0.33, 0.44}
	got, _ := Compute(testTS, sig, "inference", cfg)
	if got.Throughput != 0.11 || got.Compute != 0.22 || got.Memory != 0.33 || got.CPU != 0.44 {
		t.Fatalf("per-signal values not preserved: %+v", got)
	}
}

func TestCompute_CustomWeights(t *testing.T) {
	// Evenly weighted config produces a simple average.
	cfg := Config{
		Weights: Weights{0.25, 0.25, 0.25, 0.25},
		Floor:   Floor{DegradationZone: 0.35, PenaltyFloor: 0.25},
	}
	if err := cfg.Validate(); err != nil {
		t.Fatalf("cfg invalid: %v", err)
	}
	sig := Signals{0.80, 0.60, 0.40, 0.20}
	got, _ := Compute(testTS, sig, "", cfg)
	// min=0.20 < 0.25 -> penalty 0.5. raw = 0.25*(0.8+0.6+0.4+0.2)=0.5. value=0.25.
	if math.Abs(got.Value-0.25) > 1e-9 {
		t.Fatalf("Value = %v, want 0.25", got.Value)
	}
}

func TestCompute_Deterministic(t *testing.T) {
	cfg := DefaultConfig()
	sig := Signals{0.72, 0.81, 0.63, 0.49}
	first, _ := Compute(testTS, sig, "training", cfg)
	second, _ := Compute(testTS, sig, "training", cfg)
	if first != second {
		t.Fatalf("Compute not deterministic: %+v vs %+v", first, second)
	}
}

func TestClamp01_NaN(t *testing.T) {
	if got := clamp01(math.NaN()); got != 0 {
		t.Fatalf("clamp01(NaN) = %v, want 0", got)
	}
}

func BenchmarkCompute(b *testing.B) {
	cfg := DefaultConfig()
	sig := Signals{0.72, 0.81, 0.63, 0.49}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_, _ = Compute(testTS, sig, "training", cfg)
	}
}
