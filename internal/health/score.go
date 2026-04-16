package health

import (
	"math"
	"time"
)

// Signals carries the four raw inputs to the health score. Each value is a
// ratio in [0,1] where 1.0 is healthy. The caller is responsible for deriving
// these from live sources (CUDA trace events, /proc, cgroups, etc.).
//
// NaN or Inf inputs are tolerated: Compute coerces them to 0.0 and reports
// the coercion via its []Anomaly return so the caller can log with its own
// rate limiter.
type Signals struct {
	Throughput float64
	Compute    float64
	Memory     float64
	CPU        float64
}

// Score is the output of a single health computation. Per-signal values are
// preserved for dashboard drill-down and match the OTLP attribute schema
// from pkg/contract.
type Score struct {
	Value        float64
	Throughput   float64
	Compute      float64
	Memory       float64
	CPU          float64
	WorkloadType string
	Timestamp    time.Time
}

// Anomaly reports a coerced input signal (NaN or Inf replaced with 0.0).
// Compute returns a slice of these so the caller can log with its own
// deduplication. A nil or empty slice means all inputs were clean.
type Anomaly struct {
	Signal string // "throughput" | "compute" | "memory" | "cpu"
	Reason string // "NaN" | "Inf"
}

// Weights are the four signal weights. They should sum to 1.0; Compute does
// not normalize, so misconfigured weights silently produce a bad score.
// The caller should validate via Config.Validate.
type Weights struct {
	Throughput float64 `yaml:"throughput"`
	Compute    float64 `yaml:"compute"`
	Memory     float64 `yaml:"memory"`
	CPU        float64 `yaml:"cpu"`
}

// Floor configures the smooth floor penalty.
//
// When the minimum of the four signals falls below DegradationZone, a penalty
// multiplier ramps linearly from 1.0 down to 0.5 as the minimum approaches
// PenaltyFloor. At or below PenaltyFloor the penalty caps at 0.5 (50% drag).
// The penalty multiplies the weighted sum — it does not touch individual
// signals.
type Floor struct {
	DegradationZone float64 `yaml:"degradation_zone"`
	PenaltyFloor    float64 `yaml:"penalty_floor"`
}

// Config carries the tunable parameters for score computation. Callers should
// construct via DefaultConfig() and override fields as needed, then call
// Validate.
type Config struct {
	Weights Weights `yaml:"weights"`
	Floor   Floor   `yaml:"floor"`
}

// DefaultConfig returns the canonical weights and floor values. Weights sum to
// 1.0; floor is the shape described in docs/health-score.md.
func DefaultConfig() Config {
	return Config{
		Weights: Weights{
			Throughput: 0.40,
			Compute:    0.25,
			Memory:     0.20,
			CPU:        0.15,
		},
		Floor: Floor{
			DegradationZone: 0.35,
			PenaltyFloor:    0.25,
		},
	}
}

// Validate returns an error if the config would produce garbage scores.
// Callers must call this once at startup; Compute does not re-validate on
// each call.
func (c Config) Validate() error {
	w := c.Weights
	sum := w.Throughput + w.Compute + w.Memory + w.CPU
	if math.Abs(sum-1.0) > 1e-6 {
		return &ConfigError{Field: "weights", Msg: "weights must sum to 1.0"}
	}
	// Each weight must be strictly positive. A zero weight degenerates the
	// score to 3 signals (or fewer) and defeats the purpose of peer-relative
	// comparison — every cluster peer must score the same 4 signals.
	if w.Throughput <= 0 || w.Compute <= 0 || w.Memory <= 0 || w.CPU <= 0 {
		return &ConfigError{Field: "weights", Msg: "each weight must be > 0"}
	}
	f := c.Floor
	if f.PenaltyFloor < 0 || f.PenaltyFloor >= f.DegradationZone {
		return &ConfigError{Field: "floor", Msg: "penalty_floor must be >= 0 and < degradation_zone"}
	}
	if f.DegradationZone > 1 {
		return &ConfigError{Field: "floor", Msg: "degradation_zone must be <= 1"}
	}
	return nil
}

// ConfigError identifies a specific invalid field.
type ConfigError struct {
	Field string
	Msg   string
}

func (e *ConfigError) Error() string { return "health.Config." + e.Field + ": " + e.Msg }

// Compute produces a Score from the given Signals and workload label. NaN or
// Inf inputs are coerced to 0.0 and reported via the returned anomalies. The
// final score is clamped to [0, 1].
//
// The function is pure: same inputs produce the same output. It does not
// log, does not touch the clock other than to record ts, and does not retain
// state.
func Compute(ts time.Time, sig Signals, workload string, cfg Config) (Score, []Anomaly) {
	var anomalies []Anomaly
	thr, a := sanitize("throughput", sig.Throughput)
	if a != nil {
		anomalies = append(anomalies, *a)
	}
	cmp, a := sanitize("compute", sig.Compute)
	if a != nil {
		anomalies = append(anomalies, *a)
	}
	mem, a := sanitize("memory", sig.Memory)
	if a != nil {
		anomalies = append(anomalies, *a)
	}
	cpu, a := sanitize("cpu", sig.CPU)
	if a != nil {
		anomalies = append(anomalies, *a)
	}

	thr = clamp01(thr)
	cmp = clamp01(cmp)
	mem = clamp01(mem)
	cpu = clamp01(cpu)

	raw := cfg.Weights.Throughput*thr +
		cfg.Weights.Compute*cmp +
		cfg.Weights.Memory*mem +
		cfg.Weights.CPU*cpu

	penalty := smoothFloorPenalty(minOf4(thr, cmp, mem, cpu), cfg.Floor)
	value := clamp01(raw * penalty)

	return Score{
		Value:        value,
		Throughput:   thr,
		Compute:      cmp,
		Memory:       mem,
		CPU:          cpu,
		WorkloadType: workload,
		Timestamp:    ts,
	}, anomalies
}

// smoothFloorPenalty returns the multiplier applied to the weighted sum.
//
// If minSignal >= DegradationZone, penalty = 1.0 (no drag).
// If minSignal <= PenaltyFloor, penalty = 0.5 (max drag).
// Between the two, penalty ramps linearly from 1.0 to 0.5.
//
// Defensive: a caller that skips Validate may pass a degenerate Floor where
// DegradationZone == PenaltyFloor. In that case span is zero, so we fall
// back to the boundary rule (>= -> 1.0, < -> 0.5) instead of dividing.
func smoothFloorPenalty(minSignal float64, f Floor) float64 {
	if minSignal >= f.DegradationZone {
		return 1.0
	}
	if minSignal <= f.PenaltyFloor {
		return 0.5
	}
	span := f.DegradationZone - f.PenaltyFloor
	if span <= 0 {
		return 0.5
	}
	progress := (minSignal - f.PenaltyFloor) / span // (0,1)
	return 0.5 + 0.5*progress
}

// sanitize returns (clean, nil) for finite values, or (0, anomaly) otherwise.
func sanitize(name string, v float64) (float64, *Anomaly) {
	if math.IsNaN(v) {
		return 0, &Anomaly{Signal: name, Reason: "NaN"}
	}
	if math.IsInf(v, 0) {
		return 0, &Anomaly{Signal: name, Reason: "Inf"}
	}
	return v, nil
}

// clamp01 clamps x to [0, 1]. NaN is mapped to 0.
func clamp01(x float64) float64 {
	if math.IsNaN(x) {
		return 0
	}
	if x < 0 {
		return 0
	}
	if x > 1 {
		return 1
	}
	return x
}

func minOf4(a, b, c, d float64) float64 {
	m := a
	if b < m {
		m = b
	}
	if c < m {
		m = c
	}
	if d < m {
		m = d
	}
	return m
}
