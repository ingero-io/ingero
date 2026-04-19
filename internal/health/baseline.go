package health

import (
	"fmt"
	"log/slog"
	"math"
	"sync"
)

// RawObservation is one sample collected from live sources: raw CUDA
// throughput (kernels/sec, bytes/sec — the unit is caller-defined as long
// as it is stable over time) plus the three ratios already in [0,1].
//
// Baseliner normalizes the raw throughput against its running EMA to
// produce the throughput_ratio signal consumed by Compute.
type RawObservation struct {
	Throughput float64
	Compute    float64
	Memory     float64
	CPU        float64
}

// Baselines holds a per-signal snapshot of either the fast EMA or the
// hard-floor EMA at one point in time.
type Baselines struct {
	Throughput float64
	Compute    float64
	Memory     float64
	CPU        float64
}

// BaselineConfig tunes the two EMAs and the degradation warning rule.
type BaselineConfig struct {
	// FastAlpha is the smoothing factor for the adaptive baseline used to
	// normalize throughput. Higher = faster adaptation. Default 0.1.
	FastAlpha float64 `yaml:"fast_alpha"`
	// FloorAlpha is the smoothing factor for the long-memory hard-floor
	// baseline used for slow-degradation detection. Default 0.001.
	FloorAlpha float64 `yaml:"floor_alpha"`
	// WarningRatio triggers DegradationWarning when any signal's fast EMA
	// divided by hard-floor EMA falls below this ratio. Default 0.5.
	WarningRatio float64 `yaml:"warning_ratio"`
	// WarmupSamples gates DegradationWarning: it stays false until the
	// Baseliner has seen at least this many samples. Default 30.
	WarmupSamples int `yaml:"warmup_samples"`
	// FloorWarmthMin is the minimum hard-floor EMA value below which a
	// signal is treated as "not yet warmed" and excluded from the warning
	// ratio — prevents the warmup-edge instability where a tiny-but-nonzero
	// floor makes fast/floor explode. Default 0.01.
	FloorWarmthMin float64 `yaml:"floor_warmth_min"`
}

// DefaultBaselineConfig returns the canonical values from the story.
func DefaultBaselineConfig() BaselineConfig {
	return BaselineConfig{
		FastAlpha:      0.1,
		FloorAlpha:     0.001,
		WarningRatio:   0.5,
		WarmupSamples:  30,
		FloorWarmthMin: 0.01,
	}
}

// BaselineError is the typed error returned by Validate / Restore /
// NewBaseliner. Callers can use errors.Is / errors.As to distinguish it
// from other error kinds.
type BaselineError struct {
	Field string
	Msg   string
}

func (e *BaselineError) Error() string { return "health.Baseline." + e.Field + ": " + e.Msg }

// Validate rejects configurations that would produce garbage baselines.
func (c BaselineConfig) Validate() error {
	if c.FastAlpha <= 0 || c.FastAlpha >= 1 {
		return &BaselineError{Field: "fast_alpha", Msg: fmt.Sprintf("must be in (0,1): got %v", c.FastAlpha)}
	}
	if c.FloorAlpha <= 0 || c.FloorAlpha >= 1 {
		return &BaselineError{Field: "floor_alpha", Msg: fmt.Sprintf("must be in (0,1): got %v", c.FloorAlpha)}
	}
	if c.FloorAlpha >= c.FastAlpha {
		return &BaselineError{Field: "floor_alpha", Msg: fmt.Sprintf("(%v) must be < fast_alpha (%v)", c.FloorAlpha, c.FastAlpha)}
	}
	if c.WarningRatio <= 0 || c.WarningRatio > 1 {
		return &BaselineError{Field: "warning_ratio", Msg: fmt.Sprintf("must be in (0,1]: got %v", c.WarningRatio)}
	}
	if c.WarmupSamples < 0 {
		return &BaselineError{Field: "warmup_samples", Msg: fmt.Sprintf("must be >= 0: got %v", c.WarmupSamples)}
	}
	if c.FloorWarmthMin < 0 || c.FloorWarmthMin >= 1 {
		return &BaselineError{Field: "floor_warmth_min", Msg: fmt.Sprintf("must be in [0,1): got %v", c.FloorWarmthMin)}
	}
	return nil
}

// Baseliner tracks per-signal fast and hard-floor EMAs and surfaces them
// as the inputs to the health score. See doc.go for the role in the
// broader pipeline.
//
// All methods are safe for concurrent use — internal state is protected by
// an RWMutex.
type Baseliner interface {
	// Update folds one observation into both EMAs and increments the
	// sample counter. Bias correction is applied transparently.
	Update(obs RawObservation)
	// Signals converts a raw observation into Signals suitable for
	// Compute. Throughput is normalized against the current fast EMA.
	//
	// Call order: Signals(obs) BEFORE Update(obs). Using Signals(obs)
	// after Update(obs) means the baseline already contains the current
	// observation (alpha=0.1 pulls it 10% toward current), biasing the
	// ratio toward 1.0.
	//
	// During calibration (baseline == 0), the returned Throughput is 0.
	// Callers must gate classification on the state machine being
	// ACTIVE — not on the score alone.
	Signals(obs RawObservation) Signals
	// Current returns the bias-corrected fast EMA.
	Current() Baselines
	// HardFloor returns the bias-corrected slow EMA.
	HardFloor() Baselines
	// DegradationWarning returns true when any signal's fast EMA divided
	// by its hard-floor EMA is below WarningRatio, after warmup, and the
	// hard floor is above FloorWarmthMin (so tiny transients don't fire).
	// Edge-triggered slog.Warn fires once per false->true transition.
	DegradationWarning() bool
	// SampleCount is the total number of observations folded in.
	SampleCount() int
	// Reset clears both EMAs and the sample counter. Used when the state
	// machine transitions back to CALIBRATING.
	Reset()
	// Snapshot serializes internal state for persistence (Story 2.4).
	Snapshot() PersistedState
	// Restore replaces internal state from a snapshot. Returns an error
	// only if the snapshot is malformed, contains non-finite values, or
	// was produced under different alphas.
	Restore(s PersistedState) error
}

// PersistedState is the on-disk representation produced by Snapshot. It is
// kept minimal and versioned so file format evolution is safe. The JSON
// encoding lives in persist.go (Story 2.4).
//
// FastAlpha and FloorAlpha are carried so that Restore can reject
// snapshots produced under a different configuration — restoring a raw
// EMA built with alpha_old under a runtime with alpha_new would silently
// corrupt the bias correction.
type PersistedState struct {
	SchemaVersion int       `json:"schema_version"`
	SampleCount   int       `json:"sample_count"`
	FastAlpha     float64   `json:"fast_alpha"`
	FloorAlpha    float64   `json:"floor_alpha"`
	FastEMA       Baselines `json:"fast_ema"`
	HardFloor     Baselines `json:"hard_floor"`
}

// emaBaseliner is the production implementation of Baseliner. It stores the
// raw (uncorrected) EMA values and applies bias correction on read so that
// Snapshot / Restore are straightforward JSON.
type emaBaseliner struct {
	cfg BaselineConfig
	log *slog.Logger

	mu          sync.RWMutex
	fast        Baselines
	floor       Baselines
	sampleCount int
	// warningLatch tracks the previous DegradationWarning() return so we
	// can fire slog.Warn exactly once on the false->true edge.
	warningLatch bool
}

// NewBaseliner constructs an emaBaseliner. Returns an error if cfg is
// invalid. Pass a nil logger to use slog.Default().
func NewBaseliner(cfg BaselineConfig, log *slog.Logger) (Baseliner, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if log == nil {
		log = slog.Default()
	}
	return &emaBaseliner{cfg: cfg, log: log}, nil
}

func (b *emaBaseliner) Update(obs RawObservation) {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.sampleCount++
	b.fast = emaStep(b.fast, obs, b.cfg.FastAlpha)
	b.floor = emaStep(b.floor, obs, b.cfg.FloorAlpha)
}

func (b *emaBaseliner) Signals(obs RawObservation) Signals {
	cur := b.Current() // bias-corrected, RLock-protected internally
	var thrRatio float64
	throughput := cleanFinite(obs.Throughput)
	if cur.Throughput > 0 {
		thrRatio = throughput / cur.Throughput
	}
	return Signals{
		Throughput: thrRatio,
		Compute:    obs.Compute,
		Memory:     obs.Memory,
		CPU:        obs.CPU,
	}
}

func (b *emaBaseliner) Current() Baselines {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return biasCorrect(b.fast, b.cfg.FastAlpha, b.sampleCount)
}

func (b *emaBaseliner) HardFloor() Baselines {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return biasCorrect(b.floor, b.cfg.FloorAlpha, b.sampleCount)
}

func (b *emaBaseliner) DegradationWarning() bool {
	b.mu.Lock() // write-lock: we may mutate warningLatch
	defer b.mu.Unlock()

	if b.sampleCount < b.cfg.WarmupSamples {
		b.warningLatch = false
		return false
	}
	fast := biasCorrect(b.fast, b.cfg.FastAlpha, b.sampleCount)
	floor := biasCorrect(b.floor, b.cfg.FloorAlpha, b.sampleCount)

	// A signal contributes to the warning only when its hard floor is
	// "warm" — i.e. large enough that fast/floor is a meaningful ratio.
	// This prevents the warmup-edge case where a tiny-but-nonzero floor
	// produces an unstable ratio.
	min := b.cfg.FloorWarmthMin
	triggered := ratioBelowWarm(fast.Throughput, floor.Throughput, min, b.cfg.WarningRatio) ||
		ratioBelowWarm(fast.Compute, floor.Compute, min, b.cfg.WarningRatio) ||
		ratioBelowWarm(fast.Memory, floor.Memory, min, b.cfg.WarningRatio) ||
		ratioBelowWarm(fast.CPU, floor.CPU, min, b.cfg.WarningRatio)

	if triggered && !b.warningLatch {
		b.log.Warn("health degradation warning",
			"fast", fast,
			"hard_floor", floor,
			"sample_count", b.sampleCount,
		)
	}
	b.warningLatch = triggered
	return triggered
}

func (b *emaBaseliner) SampleCount() int {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return b.sampleCount
}

func (b *emaBaseliner) Reset() {
	b.mu.Lock()
	defer b.mu.Unlock()
	b.fast = Baselines{}
	b.floor = Baselines{}
	b.sampleCount = 0
	b.warningLatch = false
}

func (b *emaBaseliner) Snapshot() PersistedState {
	b.mu.RLock()
	defer b.mu.RUnlock()
	return PersistedState{
		SchemaVersion: 1,
		SampleCount:   b.sampleCount,
		FastAlpha:     b.cfg.FastAlpha,
		FloorAlpha:    b.cfg.FloorAlpha,
		FastEMA:       b.fast,
		HardFloor:     b.floor,
	}
}

func (b *emaBaseliner) Restore(s PersistedState) error {
	if s.SchemaVersion != 1 {
		return &BaselineError{Field: "snapshot", Msg: fmt.Sprintf("unsupported schema_version %d", s.SchemaVersion)}
	}
	if s.SampleCount < 0 {
		return &BaselineError{Field: "snapshot", Msg: fmt.Sprintf("negative sample_count %d", s.SampleCount)}
	}
	if !baselinesFinite(s.FastEMA) || !baselinesFinite(s.HardFloor) {
		return &BaselineError{Field: "snapshot", Msg: "non-finite value in fast_ema or hard_floor"}
	}
	// Alpha mismatch: restoring a raw EMA built under different alphas would
	// silently corrupt the bias correction. A zero alpha in the snapshot
	// means a pre-versioned file — reject it as well.
	if !alphasMatch(s.FastAlpha, b.cfg.FastAlpha) || !alphasMatch(s.FloorAlpha, b.cfg.FloorAlpha) {
		return &BaselineError{
			Field: "snapshot",
			Msg: fmt.Sprintf("alpha mismatch: snapshot fast=%v floor=%v, runtime fast=%v floor=%v",
				s.FastAlpha, s.FloorAlpha, b.cfg.FastAlpha, b.cfg.FloorAlpha),
		}
	}
	// Cross-field sanity: each Baselines field is a well-behaved finite
	// non-negative value. The stronger "floor within N× of fast" check is
	// not enforced — legitimate restores during healthy warmup can have
	// floor arbitrarily below fast.
	if !baselinesNonNegative(s.FastEMA) || !baselinesNonNegative(s.HardFloor) {
		return &BaselineError{Field: "snapshot", Msg: "negative value in fast_ema or hard_floor"}
	}

	b.mu.Lock()
	defer b.mu.Unlock()
	b.sampleCount = s.SampleCount
	b.fast = s.FastEMA
	b.floor = s.HardFloor
	b.warningLatch = false
	return nil
}

// emaStep folds one observation into a Baselines struct using the standard
// EMA recurrence s_t = alpha*x + (1-alpha)*s_{t-1}. NaN/Inf inputs are
// coerced to 0 — the baseline is a stability anchor, not a detector.
func emaStep(prev Baselines, obs RawObservation, alpha float64) Baselines {
	return Baselines{
		Throughput: emaUpdate(prev.Throughput, cleanFinite(obs.Throughput), alpha),
		Compute:    emaUpdate(prev.Compute, cleanFinite(obs.Compute), alpha),
		Memory:     emaUpdate(prev.Memory, cleanFinite(obs.Memory), alpha),
		CPU:        emaUpdate(prev.CPU, cleanFinite(obs.CPU), alpha),
	}
}

func emaUpdate(prev, x, alpha float64) float64 {
	return alpha*x + (1-alpha)*prev
}

// biasCorrect applies s_hat_t = s_t / (1 - (1-alpha)^t) so that samples
// near t=0 aren't pulled toward zero. Returns zero baselines when
// sampleCount is zero (nothing observed yet).
func biasCorrect(raw Baselines, alpha float64, t int) Baselines {
	if t <= 0 {
		return Baselines{}
	}
	correction := 1 - math.Pow(1-alpha, float64(t))
	if correction <= 0 {
		return Baselines{}
	}
	inv := 1 / correction
	return Baselines{
		Throughput: raw.Throughput * inv,
		Compute:    raw.Compute * inv,
		Memory:     raw.Memory * inv,
		CPU:        raw.CPU * inv,
	}
}

// ratioBelowWarm returns true iff fast/floor < warn AND floor is warm.
// Unwarmed (floor below warmthMin) signals never contribute.
func ratioBelowWarm(fast, floor, warmthMin, warn float64) bool {
	if floor < warmthMin {
		return false
	}
	return fast/floor < warn
}

func cleanFinite(x float64) float64 {
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return 0
	}
	return x
}

func baselinesFinite(b Baselines) bool {
	return isFinite(b.Throughput) && isFinite(b.Compute) && isFinite(b.Memory) && isFinite(b.CPU)
}

func baselinesNonNegative(b Baselines) bool {
	return b.Throughput >= 0 && b.Compute >= 0 && b.Memory >= 0 && b.CPU >= 0
}

func isFinite(x float64) bool {
	return !math.IsNaN(x) && !math.IsInf(x, 0)
}

// alphasMatch accepts bit-identical equality or agreement to within 1e-9.
// Snapshots stored as JSON float64 round-trip exactly, so exact equality
// is the common case; the tolerance guards against YAML parsers that
// introduce sub-ULP drift.
func alphasMatch(a, b float64) bool {
	if a == b {
		return true
	}
	return math.Abs(a-b) < 1e-9
}
