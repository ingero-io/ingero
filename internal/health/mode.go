package health

import (
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"
)

// DetectionMode labels how the agent arrived at its current classification
// threshold. The agent emits this as the `mode` attribute on the
// `ingero.agent.detection_mode` metric.
type DetectionMode string

const (
	// ModeFleet: Fleet reachable, quorum met, threshold fresh.
	ModeFleet DetectionMode = "fleet"
	// ModeFleetCached: Fleet reachable but quorum_met=false — use the
	// last-known cached value.
	ModeFleetCached DetectionMode = "fleet-cached"
	// ModeLocalCached: Fleet unreachable, cached threshold < max age.
	ModeLocalCached DetectionMode = "local-cached"
	// ModeLocalBaseline: Fleet unreachable, cached threshold stale; use
	// local EMA baseline with a conservative drop factor.
	ModeLocalBaseline DetectionMode = "local-baseline"
	// ModeNone: Fleet unreachable and the local baseliner is still
	// calibrating — classification is suspended.
	ModeNone DetectionMode = "none"
)

// FleetReachabilitySource exposes the emitter's FleetReachable() view to
// the evaluator. The Emitter interface already implements this method
// directly.
type FleetReachabilitySource interface {
	FleetReachable() bool
}

// ModeConfig tunes the evaluator's age thresholds and local-baseline
// fallback factor.
type ModeConfig struct {
	// FreshMaxAge: a cached threshold older than this is no longer
	// "fresh" — evaluator steps down to local-cached / local-baseline.
	// Default 2x PushInterval (conservatively applied elsewhere).
	FreshMaxAge time.Duration `yaml:"fresh_max_age"`
	// CachedMaxAge: a cached threshold older than this is fully stale
	// and not even usable via local-cached.
	// Default 5 minutes.
	CachedMaxAge time.Duration `yaml:"cached_max_age"`
	// LocalBaselineFactor: the multiplier applied to the local baseline
	// score to produce a classification threshold when neither fleet nor
	// cached modes are available. Default 0.85 — an agent that drops
	// more than 15% below its own historical norm is treated as a
	// straggler in the absence of peer comparison.
	LocalBaselineFactor float64 `yaml:"local_baseline_factor"`
}

// DefaultModeConfig returns the canonical values.
func DefaultModeConfig() ModeConfig {
	return ModeConfig{
		FreshMaxAge:         20 * time.Second, // default push_interval 10s × 2
		CachedMaxAge:        5 * time.Minute,
		LocalBaselineFactor: 0.85,
	}
}

// Validate rejects invalid configurations.
func (c ModeConfig) Validate() error {
	if c.FreshMaxAge <= 0 {
		return fmt.Errorf("mode.fresh_max_age must be > 0: got %s", c.FreshMaxAge)
	}
	if c.CachedMaxAge <= c.FreshMaxAge {
		return fmt.Errorf("mode.cached_max_age (%s) must be > fresh_max_age (%s)", c.CachedMaxAge, c.FreshMaxAge)
	}
	if c.LocalBaselineFactor <= 0 || c.LocalBaselineFactor >= 1 {
		return fmt.Errorf("mode.local_baseline_factor must be in (0,1): got %v", c.LocalBaselineFactor)
	}
	return nil
}

// ModeEvaluator computes the current DetectionMode and the threshold to
// apply for straggler classification (Story 3.4).
type ModeEvaluator interface {
	// Evaluate returns the current DetectionMode and the threshold value
	// to use (if any). `ok` is false when mode == ModeNone — classification
	// should be skipped.
	Evaluate(now time.Time) (mode DetectionMode, threshold float64, ok bool)
	// CurrentMode returns the most recently evaluated mode without
	// re-running the decision tree.
	CurrentMode() DetectionMode
}

// modeEvaluator is the production implementation. The decision tree:
//
//   1. Fresh cache + quorum met  -> ModeFleet
//   2. Cache present + quorum NOT met (or stale but still "fleet-cached"
//      window) -> ModeFleetCached
//   3. Fleet unreachable AND cache < CachedMaxAge -> ModeLocalCached
//   4. Fleet unreachable AND baseliner warm       -> ModeLocalBaseline
//   5. otherwise -> ModeNone
//
// "Fresh" means `now - cache.ReceivedAt <= FreshMaxAge`. "Cached" means
// `cache has ever been populated`. The local-baseline threshold is the
// mean of the fast EMA baselines times LocalBaselineFactor — a
// conservative substitute when no peer signal is available.
type modeEvaluator struct {
	cfg       ModeConfig
	cache     *ThresholdCache
	reachable FleetReachabilitySource
	baseliner Baseliner
	warmupMin int
	log       *slog.Logger

	mu      sync.Mutex
	current DetectionMode
}

// NewModeEvaluator constructs a modeEvaluator. warmupMin is the number
// of Baseliner samples required before ModeLocalBaseline is available
// (typically the same as the baseliner's WarmupSamples).
//
// All four dependencies are required. Pass a nil logger to use
// slog.Default().
func NewModeEvaluator(
	cfg ModeConfig,
	cache *ThresholdCache,
	reachable FleetReachabilitySource,
	baseliner Baseliner,
	warmupMin int,
	log *slog.Logger,
) (ModeEvaluator, error) {
	if err := cfg.Validate(); err != nil {
		return nil, err
	}
	if cache == nil {
		return nil, errors.New("mode: cache must not be nil")
	}
	if reachable == nil {
		return nil, errors.New("mode: reachable source must not be nil")
	}
	if baseliner == nil {
		return nil, errors.New("mode: baseliner must not be nil")
	}
	if warmupMin < 0 {
		warmupMin = 0
	}
	if log == nil {
		log = slog.Default()
	}
	return &modeEvaluator{
		cfg:       cfg,
		cache:     cache,
		reachable: reachable,
		baseliner: baseliner,
		warmupMin: warmupMin,
		log:       log,
		current:   ModeNone,
	}, nil
}

func (m *modeEvaluator) Evaluate(now time.Time) (DetectionMode, float64, bool) {
	m.mu.Lock()
	prev := m.current
	m.mu.Unlock()

	mode, threshold, ok := m.decide(now)

	m.mu.Lock()
	m.current = mode
	changed := prev != mode
	m.mu.Unlock()

	if changed {
		m.log.Info("detection mode transition",
			"prev", string(prev),
			"next", string(mode),
			"threshold", threshold,
			"ok", ok,
		)
	}
	return mode, threshold, ok
}

func (m *modeEvaluator) CurrentMode() DetectionMode {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.current
}

// decide is the pure decision tree. No mutation of evaluator state.
func (m *modeEvaluator) decide(now time.Time) (DetectionMode, float64, bool) {
	snap, havesnap := m.cache.Get()
	fleetReachable := m.reachable.FleetReachable()

	// Step 1: fresh cache + quorum met = ModeFleet.
	if havesnap && snap.QuorumMet && isFresh(now, snap.ReceivedAt, m.cfg.FreshMaxAge) {
		return ModeFleet, snap.Value, true
	}

	// Step 2: cache present, quorum NOT met, still within fleet-cached
	// window. "Window" here means: received from Fleet within CachedMaxAge.
	// We also reach this branch for a fresh snapshot that's quorum=false.
	if havesnap && !snap.QuorumMet && isFresh(now, snap.ReceivedAt, m.cfg.CachedMaxAge) {
		return ModeFleetCached, snap.Value, true
	}

	// Step 3: Fleet unreachable, cached value within CachedMaxAge.
	if !fleetReachable && havesnap && isFresh(now, snap.ReceivedAt, m.cfg.CachedMaxAge) {
		return ModeLocalCached, snap.Value, true
	}

	// Step 4: Fleet unreachable, cache stale or missing, baseliner warm.
	if !fleetReachable && m.baseliner.SampleCount() >= m.warmupMin {
		baseline := m.baseliner.Current()
		mean := (baseline.Throughput + baseline.Compute + baseline.Memory + baseline.CPU) / 4
		if mean <= 0 {
			return ModeNone, 0, false
		}
		return ModeLocalBaseline, mean * m.cfg.LocalBaselineFactor, true
	}

	return ModeNone, 0, false
}

func isFresh(now, at time.Time, maxAge time.Duration) bool {
	if at.IsZero() {
		return false
	}
	age := now.Sub(at)
	if age < 0 {
		// Clock backwards - treat as not fresh, like persistence layer.
		return false
	}
	return age <= maxAge
}
