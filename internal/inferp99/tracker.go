// Package inferp99 maintains a rolling p99 of inference step durations
// per workload and emits a Breach when the rolling p99 exceeds a frozen
// baseline by a configurable ratio sustained across consecutive ticks.
//
// Distinct from the per-step InferenceOutlier classifier in
// internal/infer: that path fires on a single step crossing the
// baseline-p95 by 1.5x/2x/3x. This path fires on the SLO (rolling p99)
// crossing a frozen baseline by 1.5x sustained for 15s — soft enough
// to drive `drain_lb_endpoint` (shift traffic) before any individual
// request exceeds its SLO budget visibly.
//
// EE-side dispatch: the orchestrator's InferenceSloBreach chain
// (`drain_lb_endpoint` -> `pod_drain` -> `process_recycle`) shipped in
// Phase 15 with `chain_for(InferenceSloBreach) = [...]` and
// chain-active suppression. This package emits the wire message that
// activates that chain.
//
// Baseline freeze rationale: after `warmupSamples` healthy samples
// land, the baseline_p99 is captured and frozen. A continually-
// updating baseline would chase a sustained breach upward and silence
// the signal. The frozen baseline is the operator's implicit "this is
// the normal p99 the SLO was sized against" reference.
package inferp99

import (
	"math"
	"sort"
	"sync"
	"time"
)

// Defaults match the EE-side dispatch comment (uds.rs:184: "default
// 1.5x per typical SLO-breach detection literature") and the
// observed-good shapes from Theme 4 trackers (3-sustain pattern,
// 60s suppression rearm).
const (
	DefaultBreachRatio    = 1.5
	DefaultClearRatio     = 1.1
	DefaultSustainTicks   = 3
	DefaultWarmupSamples  = 200
	DefaultWindowDuration = 60 * time.Second
	DefaultMaxSamples     = 10_000
	DefaultRearmDuration  = 60 * time.Second
)

// Config bundles the per-tracker thresholds. Defaults applied at
// NewTracker time; tests can pass an explicit Config to drive
// shorter sustain windows without waiting wall-clock seconds.
type Config struct {
	// BreachRatio: emit when current_p99 / baseline_p99 > BreachRatio
	// for SustainTicks consecutive Check() calls.
	BreachRatio float64
	// ClearRatio: re-arm (clear suppression) once current_p99 /
	// baseline_p99 < ClearRatio. Hysteresis around BreachRatio so a
	// workload hovering at exactly the threshold doesn't flap.
	ClearRatio float64
	// SustainTicks: consecutive Check() calls above BreachRatio
	// required before emission. Sized against the watcher's tick
	// cadence (5s default) so 3 ticks = 15s of sustained breach.
	SustainTicks int
	// WarmupSamples: minimum Observe count before baseline_p99 is
	// captured. Below this, the tracker neither emits nor classifies.
	WarmupSamples int
	// WindowDuration: rolling-window upper bound on sample age. Samples
	// older than now-WindowDuration are pruned on each Observe.
	WindowDuration time.Duration
	// MaxSamples: hard cap on retained samples; protects very-high-
	// throughput workloads from unbounded memory. When the cap is
	// reached, the oldest samples are dropped FIFO.
	MaxSamples int
	// RearmDuration: minimum time between successive emissions on the
	// same tracker, independent of the clear-ratio path. Guards
	// against the orchestrator's chain re-firing before its first
	// action has had a chance to settle.
	RearmDuration time.Duration
}

func (c Config) resolved() Config {
	if c.BreachRatio <= 0 {
		c.BreachRatio = DefaultBreachRatio
	}
	if c.ClearRatio <= 0 {
		c.ClearRatio = DefaultClearRatio
	}
	if c.SustainTicks <= 0 {
		c.SustainTicks = DefaultSustainTicks
	}
	if c.WarmupSamples <= 0 {
		c.WarmupSamples = DefaultWarmupSamples
	}
	if c.WindowDuration <= 0 {
		c.WindowDuration = DefaultWindowDuration
	}
	if c.MaxSamples <= 0 {
		c.MaxSamples = DefaultMaxSamples
	}
	if c.RearmDuration <= 0 {
		c.RearmDuration = DefaultRearmDuration
	}
	return c
}

// DefaultConfig returns Config{} resolved -- all defaults.
func DefaultConfig() Config { return Config{}.resolved() }

// Breach is the per-emission payload. The owning workload identity
// (PID, cgroup hash, stream, phase) is layered on top by the caller;
// this struct carries only the SLO-breach metrics so the tracker
// stays workload-key-agnostic.
type Breach struct {
	// CurrentP99Ns is the p99 over the rolling window at emission time.
	CurrentP99Ns uint64
	// BaselineP99Ns is the frozen baseline_p99 captured at warmup.
	BaselineP99Ns uint64
	// Ratio is CurrentP99Ns / BaselineP99Ns at emission time, redundant
	// with the two fields above but pinned so the consumer logs the
	// exact value the agent gated on.
	Ratio float64
	// At is the wall-clock time the emission fired. Carried in the
	// struct so the watcher's wire emit uses the detection time, not
	// the slightly-later UDS write time.
	At time.Time
}

// Tracker holds the rolling sample window + the sustain state machine
// for one workload. Caller (typically internal/infer's workloadMap)
// allocates one Tracker per workload and feeds it via Observe; the
// watcher goroutine periodically calls CheckAt to drain breaches.
type Tracker struct {
	mu  sync.Mutex
	cfg Config

	// samples is the rolling window. Two parallel slices instead of a
	// []sampleEntry struct so the sort path can copy a flat float64
	// slice without extracting fields; cost is one extra slice header
	// (~24 bytes) for the timestamp parallel array.
	stepsNs    []float64
	timestamps []time.Time

	// baselineP99Ns is captured once after warmup completes; never
	// updated thereafter. Zero means "not yet warmed".
	baselineP99Ns float64

	// sustainCount counts consecutive CheckAt calls where the ratio
	// exceeded BreachRatio. Reset when the ratio drops below
	// ClearRatio or after an emission fires.
	sustainCount int

	// suppressedUntil is the wall-clock time after which a fresh
	// emission is allowed. Zero before the first emission.
	suppressedUntil time.Time
}

// NewTracker constructs a Tracker with cfg's defaults filled in.
func NewTracker(cfg Config) *Tracker {
	return &Tracker{cfg: cfg.resolved()}
}

// Observe records one step-duration sample. Caller passes the wall-
// clock time of the sample (typically the sync event's timestamp);
// stale samples older than WindowDuration relative to `at` are pruned
// FIFO on every call so the window stays time-bounded.
//
// Below WarmupSamples, samples are still recorded so the baseline
// captures the right distribution; baseline freeze happens lazily
// inside CheckAt when len(stepsNs) >= WarmupSamples.
func (t *Tracker) Observe(stepNs float64, at time.Time) {
	if stepNs <= 0 {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	t.pruneLocked(at)
	t.stepsNs = append(t.stepsNs, stepNs)
	t.timestamps = append(t.timestamps, at)

	// Cap on retained samples. When exceeded, drop the oldest FIFO
	// before letting the slice grow further. The MaxSamples cap is a
	// safety upper bound, not the primary windowing mechanism (time
	// is); for typical workloads pruneLocked already keeps len well
	// under MaxSamples.
	if len(t.stepsNs) > t.cfg.MaxSamples {
		drop := len(t.stepsNs) - t.cfg.MaxSamples
		t.stepsNs = t.stepsNs[drop:]
		t.timestamps = t.timestamps[drop:]
	}
}

// pruneLocked drops samples older than now-WindowDuration. Caller
// holds t.mu. Linear scan from the head; the timestamp slice is
// monotonic-non-decreasing because Observe is called in causal order
// (the sync hot path), so the first index whose timestamp falls inside
// the window is the new head.
func (t *Tracker) pruneLocked(now time.Time) {
	cutoff := now.Add(-t.cfg.WindowDuration)
	keep := 0
	for keep < len(t.timestamps) && t.timestamps[keep].Before(cutoff) {
		keep++
	}
	if keep == 0 {
		return
	}
	t.stepsNs = append(t.stepsNs[:0], t.stepsNs[keep:]...)
	t.timestamps = append(t.timestamps[:0], t.timestamps[keep:]...)
}

// CheckAt evaluates the sustain state machine at time `now` and
// returns a Breach if the threshold-and-sustain condition fires this
// tick. Returns (zero, false) on no-emission ticks. Caller (the
// watcher goroutine) calls this on a fixed cadence; sustain ticks are
// counted in calls, not in wall time.
//
// Exit shapes:
//   - len(stepsNs) < WarmupSamples: no emit; baseline not yet
//     captured. The tracker is still "warming."
//   - baseline_p99 == 0 (computed during warmup-complete transition
//     this call): no emit; freezes baseline and exits.
//   - ratio < ClearRatio: reset sustainCount, clear suppression.
//   - ratio > BreachRatio AND past suppressedUntil:
//     increment sustainCount; if it reaches SustainTicks, emit and
//     set suppressedUntil = now + RearmDuration.
//   - between ClearRatio and BreachRatio: hysteresis band, no change.
func (t *Tracker) CheckAt(now time.Time) (Breach, bool) {
	t.mu.Lock()
	defer t.mu.Unlock()

	t.pruneLocked(now)

	if len(t.stepsNs) < t.cfg.WarmupSamples {
		return Breach{}, false
	}

	currentP99 := computeP99(t.stepsNs)

	// Baseline freeze on the first CheckAt that finds enough samples.
	if t.baselineP99Ns == 0 {
		t.baselineP99Ns = currentP99
		return Breach{}, false
	}

	if currentP99 <= 0 {
		return Breach{}, false
	}
	ratio := currentP99 / t.baselineP99Ns

	if ratio < t.cfg.ClearRatio {
		t.sustainCount = 0
		return Breach{}, false
	}
	if ratio <= t.cfg.BreachRatio {
		// Hysteresis band: above clear, below breach. Hold state.
		return Breach{}, false
	}

	if now.Before(t.suppressedUntil) {
		// Still inside the rearm window; sustain count stays paused
		// so a fresh sustain starts cleanly after suppression lifts.
		return Breach{}, false
	}

	t.sustainCount++
	if t.sustainCount < t.cfg.SustainTicks {
		return Breach{}, false
	}

	// Emit. Reset sustain so the next breach episode counts cleanly,
	// arm the suppression window so the orchestrator's chain has time
	// to act before we re-fire.
	t.sustainCount = 0
	t.suppressedUntil = now.Add(t.cfg.RearmDuration)
	return Breach{
		CurrentP99Ns:  uint64(currentP99),
		BaselineP99Ns: uint64(t.baselineP99Ns),
		Ratio:         ratio,
		At:            now,
	}, true
}

// Samples returns the number of samples currently in the rolling
// window. Test-only / observability accessor; not on the wire.
func (t *Tracker) Samples() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.stepsNs)
}

// BaselineP99Ns returns the frozen baseline once warmup completes.
// Returns 0 before the warmup CheckAt has fired. Test-only /
// observability accessor.
func (t *Tracker) BaselineP99Ns() uint64 {
	t.mu.Lock()
	defer t.mu.Unlock()
	return uint64(t.baselineP99Ns)
}

// computeP99 returns the 99th percentile of stepsNs using the
// nearest-rank method (sort + index). O(N log N) per CheckAt at the
// watcher cadence (5s default); for N up to MaxSamples=10000 that's
// ~100 microseconds, well below the watcher's per-tick budget. The
// sort is destructive on a COPY so the caller's window slice stays
// in causal order.
func computeP99(stepsNs []float64) float64 {
	if len(stepsNs) == 0 {
		return 0
	}
	buf := make([]float64, len(stepsNs))
	copy(buf, stepsNs)
	sort.Float64s(buf)
	// Nearest-rank: index = ceil(0.99 * N) - 1, clamped to [0, N-1].
	idx := int(math.Ceil(float64(len(buf))*0.99)) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= len(buf) {
		idx = len(buf) - 1
	}
	return buf[idx]
}
