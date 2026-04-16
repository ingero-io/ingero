package health

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"sync"
	"time"
)

// Collector is the source of raw inputs for one tick of the push loop. Real
// implementations wire to /proc, CUDA trace events, cgroups, etc. Tests
// supply a fake.
//
// Return values:
//   - RawObservation: the four signal values. Zero values are legitimate
//     during calibration.
//   - int: kernel launch count observed during this tick (used by the
//     state machine's idle detection).
//   - error: hard collection failure. The loop constructs an Observation
//     with EventReadOK=false so the state machine can transition to
//     STALE; the returned RawObservation is discarded.
//
// Implementations must honor ctx cancellation — a collector that blocks
// on /proc reads should respect ctx.Done().
type Collector interface {
	Collect(ctx context.Context, now time.Time) (obs RawObservation, kernelLaunches int, err error)
}

// LoopConfig wires the cross-cutting pieces together. All four interface
// dependencies must be non-nil; NewLoop validates this.
type LoopConfig struct {
	Baseliner    Baseliner
	StateMachine StateMachine
	Emitter      Emitter
	Collector    Collector
	ScoreConfig  Config
	// PushInterval drives the internal ticker. Must be > 0.
	PushInterval time.Duration
	// TickTimeout bounds how long a single tick's Collect+Emit sequence
	// can run before ctx is cancelled. Defaults to PushInterval*2 when zero.
	TickTimeout time.Duration
	// DetectionMode is a static label for the ingero.agent.detection_mode
	// metric until Story 3.3 lands a real evaluator. Safe default: "none".
	DetectionMode string
	// WorkloadType labels the workload for Compute. Separate from the
	// emitter's WorkloadType attribute — this one feeds Score.
	WorkloadType string
	// Log receives transition and anomaly logs. nil = slog.Default().
	Log *slog.Logger
	// Clock returns the current time. nil = time.Now. Useful for tests.
	Clock func() time.Time
	// AnomalyLogMinInterval rate-limits per-signal anomaly log lines —
	// an Inf/NaN source should not produce tens of thousands of WARN
	// lines per day. Defaults to 5 minutes when zero.
	AnomalyLogMinInterval time.Duration
}

// Loop drives one push per interval: collect -> state transition -> update
// baseline -> compute score -> emit. It is the only place these five
// primitives are wired together.
type Loop struct {
	cfg LoopConfig
	log *slog.Logger
	now func() time.Time

	mu               sync.Mutex
	lastAnomalyLogAt map[string]time.Time
}

// NewLoop validates the config and returns a runnable Loop.
func NewLoop(cfg LoopConfig) (*Loop, error) {
	if cfg.Baseliner == nil {
		return nil, errors.New("loop: Baseliner is required")
	}
	if cfg.StateMachine == nil {
		return nil, errors.New("loop: StateMachine is required")
	}
	if cfg.Emitter == nil {
		return nil, errors.New("loop: Emitter is required")
	}
	if cfg.Collector == nil {
		return nil, errors.New("loop: Collector is required")
	}
	if cfg.PushInterval <= 0 {
		return nil, errors.New("loop: PushInterval must be > 0")
	}
	if err := cfg.ScoreConfig.Validate(); err != nil {
		return nil, fmt.Errorf("loop: ScoreConfig: %w", err)
	}
	log := cfg.Log
	if log == nil {
		log = slog.Default()
	}
	now := cfg.Clock
	if now == nil {
		now = time.Now
	}
	if cfg.DetectionMode == "" {
		cfg.DetectionMode = "none"
	}
	if cfg.TickTimeout == 0 {
		cfg.TickTimeout = cfg.PushInterval * 2
	}
	if cfg.AnomalyLogMinInterval == 0 {
		cfg.AnomalyLogMinInterval = 5 * time.Minute
	}
	return &Loop{
		cfg:              cfg,
		log:              log,
		now:              now,
		lastAnomalyLogAt: make(map[string]time.Time),
	}, nil
}

// Run blocks until ctx is cancelled, ticking once per PushInterval. Each
// tick is wrapped in a derived context with TickTimeout so a slow
// Collector or HTTP server never stalls the whole loop.
func (l *Loop) Run(ctx context.Context) error {
	ticker := time.NewTicker(l.cfg.PushInterval)
	defer ticker.Stop()

	for {
		select {
		case <-ctx.Done():
			return ctx.Err()
		case <-ticker.C:
			l.tick(ctx)
		}
	}
}

// TickOnce runs one iteration of the loop synchronously. Tests drive the
// loop via this instead of Run() to avoid timing dependencies.
func (l *Loop) TickOnce(ctx context.Context) {
	l.tick(ctx)
}

func (l *Loop) tick(parentCtx context.Context) {
	// Bound this tick so a slow Collector or HTTP server cannot stall the
	// next tick indefinitely.
	ctx, cancel := context.WithTimeout(parentCtx, l.cfg.TickTimeout)
	defer cancel()

	now := l.now()

	raw, launches, collectErr := l.cfg.Collector.Collect(ctx, now)
	obs := Observation{
		KernelLaunchCount: launches,
		EventReadOK:       collectErr == nil,
		Timestamp:         now,
	}
	if collectErr != nil {
		l.log.Debug("fleet loop: collector error", "err", collectErr.Error())
	}

	// Transition first — if we land in CALIBRATING from IDLE/STALE, reset
	// the baseliner so stale state doesn't pollute the new phase, AND
	// skip emission for this tick: Signals on a just-reset baseline
	// returns throughput_ratio = 0, producing a misleading low score.
	// The next tick will have a partially-warmed baseline and meaningful
	// output.
	prev, next, _, changed := l.cfg.StateMachine.TransitionIfNeeded(obs)
	if changed && next == StateCalibrating && (prev == StateIdle || prev == StateStale) {
		l.cfg.Baseliner.Reset()
		// Seed the baseline with this tick's observation so next tick
		// has something to normalize against. Do not emit — the score
		// would be derived from a zero baseline and is meaningless.
		l.cfg.Baseliner.Update(raw)
		return
	}

	// STALE emits nothing — the agent does not know what its score is.
	if next == StateStale {
		return
	}

	// Signals FIRST, Update after — otherwise the baseline used for the
	// throughput ratio would already contain this tick's observation,
	// biasing the ratio toward 1.0.
	sigs := l.cfg.Baseliner.Signals(raw)
	l.cfg.Baseliner.Update(raw)
	score, anomalies := Compute(now, sigs, l.cfg.WorkloadType, l.cfg.ScoreConfig)
	for _, a := range anomalies {
		l.maybeLogAnomaly(now, a)
	}

	if err := l.cfg.Emitter.Push(ctx, now, score, next, l.cfg.DetectionMode, l.cfg.Baseliner.DegradationWarning()); err != nil {
		l.log.Debug("fleet loop: push error", "err", err.Error())
	}
}

// maybeLogAnomaly rate-limits per-signal anomaly logs to once per
// AnomalyLogMinInterval. Without this, a broken collector producing
// Inf/NaN every tick floods logs at push_interval cadence (86400
// entries/day at a 1s cadence).
func (l *Loop) maybeLogAnomaly(now time.Time, a Anomaly) {
	key := a.Signal + "/" + a.Reason
	l.mu.Lock()
	last, ok := l.lastAnomalyLogAt[key]
	if ok && now.Sub(last) < l.cfg.AnomalyLogMinInterval {
		l.mu.Unlock()
		return
	}
	l.lastAnomalyLogAt[key] = now
	l.mu.Unlock()
	l.log.Warn("fleet loop: signal coerced", "signal", a.Signal, "reason", a.Reason)
}
