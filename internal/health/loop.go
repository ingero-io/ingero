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

// LoopConfig wires the cross-cutting pieces together. All four primary
// interface dependencies (Baseliner, StateMachine, Emitter, Collector)
// must be non-nil; NewLoop validates this. ModeEvaluator, Classifier,
// and StragglerSink are optional.
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
	// DetectionMode is a static fallback label for the
	// ingero.agent.detection_mode metric when ModeEvaluator is nil.
	// Ignored when ModeEvaluator is set. Safe default: "none".
	DetectionMode string
	// ModeEvaluator, if non-nil, is called once per tick to compute the
	// current detection mode and threshold (Story 3.3). The returned
	// mode replaces the static DetectionMode field on every push.
	ModeEvaluator ModeEvaluator
	// Classifier, if non-nil AND ModeEvaluator is non-nil AND the state
	// machine is ACTIVE AND the detection mode is not "none", compares
	// the tick's score against the threshold and emits a straggler
	// event on straggler/recovery transitions (Story 3.4).
	Classifier Classifier
	// StragglerSink, if non-nil, receives NDJSON notifications on each
	// straggler tick AND on the recovery edge. Typically implemented by
	// internal/remediate/server.go (the UDS socket).
	StragglerSink StragglerSink
	// NodeID and ClusterID populate the StragglerEvent carried to the
	// emitter and sink. Required when Classifier is set.
	NodeID    string
	ClusterID string
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
	if cfg.Classifier != nil {
		if cfg.NodeID == "" || cfg.ClusterID == "" {
			return nil, errors.New("loop: NodeID and ClusterID are required when Classifier is set")
		}
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
	preUpdateBaseline := l.cfg.Baseliner.Current()
	l.cfg.Baseliner.Update(raw)
	score, anomalies := Compute(now, sigs, l.cfg.WorkloadType, l.cfg.ScoreConfig)
	for _, a := range anomalies {
		l.maybeLogAnomaly(now, a)
	}

	// Detection mode + threshold: evaluator (Story 3.3) overrides the
	// static config when present.
	mode := l.cfg.DetectionMode
	var threshold float64
	var thresholdOK bool
	if l.cfg.ModeEvaluator != nil {
		m, th, ok := l.cfg.ModeEvaluator.Evaluate(now)
		mode = string(m)
		threshold = th
		thresholdOK = ok
	}

	if err := l.cfg.Emitter.Push(ctx, now, score, next, mode, l.cfg.Baseliner.DegradationWarning()); err != nil {
		l.log.Debug("fleet loop: push error", "err", err.Error())
	}

	// Straggler classification (Story 3.4). Gated on:
	//   - Classifier configured
	//   - State machine in ACTIVE (skip calibrating/idle/stale)
	//   - Mode is not "none" (threshold is meaningful)
	if l.cfg.Classifier != nil && next == StateActive && thresholdOK {
		l.classifyAndEmit(ctx, now, score, threshold, mode, preUpdateBaseline)
	}
}

// classifyAndEmit runs the classifier and fires OTLP + UDS notifications.
// Safe to call every tick; internal bookkeeping enforces edge-triggered
// recovery and per-tick straggler-state emission.
func (l *Loop) classifyAndEmit(ctx context.Context, now time.Time, score Score, threshold float64, mode string, baseline Baselines) {
	isStraggler, changed := l.cfg.Classifier.Classify(score.Value, threshold, now)

	// AC4: recovery emits exactly once on the edge; subsequent healthy
	// ticks emit nothing. Straggler ticks emit every interval (AC5).
	shouldEmit := isStraggler || changed
	if !shouldEmit {
		return
	}

	// Log the edge transition at Info so operators (and e2e scripts) can
	// observe it in the agent log without enabling Debug. Non-edge
	// straggler ticks are left for the OTLP / UDS channels.
	if changed {
		state := "HEALTHY"
		if isStraggler {
			state = "STRAGGLER"
		}
		l.log.Info("straggler_state transition",
			"straggler_state", state,
			"score", score.Value,
			"threshold", threshold,
			"mode", mode,
		)
	}

	ev := StragglerEvent{
		NodeID:         l.cfg.NodeID,
		ClusterID:      l.cfg.ClusterID,
		Score:          score.Value,
		Threshold:      threshold,
		DetectionMode:  DetectionMode(mode),
		DominantSignal: dominantFromScore(score, baseline),
		Timestamp:      now,
	}

	if err := l.cfg.Emitter.EmitStragglerEvent(ctx, ev, isStraggler); err != nil {
		l.log.Debug("fleet loop: straggler emit error", "err", err.Error())
	}

	if l.cfg.StragglerSink != nil {
		if isStraggler {
			if err := l.cfg.StragglerSink.SendStragglerState(ev); err != nil {
				l.log.Debug("fleet loop: straggler sink error", "err", err.Error())
			}
		} else {
			// changed && !isStraggler => recovery edge.
			if err := l.cfg.StragglerSink.SendStragglerResolved(ev.NodeID, ev.ClusterID, now); err != nil {
				l.log.Debug("fleet loop: straggler resolved sink error", "err", err.Error())
			}
		}
	}
}

// dominantFromScore computes the dominant-signal label using the
// per-signal fields on Score (the coerced [0,1] values) compared to the
// fast EMA baseline captured BEFORE this tick's Update.
func dominantFromScore(score Score, baseline Baselines) string {
	current := Baselines{
		Throughput: score.Throughput,
		Compute:    score.Compute,
		Memory:     score.Memory,
		CPU:        score.CPU,
	}
	return DominantSignal(current, baseline)
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
