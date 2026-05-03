package health

import (
	"context"
	"errors"
	"fmt"
	"log/slog"
	"math/rand"
	"sync"
	"time"

	"github.com/google/uuid"
	"go.opentelemetry.io/otel/attribute"
	"go.opentelemetry.io/otel/trace"

	"github.com/ingero-io/ingero/internal/sampling"
	"github.com/ingero-io/ingero/pkg/contract"
)

const (
	spanClassStraggler   = "straggler"
	spanClassDegradation = "degradation"
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

// PerCGroupCollector is an optional capability surface: collectors that
// can attribute per-window event totals to cgroup_path_hash implement
// it so the loop can emit the per-cgroup CUDA + CPU-stall counter
// metrics. The signal collector backed by the SQLite event store
// implements this; lighter-weight test collectors typically do not.
type PerCGroupCollector interface {
	CollectPerCGroup(ctx context.Context, now time.Time) ([]PerCGroupStats, error)
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
	// Sampler, if non-nil, is informed of degradation-edge transitions
	// every tick. The sampler does its own edge detection so calling on
	// every tick is safe and idempotent within a state.
	Sampler *sampling.Sampler
	// Tracer, when non-nil, receives one root span per detection-event
	// rising edge (straggler, degradation) and ends it on the falling
	// edge. Nil tracer = trace operations are skipped entirely; the hot
	// path makes zero OTel allocations.
	Tracer trace.Tracer
	// CgroupPathHash and GPUModel populate detection-span attributes.
	// Caller (cli.fleet_push) sources them from the same nvidia-smi /
	// /proc/self/cgroup probes that feed the metrics emitter, so a span
	// and the metric for the same event carry identical hardware-attribution
	// tags.
	CgroupPathHash string
	GPUModel       string
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
	// rng seeds the per-tick jitter and runs under `mu`. Seeded per Loop
	// (not package-global) so multiple Loops in the same process draw
	// independent sequences.
	rng *rand.Rand
	// openSpans tracks the currently-recording detection spans keyed by
	// event class. Single-goroutine state (only mutated from tick) but
	// guarded by mu so tests calling TickOnce concurrently with shutdown
	// are race-free.
	openSpans map[string]trace.Span
	// degradedPrev tracks the prior tick's DegradationWarning so the loop
	// can detect rising/falling edges. The zero value (false) is intentional:
	// the first tick observing DegradationWarning()=true is correctly
	// treated as a rising edge against this initial false.
	degradedPrev bool
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
		// now().UnixNano() seed gives per-agent spread at process start
		// so N agents launched by the same Helm rollout don't synchronize
		// their push ticks.
		rng:       rand.New(rand.NewSource(now().UnixNano())),
		openSpans: make(map[string]trace.Span),
	}, nil
}

// Run blocks until ctx is cancelled, ticking roughly every PushInterval
// with ±10% jitter so N agents launched together do not synchronize a
// thundering-herd push every interval. Each tick is wrapped in a derived
// context with TickTimeout so a slow Collector or HTTP server never
// stalls the whole loop. If the last tick returned a RetryAfterError,
// the next fire waits at least that long before the jittered interval
// kicks back in — the server's backoff request takes precedence over
// the normal cadence.
func (l *Loop) Run(ctx context.Context) error {
	var extra time.Duration
	for {
		timer := time.NewTimer(l.nextDelay(extra))
		extra = 0
		select {
		case <-ctx.Done():
			timer.Stop()
			return ctx.Err()
		case <-timer.C:
			extra = l.tick(ctx)
		}
	}
}

// nextDelay computes the time until the next tick. Applies a ±10%
// jitter around PushInterval so rollouts don't synchronize. The
// `extra` parameter is added unconditionally when a previous tick asked
// for backoff (server-side Retry-After).
func (l *Loop) nextDelay(extra time.Duration) time.Duration {
	l.mu.Lock()
	defer l.mu.Unlock()
	pct := l.rng.Float64()*0.2 - 0.1 // -10% .. +10%
	jittered := l.cfg.PushInterval + time.Duration(float64(l.cfg.PushInterval)*pct)
	if jittered < time.Millisecond {
		jittered = time.Millisecond
	}
	return jittered + extra
}

// TickOnce runs one iteration of the loop synchronously. Tests drive the
// loop via this instead of Run() to avoid timing dependencies. The
// return value is the server-requested Retry-After backoff, if any.
// Zero means "no explicit backoff; schedule the next tick normally".
func (l *Loop) TickOnce(ctx context.Context) time.Duration {
	return l.tick(ctx)
}

// tick runs one loop iteration. Returns the backoff requested by the
// server via a Retry-After header on a 429/503 push response, or 0 if
// the push succeeded or failed without a structured backoff hint.
func (l *Loop) tick(parentCtx context.Context) time.Duration {
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
		return 0
	}

	// STALE emits nothing — the agent does not know what its score is.
	if next == StateStale {
		return 0
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

	// Read DegradationWarning once and forward it to both the push payload
	// and the sampler. The sampler does its own edge detection internally.
	degraded := l.cfg.Baseliner.DegradationWarning()
	if l.cfg.Sampler != nil {
		l.cfg.Sampler.SetDegraded(degraded)
	}
	l.handleDegradationSpan(ctx, degraded)

	var perCGroup []PerCGroupStats
	if pcc, ok := l.cfg.Collector.(PerCGroupCollector); ok {
		stats, perr := pcc.CollectPerCGroup(ctx, now)
		if perr != nil {
			l.log.Debug("fleet loop: per-cgroup collect error", "err", perr.Error())
		} else {
			perCGroup = stats
		}
	}

	var retryAfter time.Duration
	if err := l.cfg.Emitter.Push(ctx, now, score, next, mode, degraded, perCGroup); err != nil {
		l.log.Debug("fleet loop: push error", "err", err.Error())
		if ra := AsRetryAfter(err); ra != nil {
			retryAfter = ra.Delay
			l.log.Info("fleet loop: server asked for backoff",
				"status", ra.StatusCode, "retry_after", ra.Delay.String())
		}
	}

	// Straggler classification (Story 3.4). Gated on:
	//   - Classifier configured
	//   - State machine in ACTIVE (skip calibrating/idle/stale)
	//   - Mode is not "none" (threshold is meaningful)
	if l.cfg.Classifier != nil && next == StateActive && thresholdOK {
		l.classifyAndEmit(ctx, now, score, threshold, mode, preUpdateBaseline)
	}
	return retryAfter
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
		EventID:        uuid.NewString(),
	}

	if changed {
		if isStraggler {
			l.startDetectionSpan(ctx, spanClassStraggler, ev.EventID)
		} else {
			l.endDetectionSpan(spanClassStraggler)
		}
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
			// changed && !isStraggler => recovery edge. The recovery has
			// its own EventID (one per edge). Both the OTLP push and the
			// UDS message for this edge carry the same id so the two
			// channels can be reconciled per event.
			if err := l.cfg.StragglerSink.SendStragglerResolved(ev); err != nil {
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

// handleDegradationSpan opens a span on the rising edge of
// DegradationWarning and closes it on the falling edge. No-op when no
// tracer is configured.
//
// Tracking edges with l.degradedPrev (rather than the Baseliner's
// internal edge log) keeps the span lifecycle owned by the loop: the
// loop is the single observer of the warning, so a stale prev value
// cannot occur.
func (l *Loop) handleDegradationSpan(ctx context.Context, degraded bool) {
	if l.cfg.Tracer == nil {
		return
	}
	prev := l.degradedPrev
	l.degradedPrev = degraded
	switch {
	case degraded && !prev:
		l.startDetectionSpan(ctx, spanClassDegradation, uuid.NewString())
	case !degraded && prev:
		l.endDetectionSpan(spanClassDegradation)
	}
}

// startDetectionSpan opens a root span keyed by class. Safe to call when
// Tracer is nil (no-op). If a span for the same class is already open
// (rapid rising->falling->rising flap, or a logic bug) the prior span is
// ended cleanly before being replaced so the trace backend isn't left
// holding a never-closed span.
//
// TODO(v0.14, internal/health/loop.go): attach correlator chain children.
// Requires plumbing through the per-PID *correlate.Engine and the
// active cudaOps slice (see internal/correlate/correlate.go:565
// SnapshotCausalChains). The loop today does not own those handles -
// they live on the trace recorder, which is in a separate process when
// the agent is configured for the SQLite-shared-DB topology used by
// Helm. Deferred for the v0.13 first cut.
func (l *Loop) startDetectionSpan(ctx context.Context, class, eventID string) {
	if l.cfg.Tracer == nil {
		return
	}
	_, span := l.cfg.Tracer.Start(ctx, "ingero.detection."+class,
		trace.WithAttributes(
			attribute.String(contract.AttrCgroupPathHash, l.cfg.CgroupPathHash),
			attribute.String(contract.AttrWorkloadType, l.cfg.WorkloadType),
			attribute.String(contract.AttrGPUModel, l.cfg.GPUModel),
			attribute.String(contract.AttrClusterID, l.cfg.ClusterID),
			attribute.String(contract.AttrEventID, eventID),
		),
	)
	l.mu.Lock()
	prior, hadPrior := l.openSpans[class]
	l.openSpans[class] = span
	l.mu.Unlock()
	if hadPrior {
		l.log.Warn("unexpected detection-span overwrite; ending prior span", "class", class)
		prior.End()
	}
}

// endDetectionSpan attaches outcome=resolved and ends the span, if open.
func (l *Loop) endDetectionSpan(class string) {
	if l.cfg.Tracer == nil {
		return
	}
	l.mu.Lock()
	span, ok := l.openSpans[class]
	if ok {
		delete(l.openSpans, class)
	}
	l.mu.Unlock()
	if !ok {
		return
	}
	span.AddEvent("outcome", trace.WithAttributes(attribute.String("status", "resolved")))
	span.End()
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
