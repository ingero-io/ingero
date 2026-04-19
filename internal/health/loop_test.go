package health

import (
	"context"
	"errors"
	"io"
	"log/slog"
	"sync"
	"testing"
	"time"
)

// fakeCollector returns a pre-programmed sequence of observations and
// optional collect errors.
type fakeCollector struct {
	mu       sync.Mutex
	obs      []RawObservation
	launches []int
	errs     []error
	i        int
}

func (f *fakeCollector) Collect(ctx context.Context, now time.Time) (RawObservation, int, error) {
	f.mu.Lock()
	defer f.mu.Unlock()
	if f.i >= len(f.obs) {
		return RawObservation{}, 0, errors.New("collector: exhausted")
	}
	o := f.obs[f.i]
	c := 0
	if f.i < len(f.launches) {
		c = f.launches[f.i]
	}
	var err error
	if f.i < len(f.errs) {
		err = f.errs[f.i]
	}
	f.i++
	return o, c, err
}

// fakeEmitter records each Push call for assertion.
type fakeEmitter struct {
	mu             sync.Mutex
	calls          []emitCall
	stragglerCalls []stragglerCall
	err            error
	reachable      bool
	pushes         int64
	errors         int64
}

type emitCall struct {
	score       Score
	state       State
	mode        string
	degradation bool
	now         time.Time
}

type stragglerCall struct {
	ev          StragglerEvent
	isStraggler bool
}

func (f *fakeEmitter) Push(ctx context.Context, now time.Time, score Score, state State, mode string, degradation bool) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.calls = append(f.calls, emitCall{score, state, mode, degradation, now})
	if f.err != nil {
		f.errors++
		return f.err
	}
	f.pushes++
	return nil
}

func (f *fakeEmitter) EmitStragglerEvent(ctx context.Context, ev StragglerEvent, isStraggler bool) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.stragglerCalls = append(f.stragglerCalls, stragglerCall{ev, isStraggler})
	return nil
}

func (f *fakeEmitter) FleetReachable() bool { return f.reachable }
func (f *fakeEmitter) Stats() (int64, int64) {
	f.mu.Lock()
	defer f.mu.Unlock()
	return f.pushes, f.errors
}

func newQuietLoop(t *testing.T, c Collector, e Emitter) (*Loop, Baseliner, StateMachine) {
	t.Helper()
	b, _ := NewBaseliner(DefaultBaselineConfig(), nil)
	sm, _ := NewStateMachine(
		StateConfig{IdleIntervals: 2, WarmupSamples: 0, StaleReadFailures: 2},
		slog.New(slog.NewTextHandler(io.Discard, nil)),
	)
	l, err := NewLoop(LoopConfig{
		Baseliner:     b,
		StateMachine:  sm,
		Emitter:       e,
		Collector:     c,
		ScoreConfig:   DefaultConfig(),
		PushInterval:  10 * time.Millisecond,
		DetectionMode: "fleet",
		Log:           slog.New(slog.NewTextHandler(io.Discard, nil)),
		Clock:         func() time.Time { return testTS },
	})
	if err != nil {
		t.Fatalf("NewLoop: %v", err)
	}
	return l, b, sm
}

func TestNewLoop_ValidatesConfig(t *testing.T) {
	good := LoopConfig{
		Baseliner:     mustBaseliner(t),
		StateMachine:  mustStateMachine(t),
		Emitter:       &fakeEmitter{reachable: true},
		Collector:     &fakeCollector{},
		ScoreConfig:   DefaultConfig(),
		PushInterval:  time.Second,
		DetectionMode: "fleet",
	}
	if _, err := NewLoop(good); err != nil {
		t.Fatalf("good config rejected: %v", err)
	}

	tests := []struct {
		name   string
		mutate func(*LoopConfig)
	}{
		{"nil_baseliner", func(c *LoopConfig) { c.Baseliner = nil }},
		{"nil_state_machine", func(c *LoopConfig) { c.StateMachine = nil }},
		{"nil_emitter", func(c *LoopConfig) { c.Emitter = nil }},
		{"nil_collector", func(c *LoopConfig) { c.Collector = nil }},
		{"zero_interval", func(c *LoopConfig) { c.PushInterval = 0 }},
		{"invalid_score_cfg", func(c *LoopConfig) { c.ScoreConfig = Config{} }},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			c := good
			tc.mutate(&c)
			if _, err := NewLoop(c); err == nil {
				t.Fatal("expected error")
			}
		})
	}
}

func mustBaseliner(t *testing.T) Baseliner {
	t.Helper()
	b, err := NewBaseliner(DefaultBaselineConfig(), nil)
	if err != nil {
		t.Fatalf("baseliner: %v", err)
	}
	return b
}

func mustStateMachine(t *testing.T) StateMachine {
	t.Helper()
	sm, err := NewStateMachine(DefaultStateConfig(), slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err != nil {
		t.Fatalf("state machine: %v", err)
	}
	return sm
}

// One tick with a healthy observation: emitter sees a Push with the
// expected state.
func TestLoop_TickHealthy(t *testing.T) {
	col := &fakeCollector{
		obs:      []RawObservation{{Throughput: 100, Compute: 0.9, Memory: 0.9, CPU: 0.9}},
		launches: []int{10},
	}
	em := &fakeEmitter{reachable: true}
	l, _, _ := newQuietLoop(t, col, em)

	l.TickOnce(context.Background())

	em.mu.Lock()
	defer em.mu.Unlock()
	if len(em.calls) != 1 {
		t.Fatalf("expected 1 Push, got %d", len(em.calls))
	}
	c := em.calls[0]
	// WarmupSamples=0 means the state machine transitions to ACTIVE on the
	// first tick.
	if c.state != StateActive {
		t.Fatalf("state = %v, want ACTIVE", c.state)
	}
	if c.mode != "fleet" {
		t.Fatalf("mode = %q", c.mode)
	}
}

// Collector failures flip state to STALE after threshold, and no Push
// happens while in STALE. newQuietLoop uses StaleReadFailures=2, so the
// second consecutive bad tick trips STALE.
func TestLoop_CollectorFailures_LeadsToStale_NoPush(t *testing.T) {
	col := &fakeCollector{
		obs:      []RawObservation{{}, {}, {}},
		launches: []int{0, 0, 0},
		errs:     []error{errors.New("no events"), errors.New("no events"), errors.New("no events")},
	}
	em := &fakeEmitter{reachable: true}
	l, _, sm := newQuietLoop(t, col, em)

	l.TickOnce(context.Background())
	if sm.Current() == StateStale {
		t.Fatal("STALE after only 1 failure")
	}
	l.TickOnce(context.Background())
	if sm.Current() != StateStale {
		t.Fatalf("expected STALE after 2 consecutive read failures, got %v", sm.Current())
	}

	// A further tick in STALE should NOT call Push.
	baselinePushCount := len(em.calls)
	l.TickOnce(context.Background())
	if len(em.calls) != baselinePushCount {
		t.Fatalf("Push called in STALE: before=%d after=%d", baselinePushCount, len(em.calls))
	}
}

// IDLE -> CALIBRATING transition resets the baseliner AND suppresses
// emission for the transition tick (Signals on a zero baseline would
// produce a misleading low score).
func TestLoop_IdleToCalibrating_ResetsAndSuppressesEmit(t *testing.T) {
	col := &fakeCollector{}
	em := &fakeEmitter{reachable: true}
	l, baseliner, sm := newQuietLoop(t, col, em)

	col.mu.Lock()
	// 1 healthy -> ACTIVE (warmup=0), 2 idle -> IDLE (idle_intervals=2),
	// 1 healthy again -> CALIBRATING.
	col.obs = []RawObservation{
		{Throughput: 100, Compute: 0.9, Memory: 0.9, CPU: 0.9},
		{Throughput: 100, Compute: 0.9, Memory: 0.9, CPU: 0.9},
		{Throughput: 100, Compute: 0.9, Memory: 0.9, CPU: 0.9},
		{Throughput: 100, Compute: 0.9, Memory: 0.9, CPU: 0.9},
	}
	col.launches = []int{10, 0, 0, 10}
	col.mu.Unlock()

	l.TickOnce(context.Background())
	if sm.Current() != StateActive {
		t.Fatalf("tick1 state = %v, want ACTIVE", sm.Current())
	}
	l.TickOnce(context.Background())
	l.TickOnce(context.Background())
	if sm.Current() != StateIdle {
		t.Fatalf("after 2 zeros state = %v, want IDLE", sm.Current())
	}
	priorSampleCount := baseliner.SampleCount()
	if priorSampleCount == 0 {
		t.Fatal("baseliner never updated")
	}
	priorEmitCount := len(em.calls)

	// Waking observation: transition IDLE -> CALIBRATING. Reset + seed
	// Update produce SampleCount == 1. Emission is skipped for this tick.
	l.TickOnce(context.Background())
	if sm.Current() != StateCalibrating {
		t.Fatalf("wake tick state = %v, want CALIBRATING", sm.Current())
	}
	if baseliner.SampleCount() != 1 {
		t.Fatalf("baseliner not reset+seeded: SampleCount = %d, want 1", baseliner.SampleCount())
	}
	em.mu.Lock()
	defer em.mu.Unlock()
	if len(em.calls) != priorEmitCount {
		t.Fatalf("emission fired on transition tick: before=%d after=%d", priorEmitCount, len(em.calls))
	}
}

// Anomalies from Compute are logged but do not break the loop.
func TestLoop_AnomaliesDoNotStopTick(t *testing.T) {
	col := &fakeCollector{
		obs:      []RawObservation{{Throughput: 1e300, Compute: 0.9, Memory: 0.9, CPU: 0.9}},
		launches: []int{10},
	}
	em := &fakeEmitter{reachable: true}
	l, _, _ := newQuietLoop(t, col, em)
	l.TickOnce(context.Background())
	em.mu.Lock()
	defer em.mu.Unlock()
	if len(em.calls) != 1 {
		t.Fatalf("Push should still fire despite large input; got %d calls", len(em.calls))
	}
}

// Emitter errors are tolerated — loop keeps going on next tick.
func TestLoop_EmitterError_Tolerated(t *testing.T) {
	col := &fakeCollector{
		obs: []RawObservation{
			{Throughput: 100, Compute: 0.9, Memory: 0.9, CPU: 0.9},
			{Throughput: 100, Compute: 0.9, Memory: 0.9, CPU: 0.9},
		},
		launches: []int{10, 10},
	}
	em := &fakeEmitter{reachable: false, err: errors.New("server 500")}
	l, _, _ := newQuietLoop(t, col, em)
	l.TickOnce(context.Background())
	l.TickOnce(context.Background())
	if em.errors != 2 {
		t.Fatalf("emitter errors = %d, want 2", em.errors)
	}
}

// Fake ModeEvaluator for loop integration tests.
type fakeMode struct {
	mode      DetectionMode
	threshold float64
	ok        bool
}

func (f *fakeMode) Evaluate(now time.Time) (DetectionMode, float64, bool) {
	return f.mode, f.threshold, f.ok
}
func (f *fakeMode) CurrentMode() DetectionMode { return f.mode }

// Fake StragglerSink for loop integration tests.
type fakeSink struct {
	mu        sync.Mutex
	states    []StragglerEvent
	resolved  []struct{ node, cluster string }
	stateErr  error
}

func (f *fakeSink) SendStragglerState(ev StragglerEvent) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.states = append(f.states, ev)
	return f.stateErr
}
func (f *fakeSink) SendStragglerResolved(nodeID, clusterID string, ts time.Time) error {
	f.mu.Lock()
	defer f.mu.Unlock()
	f.resolved = append(f.resolved, struct{ node, cluster string }{nodeID, clusterID})
	return nil
}

// Story 3.3 integration: ModeEvaluator controls the mode passed to Push.
func TestLoop_ModeEvaluatorDrivesPushMode(t *testing.T) {
	col := &fakeCollector{
		obs:      []RawObservation{{Throughput: 100, Compute: 0.9, Memory: 0.9, CPU: 0.9}},
		launches: []int{10},
	}
	em := &fakeEmitter{reachable: true}
	bl, _ := NewBaseliner(DefaultBaselineConfig(), discardLogger())
	sm, _ := NewStateMachine(
		StateConfig{IdleIntervals: 2, WarmupSamples: 0, StaleReadFailures: 3},
		discardLogger(),
	)
	mode := &fakeMode{mode: ModeFleet, threshold: 0.85, ok: true}

	l, err := NewLoop(LoopConfig{
		Baseliner:     bl,
		StateMachine:  sm,
		Emitter:       em,
		Collector:     col,
		ModeEvaluator: mode,
		ScoreConfig:   DefaultConfig(),
		PushInterval:  10 * time.Millisecond,
		DetectionMode: "ignored-static-fallback",
		Log:           discardLogger(),
		Clock:         func() time.Time { return testTS },
	})
	if err != nil {
		t.Fatalf("NewLoop: %v", err)
	}
	l.TickOnce(context.Background())

	em.mu.Lock()
	defer em.mu.Unlock()
	if len(em.calls) != 1 {
		t.Fatalf("want 1 push, got %d", len(em.calls))
	}
	if em.calls[0].mode != "fleet" {
		t.Fatalf("mode = %q, want fleet (from ModeEvaluator, not static)", em.calls[0].mode)
	}
}

// Story 3.4 integration: classifier fires straggler emission (OTLP + sink)
// while score < threshold; recovery emits exactly once on the edge.
func TestLoop_ClassifierStragglerAndRecovery(t *testing.T) {
	col := &fakeCollector{}
	em := &fakeEmitter{reachable: true}
	sink := &fakeSink{}
	bl, _ := NewBaseliner(DefaultBaselineConfig(), discardLogger())
	sm, _ := NewStateMachine(
		StateConfig{IdleIntervals: 5, WarmupSamples: 0, StaleReadFailures: 5},
		discardLogger(),
	)
	mode := &fakeMode{mode: ModeFleet, threshold: 0.80, ok: true}
	clf, _ := NewClassifier(DefaultClassifierConfig())

	// Feed three ticks: two straggler, one recovery.
	col.mu.Lock()
	col.obs = []RawObservation{
		{Throughput: 50, Compute: 0.5, Memory: 0.5, CPU: 0.5},   // low score -> straggler
		{Throughput: 50, Compute: 0.5, Memory: 0.5, CPU: 0.5},   // still straggler
		{Throughput: 100, Compute: 0.95, Memory: 0.95, CPU: 0.95}, // recovered
	}
	col.launches = []int{10, 10, 10}
	col.mu.Unlock()

	l, err := NewLoop(LoopConfig{
		Baseliner:     bl,
		StateMachine:  sm,
		Emitter:       em,
		Collector:     col,
		ModeEvaluator: mode,
		Classifier:    clf,
		StragglerSink: sink,
		NodeID:        "gpu-node-42",
		ClusterID:     "prod",
		ScoreConfig:   DefaultConfig(),
		PushInterval:  10 * time.Millisecond,
		Log:           discardLogger(),
		Clock:         func() time.Time { return testTS },
	})
	if err != nil {
		t.Fatalf("NewLoop: %v", err)
	}

	l.TickOnce(context.Background()) // ACTIVE (warmup=0), straggler
	l.TickOnce(context.Background()) // ACTIVE, still straggler
	l.TickOnce(context.Background()) // recovery

	em.mu.Lock()
	sCalls := append([]stragglerCall{}, em.stragglerCalls...)
	em.mu.Unlock()
	sink.mu.Lock()
	states := append([]StragglerEvent{}, sink.states...)
	resolved := append([]struct{ node, cluster string }{}, sink.resolved...)
	sink.mu.Unlock()

	// Two straggler emissions (tick 1 = edge, tick 2 = while straggler),
	// plus one recovery emission (tick 3 edge) = 3 total events.
	if len(sCalls) != 3 {
		t.Fatalf("straggler emit calls = %d, want 3 (2 straggler + 1 recovery)", len(sCalls))
	}
	if !sCalls[0].isStraggler || !sCalls[1].isStraggler {
		t.Fatalf("first two calls should be isStraggler=true: %+v", sCalls[:2])
	}
	if sCalls[2].isStraggler {
		t.Fatalf("third call should be recovery (isStraggler=false): %+v", sCalls[2])
	}

	// Sink: two state messages (stragger ticks), one resolved.
	if len(states) != 2 {
		t.Fatalf("sink states = %d, want 2", len(states))
	}
	if len(resolved) != 1 {
		t.Fatalf("sink resolved = %d, want 1", len(resolved))
	}
	if resolved[0].node != "gpu-node-42" || resolved[0].cluster != "prod" {
		t.Fatalf("resolved identifiers wrong: %+v", resolved[0])
	}
}

// Classifier stays silent when state machine is not ACTIVE.
func TestLoop_ClassifierSkippedWhenNotActive(t *testing.T) {
	col := &fakeCollector{
		obs:      []RawObservation{{Throughput: 50, Compute: 0.5, Memory: 0.5, CPU: 0.5}},
		launches: []int{10},
	}
	em := &fakeEmitter{reachable: true}
	sink := &fakeSink{}
	bl, _ := NewBaseliner(DefaultBaselineConfig(), discardLogger())
	// warmup=5 so the state machine stays CALIBRATING after one tick.
	sm, _ := NewStateMachine(
		StateConfig{IdleIntervals: 5, WarmupSamples: 5, StaleReadFailures: 5},
		discardLogger(),
	)
	mode := &fakeMode{mode: ModeFleet, threshold: 0.80, ok: true}
	clf, _ := NewClassifier(DefaultClassifierConfig())

	l, _ := NewLoop(LoopConfig{
		Baseliner:     bl,
		StateMachine:  sm,
		Emitter:       em,
		Collector:     col,
		ModeEvaluator: mode,
		Classifier:    clf,
		StragglerSink: sink,
		NodeID:        "node-0",
		ClusterID:     "prod",
		ScoreConfig:   DefaultConfig(),
		PushInterval:  10 * time.Millisecond,
		Log:           discardLogger(),
		Clock:         func() time.Time { return testTS },
	})
	l.TickOnce(context.Background())

	em.mu.Lock()
	defer em.mu.Unlock()
	if len(em.stragglerCalls) != 0 {
		t.Fatal("classifier should be skipped while CALIBRATING")
	}
	sink.mu.Lock()
	defer sink.mu.Unlock()
	if len(sink.states) != 0 {
		t.Fatal("sink should be untouched while CALIBRATING")
	}
}

// Classifier stays silent when mode == none (threshold not meaningful).
func TestLoop_ClassifierSkippedWhenModeNone(t *testing.T) {
	col := &fakeCollector{
		obs:      []RawObservation{{Throughput: 100, Compute: 0.9, Memory: 0.9, CPU: 0.9}},
		launches: []int{10},
	}
	em := &fakeEmitter{reachable: true}
	bl, _ := NewBaseliner(DefaultBaselineConfig(), discardLogger())
	sm, _ := NewStateMachine(
		StateConfig{IdleIntervals: 5, WarmupSamples: 0, StaleReadFailures: 5},
		discardLogger(),
	)
	mode := &fakeMode{mode: ModeNone, threshold: 0, ok: false}
	clf, _ := NewClassifier(DefaultClassifierConfig())

	l, _ := NewLoop(LoopConfig{
		Baseliner:     bl,
		StateMachine:  sm,
		Emitter:       em,
		Collector:     col,
		ModeEvaluator: mode,
		Classifier:    clf,
		NodeID:        "node-0",
		ClusterID:     "prod",
		ScoreConfig:   DefaultConfig(),
		PushInterval:  10 * time.Millisecond,
		Log:           discardLogger(),
		Clock:         func() time.Time { return testTS },
	})
	l.TickOnce(context.Background())

	em.mu.Lock()
	defer em.mu.Unlock()
	if len(em.stragglerCalls) != 0 {
		t.Fatalf("classifier should be skipped when mode=none, got %d calls", len(em.stragglerCalls))
	}
}

// NewLoop rejects Classifier without NodeID or ClusterID.
func TestNewLoop_ClassifierRequiresIdentity(t *testing.T) {
	bl, _ := NewBaseliner(DefaultBaselineConfig(), discardLogger())
	sm, _ := NewStateMachine(DefaultStateConfig(), discardLogger())
	clf, _ := NewClassifier(DefaultClassifierConfig())
	_, err := NewLoop(LoopConfig{
		Baseliner:    bl,
		StateMachine: sm,
		Emitter:      &fakeEmitter{reachable: true},
		Collector:    &fakeCollector{},
		Classifier:   clf,
		ScoreConfig:  DefaultConfig(),
		PushInterval: time.Second,
	})
	if err == nil {
		t.Fatal("expected error when Classifier is set without NodeID/ClusterID")
	}
}

// Run() ticks at least once before ctx cancel, and the Emitter is
// actually invoked. The earlier version of this test only verified
// ctx.Canceled, which would pass even with zero ticks.
func TestLoop_RunTicksThenExits(t *testing.T) {
	// Seed enough observations to outlast any plausible ticker schedule.
	const n = 32
	obs := make([]RawObservation, n)
	launches := make([]int, n)
	for i := range obs {
		obs[i] = RawObservation{Throughput: 100, Compute: 0.9, Memory: 0.9, CPU: 0.9}
		launches[i] = 10
	}
	col := &fakeCollector{obs: obs, launches: launches}
	em := &fakeEmitter{reachable: true}
	l, _, _ := newQuietLoop(t, col, em)

	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan error, 1)
	go func() { done <- l.Run(ctx) }()

	// Wait until at least one tick has been pushed, or fail the test.
	deadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) {
		em.mu.Lock()
		hit := len(em.calls) > 0
		em.mu.Unlock()
		if hit {
			break
		}
		time.Sleep(2 * time.Millisecond)
	}
	em.mu.Lock()
	pushes := len(em.calls)
	em.mu.Unlock()
	cancel()

	select {
	case err := <-done:
		if !errors.Is(err, context.Canceled) {
			t.Fatalf("Run err = %v, want context.Canceled", err)
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("Run did not return after cancel")
	}
	if pushes == 0 {
		t.Fatal("Run never invoked Emitter.Push — ticker did not fire at all")
	}
}

// BenchmarkLoopTickOnce measures the per-tick cost of the full pipeline
// (collect stub + baseline update + compute + emit to no-op sink).
// Zero-allocation in steady state is the design target; see AC8.
func BenchmarkLoopTickOnce(b *testing.B) {
	col := &benchCollector{}
	em := &benchEmitter{}
	bl, _ := NewBaseliner(DefaultBaselineConfig(), nil)
	sm, _ := NewStateMachine(
		StateConfig{IdleIntervals: 3, WarmupSamples: 0, StaleReadFailures: 3},
		discardLogger(),
	)
	l, err := NewLoop(LoopConfig{
		Baseliner:     bl,
		StateMachine:  sm,
		Emitter:       em,
		Collector:     col,
		ScoreConfig:   DefaultConfig(),
		PushInterval:  time.Second,
		DetectionMode: "fleet",
		Log:           discardLogger(),
		Clock:         func() time.Time { return testTS },
	})
	if err != nil {
		b.Fatal(err)
	}
	// Warm up past CALIBRATING so the hot path is the ACTIVE emission
	// path, not the one-time transition.
	for i := 0; i < 2; i++ {
		l.TickOnce(context.Background())
	}
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		l.TickOnce(context.Background())
	}
}

// benchCollector is allocation-free: it returns the same observation
// every call without touching any heap state.
type benchCollector struct{}

func (benchCollector) Collect(ctx context.Context, now time.Time) (RawObservation, int, error) {
	return RawObservation{Throughput: 100, Compute: 0.9, Memory: 0.9, CPU: 0.9}, 10, nil
}

// benchEmitter captures nothing, allocates nothing.
type benchEmitter struct{}

func (benchEmitter) Push(ctx context.Context, now time.Time, score Score, state State, mode string, degradation bool) error {
	return nil
}
func (benchEmitter) EmitStragglerEvent(ctx context.Context, ev StragglerEvent, isStraggler bool) error {
	return nil
}
func (benchEmitter) FleetReachable() bool  { return true }
func (benchEmitter) Stats() (int64, int64) { return 0, 0 }

func discardLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}
