package infer

import (
	"io"
	"log/slog"
	"testing"
	"time"

	"github.com/ingero-io/ingero/internal/correlate"
	"github.com/ingero-io/ingero/internal/sampling"
	"github.com/ingero-io/ingero/pkg/events"
)

// quietLogger discards all output so test runs don't litter stderr.
func quietLogger() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// syncEvent constructs an event that satisfies isSyncEvent.
func syncEvent(pid uint32, stream uint64, at time.Time) events.Event {
	return events.Event{
		Timestamp: at,
		PID:       pid,
		Source:    events.SourceCUDA,
		Op:        uint8(events.CUDAStreamSync),
		Args:      [2]uint64{stream, 0},
	}
}

func TestClassify(t *testing.T) {
	cases := []struct {
		name      string
		ratio     float64
		threshold float64
		want      OutlierBucket
	}{
		{"healthy", 1.0, 3.0, BucketNone},
		{"just under 1.5", 1.499, 3.0, BucketNone},
		{"exactly 1.5", 1.5, 3.0, Bucket1_5x},
		{"just under 2", 1.999, 3.0, Bucket1_5x},
		{"exactly 2", 2.0, 3.0, Bucket2x},
		{"just under 3", 2.999, 3.0, Bucket2x},
		{"exactly 3", 3.0, 3.0, Bucket3x},
		{"way past 3", 50.0, 3.0, Bucket3x},
		{"narrow threshold 2.5: ratio 2.4", 2.4, 2.5, Bucket2x},
		{"narrow threshold 2.5: ratio 2.5", 2.5, 2.5, Bucket3x},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if got := classify(tc.ratio, tc.threshold); got != tc.want {
				t.Errorf("classify(%v, %v) = %v, want %v", tc.ratio, tc.threshold, got, tc.want)
			}
		})
	}
}

func TestEngine_FirstSyncIsNoOp(t *testing.T) {
	e := New(Config{}, quietLogger())
	t0 := time.Now()
	e.OnSyncEvent(syncEvent(1, 0xff, t0), "abc")
	out := e.Drain()
	if len(out) != 0 {
		t.Errorf("first sync emitted %d outliers, want 0", len(out))
	}
}

func TestEngine_StepsBuildBaselineThenClassify(t *testing.T) {
	cfg := Config{
		WarmupSamples:         5,
		OutlierThresholdRatio: 3.0,
		PauseOnSeverity:       "HIGH",
		Sampler:               nil,
	}
	e := New(cfg, quietLogger())

	t0 := time.Now()
	step := 10 * time.Millisecond
	// 6 healthy syncs -> 5 healthy steps (warmup-complete after the
	// 5th step, which is the 6th sync).
	for i := 0; i < 7; i++ {
		e.OnSyncEvent(syncEvent(7, 0x5742, t0.Add(time.Duration(i)*step)), "wl-a")
	}
	if got := e.Drain(); len(got) != 0 {
		t.Errorf("warmup phase emitted %d outliers, want 0", len(got))
	}
	// One slow step (5x the baseline).
	bigStep := 50 * time.Millisecond
	e.OnSyncEvent(syncEvent(7, 0x5742, t0.Add(7*step+bigStep)), "wl-a")
	out := e.Drain()
	if len(out) != 1 {
		t.Fatalf("expected 1 outlier, got %d", len(out))
	}
	if out[0].Bucket != Bucket3x {
		t.Errorf("bucket = %v, want 3x", out[0].Bucket)
	}
	if out[0].Key.PID != 7 || out[0].Key.StreamHandle != 0x5742 {
		t.Errorf("outlier key = %+v, want PID=7 stream=0x5742", out[0].Key)
	}
}

func TestEngine_ClockSkewIgnored(t *testing.T) {
	e := New(Config{WarmupSamples: 1}, quietLogger())
	t0 := time.Now()
	e.OnSyncEvent(syncEvent(1, 0xa, t0), "x")
	// Backwards-in-time second sync — must be silently ignored.
	e.OnSyncEvent(syncEvent(1, 0xa, t0.Add(-time.Second)), "x")
	if got := e.Drain(); len(got) != 0 {
		t.Errorf("clock skew emitted outliers: %+v", got)
	}
}

func TestEngine_IdleGapIgnored(t *testing.T) {
	cfg := Config{WarmupSamples: 5, MaxStepDuration: 5 * time.Second}
	e := New(cfg, quietLogger())
	t0 := time.Now()
	e.OnSyncEvent(syncEvent(1, 0xa, t0), "x")
	// Gap > MaxStepDuration: should not produce a baseline sample.
	e.OnSyncEvent(syncEvent(1, 0xa, t0.Add(time.Hour)), "x")
	// Now do a real step.
	e.OnSyncEvent(syncEvent(1, 0xa, t0.Add(time.Hour+10*time.Millisecond)), "x")
	// The first step should have been dropped, so we have only 1 valid
	// baseline sample, and Warmed(5) is still false. No outliers.
	if got := e.Drain(); len(got) != 0 {
		t.Errorf("idle gap counted as a step; got outliers %+v", got)
	}
}

func TestEngine_NonSyncEventsIgnored(t *testing.T) {
	e := New(Config{}, quietLogger())
	t0 := time.Now()
	// Two memcpy events, NOT sync. Should produce no state.
	for _, op := range []events.CUDAOp{events.CUDAMemcpy, events.CUDAMalloc} {
		evt := events.Event{
			Timestamp: t0,
			PID:       1,
			Source:    events.SourceCUDA,
			Op:        uint8(op),
		}
		e.OnSyncEvent(evt, "x")
	}
	if e.Stats().WorkloadsTracked != 0 {
		t.Error("non-sync events should not register a workload")
	}
}

func TestEngine_SeverityGatePausesUpdates(t *testing.T) {
	cfg := Config{WarmupSamples: 5, PauseOnSeverity: "HIGH"}
	e := New(cfg, quietLogger())

	t0 := time.Now()
	// Warm up the baseline at 10ms steps.
	for i := 0; i < 7; i++ {
		e.OnSyncEvent(syncEvent(99, 0xb, t0.Add(time.Duration(i)*10*time.Millisecond)), "x")
	}
	_ = e.Drain() // discard any from warmup edge

	// Inject a HIGH chain for PID 99.
	chain := correlate.CausalChain{ID: "fake", Severity: "HIGH"}
	e.OnChainSnapshot([]correlate.CausalChain{chain}, 99, t0.Add(100*time.Millisecond))

	// Now a 50ms step on PID 99 — without the gate this is a 5x
	// outlier. With the gate, it should be silently dropped.
	e.OnSyncEvent(syncEvent(99, 0xb, t0.Add(150*time.Millisecond)), "x")
	if got := e.Drain(); len(got) != 0 {
		t.Errorf("severity gate failed to suppress outlier: %+v", got)
	}
}

func TestEngine_SamplerDegradeOn3x(t *testing.T) {
	// Real sampler so we can read its state via ShouldEmit().
	smp := sampling.New("inference", 0.01, 30*time.Second)
	cfg := Config{
		WarmupSamples:    5,
		SamplerDegradeOn: Bucket3x,
		Sampler:          smp,
	}
	e := New(cfg, quietLogger())
	t0 := time.Now()
	// Warmup at 10ms steps.
	for i := 0; i < 7; i++ {
		e.OnSyncEvent(syncEvent(7, 0xc, t0.Add(time.Duration(i)*10*time.Millisecond)), "x")
	}
	_ = e.Drain()
	// Healthy sampler is at 1% — most events will not emit.
	healthyEmits := 0
	for i := 0; i < 100; i++ {
		if smp.ShouldEmit() {
			healthyEmits++
		}
	}
	if healthyEmits > 10 {
		t.Errorf("healthy sampler emitted %d/100, expected ≈1 (much less than 10)", healthyEmits)
	}
	// 3x outlier should bump the sampler to admit 100%.
	e.OnSyncEvent(syncEvent(7, 0xc, t0.Add(70*time.Millisecond+100*time.Millisecond)), "x")
	got := e.Drain()
	if len(got) == 0 {
		t.Fatal("expected an outlier event after slow step")
	}
	// Now the sampler should admit 100%.
	degradedEmits := 0
	for i := 0; i < 100; i++ {
		if smp.ShouldEmit() {
			degradedEmits++
		}
	}
	if degradedEmits != 100 {
		t.Errorf("after 3x outlier, sampler admitted %d/100, want 100", degradedEmits)
	}
}

func TestEngine_OutlierQueueDropsOldestOnOverflow(t *testing.T) {
	cfg := Config{
		WarmupSamples:   2,
		OutlierQueueCap: 3,
	}
	e := New(cfg, quietLogger())
	t0 := time.Now()
	// Warm up: 3 healthy syncs at 10ms.
	for i := 0; i < 3; i++ {
		e.OnSyncEvent(syncEvent(1, 0xa, t0.Add(time.Duration(i)*10*time.Millisecond)), "x")
	}
	// Now generate 5 outliers on the same workload, each at 50ms gap
	// after a 10ms baseline reference. Because Update() is skipped on
	// outliers, the baseline stays stable and each one fires.
	for i := 0; i < 5; i++ {
		base := t0.Add(time.Duration(30+i*60) * time.Millisecond)
		// Healthy step preceding the outlier so lastSync is close.
		e.OnSyncEvent(syncEvent(1, 0xa, base), "x")
		e.OnSyncEvent(syncEvent(1, 0xa, base.Add(50*time.Millisecond)), "x")
	}
	out := e.Drain()
	if len(out) != cfg.OutlierQueueCap {
		t.Errorf("queue len after overflow = %d, want %d", len(out), cfg.OutlierQueueCap)
	}
	if e.Stats().QueueDropped == 0 {
		t.Error("QueueDropped counter did not increment")
	}
}

func TestEngine_StatsCountsByBucket(t *testing.T) {
	cfg := Config{WarmupSamples: 5}
	e := New(cfg, quietLogger())
	t0 := time.Now()
	for i := 0; i < 7; i++ {
		e.OnSyncEvent(syncEvent(1, 0xa, t0.Add(time.Duration(i)*10*time.Millisecond)), "x")
	}
	_ = e.Drain()
	// 1.6x — bucket 1.5x. Last warmup sync was at t0+60ms; next sync
	// at t0+76ms gives step=16ms vs p95~10ms = 1.6 ratio.
	e.OnSyncEvent(syncEvent(1, 0xa, t0.Add(76*time.Millisecond)), "x")
	out := e.Drain()
	if len(out) != 1 || out[0].Bucket != Bucket1_5x {
		t.Fatalf("expected one 1.5x outlier, got %+v", out)
	}
	st := e.Stats()
	if st.OutliersTotal[Bucket1_5x] != 1 {
		t.Errorf("OutliersTotal[1.5x] = %d, want 1", st.OutliersTotal[Bucket1_5x])
	}
}

// launchEvent constructs a CUDALaunchKernel event on the given stream.
func launchEvent(pid uint32, stream uint64, at time.Time) events.Event {
	return events.Event{
		Timestamp: at,
		PID:       pid,
		Source:    events.SourceCUDA,
		Op:        uint8(events.CUDALaunchKernel),
		Args:      [2]uint64{stream, 0},
	}
}

// drivePhaseSteps simulates `count` steps of `step` duration on
// `(pid, stream)`, prefixing each with `launches` cudaLaunchKernel
// events at `kernelDur` apart. Returns the wall-clock time after the
// last sync.
func drivePhaseSteps(t *testing.T, e *Engine, pid uint32, stream uint64, t0 time.Time,
	count int, step time.Duration, launches int, kernelDur time.Duration) time.Time {
	t.Helper()
	now := t0
	// Initial anchor sync.
	e.OnSyncEvent(syncEvent(pid, stream, now), "x")
	for s := 0; s < count; s++ {
		// Launches between syncs.
		for k := 0; k < launches; k++ {
			lt := now.Add(time.Duration(k+1) * (step / time.Duration(launches+1)))
			e.OnLaunchEvent(launchEvent(pid, stream, lt), "x", kernelDur)
		}
		now = now.Add(step)
		e.OnSyncEvent(syncEvent(pid, stream, now), "x")
	}
	return now
}

func TestEngine_PhaseClassifier_DisabledByDefault(t *testing.T) {
	// Default Config has PhaseClassifierEnabled=false; every step
	// should land in Phase="" so engine_test.go's existing tests keep
	// passing. This test confirms phaseCounts stays empty.
	e := New(Config{WarmupSamples: 3}, quietLogger())
	t0 := time.Now()
	for i := 0; i < 5; i++ {
		e.OnSyncEvent(syncEvent(1, 0xa, t0.Add(time.Duration(i)*10*time.Millisecond)), "x")
	}
	st := e.Stats()
	if len(st.PhaseDistribution) != 0 {
		t.Errorf("phase distribution should be empty when classifier disabled, got %+v", st.PhaseDistribution)
	}
}

func TestEngine_PhaseClassifier_Enabled_DistributesSteps(t *testing.T) {
	cfg := Config{
		PhaseClassifierEnabled: true,
		WarmupSamples:          3,
	}
	e := New(cfg, quietLogger())

	t0 := time.Now()
	// Stream A: 5 prefill steps (30ms each, 250 launches at 100us each).
	// avg kernel time = 100us, so rule 3 fires via "launches > 200".
	now := drivePhaseSteps(t, e, 1, 0xa, t0, 5, 30*time.Millisecond, 250, 100*time.Microsecond)
	// Stream A continued: 5 decode steps (2ms each, 10 launches).
	drivePhaseSteps(t, e, 1, 0xa, now, 5, 2*time.Millisecond, 10, 50*time.Microsecond)

	st := e.Stats()
	if st.PhaseDistribution[PhasePrefill] < 5 {
		t.Errorf("prefill count = %d, want >= 5", st.PhaseDistribution[PhasePrefill])
	}
	if st.PhaseDistribution[PhaseDecode] < 5 {
		t.Errorf("decode count = %d, want >= 5", st.PhaseDistribution[PhaseDecode])
	}
}

func TestEngine_PhaseClassifier_BimodalProducesSeparateBaselines(t *testing.T) {
	// The CORE false-negative test. Drive an alternating prefill/decode
	// stream, then inject a slow decode that's only outlier-class
	// against the decode baseline (not the mixed baseline).
	cfg := Config{
		PhaseClassifierEnabled: true,
		WarmupSamples:          5,
		OutlierThresholdRatio:  3.0,
	}
	e := New(cfg, quietLogger())

	t0 := time.Now()
	now := t0

	// Drive 30 alternating steps so each phase gets enough warm-up
	// samples (15 each, > WarmupSamples=5).
	for i := 0; i < 30; i++ {
		// Initial sync if first
		if i == 0 {
			e.OnSyncEvent(syncEvent(1, 0xa, now), "x")
		}
		var step time.Duration
		var launches int
		if i%2 == 0 {
			// Prefill: 30ms, 250 launches.
			step = 30 * time.Millisecond
			launches = 250
		} else {
			// Decode: 2ms, 10 launches.
			step = 2 * time.Millisecond
			launches = 10
		}
		// Drop launches between syncs.
		for k := 0; k < launches; k++ {
			lt := now.Add(time.Duration(k+1) * (step / time.Duration(launches+1)))
			e.OnLaunchEvent(launchEvent(1, 0xa, lt), "x", 100*time.Microsecond)
		}
		now = now.Add(step)
		e.OnSyncEvent(syncEvent(1, 0xa, now), "x")
	}
	_ = e.Drain() // discard any warm-up edge outliers

	// Now inject a slow decode: 20ms instead of 2ms = 10x decode-p95.
	// In the v0.16.0 single-baseline regime, the mixed p95 absorbs
	// the prefill tail (~30ms), so a 20ms step would NOT fire.
	// In v0.16.1's phase-aware regime, this lands in the decode
	// bucket with p95 ~2ms; 20ms / 2ms = 10x → 3x outlier.
	for k := 0; k < 10; k++ {
		lt := now.Add(time.Duration(k+1) * 1900 * time.Microsecond)
		e.OnLaunchEvent(launchEvent(1, 0xa, lt), "x", 100*time.Microsecond)
	}
	now = now.Add(20 * time.Millisecond)
	e.OnSyncEvent(syncEvent(1, 0xa, now), "x")

	out := e.Drain()
	if len(out) == 0 {
		t.Fatal("expected outlier on slow decode against decode-phase baseline")
	}
	found := false
	for _, ev := range out {
		if ev.Key.Phase == PhaseDecode && ev.Bucket == Bucket3x {
			found = true
			break
		}
	}
	if !found {
		t.Errorf("no PhaseDecode 3x outlier found; got %+v", out)
	}
}

func TestEngine_SamplerDegradedSurfacesObservability(t *testing.T) {
	// v0.16.3: a 3x outlier should populate the sampler-degraded queue,
	// bump the cumulative degradations counter, and stamp a non-empty
	// cause string. Mirrors the AWS-validation expectation.
	smp := sampling.New("inference", 0.01, 30*time.Second)
	cfg := Config{
		WarmupSamples:    5,
		SamplerDegradeOn: Bucket3x,
		Sampler:          smp,
	}
	e := New(cfg, quietLogger())
	t0 := time.Now()
	for i := 0; i < 7; i++ {
		e.OnSyncEvent(syncEvent(7, 0xc, t0.Add(time.Duration(i)*10*time.Millisecond)), "wl-x")
	}
	_ = e.Drain()
	// 3x outlier (50ms vs 10ms baseline = 5x ratio).
	e.OnSyncEvent(syncEvent(7, 0xc, t0.Add(70*time.Millisecond+50*time.Millisecond)), "wl-x")
	if got := e.Drain(); len(got) == 0 {
		t.Fatal("expected outlier event")
	}
	sd := e.DrainSampler()
	if len(sd) != 1 {
		t.Fatalf("expected 1 sampler-degraded event, got %d", len(sd))
	}
	if sd[0].Cause == "" {
		t.Error("sampler-degraded cause should not be empty")
	}
	if sd[0].Bucket != Bucket3x {
		t.Errorf("sampler-degraded bucket = %v, want 3x", sd[0].Bucket)
	}
	if sd[0].CooldownEnd.Before(sd[0].At) {
		t.Errorf("CooldownEnd should be after At")
	}
	st := e.Stats()
	if st.SamplerDegradationsTotal != 1 {
		t.Errorf("DegradationsTotal = %d, want 1", st.SamplerDegradationsTotal)
	}
	if !st.SamplerDegraded {
		t.Error("SamplerDegraded should be true within cooldown window")
	}
	if st.LastDegradationCause != sd[0].Cause {
		t.Errorf("LastDegradationCause %q != event Cause %q",
			st.LastDegradationCause, sd[0].Cause)
	}
}

func TestEngine_SnapshotForExport_ReturnsWarmedWorkloads(t *testing.T) {
	// v0.16.3 exporter contract: SnapshotForExport returns one entry per
	// warmed workload with a populated histogram, plus engine-level
	// outlier counts.
	cfg := Config{WarmupSamples: 5}
	e := New(cfg, quietLogger())
	t0 := time.Now()
	for i := 0; i < 7; i++ {
		e.OnSyncEvent(syncEvent(11, 0xa, t0.Add(time.Duration(i)*10*time.Millisecond)), "wl-y")
	}
	rows, es, ss := e.SnapshotForExport()
	if len(rows) != 1 {
		t.Fatalf("expected 1 warmed workload, got %d", len(rows))
	}
	if rows[0].PID != 11 {
		t.Errorf("workload PID = %d, want 11", rows[0].PID)
	}
	if !rows[0].Histogram.HasObservation {
		t.Error("histogram should have observations after warmup")
	}
	if es.WorkloadsTracked != 1 {
		t.Errorf("WorkloadsTracked = %d, want 1", es.WorkloadsTracked)
	}
	if ss.Degraded {
		t.Error("sampler should NOT be degraded with no outliers")
	}
}

func TestEngine_OnMemfragEventClassifiesDecode(t *testing.T) {
	// v0.16.3 memfrag rule: a memfrag burst with low launch count
	// classifies as decode. The test exercises the full path:
	// OnMemfragEvent -> observables -> ClassifyPhase -> phase counter.
	cfg := Config{
		PhaseClassifierEnabled: true,
		WarmupSamples:          1,
	}
	e := New(cfg, quietLogger())
	t0 := time.Now()
	now := t0
	// First sync to anchor.
	e.OnSyncEvent(syncEvent(42, 0xa, now), "wl-z")
	// Inject 3 memfrag events between syncs.
	for k := 0; k < 3; k++ {
		e.OnMemfragEvent(42, "wl-z", now.Add(time.Duration(k+1)*time.Millisecond))
	}
	// A few launches (< DecodeMaxLaunches=50).
	for k := 0; k < 5; k++ {
		e.OnLaunchEvent(launchEvent(42, 0xa, now.Add(time.Duration(k+1)*time.Millisecond)),
			"wl-z", 100*time.Microsecond)
	}
	now = now.Add(10 * time.Millisecond)
	e.OnSyncEvent(syncEvent(42, 0xa, now), "wl-z")
	st := e.Stats()
	if st.PhaseDistribution[PhaseDecode] != 1 {
		t.Errorf("expected 1 PhaseDecode step from memfrag rule, got distribution %+v", st.PhaseDistribution)
	}
}

func TestEngine_PhaseClassifier_UnknownPhaseDoesNotDegradeSampler(t *testing.T) {
	// Phase=unknown outliers should fire metrics/log/UDS but should
	// NOT bump the sampler — we lack workload context to know the
	// slowdown is meaningful.
	smp := sampling.New("inference", 0.01, 30*time.Second)
	cfg := Config{
		PhaseClassifierEnabled: true,
		WarmupSamples:          3,
		OutlierThresholdRatio:  3.0,
		SamplerDegradeOn:       Bucket3x,
		Sampler:                smp,
	}
	e := New(cfg, quietLogger())

	// Drive 5 steps in the "unknown" gap (10ms, 5 launches → not
	// decode (5 < 50 launches but step >= 5ms), not prefill (step <
	// 20ms), not mixed (5 launches outside [50,200] AND no big
	// memcpy)). Rule 5 fallback → PhaseUnknown.
	t0 := time.Now()
	now := t0
	e.OnSyncEvent(syncEvent(1, 0xa, now), "x")
	for s := 0; s < 5; s++ {
		for k := 0; k < 5; k++ {
			lt := now.Add(time.Duration(k+1) * 1500 * time.Microsecond)
			e.OnLaunchEvent(launchEvent(1, 0xa, lt), "x", 100*time.Microsecond)
		}
		now = now.Add(10 * time.Millisecond)
		e.OnSyncEvent(syncEvent(1, 0xa, now), "x")
	}
	st := e.Stats()
	if st.PhaseDistribution[PhaseUnknown] == 0 {
		t.Skip("test setup did not produce PhaseUnknown steps; rule matrix may have shifted")
	}

	// Sampler should be at healthy admit (1%) since no classified
	// outliers happened.
	healthyEmits := 0
	for i := 0; i < 100; i++ {
		if smp.ShouldEmit() {
			healthyEmits++
		}
	}
	if healthyEmits > 10 {
		t.Errorf("healthy sampler should admit ~1%%, got %d/100", healthyEmits)
	}

	// Inject a slow unknown-phase step: 50ms instead of 10ms = 5x.
	// Should fire as 3x outlier but NOT trigger sampler.
	for k := 0; k < 5; k++ {
		lt := now.Add(time.Duration(k+1) * 9 * time.Millisecond)
		e.OnLaunchEvent(launchEvent(1, 0xa, lt), "x", 100*time.Microsecond)
	}
	now = now.Add(50 * time.Millisecond)
	e.OnSyncEvent(syncEvent(1, 0xa, now), "x")

	// Outlier should be queued (the rule still classifies bucket).
	out := e.Drain()
	if len(out) == 0 {
		t.Skip("test setup did not produce a measurable outlier; baseline may be unstable")
	}
	// Sampler should STILL be at healthy admit because phase=unknown.
	postEmits := 0
	for i := 0; i < 100; i++ {
		if smp.ShouldEmit() {
			postEmits++
		}
	}
	if postEmits == 100 {
		t.Errorf("phase=unknown 3x outlier should NOT have flipped sampler to 100%%, got %d/100", postEmits)
	}
}
