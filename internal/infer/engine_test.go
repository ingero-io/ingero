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
