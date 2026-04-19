package health

import (
	"log/slog"
	"math"
	"sync"
	"testing"
)

// freshBaseliner returns a baseliner with default config for tests.
func freshBaseliner(t *testing.T) Baseliner {
	t.Helper()
	b, err := NewBaseliner(DefaultBaselineConfig(), nil)
	if err != nil {
		t.Fatalf("NewBaseliner: %v", err)
	}
	return b
}

func TestDefaultBaselineConfig_Valid(t *testing.T) {
	if err := DefaultBaselineConfig().Validate(); err != nil {
		t.Fatalf("default config invalid: %v", err)
	}
}

func TestBaselineConfig_Validate(t *testing.T) {
	ok := DefaultBaselineConfig()
	tests := []struct {
		name    string
		cfg     BaselineConfig
		wantErr bool
	}{
		{"defaults", ok, false},
		{"fast_alpha_zero", BaselineConfig{FastAlpha: 0, FloorAlpha: 0.001, WarningRatio: 0.5, WarmupSamples: 30, FloorWarmthMin: 0.01}, true},
		{"fast_alpha_one", BaselineConfig{FastAlpha: 1.0, FloorAlpha: 0.001, WarningRatio: 0.5, WarmupSamples: 30, FloorWarmthMin: 0.01}, true},
		{"floor_ge_fast", BaselineConfig{FastAlpha: 0.1, FloorAlpha: 0.2, WarningRatio: 0.5, WarmupSamples: 30, FloorWarmthMin: 0.01}, true},
		{"floor_alpha_zero", BaselineConfig{FastAlpha: 0.1, FloorAlpha: 0, WarningRatio: 0.5, WarmupSamples: 30, FloorWarmthMin: 0.01}, true},
		{"warning_ratio_zero", BaselineConfig{FastAlpha: 0.1, FloorAlpha: 0.001, WarningRatio: 0, WarmupSamples: 30, FloorWarmthMin: 0.01}, true},
		{"warning_ratio_over_one", BaselineConfig{FastAlpha: 0.1, FloorAlpha: 0.001, WarningRatio: 1.5, WarmupSamples: 30, FloorWarmthMin: 0.01}, true},
		{"warmup_negative", BaselineConfig{FastAlpha: 0.1, FloorAlpha: 0.001, WarningRatio: 0.5, WarmupSamples: -1, FloorWarmthMin: 0.01}, true},
		{"floor_warmth_negative", BaselineConfig{FastAlpha: 0.1, FloorAlpha: 0.001, WarningRatio: 0.5, WarmupSamples: 30, FloorWarmthMin: -0.1}, true},
		{"floor_warmth_one", BaselineConfig{FastAlpha: 0.1, FloorAlpha: 0.001, WarningRatio: 0.5, WarmupSamples: 30, FloorWarmthMin: 1.0}, true},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.cfg.Validate()
			if tc.wantErr && err == nil {
				t.Fatal("expected error, got nil")
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

// AC1: Bias-corrected EMA should yield the input value at sample 1,
// not the raw EMA value (alpha*x).
func TestBiasCorrection_FirstSampleIsMeaningful(t *testing.T) {
	b := freshBaseliner(t)
	obs := RawObservation{Throughput: 100, Compute: 0.9, Memory: 0.8, CPU: 0.7}
	b.Update(obs)
	cur := b.Current()
	if math.Abs(cur.Throughput-100) > 1e-9 {
		t.Fatalf("Throughput bias-corrected = %v, want 100", cur.Throughput)
	}
	if math.Abs(cur.Compute-0.9) > 1e-9 {
		t.Fatalf("Compute = %v, want 0.9", cur.Compute)
	}
}

// After sample 1, raw EMA equals alpha*x — a biased underestimate.
// Bias-corrected EMA equals x. Asserting the gap proves bias correction
// is actually being applied (not a silent no-op).
func TestBiasCorrection_DoesSomethingAtSample1(t *testing.T) {
	b := freshBaseliner(t)
	b.Update(RawObservation{Throughput: 100, Compute: 0.9, Memory: 0.8, CPU: 0.7})
	snap := b.Snapshot()
	corrected := b.Current()
	// Raw EMA = alpha * x = 0.1 * 100 = 10. Corrected = 100.
	if math.Abs(snap.FastEMA.Throughput-10) > 1e-9 {
		t.Fatalf("raw fast EMA at t=1 = %v, want 10 (alpha*x)", snap.FastEMA.Throughput)
	}
	if math.Abs(corrected.Throughput-100) > 1e-9 {
		t.Fatalf("corrected fast EMA at t=1 = %v, want 100 (x)", corrected.Throughput)
	}
	// Sanity: raw != corrected proves bias correction is actually applied.
	if snap.FastEMA.Throughput == corrected.Throughput {
		t.Fatal("bias correction appears to be a no-op (raw == corrected)")
	}
}

// AC2: Fast EMA with alpha=0.1 should visibly adapt over ~30 samples to a
// step change.
func TestFastEMA_AdaptsToStepChange(t *testing.T) {
	b := freshBaseliner(t)
	// Warm to 0.9.
	for i := 0; i < 100; i++ {
		b.Update(RawObservation{Throughput: 100, Compute: 0.9, Memory: 0.9, CPU: 0.9})
	}
	before := b.Current().Compute
	if math.Abs(before-0.9) > 1e-3 {
		t.Fatalf("pre-step EMA = %v, want ~0.9", before)
	}
	// Step down to 0.5 for 30 samples.
	for i := 0; i < 30; i++ {
		b.Update(RawObservation{Throughput: 100, Compute: 0.5, Memory: 0.9, CPU: 0.9})
	}
	after := b.Current().Compute
	if after >= before {
		t.Fatalf("fast EMA should have moved down: before=%v after=%v", before, after)
	}
	// Should move substantially toward 0.5 (past halfway).
	if after > 0.7 {
		t.Fatalf("fast EMA did not adapt enough after 30 samples: %v", after)
	}
}

// AC3: Hard-floor EMA (alpha=0.001) must be substantially less reactive
// than the fast EMA (alpha=0.1) to a step change. Verify by position:
// after 30 samples at 0.5 the fast EMA should be close to 0.5 while the
// hard floor should remain closer to the prior baseline 0.9.
func TestHardFloor_FarLessReactiveThanFastEMA(t *testing.T) {
	b := freshBaseliner(t)
	for i := 0; i < 100; i++ {
		b.Update(RawObservation{Throughput: 100, Compute: 0.9, Memory: 0.9, CPU: 0.9})
	}
	for i := 0; i < 30; i++ {
		b.Update(RawObservation{Throughput: 100, Compute: 0.5, Memory: 0.9, CPU: 0.9})
	}
	fast := b.Current().Compute
	floor := b.HardFloor().Compute
	// Fast EMA should have adapted most of the way to 0.5.
	if fast > 0.6 {
		t.Fatalf("fast EMA did not adapt to step: fast=%v, want <0.6", fast)
	}
	// Hard floor should still be well above the step value — at least 70%
	// of the way back to the pre-step baseline of 0.9. That is the whole
	// point of having a long-memory baseline for slow-degradation detection.
	if floor < 0.78 {
		t.Fatalf("hard floor tracked step too quickly: floor=%v, want >=0.78", floor)
	}
}

// AC4: After slow degradation fast_ema/hard_floor < 0.5 triggers warning.
func TestDegradationWarning_TriggersOnSlowDrift(t *testing.T) {
	b := freshBaseliner(t)
	// Warm with healthy samples so hard floor settles near 0.9.
	for i := 0; i < 200; i++ {
		b.Update(RawObservation{Throughput: 100, Compute: 0.9, Memory: 0.9, CPU: 0.9})
	}
	if b.DegradationWarning() {
		t.Fatal("warning triggered during healthy warmup")
	}
	// Drop compute to 0.2 and let fast EMA adapt while hard floor holds.
	for i := 0; i < 100; i++ {
		b.Update(RawObservation{Throughput: 100, Compute: 0.2, Memory: 0.9, CPU: 0.9})
	}
	if !b.DegradationWarning() {
		fast := b.Current().Compute
		floor := b.HardFloor().Compute
		t.Fatalf("warning not triggered: fast=%v floor=%v ratio=%v", fast, floor, fast/floor)
	}
}

// AC: Warmup gating. Warning must stay false until sample_count >= warmup.
func TestDegradationWarning_SuppressedDuringWarmup(t *testing.T) {
	cfg := DefaultBaselineConfig()
	cfg.WarmupSamples = 50
	b, err := NewBaseliner(cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	// Push 10 terrible samples — ratio would trigger, but warmup gate holds.
	for i := 0; i < 10; i++ {
		b.Update(RawObservation{Throughput: 10, Compute: 0.01, Memory: 0.01, CPU: 0.01})
	}
	if b.DegradationWarning() {
		t.Fatal("warning fired before warmup threshold")
	}
}

// AC5: Baseliner exposes the methods listed in the interface.
func TestBaseliner_InterfaceMethods(t *testing.T) {
	b := freshBaseliner(t)
	obs := RawObservation{Throughput: 50, Compute: 0.8, Memory: 0.7, CPU: 0.6}
	b.Update(obs)
	if b.SampleCount() != 1 {
		t.Fatalf("SampleCount = %d, want 1", b.SampleCount())
	}
	if b.Current() == (Baselines{}) {
		t.Fatal("Current() returned zero baselines after 1 update")
	}
	if b.HardFloor() == (Baselines{}) {
		t.Fatal("HardFloor() returned zero baselines after 1 update")
	}
	snap := b.Snapshot()
	if snap.SampleCount != 1 {
		t.Fatalf("Snapshot.SampleCount = %d", snap.SampleCount)
	}
	b.Reset()
	if b.SampleCount() != 0 {
		t.Fatalf("after Reset SampleCount = %d, want 0", b.SampleCount())
	}
}

// AC6: Signals throughput is normalized against the baseline.
func TestSignals_ThroughputNormalized(t *testing.T) {
	b := freshBaseliner(t)
	// Warm to baseline 100.
	for i := 0; i < 50; i++ {
		b.Update(RawObservation{Throughput: 100, Compute: 0.9, Memory: 0.9, CPU: 0.9})
	}
	// Current observation at 50 should give ratio ~0.5 (after further
	// updating the fast EMA moves, but the pre-Signals baseline ratio is
	// captured from Current()).
	sig := b.Signals(RawObservation{Throughput: 50, Compute: 0.9, Memory: 0.9, CPU: 0.9})
	if sig.Throughput < 0.4 || sig.Throughput > 0.6 {
		t.Fatalf("throughput ratio = %v, want ~0.5", sig.Throughput)
	}
	if sig.Compute != 0.9 {
		t.Fatalf("Compute pass-through = %v, want 0.9", sig.Compute)
	}
}

// AC6 edge: during calibration (no samples yet) baseline is zero and ratio
// falls to 0 cleanly — caller will see a low score but CALIBRATING state
// machine gate (Story 2.3) prevents classification.
func TestSignals_ZeroBaselineProducesZeroRatio(t *testing.T) {
	b := freshBaseliner(t)
	sig := b.Signals(RawObservation{Throughput: 100, Compute: 0.9, Memory: 0.9, CPU: 0.9})
	if sig.Throughput != 0 {
		t.Fatalf("zero-baseline ratio = %v, want 0", sig.Throughput)
	}
}

func TestUpdate_NaNCoercedToZero(t *testing.T) {
	b := freshBaseliner(t)
	b.Update(RawObservation{Throughput: math.NaN(), Compute: math.Inf(1), Memory: 0.5, CPU: 0.5})
	if math.IsNaN(b.Current().Throughput) || math.IsInf(b.Current().Throughput, 0) {
		t.Fatalf("NaN/Inf leaked into EMA: %+v", b.Current())
	}
}

// Stronger: a NaN dropped into an already-warm EMA must not poison the
// running baseline. After the bad sample, the EMA should be shifted only
// by the alpha-weighted-zero coercion, never contain NaN.
func TestUpdate_NaNDoesNotPoisonWarmEMA(t *testing.T) {
	b := freshBaseliner(t)
	for i := 0; i < 100; i++ {
		b.Update(RawObservation{Throughput: 100, Compute: 0.9, Memory: 0.9, CPU: 0.9})
	}
	before := b.Current()
	b.Update(RawObservation{Throughput: math.NaN(), Compute: math.Inf(1), Memory: math.Inf(-1), CPU: math.NaN()})
	after := b.Current()
	if !isFinite(after.Throughput) || !isFinite(after.Compute) || !isFinite(after.Memory) || !isFinite(after.CPU) {
		t.Fatalf("NaN/Inf poisoned warm EMA: %+v", after)
	}
	// NaN inputs are coerced to zero, so the EMA should have dropped by
	// exactly one alpha-step from the prior value.
	wantThroughput := 0.9*before.Throughput + 0.1*0
	if math.Abs(after.Throughput-wantThroughput) > 1e-3 {
		t.Fatalf("throughput after coerced NaN = %v, want ~%v (1 alpha-step toward 0)", after.Throughput, wantThroughput)
	}
}

func TestSnapshotRestore_Roundtrip(t *testing.T) {
	src := freshBaseliner(t)
	for i := 0; i < 40; i++ {
		src.Update(RawObservation{Throughput: 100, Compute: 0.9, Memory: 0.8, CPU: 0.7})
	}
	snap := src.Snapshot()

	dst := freshBaseliner(t)
	if err := dst.Restore(snap); err != nil {
		t.Fatalf("Restore: %v", err)
	}
	if dst.SampleCount() != src.SampleCount() {
		t.Fatalf("SampleCount mismatch: src=%d dst=%d", src.SampleCount(), dst.SampleCount())
	}
	if dst.Current() != src.Current() {
		t.Fatalf("Current mismatch: src=%+v dst=%+v", src.Current(), dst.Current())
	}
	if dst.HardFloor() != src.HardFloor() {
		t.Fatalf("HardFloor mismatch: src=%+v dst=%+v", src.HardFloor(), dst.HardFloor())
	}
}

func TestRestore_RejectsBadSchema(t *testing.T) {
	b := freshBaseliner(t)
	err := b.Restore(PersistedState{SchemaVersion: 99})
	if err == nil {
		t.Fatal("expected error for bad schema version")
	}
}

func TestRestore_RejectsNegativeSampleCount(t *testing.T) {
	b := freshBaseliner(t)
	err := b.Restore(PersistedState{SchemaVersion: 1, SampleCount: -5})
	if err == nil {
		t.Fatal("expected error for negative sample_count")
	}
}

func TestReset_ClearsState(t *testing.T) {
	b := freshBaseliner(t)
	for i := 0; i < 50; i++ {
		b.Update(RawObservation{Throughput: 100, Compute: 0.9, Memory: 0.8, CPU: 0.7})
	}
	b.Reset()
	if b.SampleCount() != 0 {
		t.Fatalf("SampleCount after Reset = %d, want 0", b.SampleCount())
	}
	if b.Current() != (Baselines{}) {
		t.Fatalf("Current after Reset = %+v, want zero", b.Current())
	}
	if b.HardFloor() != (Baselines{}) {
		t.Fatalf("HardFloor after Reset = %+v, want zero", b.HardFloor())
	}
}

// Alpha change between save and restore must be rejected — the raw EMA
// saved under one alpha cannot be interpreted under another.
func TestRestore_RejectsAlphaMismatch(t *testing.T) {
	a := freshBaseliner(t)
	a.Update(RawObservation{Throughput: 100, Compute: 0.9, Memory: 0.9, CPU: 0.9})
	snap := a.Snapshot()

	// Rebuild with a different FastAlpha. Restore should reject.
	cfg := DefaultBaselineConfig()
	cfg.FastAlpha = 0.3
	cfg.FloorAlpha = 0.01
	b, err := NewBaseliner(cfg, nil)
	if err != nil {
		t.Fatal(err)
	}
	if err := b.Restore(snap); err == nil {
		t.Fatal("expected alpha-mismatch error, got nil")
	}
}

// Forged NaN/Inf snapshots must be rejected at Restore.
func TestRestore_RejectsNonFinite(t *testing.T) {
	b := freshBaseliner(t)
	cfg := DefaultBaselineConfig()
	bad := PersistedState{
		SchemaVersion: 1,
		SampleCount:   10,
		FastAlpha:     cfg.FastAlpha,
		FloorAlpha:    cfg.FloorAlpha,
		FastEMA:       Baselines{Throughput: math.NaN(), Compute: 0, Memory: 0, CPU: 0},
	}
	if err := b.Restore(bad); err == nil {
		t.Fatal("expected NaN-rejection error, got nil")
	}
	bad2 := PersistedState{
		SchemaVersion: 1,
		SampleCount:   10,
		FastAlpha:     cfg.FastAlpha,
		FloorAlpha:    cfg.FloorAlpha,
		HardFloor:     Baselines{Throughput: math.Inf(1)},
	}
	if err := b.Restore(bad2); err == nil {
		t.Fatal("expected Inf-rejection error, got nil")
	}
}

// Negative baseline values must be rejected.
func TestRestore_RejectsNegativeBaseline(t *testing.T) {
	b := freshBaseliner(t)
	cfg := DefaultBaselineConfig()
	bad := PersistedState{
		SchemaVersion: 1,
		SampleCount:   10,
		FastAlpha:     cfg.FastAlpha,
		FloorAlpha:    cfg.FloorAlpha,
		FastEMA:       Baselines{Throughput: -1, Compute: 0, Memory: 0, CPU: 0},
	}
	if err := b.Restore(bad); err == nil {
		t.Fatal("expected negative-rejection error, got nil")
	}
}

// FloorWarmthMin gates DegradationWarning: during warmup, a signal whose
// hard_floor is below warmth should not trigger even if the ratio is bad.
func TestDegradationWarning_GatedByFloorWarmth(t *testing.T) {
	cfg := DefaultBaselineConfig()
	cfg.WarmupSamples = 5
	cfg.FloorWarmthMin = 0.5
	b, _ := NewBaseliner(cfg, nil)
	// 5 samples of tiny healthy value — warmup crosses but floor is still low.
	for i := 0; i < 5; i++ {
		b.Update(RawObservation{Throughput: 0.1, Compute: 0.01, Memory: 0.01, CPU: 0.01})
	}
	// Ratio fast/floor would be ~1.0 (same input), but regardless, floor is
	// below warmth threshold so no warning.
	if b.DegradationWarning() {
		t.Fatal("warning should be gated by floor warmth")
	}
}

// Edge-triggered slog.Warn: only fires on false->true transition, not on
// every sample while warning is true.
func TestDegradationWarning_EdgeTriggeredLog(t *testing.T) {
	cfg := DefaultBaselineConfig()
	cfg.WarmupSamples = 20
	// Capture the log output.
	var buf testLogBuffer
	log := slog.New(slog.NewTextHandler(&buf, &slog.HandlerOptions{Level: slog.LevelWarn}))
	b, _ := NewBaseliner(cfg, log)
	// Warm with healthy data for 200 samples — floor should settle high
	// enough to clear FloorWarmthMin.
	for i := 0; i < 200; i++ {
		b.Update(RawObservation{Throughput: 100, Compute: 0.9, Memory: 0.9, CPU: 0.9})
	}
	if b.DegradationWarning() {
		t.Fatal("unexpected warning during warmup")
	}
	// Drop compute to 0.2 and watch for the warning to fire exactly once.
	for i := 0; i < 100; i++ {
		b.Update(RawObservation{Throughput: 100, Compute: 0.2, Memory: 0.9, CPU: 0.9})
	}
	// Query several times: should fire once.
	for i := 0; i < 5; i++ {
		_ = b.DegradationWarning()
	}
	warnings := buf.CountContaining("health degradation warning")
	if warnings != 1 {
		t.Fatalf("expected exactly 1 edge-triggered warning log, got %d", warnings)
	}
}

// Concurrent Update and Snapshot must not race (verified by test not
// panicking or producing torn reads — -race detector adds the real signal
// when available on Linux CI).
func TestBaseliner_ConcurrentAccess(t *testing.T) {
	b := freshBaseliner(t)
	done := make(chan struct{})
	// Writer
	go func() {
		for i := 0; i < 1000; i++ {
			b.Update(RawObservation{Throughput: 100, Compute: 0.9, Memory: 0.9, CPU: 0.9})
		}
		close(done)
	}()
	// Concurrent readers
	readDone := make(chan struct{}, 4)
	for r := 0; r < 4; r++ {
		go func() {
			for i := 0; i < 500; i++ {
				_ = b.Current()
				_ = b.HardFloor()
				_ = b.Snapshot()
				_ = b.SampleCount()
				_ = b.DegradationWarning()
			}
			readDone <- struct{}{}
		}()
	}
	<-done
	for r := 0; r < 4; r++ {
		<-readDone
	}
	if b.SampleCount() != 1000 {
		t.Fatalf("SampleCount = %d, want 1000", b.SampleCount())
	}
}

// testLogBuffer is a simple concurrent-safe byte slice used to capture slog output.
type testLogBuffer struct {
	mu  sync.Mutex
	buf []byte
}

func (t *testLogBuffer) Write(p []byte) (int, error) {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.buf = append(t.buf, p...)
	return len(p), nil
}

func (t *testLogBuffer) String() string {
	t.mu.Lock()
	defer t.mu.Unlock()
	return string(t.buf)
}

func (t *testLogBuffer) CountContaining(substr string) int {
	s := t.String()
	n := 0
	for i := 0; i+len(substr) <= len(s); i++ {
		if s[i:i+len(substr)] == substr {
			n++
			i += len(substr) - 1
		}
	}
	return n
}

func BenchmarkBaseliner_Update(b *testing.B) {
	bl, _ := NewBaseliner(DefaultBaselineConfig(), nil)
	obs := RawObservation{Throughput: 100, Compute: 0.9, Memory: 0.9, CPU: 0.9}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		bl.Update(obs)
	}
}
