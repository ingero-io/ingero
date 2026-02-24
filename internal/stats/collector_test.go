package stats

import (
	"math"
	"testing"
	"time"

	"github.com/ingero-io/ingero/pkg/events"
)

// ---------------------------------------------------------------------------
// Helper to create test events
// ---------------------------------------------------------------------------

// makeEvent creates an event with the given op and duration.
func makeEvent(op events.CUDAOp, dur time.Duration) events.Event {
	return events.Event{
		Timestamp: time.Now(),
		PID:       1234,
		TID:       1234,
		Source:    events.SourceCUDA,
		Op:        uint8(op),
		Duration:  dur,
	}
}

// ---------------------------------------------------------------------------
// Percentile computation tests
// ---------------------------------------------------------------------------

// TestPercentileEmpty verifies that percentile of an empty buffer returns 0.
func TestPercentileEmpty(t *testing.T) {
	samples := make([]time.Duration, 10)
	got := computePercentile(samples, 0, false, 0.50)
	if got != 0 {
		t.Errorf("expected 0 for empty buffer, got %v", got)
	}
}

// TestPercentileSingleElement verifies percentile with one sample.
func TestPercentileSingleElement(t *testing.T) {
	samples := make([]time.Duration, 10)
	samples[0] = 5 * time.Millisecond
	got := computePercentile(samples, 1, false, 0.50)
	if got != 5*time.Millisecond {
		t.Errorf("expected 5ms for single element, got %v", got)
	}
}

// TestPercentileKnownValues tests percentiles against hand-computed values.
func TestPercentileKnownValues(t *testing.T) {
	// 10 sorted samples: 1, 2, 3, 4, 5, 6, 7, 8, 9, 10 milliseconds
	samples := make([]time.Duration, 10)
	for i := range samples {
		samples[i] = time.Duration(i+1) * time.Millisecond
	}

	tests := []struct {
		name string
		pct  float64
		want time.Duration
	}{
		{"p50 of 10 elements", 0.50, 5 * time.Millisecond},
		{"p95 of 10 elements", 0.95, 10 * time.Millisecond},
		{"p99 of 10 elements", 0.99, 10 * time.Millisecond},
		{"p10 of 10 elements", 0.10, 1 * time.Millisecond},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			// full=true means all 10 elements are valid
			got := computePercentile(samples, 0, true, tt.pct)
			if got != tt.want {
				t.Errorf("computePercentile(pct=%.2f) = %v, want %v", tt.pct, got, tt.want)
			}
		})
	}
}

// TestPercentilePartialBuffer tests percentile when buffer isn't full yet.
func TestPercentilePartialBuffer(t *testing.T) {
	samples := make([]time.Duration, 100)
	// Only 5 elements written: 10, 20, 30, 40, 50 µs
	for i := 0; i < 5; i++ {
		samples[i] = time.Duration((i+1)*10) * time.Microsecond
	}

	// p50 of [10, 20, 30, 40, 50] should be 30 µs (middle element)
	got := computePercentile(samples, 5, false, 0.50)
	if got != 30*time.Microsecond {
		t.Errorf("expected 30µs for partial buffer p50, got %v", got)
	}
}

// TestPercentileDoesNotMutateSamples ensures sort doesn't corrupt the buffer.
func TestPercentileDoesNotMutateSamples(t *testing.T) {
	samples := []time.Duration{5, 3, 1, 4, 2} // unsorted
	original := make([]time.Duration, len(samples))
	copy(original, samples)

	computePercentile(samples, 0, true, 0.50)

	for i, v := range samples {
		if v != original[i] {
			t.Errorf("samples[%d] changed from %v to %v", i, original[i], v)
		}
	}
}

// ---------------------------------------------------------------------------
// Collector basic tests
// ---------------------------------------------------------------------------

// TestCollectorRecord verifies that Record() tracks events correctly.
func TestCollectorRecord(t *testing.T) {
	c := New(WithWindowSize(100))

	// Record 10 cudaMalloc events at 1ms each.
	for i := 0; i < 10; i++ {
		c.Record(makeEvent(events.CUDAMalloc, 1*time.Millisecond))
	}

	if c.TotalEvents() != 10 {
		t.Errorf("TotalEvents() = %d, want 10", c.TotalEvents())
	}

	snap := c.Snapshot()
	if len(snap.Ops) != 1 {
		t.Fatalf("expected 1 operation, got %d", len(snap.Ops))
	}

	op := snap.Ops[0]
	if op.Op != "cudaMalloc" {
		t.Errorf("op name = %q, want %q", op.Op, "cudaMalloc")
	}
	if op.Count != 10 {
		t.Errorf("count = %d, want 10", op.Count)
	}
	if op.P50 != 1*time.Millisecond {
		t.Errorf("p50 = %v, want 1ms", op.P50)
	}
}

// TestCollectorMultipleOps verifies tracking multiple operation types.
func TestCollectorMultipleOps(t *testing.T) {
	// Use a fixed start time far enough in the past so wallClock is large
	// enough for TimeFraction values to be finite and meaningfully different.
	c := New(WithWindowSize(100), WithStartTime(time.Now().Add(-10*time.Second)))

	// Mix of operations with different latencies.
	for i := 0; i < 50; i++ {
		c.Record(makeEvent(events.CUDAMalloc, 1*time.Millisecond))
		c.Record(makeEvent(events.CUDAMemcpy, 5*time.Millisecond))
		c.Record(makeEvent(events.CUDALaunchKernel, 100*time.Microsecond))
	}

	snap := c.Snapshot()
	if snap.TotalEvents != 150 {
		t.Errorf("TotalEvents = %d, want 150", snap.TotalEvents)
	}
	if len(snap.Ops) != 3 {
		t.Errorf("expected 3 operations, got %d", len(snap.Ops))
	}

	// Ops should be sorted by TimeFraction descending.
	// cudaMemcpy (5ms each) should dominate.
	if snap.Ops[0].Op != "cudaMemcpy" {
		t.Errorf("highest time-fraction op = %q, want cudaMemcpy", snap.Ops[0].Op)
	}
}

// TestCollectorMinMax verifies min/max tracking.
func TestCollectorMinMax(t *testing.T) {
	c := New(WithWindowSize(100))

	c.Record(makeEvent(events.CUDAMalloc, 5*time.Millisecond))
	c.Record(makeEvent(events.CUDAMalloc, 1*time.Millisecond))
	c.Record(makeEvent(events.CUDAMalloc, 10*time.Millisecond))

	snap := c.Snapshot()
	op := snap.Ops[0]

	if op.Min != 1*time.Millisecond {
		t.Errorf("min = %v, want 1ms", op.Min)
	}
	if op.Max != 10*time.Millisecond {
		t.Errorf("max = %v, want 10ms", op.Max)
	}
}

// ---------------------------------------------------------------------------
// Time-fraction tests
// ---------------------------------------------------------------------------

// TestTimeFraction verifies time-fraction computation.
func TestTimeFraction(t *testing.T) {
	// Use a fixed start time so we control wallClock.
	startTime := time.Now().Add(-10 * time.Second)
	c := New(WithWindowSize(100), WithStartTime(startTime))

	// Record events totaling 5 seconds of duration.
	// With 10 seconds wallClock, that's 50% time fraction.
	for i := 0; i < 50; i++ {
		c.Record(makeEvent(events.CUDAMemcpy, 100*time.Millisecond))
	}

	snap := c.Snapshot()
	op := snap.Ops[0]

	// Time fraction should be approximately 0.5 (5s / ~10s).
	// Allow some tolerance since wall clock keeps ticking.
	if op.TimeFraction < 0.4 || op.TimeFraction > 0.6 {
		t.Errorf("TimeFraction = %.3f, want ~0.5", op.TimeFraction)
	}
}

// ---------------------------------------------------------------------------
// Anomaly detection tests
// ---------------------------------------------------------------------------

// TestAnomalyDetection verifies that anomalous events are flagged.
func TestAnomalyDetection(t *testing.T) {
	c := New(WithWindowSize(100), WithAnomalyThreshold(3.0))

	// Record 20 normal events at 1ms to establish baseline.
	for i := 0; i < 20; i++ {
		c.Record(makeEvent(events.CUDAMalloc, 1*time.Millisecond))
	}

	// Take a snapshot to populate cachedP50.
	snap1 := c.Snapshot()
	if snap1.Ops[0].P50 != 1*time.Millisecond {
		t.Fatalf("baseline p50 = %v, want 1ms", snap1.Ops[0].P50)
	}

	// Now record an anomalous event (10ms = 10x the median).
	anomalousEvt := makeEvent(events.CUDAMalloc, 10*time.Millisecond)
	c.Record(anomalousEvt)

	// Check IsAnomaly.
	if !c.IsAnomaly(anomalousEvt) {
		t.Error("expected 10ms event to be flagged as anomaly (threshold=3x, median=1ms)")
	}

	// Normal event should not be anomalous.
	normalEvt := makeEvent(events.CUDAMalloc, 2*time.Millisecond)
	if c.IsAnomaly(normalEvt) {
		t.Error("expected 2ms event to NOT be flagged as anomaly")
	}

	// Check anomaly count in snapshot.
	snap2 := c.Snapshot()
	if snap2.Ops[0].AnomalyCount != 1 {
		t.Errorf("AnomalyCount = %d, want 1", snap2.Ops[0].AnomalyCount)
	}
	if snap2.AnomalyEvents != 1 {
		t.Errorf("total AnomalyEvents = %d, want 1", snap2.AnomalyEvents)
	}
}

// TestAnomalyNotFlaggedWithoutBaseline verifies no false anomalies early on.
func TestAnomalyNotFlaggedWithoutBaseline(t *testing.T) {
	c := New(WithWindowSize(100))

	// Record only 5 events — too few for anomaly detection.
	for i := 0; i < 5; i++ {
		c.Record(makeEvent(events.CUDAMalloc, 1*time.Millisecond))
	}

	// Even a 1000ms event shouldn't be flagged — not enough data.
	bigEvt := makeEvent(events.CUDAMalloc, 1000*time.Millisecond)
	if c.IsAnomaly(bigEvt) {
		t.Error("should not flag anomaly with only 5 baseline events")
	}
}

// ---------------------------------------------------------------------------
// Spike pattern detection tests
// ---------------------------------------------------------------------------

// TestSpikePatternDetection tests periodic spike detection.
func TestSpikePatternDetection(t *testing.T) {
	// Spikes at regular intervals: 100, 200, 300, 400, 500
	positions := []int64{100, 200, 300, 400, 500}
	pattern := detectSpikePattern(positions)

	if pattern != "every ~100 events" {
		t.Errorf("expected 'every ~100 events', got %q", pattern)
	}
}

// TestSpikePatternNoPattern tests that irregular spikes don't produce a pattern.
func TestSpikePatternNoPattern(t *testing.T) {
	// Random intervals: 50, 200, 51, 400, 52
	positions := []int64{50, 100, 300, 351, 751}
	pattern := detectSpikePattern(positions)

	if pattern != "" {
		t.Errorf("expected no pattern for irregular spikes, got %q", pattern)
	}
}

// TestSpikePatternTooFewSpikes tests that <3 spikes produce no pattern.
func TestSpikePatternTooFewSpikes(t *testing.T) {
	positions := []int64{100, 200}
	pattern := detectSpikePattern(positions)

	if pattern != "" {
		t.Errorf("expected no pattern for 2 spikes, got %q", pattern)
	}
}

// TestSpikePatternWithNoise tests tolerance for noisy intervals.
func TestSpikePatternWithNoise(t *testing.T) {
	// Intervals roughly 200 but with ±25% noise.
	positions := []int64{100, 290, 510, 690, 900}
	// Intervals: 190, 220, 180, 210 — median ~200, all within 30% tolerance
	pattern := detectSpikePattern(positions)

	if pattern == "" {
		t.Error("expected a periodic pattern with noisy but consistent intervals")
	}
}

// ---------------------------------------------------------------------------
// Circular buffer tests
// ---------------------------------------------------------------------------

// TestCircularBufferWrap verifies correct behavior when the window wraps around.
func TestCircularBufferWrap(t *testing.T) {
	c := New(WithWindowSize(5)) // tiny window for testing

	// Record 10 events: 1ms, 2ms, ..., 10ms.
	// Only the last 5 should be in the window: 6, 7, 8, 9, 10 ms.
	for i := 1; i <= 10; i++ {
		c.Record(makeEvent(events.CUDAMalloc, time.Duration(i)*time.Millisecond))
	}

	snap := c.Snapshot()
	op := snap.Ops[0]

	// Count should be 10 (all-time), not 5 (window).
	if op.Count != 10 {
		t.Errorf("count = %d, want 10", op.Count)
	}

	// Percentiles should be based on window [6, 7, 8, 9, 10].
	if op.P50 != 8*time.Millisecond {
		t.Errorf("p50 after wrap = %v, want 8ms (median of [6,7,8,9,10])", op.P50)
	}

	// Min/max should be all-time.
	if op.Min != 1*time.Millisecond {
		t.Errorf("min = %v, want 1ms (all-time)", op.Min)
	}
	if op.Max != 10*time.Millisecond {
		t.Errorf("max = %v, want 10ms (all-time)", op.Max)
	}
}

// ---------------------------------------------------------------------------
// Edge cases
// ---------------------------------------------------------------------------

// TestSnapshotEmpty verifies Snapshot on a fresh collector.
func TestSnapshotEmpty(t *testing.T) {
	c := New()
	snap := c.Snapshot()

	if snap.TotalEvents != 0 {
		t.Errorf("TotalEvents = %d, want 0", snap.TotalEvents)
	}
	if len(snap.Ops) != 0 {
		t.Errorf("expected 0 ops, got %d", len(snap.Ops))
	}
}

// TestZeroDurationEvents verifies events with 0 duration are handled.
func TestZeroDurationEvents(t *testing.T) {
	c := New(WithWindowSize(100))

	for i := 0; i < 10; i++ {
		c.Record(makeEvent(events.CUDALaunchKernel, 0))
	}

	snap := c.Snapshot()
	if len(snap.Ops) != 1 {
		t.Fatalf("expected 1 op, got %d", len(snap.Ops))
	}
	if snap.Ops[0].P50 != 0 {
		t.Errorf("p50 of zero-duration events = %v, want 0", snap.Ops[0].P50)
	}
}

// TestTimeFractionSumsReasonably checks that time fractions are bounded.
func TestTimeFractionSumsReasonably(t *testing.T) {
	startTime := time.Now().Add(-1 * time.Second)
	c := New(WithWindowSize(100), WithStartTime(startTime))

	// Record 10 events of 10ms each = 100ms total duration.
	// Wall clock ~1s, so fraction should be ~0.1.
	for i := 0; i < 10; i++ {
		c.Record(makeEvent(events.CUDAMalloc, 10*time.Millisecond))
	}

	snap := c.Snapshot()
	total := 0.0
	for _, op := range snap.Ops {
		total += op.TimeFraction
		if op.TimeFraction < 0 || op.TimeFraction > 10.0 {
			t.Errorf("unreasonable time fraction %.3f for %s", op.TimeFraction, op.Op)
		}
	}
	if math.IsNaN(total) || math.IsInf(total, 0) {
		t.Errorf("total time fraction is NaN or Inf: %v", total)
	}
}

// ---------------------------------------------------------------------------
// Functional options tests
// ---------------------------------------------------------------------------

// TestWithWindowSize verifies the window size option.
func TestWithWindowSize(t *testing.T) {
	c := New(WithWindowSize(3))

	// Record 5 events.
	for i := 1; i <= 5; i++ {
		c.Record(makeEvent(events.CUDAMalloc, time.Duration(i)*time.Millisecond))
	}

	snap := c.Snapshot()
	// Window should contain [3, 4, 5], p50 = 4ms.
	if snap.Ops[0].P50 != 4*time.Millisecond {
		t.Errorf("p50 with window=3 = %v, want 4ms", snap.Ops[0].P50)
	}
}

// TestWithAnomalyThreshold verifies the anomaly threshold option.
func TestWithAnomalyThreshold(t *testing.T) {
	// Very low threshold — even 2x median should trigger.
	c := New(WithWindowSize(100), WithAnomalyThreshold(2.0))

	for i := 0; i < 20; i++ {
		c.Record(makeEvent(events.CUDAMalloc, 1*time.Millisecond))
	}
	c.Snapshot() // populate cachedP50

	// 3ms = 3x median → should be anomalous at 2x threshold.
	if !c.IsAnomaly(makeEvent(events.CUDAMalloc, 3*time.Millisecond)) {
		t.Error("3ms should be anomalous with 2x threshold and 1ms median")
	}
	// 1.5ms = 1.5x median → should NOT be anomalous at 2x threshold.
	if c.IsAnomaly(makeEvent(events.CUDAMalloc, 1500*time.Microsecond)) {
		t.Error("1.5ms should not be anomalous with 2x threshold and 1ms median")
	}
}

// ---------------------------------------------------------------------------
// Mixed source tests (v0.2: CUDA + Host events)
// ---------------------------------------------------------------------------

// makeHostEvent creates a host event with the given op and duration.
func makeHostEvent(op events.HostOp, dur time.Duration) events.Event {
	return events.Event{
		Timestamp: time.Now(),
		PID:       1234,
		TID:       1234,
		Source:    events.SourceHost,
		Op:        uint8(op),
		Duration:  dur,
	}
}

// TestMixedSourcesNoCollision verifies that CUDA and Host ops with the same
// numeric code are tracked separately (cudaMalloc=1 vs sched_switch=1).
func TestMixedSourcesNoCollision(t *testing.T) {
	c := New(WithWindowSize(100), WithStartTime(time.Now().Add(-10*time.Second)))

	// Both CUDAMalloc and HostSchedSwitch have Op code = 1.
	for i := 0; i < 20; i++ {
		c.Record(makeEvent(events.CUDAMalloc, 1*time.Millisecond))
		c.Record(makeHostEvent(events.HostSchedSwitch, 5*time.Millisecond))
	}

	snap := c.Snapshot()
	if len(snap.Ops) != 2 {
		t.Fatalf("expected 2 operations, got %d", len(snap.Ops))
	}

	// Find each op by name.
	var cudaOp, hostOp *OpStats
	for i := range snap.Ops {
		switch snap.Ops[i].Op {
		case "cudaMalloc":
			cudaOp = &snap.Ops[i]
		case "sched_switch":
			hostOp = &snap.Ops[i]
		}
	}

	if cudaOp == nil {
		t.Fatal("missing cudaMalloc in snapshot")
	}
	if hostOp == nil {
		t.Fatal("missing sched_switch in snapshot")
	}

	if cudaOp.Count != 20 {
		t.Errorf("cudaMalloc count = %d, want 20", cudaOp.Count)
	}
	if hostOp.Count != 20 {
		t.Errorf("sched_switch count = %d, want 20", hostOp.Count)
	}
	if cudaOp.P50 != 1*time.Millisecond {
		t.Errorf("cudaMalloc p50 = %v, want 1ms", cudaOp.P50)
	}
	if hostOp.P50 != 5*time.Millisecond {
		t.Errorf("sched_switch p50 = %v, want 5ms", hostOp.P50)
	}
	if cudaOp.Source != events.SourceCUDA {
		t.Errorf("cudaMalloc Source = %v, want SourceCUDA", cudaOp.Source)
	}
	if hostOp.Source != events.SourceHost {
		t.Errorf("sched_switch Source = %v, want SourceHost", hostOp.Source)
	}
}

// TestHostOpNames verifies correct op name resolution for host events.
func TestHostOpNames(t *testing.T) {
	c := New(WithWindowSize(100))

	c.Record(makeHostEvent(events.HostSchedSwitch, 1*time.Millisecond))
	c.Record(makeHostEvent(events.HostPageAlloc, 2*time.Millisecond))
	c.Record(makeHostEvent(events.HostOOMKill, 3*time.Millisecond))

	snap := c.Snapshot()
	if len(snap.Ops) != 3 {
		t.Fatalf("expected 3 operations, got %d", len(snap.Ops))
	}

	names := map[string]bool{}
	for _, op := range snap.Ops {
		names[op.Op] = true
	}

	for _, want := range []string{"sched_switch", "mm_page_alloc", "oom_kill"} {
		if !names[want] {
			t.Errorf("missing op name %q in snapshot", want)
		}
	}
}
