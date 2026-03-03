package correlate

import (
	"strings"
	"testing"
	"time"

	"github.com/ingero-io/ingero/internal/stats"
	"github.com/ingero-io/ingero/pkg/events"
)

func makeHostEvt(op events.HostOp, pid uint32, dur time.Duration, args [2]uint64) events.Event {
	return events.Event{
		Timestamp: time.Now(),
		PID:       pid,
		TID:       pid,
		Source:    events.SourceHost,
		Op:        uint8(op),
		Duration:  dur,
		Args:      args,
	}
}

func TestSchedSwitchCorrelation(t *testing.T) {
	eng := New()

	// Simulate 10 sched_switch events for PID 1234, each 5ms off-CPU.
	for i := 0; i < 10; i++ {
		eng.RecordHost(makeHostEvt(events.HostSchedSwitch, 1234, 5*time.Millisecond, [2]uint64{0, 1234}))
	}

	// CUDA stats with anomalous tail: p99=100ms, p50=1ms (100x ratio).
	cudaOps := []stats.OpStats{
		{
			Op:     "cudaStreamSync",
			OpCode: uint8(events.CUDAStreamSync),
			Source: events.SourceCUDA,
			Count:  100,
			P50:    1 * time.Millisecond,
			P99:    100 * time.Millisecond,
		},
	}

	corrs := eng.SnapshotCorrelations(cudaOps, 1234)
	if len(corrs) == 0 {
		t.Fatal("expected sched_switch correlation, got none")
	}

	found := false
	for _, c := range corrs {
		if c.HostOp == "sched_switch" && c.CUDAOp == "cudaStreamSync" {
			found = true
			if c.HostOpCount != 10 {
				t.Errorf("HostOpCount = %d, want 10", c.HostOpCount)
			}
			if c.TailRatio < 90 {
				t.Errorf("TailRatio = %.1f, want ~100", c.TailRatio)
			}
		}
	}
	if !found {
		t.Error("missing sched_switch correlation for cudaStreamSync")
	}
}

func TestNoCorrelationBelowThreshold(t *testing.T) {
	eng := New()

	// Only 3 sched_switch events (below threshold of 5).
	for i := 0; i < 3; i++ {
		eng.RecordHost(makeHostEvt(events.HostSchedSwitch, 1234, 1*time.Millisecond, [2]uint64{0, 1234}))
	}

	cudaOps := []stats.OpStats{
		{
			Op:     "cudaStreamSync",
			OpCode: uint8(events.CUDAStreamSync),
			Source: events.SourceCUDA,
			Count:  100,
			P50:    1 * time.Millisecond,
			P99:    100 * time.Millisecond,
		},
	}

	corrs := eng.SnapshotCorrelations(cudaOps, 1234)
	for _, c := range corrs {
		if c.HostOp == "sched_switch" {
			t.Error("should not correlate with only 3 sched_switch events")
		}
	}
}

func TestNoCorrelationHealthyTail(t *testing.T) {
	eng := New()

	// Many sched_switch events, but CUDA tail is healthy (ratio < 3).
	for i := 0; i < 20; i++ {
		eng.RecordHost(makeHostEvt(events.HostSchedSwitch, 1234, 1*time.Millisecond, [2]uint64{0, 1234}))
	}

	cudaOps := []stats.OpStats{
		{
			Op:     "cudaMemcpy",
			OpCode: uint8(events.CUDAMemcpy),
			Source: events.SourceCUDA,
			Count:  100,
			P50:    1 * time.Millisecond,
			P99:    2 * time.Millisecond, // ratio = 2, below threshold of 3
		},
	}

	corrs := eng.SnapshotCorrelations(cudaOps, 1234)
	for _, c := range corrs {
		if c.HostOp == "sched_switch" {
			t.Error("should not correlate when tail ratio < 3")
		}
	}
}

func TestOOMAlwaysCorrelates(t *testing.T) {
	eng := New()

	eng.RecordHost(makeHostEvt(events.HostOOMKill, 100, 0, [2]uint64{0, 9999}))

	// Even with no anomalous CUDA stats, OOM should produce a correlation.
	cudaOps := []stats.OpStats{
		{
			Op:     "cudaMalloc",
			OpCode: uint8(events.CUDAMalloc),
			Source: events.SourceCUDA,
			Count:  100,
			P50:    1 * time.Millisecond,
			P99:    1 * time.Millisecond, // healthy
		},
	}

	corrs := eng.SnapshotCorrelations(cudaOps, 1234)
	found := false
	for _, c := range corrs {
		if c.HostOp == "oom_kill" {
			found = true
		}
	}
	if !found {
		t.Error("OOM event should always produce a correlation")
	}
}

func TestPageAllocCorrelation(t *testing.T) {
	eng := New()

	// Simulate large page allocations totaling > 1GB.
	for i := 0; i < 100; i++ {
		eng.RecordHost(makeHostEvt(events.HostPageAlloc, 1234, 0,
			[2]uint64{16 * 1024 * 1024, 0})) // 16MB each = 1.6GB total
	}

	cudaOps := []stats.OpStats{
		{
			Op:     "cudaMalloc",
			OpCode: uint8(events.CUDAMalloc),
			Source: events.SourceCUDA,
			Count:  100,
			P50:    1 * time.Millisecond,
			P99:    50 * time.Millisecond, // 50x ratio
		},
	}

	corrs := eng.SnapshotCorrelations(cudaOps, 1234)
	found := false
	for _, c := range corrs {
		if c.HostOp == "mm_page_alloc" {
			found = true
			if c.HostOpCount != 100 {
				t.Errorf("HostOpCount = %d, want 100", c.HostOpCount)
			}
		}
	}
	if !found {
		t.Error("expected mm_page_alloc correlation for > 1GB allocations")
	}
}

func TestWindowPruning(t *testing.T) {
	eng := New()
	eng.maxAge = 100 * time.Millisecond

	// Add events that will expire quickly.
	for i := 0; i < 5; i++ {
		eng.RecordHost(makeHostEvt(events.HostSchedSwitch, 1234, 1*time.Millisecond, [2]uint64{0, 1234}))
	}

	if eng.HostEventCount() != 5 {
		t.Fatalf("expected 5 events, got %d", eng.HostEventCount())
	}

	// Wait for events to expire.
	time.Sleep(150 * time.Millisecond)

	// Add one fresh event to trigger pruning.
	eng.RecordHost(makeHostEvt(events.HostSchedSwitch, 1234, 1*time.Millisecond, [2]uint64{0, 1234}))

	if eng.HostEventCount() != 1 {
		t.Errorf("after pruning, expected 1 event, got %d", eng.HostEventCount())
	}
}

func TestCorrelationString(t *testing.T) {
	c := Correlation{
		CUDAOp:    "cudaStreamSync",
		P99:       142 * time.Millisecond,
		P50:       5 * time.Millisecond,
		TailRatio: 28.4,
		Cause:     "correlated with 23 sched_switch events (115ms off-CPU)",
	}

	s := c.String()
	if s == "" {
		t.Error("Correlation.String() returned empty string")
	}
}

func TestEmptyCorrelations(t *testing.T) {
	eng := New()

	// No host events, no CUDA ops — should return nil/empty.
	corrs := eng.SnapshotCorrelations(nil, 0)
	if len(corrs) != 0 {
		t.Errorf("expected 0 correlations, got %d", len(corrs))
	}
}

// ---------------------------------------------------------------------------
// Causal Chain Tests (v0.3)
// ---------------------------------------------------------------------------

func TestCausalChainWithSystemContext(t *testing.T) {
	eng := New()

	// Set high CPU + swap system context.
	eng.SetSystemSnapshot(&SystemContext{
		CPUPercent: 94,
		MemUsedPct: 97,
		MemAvailMB: 512,
		SwapUsedMB: 2100,
		LoadAvg1:   12.1,
	})

	// Add sched_switch events.
	for i := 0; i < 20; i++ {
		eng.RecordHost(makeHostEvt(events.HostSchedSwitch, 1234, 2*time.Millisecond, [2]uint64{8821, 1234}))
	}

	cudaOps := []stats.OpStats{
		{
			Op:     "cudaStreamSync",
			OpCode: uint8(events.CUDAStreamSync),
			Source: events.SourceCUDA,
			Count:  100,
			P50:    16 * time.Millisecond,
			P99:    142 * time.Millisecond,
		},
	}

	chains := eng.SnapshotCausalChains(cudaOps, 1234)
	if len(chains) == 0 {
		t.Fatal("expected at least one causal chain")
	}

	ch := chains[0]
	if ch.Severity != "HIGH" {
		t.Errorf("severity = %q, want HIGH", ch.Severity)
	}

	// Check that timeline includes all layers.
	hasSystem := false
	hasHost := false
	hasCUDA := false
	for _, evt := range ch.Timeline {
		switch evt.Layer {
		case "SYSTEM":
			hasSystem = true
		case "HOST":
			hasHost = true
		case "CUDA":
			hasCUDA = true
		}
	}
	if !hasSystem {
		t.Error("expected SYSTEM layer in timeline")
	}
	if !hasHost {
		t.Error("expected HOST layer in timeline")
	}
	if !hasCUDA {
		t.Error("expected CUDA layer in timeline")
	}

	if len(ch.Recommendations) == 0 {
		t.Error("expected at least one recommendation")
	}
	if ch.RootCause == "" {
		t.Error("expected non-empty root cause")
	}
}

func TestCausalChainNoSystemContext(t *testing.T) {
	eng := New()

	// No system context set — chain should still work with host events only.
	for i := 0; i < 10; i++ {
		eng.RecordHost(makeHostEvt(events.HostSchedSwitch, 1234, 5*time.Millisecond, [2]uint64{0, 1234}))
	}

	cudaOps := []stats.OpStats{
		{
			Op:     "cudaStreamSync",
			OpCode: uint8(events.CUDAStreamSync),
			Source: events.SourceCUDA,
			Count:  100,
			P50:    1 * time.Millisecond,
			P99:    100 * time.Millisecond,
		},
	}

	chains := eng.SnapshotCausalChains(cudaOps, 1234)
	if len(chains) == 0 {
		t.Fatal("expected causal chain from host events alone")
	}

	ch := chains[0]
	// Without system context showing critical levels, severity should be MEDIUM.
	if ch.Severity != "MEDIUM" {
		t.Errorf("severity = %q, want MEDIUM (no system context triggers)", ch.Severity)
	}
}

func TestCausalChainOOM(t *testing.T) {
	eng := New()

	eng.RecordHost(makeHostEvt(events.HostOOMKill, 100, 0, [2]uint64{0, 9999}))

	// Even with healthy CUDA, OOM should produce a HIGH chain.
	cudaOps := []stats.OpStats{
		{
			Op:     "cudaMalloc",
			OpCode: uint8(events.CUDAMalloc),
			Source: events.SourceCUDA,
			Count:  100,
			P50:    1 * time.Millisecond,
			P99:    1 * time.Millisecond, // healthy
		},
	}

	chains := eng.SnapshotCausalChains(cudaOps, 1234)
	found := false
	for _, ch := range chains {
		if ch.Severity == "HIGH" && len(ch.Timeline) > 0 {
			for _, evt := range ch.Timeline {
				if evt.Op == "oom_kill" {
					found = true
				}
			}
		}
	}
	if !found {
		t.Error("expected HIGH severity OOM causal chain")
	}
}

func TestCausalChainNoChainsWhenHealthy(t *testing.T) {
	eng := New()

	// Low system metrics, few host events, healthy CUDA tail.
	eng.SetSystemSnapshot(&SystemContext{
		CPUPercent: 30,
		MemUsedPct: 50,
	})

	cudaOps := []stats.OpStats{
		{
			Op:     "cudaLaunchKernel",
			OpCode: uint8(events.CUDALaunchKernel),
			Source: events.SourceCUDA,
			Count:  100,
			P50:    10 * time.Microsecond,
			P99:    20 * time.Microsecond, // ratio = 2, below threshold
		},
	}

	chains := eng.SnapshotCausalChains(cudaOps, 1234)
	if len(chains) != 0 {
		t.Errorf("expected 0 chains when everything is healthy, got %d", len(chains))
	}
}

func TestSetSystemSnapshot(t *testing.T) {
	eng := New()

	eng.SetSystemSnapshot(&SystemContext{CPUPercent: 50})

	eng.mu.Lock()
	if eng.sysCtx == nil {
		t.Fatal("sysCtx is nil after SetSystemSnapshot")
	}
	if eng.sysCtx.CPUPercent != 50 {
		t.Errorf("CPUPercent = %f, want 50", eng.sysCtx.CPUPercent)
	}
	eng.mu.Unlock()
}

func TestWithMaxAgeOption(t *testing.T) {
	eng := New(WithMaxAge(5 * time.Second))

	if eng.maxAge != 5*time.Second {
		t.Fatalf("maxAge = %v, want 5s", eng.maxAge)
	}

	// Verify the custom window is applied: events within 5s are kept.
	for i := 0; i < 5; i++ {
		eng.RecordHost(makeHostEvt(events.HostSchedSwitch, 1234, 1*time.Millisecond, [2]uint64{0, 1234}))
	}
	if eng.HostEventCount() != 5 {
		t.Errorf("expected 5 events, got %d", eng.HostEventCount())
	}
}

func TestWithMaxAgeZeroDisablesPruning(t *testing.T) {
	eng := New(WithMaxAge(0))

	// Add events with timestamps far in the past (simulating historical replay).
	pastEvt := events.Event{
		Timestamp: time.Now().Add(-1 * time.Hour),
		PID:       1234,
		TID:       1234,
		Source:    events.SourceHost,
		Op:        uint8(events.HostSchedSwitch),
		Duration:  5 * time.Millisecond,
		Args:      [2]uint64{0, 1234},
	}

	for i := 0; i < 10; i++ {
		eng.RecordHost(pastEvt)
	}

	// With maxAge=0, none should be pruned despite being 1 hour old.
	if eng.HostEventCount() != 10 {
		t.Errorf("expected 10 events (pruning disabled), got %d", eng.HostEventCount())
	}

	// Verify they survive SnapshotCausalChains too (which also calls prune).
	cudaOps := []stats.OpStats{
		{
			Op:     "cudaStreamSync",
			OpCode: uint8(events.CUDAStreamSync),
			Source: events.SourceCUDA,
			Count:  100,
			P50:    1 * time.Millisecond,
			P99:    100 * time.Millisecond,
		},
	}
	chains := eng.SnapshotCausalChains(cudaOps, 1234)
	if len(chains) == 0 {
		t.Fatal("expected causal chain from historical events with pruning disabled")
	}
}

func TestHostOpsIgnoredInCUDAAnalysis(t *testing.T) {
	eng := New()

	for i := 0; i < 20; i++ {
		eng.RecordHost(makeHostEvt(events.HostSchedSwitch, 1234, 5*time.Millisecond, [2]uint64{0, 1234}))
	}

	// Only host ops in the stats — should not produce sched_switch correlations
	// (correlation only triggers for CUDA ops with anomalous tail).
	hostOps := []stats.OpStats{
		{
			Op:     "sched_switch",
			OpCode: uint8(events.HostSchedSwitch),
			Source: events.SourceHost,
			Count:  100,
			P50:    1 * time.Millisecond,
			P99:    50 * time.Millisecond,
		},
	}

	corrs := eng.SnapshotCorrelations(hostOps, 1234)
	for _, c := range corrs {
		if c.HostOp == "sched_switch" {
			t.Error("should not correlate host ops with themselves")
		}
	}
}

// ---------------------------------------------------------------------------
// Cross-Source Chain Tests: Block I/O, TCP, Network (v0.8)
// ---------------------------------------------------------------------------

// makeIOEvt creates a block I/O event with configurable op, duration, and sector count.
// Args[0] = nr_sector (bytes = nr_sector * 512), Args[1] = sector number.
func makeIOEvt(op events.IOOp, dur time.Duration, nrSector uint64) events.Event {
	return events.Event{
		Timestamp: time.Now(),
		PID:       1234,
		Source:    events.SourceIO,
		Op:        uint8(op),
		Duration:  dur,
		Args:      [2]uint64{nrSector, 0},
	}
}

// makeTCPEvt creates a TCP retransmit event.
func makeTCPEvt() events.Event {
	return events.Event{
		Timestamp: time.Now(),
		PID:       1234,
		Source:    events.SourceTCP,
		Op:        uint8(events.TCPRetransmit),
	}
}

// makeNetEvt creates a network socket event with direction and bytes transferred.
// Args[0] = fd, Args[1] = bytes.
func makeNetEvt(op events.NetOp, dur time.Duration, bytes uint64) events.Event {
	return events.Event{
		Timestamp: time.Now(),
		PID:       1234,
		Source:    events.SourceNet,
		Op:        uint8(op),
		Duration:  dur,
		Args:      [2]uint64{3, bytes}, // fd=3
	}
}

// anomalousCUDAOps returns CUDA ops with p99 >> p50 (triggers chain analysis).
func anomalousCUDAOps() []stats.OpStats {
	return []stats.OpStats{
		{
			Op:     "cudaStreamSync",
			OpCode: uint8(events.CUDAStreamSync),
			Source: events.SourceCUDA,
			Count:  100,
			P50:    1 * time.Millisecond,
			P99:    100 * time.Millisecond, // 100x ratio
		},
	}
}

// healthyCUDAOps returns CUDA ops with p99 ≈ p50 (no anomaly).
func healthyCUDAOps() []stats.OpStats {
	return []stats.OpStats{
		{
			Op:     "cudaStreamSync",
			OpCode: uint8(events.CUDAStreamSync),
			Source: events.SourceCUDA,
			Count:  100,
			P50:    1 * time.Millisecond,
			P99:    2 * time.Millisecond, // ratio = 2, below threshold
		},
	}
}

// chainHasLayer returns true if any chain has a timeline event with the given layer.
func chainHasLayer(chains []CausalChain, layer string) bool {
	for _, ch := range chains {
		for _, evt := range ch.Timeline {
			if evt.Layer == layer {
				return true
			}
		}
	}
	return false
}

// chainRecommendationContains returns true if any chain recommendation contains substr.
func chainRecommendationContains(chains []CausalChain, substr string) bool {
	for _, ch := range chains {
		for _, rec := range ch.Recommendations {
			if strings.Contains(rec, substr) {
				return true
			}
		}
	}
	return false
}

// chainExplanationContains returns true if any chain explanation contains substr.
func chainExplanationContains(chains []CausalChain, substr string) bool {
	for _, ch := range chains {
		if strings.Contains(ch.Explanation, substr) {
			return true
		}
	}
	return false
}

func TestCausalChainBlockIO(t *testing.T) {
	eng := New()

	// 60 block read events (above threshold of 50), each 10ms, 256 sectors.
	for i := 0; i < 60; i++ {
		eng.RecordEvent(makeIOEvt(events.IORead, 10*time.Millisecond, 256))
	}

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 0)
	if len(chains) == 0 {
		t.Fatal("expected causal chain with heavy block I/O, got none")
	}
	if !chainHasLayer(chains, "IO") {
		t.Error("expected IO layer in chain timeline")
	}
	if !chainHasLayer(chains, "CUDA") {
		t.Error("expected CUDA layer in chain timeline")
	}
	// Verify root cause mentions I/O.
	found := false
	for _, ch := range chains {
		if strings.Contains(ch.RootCause, "block I/O") {
			found = true
		}
	}
	if !found {
		t.Error("expected root cause to mention 'block I/O'")
	}
}

func TestCausalChainBlockIOWithoutGPUAnomaly(t *testing.T) {
	eng := New()

	// Heavy I/O but no CUDA anomaly — should NOT produce a chain.
	// This tests the "I/O spike without GPU degradation" scenario.
	for i := 0; i < 100; i++ {
		eng.RecordEvent(makeIOEvt(events.IORead, 50*time.Millisecond, 1024))
	}

	chains := eng.SnapshotCausalChains(healthyCUDAOps(), 0)
	if chainHasLayer(chains, "IO") {
		t.Error("should not produce IO chain when CUDA tail is healthy")
	}
}

func TestCausalChainBlockIOBelowThreshold(t *testing.T) {
	eng := New()

	// 10 I/O events with sub-ms latency: not relevant to a 100ms CUDA p99.
	// Filters: count < 200, total 1ms < 1s, peak 100us < 1% of CUDA p99 (1ms).
	for i := 0; i < 10; i++ {
		eng.RecordEvent(makeIOEvt(events.IORead, 100*time.Microsecond, 8))
	}

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 0)
	if chainHasLayer(chains, "IO") {
		t.Error("should not produce IO chain when I/O is irrelevant to CUDA tail latency")
	}
}

func TestCausalChainBlockIOByDuration(t *testing.T) {
	eng := New()

	// Only 5 I/O events (below count threshold) but total duration > 500ms.
	for i := 0; i < 5; i++ {
		eng.RecordEvent(makeIOEvt(events.IOWrite, 200*time.Millisecond, 2048))
	}

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 0)
	if !chainHasLayer(chains, "IO") {
		t.Error("expected IO chain when total I/O duration > 500ms (5 × 200ms = 1s)")
	}
}

func TestCausalChainBlockIOReadHeavy(t *testing.T) {
	eng := New()

	// 40 reads + 15 writes = read-dominant pattern.
	for i := 0; i < 40; i++ {
		eng.RecordEvent(makeIOEvt(events.IORead, 10*time.Millisecond, 256))
	}
	for i := 0; i < 15; i++ {
		eng.RecordEvent(makeIOEvt(events.IOWrite, 10*time.Millisecond, 256))
	}

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 0)
	if !chainRecommendationContains(chains, "DataLoader") {
		t.Error("expected 'DataLoader' recommendation for read-heavy I/O pattern")
	}
}

func TestCausalChainBlockIOWriteHeavy(t *testing.T) {
	eng := New()

	// 40 writes + 15 reads = write-dominant pattern.
	for i := 0; i < 15; i++ {
		eng.RecordEvent(makeIOEvt(events.IORead, 10*time.Millisecond, 256))
	}
	for i := 0; i < 40; i++ {
		eng.RecordEvent(makeIOEvt(events.IOWrite, 10*time.Millisecond, 256))
	}

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 0)
	if !chainRecommendationContains(chains, "checkpoint") {
		t.Error("expected 'checkpoint' recommendation for write-heavy I/O pattern")
	}
}

func TestCausalChainBlockIOSpinningDisk(t *testing.T) {
	eng := New()

	// 60 reads, peak latency 30ms (> 20ms threshold → spinning disk inference).
	for i := 0; i < 59; i++ {
		eng.RecordEvent(makeIOEvt(events.IORead, 5*time.Millisecond, 128))
	}
	// One slow I/O: 30ms peak → indicates spinning disk.
	eng.RecordEvent(makeIOEvt(events.IORead, 30*time.Millisecond, 128))

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 0)
	if !chainRecommendationContains(chains, "NVMe") {
		t.Error("expected 'NVMe' recommendation when peak I/O latency > 20ms")
	}
	if !chainExplanationContains(chains, "spinning disk") {
		t.Error("expected explanation to mention 'spinning disk' for high I/O latency")
	}
}

func TestCausalChainBlockIOSeverityHigh(t *testing.T) {
	eng := New()

	// Peak I/O latency > 50ms → should escalate to HIGH severity.
	for i := 0; i < 55; i++ {
		eng.RecordEvent(makeIOEvt(events.IORead, 10*time.Millisecond, 128))
	}
	eng.RecordEvent(makeIOEvt(events.IORead, 80*time.Millisecond, 128)) // peak 80ms > 50ms threshold

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 0)
	found := false
	for _, ch := range chains {
		if chainHasLayerInChain(ch, "IO") && ch.Severity == "HIGH" {
			found = true
		}
	}
	if !found {
		t.Error("expected HIGH severity chain when I/O peak latency > 50ms")
	}
}

func TestCausalChainBlockIOThroughput(t *testing.T) {
	eng := New()

	// 60 reads × 2048 sectors × 512 bytes = 60 MB → should appear in detail.
	for i := 0; i < 60; i++ {
		eng.RecordEvent(makeIOEvt(events.IORead, 10*time.Millisecond, 2048))
	}

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 0)
	// Check that the IO layer detail mentions MB throughput.
	for _, ch := range chains {
		for _, evt := range ch.Timeline {
			if evt.Layer == "IO" && strings.Contains(evt.Detail, "MB") {
				return // pass
			}
		}
	}
	t.Error("expected IO chain detail to include MB throughput")
}

func TestCausalChainTCPRetransmit(t *testing.T) {
	eng := New()

	// 15 TCP retransmits (> threshold of 10 for standalone chain, > 5 for buildChain).
	for i := 0; i < 15; i++ {
		eng.RecordEvent(makeTCPEvt())
	}

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 0)
	if len(chains) == 0 {
		t.Fatal("expected causal chain with TCP retransmits, got none")
	}
	if !chainHasLayer(chains, "NET") {
		t.Error("expected NET layer in chain timeline")
	}
	// Should produce standalone TCP chain + TCP layer in CUDA chain.
	tcpChainCount := 0
	for _, ch := range chains {
		if strings.Contains(ch.RootCause, "TCP retransmit") ||
			strings.Contains(ch.RootCause, "TCP retransmits") {
			tcpChainCount++
		}
	}
	if tcpChainCount == 0 {
		t.Error("expected chain with TCP retransmit root cause")
	}
}

func TestCausalChainTCPStandalone(t *testing.T) {
	eng := New()

	// 20 retransmits + healthy CUDA → should still produce standalone TCP chain.
	// Standalone chains fire when tcpRetransmitCount > 10 regardless of CUDA tail.
	for i := 0; i < 20; i++ {
		eng.RecordEvent(makeTCPEvt())
	}

	chains := eng.SnapshotCausalChains(healthyCUDAOps(), 0)
	found := false
	for _, ch := range chains {
		if strings.Contains(ch.Summary, "TCP retransmit") {
			found = true
		}
	}
	if !found {
		t.Error("expected standalone TCP retransmit chain even with healthy CUDA")
	}
}

func TestCausalChainTCPHighCountSeverity(t *testing.T) {
	eng := New()

	// 120 retransmits → HIGH severity.
	for i := 0; i < 120; i++ {
		eng.RecordEvent(makeTCPEvt())
	}

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 0)
	found := false
	for _, ch := range chains {
		if strings.Contains(ch.Summary, "TCP retransmit") && ch.Severity == "HIGH" {
			found = true
		}
	}
	if !found {
		t.Error("expected HIGH severity TCP chain for > 100 retransmits")
	}
}

func TestCausalChainTCPExplanation(t *testing.T) {
	eng := New()

	for i := 0; i < 60; i++ {
		eng.RecordEvent(makeTCPEvt())
	}

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 0)
	if !chainExplanationContains(chains, "retransmission") {
		t.Error("expected explanation to mention 'retransmission'")
	}
	if !chainExplanationContains(chains, "NCCL") {
		t.Error("expected explanation to mention 'NCCL' for high retransmit count")
	}
}

func TestCausalChainNetSocket(t *testing.T) {
	eng := New()

	// 120 net events (> 100 threshold), 20KB each (total 2.4MB > 1MB threshold).
	for i := 0; i < 120; i++ {
		eng.RecordEvent(makeNetEvt(events.NetSend, 1*time.Millisecond, 20*1024))
	}

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 0)
	if !chainHasLayer(chains, "NET") {
		t.Error("expected NET layer in chain timeline for heavy network socket I/O")
	}
	if !chainRecommendationContains(chains, "network") {
		t.Error("expected network-related recommendation")
	}
}

func TestCausalChainNetBelowThreshold(t *testing.T) {
	eng := New()

	// 50 net events (below 100 threshold) — should not produce NET layer.
	for i := 0; i < 50; i++ {
		eng.RecordEvent(makeNetEvt(events.NetRecv, 1*time.Millisecond, 20*1024))
	}

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 0)
	// NET layer should NOT appear (socket_io has high threshold to avoid noise).
	for _, ch := range chains {
		for _, evt := range ch.Timeline {
			if evt.Layer == "NET" && evt.Op == "socket_io" {
				t.Error("should not produce NET/socket_io layer below 100 events")
			}
		}
	}
}

func TestCausalChainIOPlusTCP(t *testing.T) {
	eng := New()

	// Combined: heavy I/O + TCP retransmits + anomalous CUDA → multi-layer chain.
	for i := 0; i < 60; i++ {
		eng.RecordEvent(makeIOEvt(events.IORead, 15*time.Millisecond, 512))
	}
	for i := 0; i < 15; i++ {
		eng.RecordEvent(makeTCPEvt())
	}

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 0)
	if len(chains) == 0 {
		t.Fatal("expected chains with combined IO + TCP")
	}

	// The CUDA-correlated chain should have both IO and NET layers.
	foundIO := false
	foundNET := false
	for _, ch := range chains {
		for _, evt := range ch.Timeline {
			if evt.Layer == "IO" {
				foundIO = true
			}
			if evt.Layer == "NET" {
				foundNET = true
			}
		}
	}
	if !foundIO {
		t.Error("expected IO layer in compound chain")
	}
	if !foundNET {
		t.Error("expected NET layer in compound chain")
	}
}

func TestCausalChainIOPlusTCPPlusNet(t *testing.T) {
	eng := New()

	// All three infrastructure layers active simultaneously.
	for i := 0; i < 60; i++ {
		eng.RecordEvent(makeIOEvt(events.IOWrite, 10*time.Millisecond, 256))
	}
	for i := 0; i < 15; i++ {
		eng.RecordEvent(makeTCPEvt())
	}
	for i := 0; i < 120; i++ {
		eng.RecordEvent(makeNetEvt(events.NetSend, 500*time.Microsecond, 50*1024))
	}

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 0)
	if len(chains) == 0 {
		t.Fatal("expected chains with IO + TCP + Net")
	}

	// Check all three infrastructure layers + CUDA appear.
	layersSeen := map[string]bool{}
	for _, ch := range chains {
		for _, evt := range ch.Timeline {
			layersSeen[evt.Layer] = true
		}
	}
	for _, expected := range []string{"IO", "NET", "CUDA"} {
		if !layersSeen[expected] {
			t.Errorf("expected %s layer across all chains", expected)
		}
	}
}

func TestCausalChainExplanationBlockIODetail(t *testing.T) {
	eng := New()

	// Read-heavy, slow I/O → explanation should mention DataLoader and disk type.
	for i := 0; i < 55; i++ {
		eng.RecordEvent(makeIOEvt(events.IORead, 25*time.Millisecond, 512))
	}

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 0)
	if !chainExplanationContains(chains, "Block I/O activity") {
		t.Error("expected explanation to mention 'Block I/O activity'")
	}
	if !chainExplanationContains(chains, "spinning disk") {
		t.Error("expected explanation to mention 'spinning disk' for 25ms peak latency")
	}
	if !chainExplanationContains(chains, "Read-dominant") {
		t.Error("expected explanation to mention 'Read-dominant' for read-heavy pattern")
	}
}

func TestCausalChainExplanationNetworkIO(t *testing.T) {
	eng := New()

	// Heavy network I/O → explanation should mention HTTP/gRPC.
	for i := 0; i < 150; i++ {
		eng.RecordEvent(makeNetEvt(events.NetRecv, 2*time.Millisecond, 30*1024))
	}

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 0)
	if !chainExplanationContains(chains, "network socket I/O") {
		t.Error("expected explanation to mention 'network socket I/O'")
	}
}

// chainHasLayerInChain checks a single chain for a layer (helper for severity tests).
func chainHasLayerInChain(ch CausalChain, layer string) bool {
	for _, evt := range ch.Timeline {
		if evt.Layer == layer {
			return true
		}
	}
	return false
}

// ---------------------------------------------------------------------------
// Full Cross-Source Compound Chain Tests: Old Sensors + New Sensors
//
// The chain engine has 4 layers:
//   Layer 1 (SYSTEM): CPU%, Memory%, Swap, Load    — from /proc
//   Layer 2 (HOST):   sched_switch, mm_page_alloc   — kernel tracepoints
//   Layer 2b (INFRA): IO, TCP, Net                  — v0.8 eBPF probes
//   Layer 3 (CUDA):   the observed GPU symptom       — uprobes
//
// Plus standalone chains: OOM, pod_restart, pod_eviction, TCP burst, noisy neighbor.
//
// Previous tests validated each layer in isolation. These tests validate
// that ALL layers compose correctly in realistic compound scenarios.
// ---------------------------------------------------------------------------

// TestCompoundCPUPlusIO validates "checkpoint write during CPU contention" —
// the most common production scenario where sched_switch + block I/O both
// contribute to GPU stalls.
func TestCompoundCPUPlusIO(t *testing.T) {
	eng := New()
	eng.SetSystemSnapshot(&SystemContext{CPUPercent: 94, MemUsedPct: 60})

	// Host: 20 sched_switch events (CPU contention).
	for i := 0; i < 20; i++ {
		eng.RecordHost(makeHostEvt(events.HostSchedSwitch, 1234, 3*time.Millisecond, [2]uint64{0, 1234}))
	}
	// IO: 60 write events (checkpoint saves).
	for i := 0; i < 60; i++ {
		eng.RecordEvent(makeIOEvt(events.IOWrite, 15*time.Millisecond, 1024))
	}

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 1234)
	if len(chains) == 0 {
		t.Fatal("expected compound chain, got none")
	}

	// The CUDA-correlated chain should have SYSTEM + HOST + IO + CUDA layers.
	layers := map[string]bool{}
	for _, ch := range chains {
		for _, evt := range ch.Timeline {
			layers[evt.Layer] = true
		}
	}
	for _, expected := range []string{"SYSTEM", "HOST", "IO", "CUDA"} {
		if !layers[expected] {
			t.Errorf("expected %s layer in compound chain", expected)
		}
	}
	// Root cause should mention both CPU and I/O.
	foundCPU := false
	foundIO := false
	for _, ch := range chains {
		if strings.Contains(ch.RootCause, "CPU") || strings.Contains(ch.RootCause, "sched_switch") {
			foundCPU = true
		}
		if strings.Contains(ch.RootCause, "block I/O") {
			foundIO = true
		}
	}
	if !foundCPU {
		t.Error("expected root cause to mention CPU/sched_switch")
	}
	if !foundIO {
		t.Error("expected root cause to mention block I/O")
	}
}

// TestCompoundCPUPlusTCP validates "NCCL hang during CPU contention" —
// network retransmits + scheduler starvation both degrading GPU.
func TestCompoundCPUPlusTCP(t *testing.T) {
	eng := New()
	eng.SetSystemSnapshot(&SystemContext{CPUPercent: 92})

	for i := 0; i < 15; i++ {
		eng.RecordHost(makeHostEvt(events.HostSchedSwitch, 1234, 4*time.Millisecond, [2]uint64{0, 1234}))
	}
	for i := 0; i < 20; i++ {
		eng.RecordEvent(makeTCPEvt())
	}

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 1234)
	if !chainHasLayer(chains, "SYSTEM") {
		t.Error("expected SYSTEM layer")
	}
	if !chainHasLayer(chains, "HOST") {
		t.Error("expected HOST layer for sched_switch")
	}
	if !chainHasLayer(chains, "NET") {
		t.Error("expected NET layer for TCP retransmits")
	}
}

// TestCompoundMemoryPressurePlusIO validates "the slow VM scenario" —
// swap + memory pressure + disk I/O: the chain should say "get a bigger VM".
func TestCompoundMemoryPressurePlusIO(t *testing.T) {
	eng := New()
	eng.SetSystemSnapshot(&SystemContext{
		CPUPercent: 60,
		MemUsedPct: 97,
		MemAvailMB: 200,
		SwapUsedMB: 4096,
	})

	// Host: large page allocations (memory pressure).
	for i := 0; i < 50; i++ {
		eng.RecordHost(makeHostEvt(events.HostPageAlloc, 1234, 0,
			[2]uint64{32 * 1024 * 1024, 0})) // 32MB each = 1.6GB total
	}
	// IO: slow reads (spinning disk under memory pressure).
	for i := 0; i < 55; i++ {
		eng.RecordEvent(makeIOEvt(events.IORead, 40*time.Millisecond, 256))
	}

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 1234)
	if len(chains) == 0 {
		t.Fatal("expected compound memory+IO chain")
	}

	layers := map[string]bool{}
	for _, ch := range chains {
		for _, evt := range ch.Timeline {
			layers[evt.Layer] = true
		}
	}
	for _, expected := range []string{"SYSTEM", "HOST", "IO", "CUDA"} {
		if !layers[expected] {
			t.Errorf("expected %s layer in memory+IO compound chain", expected)
		}
	}
	// Severity must be HIGH (swap + high IO latency + memory pressure).
	for _, ch := range chains {
		if chainHasLayerInChain(ch, "IO") && ch.Severity != "HIGH" {
			t.Errorf("expected HIGH severity for memory pressure + slow IO, got %s", ch.Severity)
		}
	}
	// Should recommend both RAM and faster disk.
	if !chainRecommendationContains(chains, "RAM") && !chainRecommendationContains(chains, "swap") {
		t.Error("expected recommendation to mention RAM or swap for memory pressure")
	}
	if !chainRecommendationContains(chains, "NVMe") {
		t.Error("expected recommendation to mention NVMe for 40ms IO latency")
	}
}

// TestCompoundOOMPlusIO validates "OOM during checkpoint writes" —
// host memory exhausted while I/O is heavy.
func TestCompoundOOMPlusIO(t *testing.T) {
	eng := New()

	eng.RecordHost(makeHostEvt(events.HostOOMKill, 100, 0, [2]uint64{0, 9999}))
	for i := 0; i < 60; i++ {
		eng.RecordEvent(makeIOEvt(events.IOWrite, 20*time.Millisecond, 2048))
	}

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 0)
	// Should produce OOM chain (standalone) AND IO-correlated CUDA chain.
	hasOOM := false
	hasIO := false
	for _, ch := range chains {
		for _, evt := range ch.Timeline {
			if evt.Op == "oom_kill" {
				hasOOM = true
			}
			if evt.Layer == "IO" {
				hasIO = true
			}
		}
	}
	if !hasOOM {
		t.Error("expected OOM chain")
	}
	if !hasIO {
		t.Error("expected IO layer in CUDA-correlated chain")
	}
}

// TestCompoundPodRestartPlusIO validates "K8s pod restart during disk activity".
func TestCompoundPodRestartPlusIO(t *testing.T) {
	eng := New()

	eng.RecordHost(makeHostEvt(events.HostPodRestart, 0, 0, [2]uint64{}))
	for i := 0; i < 60; i++ {
		eng.RecordEvent(makeIOEvt(events.IORead, 10*time.Millisecond, 512))
	}

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 0)
	hasPodRestart := false
	hasIO := false
	for _, ch := range chains {
		for _, evt := range ch.Timeline {
			if evt.Op == "pod_restart" {
				hasPodRestart = true
			}
			if evt.Layer == "IO" {
				hasIO = true
			}
		}
	}
	if !hasPodRestart {
		t.Error("expected pod_restart chain")
	}
	if !hasIO {
		t.Error("expected IO layer in CUDA-correlated chain")
	}
}

// TestCompoundPodEvictionPlusTCP validates "K8s eviction during network issues".
func TestCompoundPodEvictionPlusTCP(t *testing.T) {
	eng := New()

	eng.RecordHost(makeHostEvt(events.HostPodEviction, 0, 0, [2]uint64{}))
	for i := 0; i < 20; i++ {
		eng.RecordEvent(makeTCPEvt())
	}

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 0)
	hasPodEviction := false
	hasTCP := false
	for _, ch := range chains {
		for _, evt := range ch.Timeline {
			if evt.Op == "pod_eviction" {
				hasPodEviction = true
			}
			if evt.Op == "tcp_retransmit" {
				hasTCP = true
			}
		}
	}
	if !hasPodEviction {
		t.Error("expected pod_eviction chain")
	}
	if !hasTCP {
		t.Error("expected TCP retransmit chain or layer")
	}
}

// TestCompoundFullStack validates "everything at once" — all 4 layers active.
// This is the ultimate compound test: System + Host + IO + TCP + Net + CUDA.
//
// Real-world scenario: Node under pressure — high CPU, memory exhaustion,
// checkpoint writes to slow disk, network congestion, GPU stalling.
func TestCompoundFullStack(t *testing.T) {
	eng := New()
	eng.SetSystemSnapshot(&SystemContext{
		CPUPercent: 95,
		MemUsedPct: 98,
		MemAvailMB: 128,
		SwapUsedMB: 2048,
		LoadAvg1:   16.5,
	})

	// Host: sched_switch (CPU contention) + page_alloc (memory pressure).
	for i := 0; i < 25; i++ {
		eng.RecordHost(makeHostEvt(events.HostSchedSwitch, 1234, 4*time.Millisecond, [2]uint64{0, 1234}))
	}
	for i := 0; i < 30; i++ {
		eng.RecordHost(makeHostEvt(events.HostPageAlloc, 1234, 0,
			[2]uint64{64 * 1024 * 1024, 0})) // 64MB each = 1.9GB total
	}

	// IO: slow writes (checkpoint to spinning disk).
	for i := 0; i < 70; i++ {
		eng.RecordEvent(makeIOEvt(events.IOWrite, 35*time.Millisecond, 2048))
	}
	// TCP: retransmit burst (NCCL congestion).
	for i := 0; i < 60; i++ {
		eng.RecordEvent(makeTCPEvt())
	}
	// Net: heavy socket I/O (inference serving).
	for i := 0; i < 150; i++ {
		eng.RecordEvent(makeNetEvt(events.NetSend, 500*time.Microsecond, 50*1024))
	}

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 1234)
	if len(chains) == 0 {
		t.Fatal("expected chains in full-stack compound scenario")
	}

	// Collect ALL layers seen across ALL chains.
	allLayers := map[string]bool{}
	for _, ch := range chains {
		for _, evt := range ch.Timeline {
			allLayers[evt.Layer] = true
		}
	}

	// Every layer should appear somewhere across all chains.
	for _, expected := range []string{"SYSTEM", "HOST", "IO", "NET", "CUDA"} {
		if !allLayers[expected] {
			t.Errorf("expected %s layer across all chains in full-stack scenario", expected)
		}
	}

	// Severity must be HIGH (swap + high CPU + memory pressure + slow IO + heavy TCP).
	hasHigh := false
	for _, ch := range chains {
		if ch.Severity == "HIGH" {
			hasHigh = true
		}
	}
	if !hasHigh {
		t.Error("expected at least one HIGH severity chain in full-stack scenario")
	}

	// Explanation should mention multiple contributing factors.
	hasIOExplanation := chainExplanationContains(chains, "Block I/O")
	hasTCPExplanation := chainExplanationContains(chains, "retransmission")
	hasCPUExplanation := chainExplanationContains(chains, "CPU")
	hasSwapExplanation := chainExplanationContains(chains, "swap")
	if !hasIOExplanation {
		t.Error("expected explanation to mention Block I/O")
	}
	if !hasTCPExplanation {
		t.Error("expected explanation to mention TCP retransmissions")
	}
	if !hasCPUExplanation {
		t.Error("expected explanation to mention CPU")
	}
	if !hasSwapExplanation {
		t.Error("expected explanation to mention swap")
	}
}

// TestCompoundFullStackNoGPUAnomaly validates that heavy infrastructure load
// without GPU tail anomaly does NOT produce false positive CUDA-correlated chains.
// Standalone chains (TCP burst >10, OOM, pod events) may still fire — they use
// Op="all" on their CUDA layer. Correlated chains use specific op names (e.g.,
// "cudaStreamSync") because buildChain only triggers for anomalous tail ratios.
func TestCompoundFullStackNoGPUAnomaly(t *testing.T) {
	eng := New()
	eng.SetSystemSnapshot(&SystemContext{
		CPUPercent: 95,
		MemUsedPct: 98,
		SwapUsedMB: 2048,
	})

	for i := 0; i < 25; i++ {
		eng.RecordHost(makeHostEvt(events.HostSchedSwitch, 1234, 4*time.Millisecond, [2]uint64{0, 1234}))
	}
	for i := 0; i < 70; i++ {
		eng.RecordEvent(makeIOEvt(events.IOWrite, 35*time.Millisecond, 2048))
	}
	for i := 0; i < 60; i++ {
		eng.RecordEvent(makeTCPEvt())
	}

	// CUDA is healthy — no tail anomaly.
	chains := eng.SnapshotCausalChains(healthyCUDAOps(), 1234)

	// Standalone chains (TCP burst) are expected — they fire regardless of CUDA tail.
	// But NO chain from buildChain should exist (those have a specific CUDA op name,
	// not "all", as the last timeline entry).
	for _, ch := range chains {
		last := ch.Timeline[len(ch.Timeline)-1]
		if last.Layer == "CUDA" && last.Op != "all" {
			t.Errorf("should not produce CUDA-correlated chain (buildChain) when GPU tail is healthy, got: %s", ch.Summary)
		}
	}

	// Verify standalone TCP chain DID fire (>10 retransmits always warns).
	hasTCPStandalone := false
	for _, ch := range chains {
		if strings.Contains(ch.Summary, "TCP retransmit") {
			hasTCPStandalone = true
		}
	}
	if !hasTCPStandalone {
		t.Error("expected standalone TCP retransmit chain (fires regardless of CUDA tail)")
	}

	// Verify NO chain has IO or HOST layers (those only come from buildChain).
	if chainHasLayer(chains, "IO") {
		t.Error("should not have IO layer when CUDA tail is healthy (IO chains need anomalous CUDA)")
	}
}

// TestCompoundNoisyNeighborPlusIO validates noisy neighbor detection combined
// with block I/O — "peer cgroup stealing CPU AND disk is slow".
func TestCompoundNoisyNeighborPlusIO(t *testing.T) {
	eng := New()

	// Target cgroup (100): high off-CPU p99.
	eng.SetTargetCGroup(100)
	for i := 0; i < 20; i++ {
		eng.RecordCGroupSchedSwitch(100, 10*time.Millisecond) // pre-filter path
		eng.RecordHost(events.Event{
			Timestamp: time.Now(),
			PID:       1234,
			Source:    events.SourceHost,
			Op:        uint8(events.HostSchedSwitch),
			Duration:  10 * time.Millisecond, // 10ms off-CPU (high)
			CGroupID:  100,
			Args:      [2]uint64{0, 1234},
		})
	}
	// Peer cgroup (200): low off-CPU p99.
	for i := 0; i < 20; i++ {
		eng.RecordCGroupSchedSwitch(200, 500*time.Microsecond) // pre-filter path
		eng.RecordHost(events.Event{
			Timestamp: time.Now(),
			PID:       5678,
			Source:    events.SourceHost,
			Op:        uint8(events.HostSchedSwitch),
			Duration:  500 * time.Microsecond, // 0.5ms off-CPU (low)
			CGroupID:  200,
			Args:      [2]uint64{0, 5678},
		})
	}

	// IO: slow reads during noisy neighbor.
	for i := 0; i < 60; i++ {
		eng.RecordEvent(makeIOEvt(events.IORead, 25*time.Millisecond, 512))
	}

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 1234)

	hasNoisyNeighbor := false
	hasIO := false
	for _, ch := range chains {
		if strings.Contains(ch.Summary, "noisy neighbor") {
			hasNoisyNeighbor = true
		}
		for _, evt := range ch.Timeline {
			if evt.Layer == "IO" {
				hasIO = true
			}
		}
	}
	if !hasNoisyNeighbor {
		t.Error("expected noisy neighbor chain (target 10ms p99 vs peer 0.5ms)")
	}
	if !hasIO {
		t.Error("expected IO layer in CUDA-correlated chain")
	}
}

// TestPodRestartChainStandalone validates that pod_restart produces a chain
// even without other infrastructure events.
func TestPodRestartChainStandalone(t *testing.T) {
	eng := New()

	eng.RecordHost(makeHostEvt(events.HostPodRestart, 0, 0, [2]uint64{}))

	chains := eng.SnapshotCausalChains(healthyCUDAOps(), 0)
	found := false
	for _, ch := range chains {
		if strings.Contains(ch.Summary, "pod restart") {
			found = true
			if ch.Severity != "HIGH" {
				t.Errorf("pod restart severity = %q, want HIGH", ch.Severity)
			}
		}
	}
	if !found {
		t.Error("expected standalone pod restart chain")
	}
}

// TestPodEvictionChainStandalone validates that pod_eviction produces a chain.
func TestPodEvictionChainStandalone(t *testing.T) {
	eng := New()

	eng.RecordHost(makeHostEvt(events.HostPodEviction, 0, 0, [2]uint64{}))

	chains := eng.SnapshotCausalChains(healthyCUDAOps(), 0)
	found := false
	for _, ch := range chains {
		if strings.Contains(ch.Summary, "pod eviction") {
			found = true
			if ch.Severity != "HIGH" {
				t.Errorf("pod eviction severity = %q, want HIGH", ch.Severity)
			}
		}
	}
	if !found {
		t.Error("expected standalone pod eviction chain")
	}
}

// TestPodOOMKillChain validates that pod OOM kill (K8s) produces a chain
// identical to kernel oom_kill.
func TestPodOOMKillChain(t *testing.T) {
	eng := New()

	eng.RecordHost(makeHostEvt(events.HostPodOOMKill, 0, 0, [2]uint64{}))

	chains := eng.SnapshotCausalChains(healthyCUDAOps(), 0)
	found := false
	for _, ch := range chains {
		if ch.Severity == "HIGH" && strings.Contains(ch.Summary, "OOM") {
			found = true
		}
	}
	if !found {
		t.Error("expected HIGH severity OOM chain from pod_oom_kill")
	}
}

// TestNoisyNeighborWithPreFilterTracking validates the split between
// RecordCGroupSchedSwitch (pre-PID-filter, populates cgroupOffCPU for
// ALL cgroups) and RecordHost (post-PID-filter, only target PID events
// in hostWindow). This simulates the trace.go event loop:
//
//  1. ALL sched_switch events → RecordCGroupSchedSwitch (both target & peer)
//  2. PID filter drops peer events
//  3. Target-only sched_switch events → RecordHost (host sliding window)
//
// Without the pre-filter, peer cgroup data never reaches cgroupOffCPU
// and detectNoisyNeighbor always returns nil.
func TestNoisyNeighborWithPreFilterTracking(t *testing.T) {
	eng := New(WithMaxAge(0)) // disable pruning for test stability
	eng.SetTargetCGroup(100)

	// Step 1: ALL sched_switch events go through RecordCGroupSchedSwitch.
	// Target cgroup (100): high off-CPU (10ms per event).
	for i := 0; i < 20; i++ {
		eng.RecordCGroupSchedSwitch(100, 10*time.Millisecond)
	}
	// Peer cgroup (200): low off-CPU (0.5ms per event).
	for i := 0; i < 20; i++ {
		eng.RecordCGroupSchedSwitch(200, 500*time.Microsecond)
	}

	// Step 2: PID filter drops peer events. Only target events reach RecordHost.
	for i := 0; i < 20; i++ {
		eng.RecordHost(events.Event{
			Timestamp: time.Now(),
			PID:       1234,
			Source:    events.SourceHost,
			Op:        uint8(events.HostSchedSwitch),
			Duration:  10 * time.Millisecond,
			CGroupID:  100,
			Args:      [2]uint64{0, 1234},
		})
	}
	// Peer events (PID 5678) are NOT passed to RecordHost — PID filter dropped them.

	// Step 3: Verify noisy neighbor fires.
	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 1234)

	found := false
	for _, ch := range chains {
		if strings.Contains(ch.Summary, "noisy neighbor") {
			found = true
			// Target p99=10ms, peer p99=0.5ms → ratio = 20x, should be HIGH.
			if ch.Severity != "HIGH" {
				t.Errorf("noisy neighbor severity = %q, want HIGH (20x ratio)", ch.Severity)
			}
		}
	}
	if !found {
		t.Error("expected noisy neighbor chain: RecordCGroupSchedSwitch should populate " +
			"peer cgroup data independently of RecordHost (which only sees PID-filtered events)")
	}
}

// TestNoisyNeighborWithoutPreFilter verifies that when only RecordHost is
// called (no RecordCGroupSchedSwitch for peer events), noisy neighbor
// detection does NOT fire — because peer cgroup data is missing.
func TestNoisyNeighborWithoutPreFilter(t *testing.T) {
	eng := New(WithMaxAge(0))
	eng.SetTargetCGroup(100)

	// Only target events reach RecordHost (simulating PID filter).
	for i := 0; i < 20; i++ {
		eng.RecordHost(events.Event{
			Timestamp: time.Now(),
			PID:       1234,
			Source:    events.SourceHost,
			Op:        uint8(events.HostSchedSwitch),
			Duration:  10 * time.Millisecond,
			CGroupID:  100,
			Args:      [2]uint64{0, 1234},
		})
	}
	// No peer events at all — RecordHost doesn't populate cgroupOffCPU anymore.

	chains := eng.SnapshotCausalChains(anomalousCUDAOps(), 1234)

	for _, ch := range chains {
		if strings.Contains(ch.Summary, "noisy neighbor") {
			t.Error("noisy neighbor should NOT fire without peer cgroup data " +
				"(RecordHost no longer populates cgroupOffCPU)")
		}
	}
}
