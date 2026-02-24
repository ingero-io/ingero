package correlate

import (
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
