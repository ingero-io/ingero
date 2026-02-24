package synth

import (
	"context"
	"testing"
	"time"

	"github.com/ingero-io/ingero/internal/stats"
	"github.com/ingero-io/ingero/pkg/events"
)

// TestRegistryCompleteness verifies all 6 scenarios are registered with
// non-empty fields: incident, cold-start, memcpy-bottleneck, periodic-spike,
// cpu-contention, gpu-steal.
func TestRegistryCompleteness(t *testing.T) {
	if len(Registry) != 6 {
		t.Fatalf("expected 6 scenarios, got %d", len(Registry))
	}

	// Incident must be first (the WOW demo).
	if Registry[0].Name != "incident" {
		t.Errorf("first scenario should be 'incident', got %q", Registry[0].Name)
	}

	for _, s := range Registry {
		if s.Name == "" {
			t.Error("scenario has empty Name")
		}
		if s.Title == "" {
			t.Errorf("scenario %q has empty Title", s.Name)
		}
		if s.Description == "" {
			t.Errorf("scenario %q has empty Description", s.Name)
		}
		if s.Insight == "" {
			t.Errorf("scenario %q has empty Insight", s.Name)
		}
		if s.Generate == nil {
			t.Errorf("scenario %q has nil Generate function", s.Name)
		}
	}
}

// TestFindScenario verifies Find() returns the correct scenario by name or alias.
func TestFindScenario(t *testing.T) {
	s := Find("cold-start")
	if s == nil {
		t.Fatal("Find(\"cold-start\") returned nil")
	}
	if s.Name != "cold-start" {
		t.Errorf("Find(\"cold-start\").Name = %q", s.Name)
	}

	// Test alias lookup: "gpu-contention" → "gpu-steal".
	s = Find("gpu-contention")
	if s == nil {
		t.Fatal("Find(\"gpu-contention\") returned nil (alias)")
	}
	if s.Name != "gpu-steal" {
		t.Errorf("Find(\"gpu-contention\").Name = %q, want gpu-steal", s.Name)
	}

	// Test alias lookup: "contention" → "gpu-steal".
	s = Find("contention")
	if s == nil {
		t.Fatal("Find(\"contention\") returned nil (alias)")
	}
	if s.Name != "gpu-steal" {
		t.Errorf("Find(\"contention\").Name = %q, want gpu-steal", s.Name)
	}

	s = Find("nonexistent")
	if s != nil {
		t.Errorf("Find(\"nonexistent\") should return nil, got %v", s.Name)
	}
}

// TestMakeEvent verifies makeEvent produces correct event fields.
func TestMakeEvent(t *testing.T) {
	evt := makeEvent(events.CUDAMalloc, 100*time.Microsecond)

	if evt.PID != SyntheticPID {
		t.Errorf("PID = %d, want %d", evt.PID, SyntheticPID)
	}
	if evt.TID != SyntheticTID {
		t.Errorf("TID = %d, want %d", evt.TID, SyntheticTID)
	}
	if evt.Source != events.SourceCUDA {
		t.Errorf("Source = %v, want SourceCUDA", evt.Source)
	}
	if evt.Op != uint8(events.CUDAMalloc) {
		t.Errorf("Op = %d, want %d", evt.Op, events.CUDAMalloc)
	}
	if evt.Duration != 100*time.Microsecond {
		t.Errorf("Duration = %v, want 100µs", evt.Duration)
	}
	if evt.Timestamp.IsZero() {
		t.Error("Timestamp should not be zero")
	}
}

// TestContextCancellation verifies generators stop promptly on context cancel.
func TestContextCancellation(t *testing.T) {
	for _, s := range Registry {
		t.Run(s.Name, func(t *testing.T) {
			ctx, cancel := context.WithTimeout(context.Background(), 200*time.Millisecond)
			defer cancel()

			ch := make(chan events.Event, 256)
			done := make(chan struct{})

			go func() {
				s.Generate(ctx, ch, 10.0) // fast speed
				close(done)
			}()

			// Should complete within 1 second (generous margin).
			select {
			case <-done:
				// OK
			case <-time.After(2 * time.Second):
				t.Fatal("generator did not stop within 2s of context cancellation")
			}
		})
	}
}

// TestColdStartMaxExceedsP50 verifies the cold-start scenario produces events
// where the maximum latency far exceeds the median — the hallmark of a
// cold-start penalty.
func TestColdStartMaxExceedsP50(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ch := make(chan events.Event, 1024)
	go func() {
		generateColdStart(ctx, ch, 50.0) // 50x speed for fast test
		close(ch)
	}()

	collector := stats.New()
	for evt := range ch {
		collector.Record(evt)
	}

	snap := collector.Snapshot()
	if len(snap.Ops) == 0 {
		t.Fatal("no ops recorded")
	}

	// At least one operation should have max >> p50.
	var foundLargeRatio bool
	for _, op := range snap.Ops {
		if op.P50 > 0 && op.Max > 0 {
			ratio := float64(op.Max) / float64(op.P50)
			if ratio > 10 {
				foundLargeRatio = true
				break
			}
		}
	}

	if !foundLargeRatio {
		t.Error("cold-start: expected at least one op with max/p50 ratio > 10")
	}
}

// TestMemcpyBottleneckDominatesWallTime verifies that cudaMemcpy has the
// highest time-fraction in the memcpy-bottleneck scenario.
func TestMemcpyBottleneckDominatesWallTime(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ch := make(chan events.Event, 1024)
	go func() {
		generateMemcpyBottleneck(ctx, ch, 50.0)
		close(ch)
	}()

	collector := stats.New()
	for evt := range ch {
		collector.Record(evt)
	}

	snap := collector.Snapshot()
	if len(snap.Ops) == 0 {
		t.Fatal("no ops recorded")
	}

	// Ops are sorted by TimeFraction descending. The top one should be cudaMemcpy.
	top := snap.Ops[0]
	if top.Op != "cudaMemcpy" {
		t.Errorf("expected cudaMemcpy to dominate wall time, got %q (%.1f%%)",
			top.Op, top.TimeFraction*100)
	}

	// cudaMemcpy should have >30% time fraction.
	if top.TimeFraction < 0.30 {
		t.Errorf("cudaMemcpy TimeFraction = %.1f%%, want >30%%", top.TimeFraction*100)
	}
}

// TestPeriodicSpikeGeneratesAnomalies verifies the periodic-spike scenario
// produces events that the stats engine flags as anomalous.
//
// Note: anomaly detection requires Snapshot() to populate cachedP50 first.
// In the real watch command, the display ticker calls Snapshot() every second.
// We simulate this by taking a snapshot after the first 100 events.
func TestPeriodicSpikeGeneratesAnomalies(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ch := make(chan events.Event, 1024)
	go func() {
		generatePeriodicSpike(ctx, ch, 50.0)
		close(ch)
	}()

	collector := stats.New()
	count := 0
	for evt := range ch {
		collector.Record(evt)
		count++
		// Take snapshot after first 100 events to populate cachedP50,
		// so subsequent spike events get flagged as anomalies.
		if count == 100 {
			collector.Snapshot()
		}
	}

	snap := collector.Snapshot()
	if snap.AnomalyEvents == 0 {
		t.Error("periodic-spike: expected anomaly events to be > 0")
	}
}

// TestCPUContentionMixedSources verifies the cpu-contention scenario generates
// both CUDA and host events, and that the host events trigger correlations.
func TestCPUContentionMixedSources(t *testing.T) {
	ctx, cancel := context.WithTimeout(context.Background(), 10*time.Second)
	defer cancel()

	ch := make(chan events.Event, 1024)
	go func() {
		generateCPUContention(ctx, ch, 50.0) // 50x speed for fast test
		close(ch)
	}()

	collector := stats.New()
	var cudaCount, hostCount int
	for evt := range ch {
		collector.Record(evt)
		switch evt.Source {
		case events.SourceCUDA:
			cudaCount++
		case events.SourceHost:
			hostCount++
		}
	}

	if cudaCount == 0 {
		t.Error("cpu-contention: expected CUDA events")
	}
	if hostCount == 0 {
		t.Error("cpu-contention: expected host events")
	}

	snap := collector.Snapshot()

	// Should have both CUDA and Host ops in the snapshot.
	var hasCUDA, hasHost bool
	for _, op := range snap.Ops {
		if op.Source == events.SourceCUDA {
			hasCUDA = true
		}
		if op.Source == events.SourceHost {
			hasHost = true
		}
	}
	if !hasCUDA {
		t.Error("cpu-contention: expected CUDA ops in snapshot")
	}
	if !hasHost {
		t.Error("cpu-contention: expected Host ops in snapshot")
	}
}
