package straggler

import (
	"sync"
	"testing"
	"time"

	"github.com/ingero-io/ingero/pkg/events"
)

// testSink captures emitted StraggleState messages for assertions.
type testSink struct {
	mu       sync.Mutex
	received []StraggleState
}

func (s *testSink) SendStraggle(ss StraggleState) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.received = append(s.received, ss)
	return nil
}

func (s *testSink) count() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.received)
}

func (s *testSink) last() StraggleState {
	s.mu.Lock()
	defer s.mu.Unlock()
	return s.received[len(s.received)-1]
}

// feedLaunchKernels sends n cudaLaunchKernel events for the given PID.
func feedLaunchKernels(d *Detector, pid uint32, n int) {
	for i := 0; i < n; i++ {
		d.ProcessEvent(events.Event{
			Source: events.SourceCUDA,
			Op:     uint8(events.CUDALaunchKernel),
			PID:    pid,
		})
	}
}

// feedSchedSwitches sends n sched_switch events where the given PID is
// being switched out (preempted). preemptorPID is the next PID.
func feedSchedSwitches(d *Detector, pid uint32, preemptorPID uint32, n int) {
	for i := 0; i < n; i++ {
		d.ProcessEvent(events.Event{
			Source: events.SourceHost,
			Op:     uint8(events.HostSchedSwitch),
			PID:    pid,
			Args:   [2]uint64{0, uint64(preemptorPID)},
		})
	}
}

// testClock provides deterministic timestamps with 1s spacing for tests.
type testClock struct {
	t time.Time
}

func newTestClock() *testClock {
	return &testClock{t: time.Date(2026, 1, 1, 0, 0, 0, 0, time.UTC)}
}

func (c *testClock) tick() time.Time {
	c.t = c.t.Add(time.Second)
	return c.t
}

func TestDetector(t *testing.T) {
	const pid uint32 = 1234
	const preemptor uint32 = 5678

	t.Run("throughput_drop_plus_high_sched_switch_emits", func(t *testing.T) {
		sink := &testSink{}
		cfg := DetectorConfig{
			ThroughputDropPct:    30,
			SchedSwitchThreshold: 5,
			CheckInterval:        time.Second,
		}
		d := NewDetector(cfg, sink)
		clk := newTestClock()

		// Build baseline: 3 intervals of 100 ops each (100 ops/sec).
		for i := 0; i < minBaselineSamples; i++ {
			feedLaunchKernels(d, pid, 100)
			d.checkAt(clk.tick())
		}
		if sink.count() != 0 {
			t.Fatalf("expected no emission during baseline, got %d", sink.count())
		}

		// Simulate straggler: drastically reduce throughput + high sched_switch.
		feedLaunchKernels(d, pid, 10) // ~90% drop
		feedSchedSwitches(d, pid, preemptor, 20)
		d.checkAt(clk.tick())

		if sink.count() != 1 {
			t.Fatalf("expected 1 emission, got %d", sink.count())
		}
		ss := sink.last()
		if ss.PID != pid {
			t.Errorf("expected PID %d, got %d", pid, ss.PID)
		}
		if ss.ThroughputDropPct < 30 {
			t.Errorf("expected drop > 30%%, got %.1f%%", ss.ThroughputDropPct)
		}
		if ss.SchedSwitchCount < 20 {
			t.Errorf("expected sched_switch >= 20, got %d", ss.SchedSwitchCount)
		}
		foundPreemptor := false
		for _, p := range ss.PreemptingPIDs {
			if p == preemptor {
				foundPreemptor = true
			}
		}
		if !foundPreemptor {
			t.Errorf("expected preempting PID %d in %v", preemptor, ss.PreemptingPIDs)
		}
		if ss.TimestampNs <= 0 {
			t.Errorf("expected positive timestamp, got %d", ss.TimestampNs)
		}
	})

	t.Run("throughput_drop_plus_low_sched_switch_no_emission", func(t *testing.T) {
		sink := &testSink{}
		cfg := DetectorConfig{
			ThroughputDropPct:    30,
			SchedSwitchThreshold: 5,
			CheckInterval:        time.Second,
		}
		d := NewDetector(cfg, sink)
		clk := newTestClock()

		// Build baseline.
		for i := 0; i < minBaselineSamples; i++ {
			feedLaunchKernels(d, pid, 100)
			d.checkAt(clk.tick())
		}

		// Throughput drop but few sched_switch events (below threshold).
		feedLaunchKernels(d, pid, 10)
		feedSchedSwitches(d, pid, preemptor, 2) // below threshold of 5
		d.checkAt(clk.tick())

		if sink.count() != 0 {
			t.Fatalf("expected no emission (low sched_switch), got %d", sink.count())
		}
	})

	t.Run("no_throughput_drop_plus_high_sched_switch_no_emission", func(t *testing.T) {
		sink := &testSink{}
		cfg := DetectorConfig{
			ThroughputDropPct:    30,
			SchedSwitchThreshold: 5,
			CheckInterval:        time.Second,
		}
		d := NewDetector(cfg, sink)
		clk := newTestClock()

		// Build baseline at 100 ops/sec.
		for i := 0; i < minBaselineSamples; i++ {
			feedLaunchKernels(d, pid, 100)
			d.checkAt(clk.tick())
		}

		// Normal throughput but lots of sched_switch.
		feedLaunchKernels(d, pid, 100)
		feedSchedSwitches(d, pid, preemptor, 50)
		d.checkAt(clk.tick())

		if sink.count() != 0 {
			t.Fatalf("expected no emission (no throughput drop), got %d", sink.count())
		}
	})

	t.Run("healthy_state_no_emission", func(t *testing.T) {
		sink := &testSink{}
		cfg := DetectorConfig{
			ThroughputDropPct:    30,
			SchedSwitchThreshold: 5,
			CheckInterval:        time.Second,
		}
		d := NewDetector(cfg, sink)
		clk := newTestClock()

		// Build baseline and keep steady throughput with no contention.
		for i := 0; i < minBaselineSamples+5; i++ {
			feedLaunchKernels(d, pid, 100)
			d.checkAt(clk.tick())
		}

		if sink.count() != 0 {
			t.Fatalf("expected no emission in healthy state, got %d", sink.count())
		}
	})

	t.Run("configurable_thresholds", func(t *testing.T) {
		sink := &testSink{}
		// Stricter thresholds: 10% drop, 2 sched_switch.
		cfg := DetectorConfig{
			ThroughputDropPct:    10,
			SchedSwitchThreshold: 2,
			CheckInterval:        time.Second,
		}
		d := NewDetector(cfg, sink)
		clk := newTestClock()

		// Build baseline at 100 ops/sec.
		for i := 0; i < minBaselineSamples; i++ {
			feedLaunchKernels(d, pid, 100)
			d.checkAt(clk.tick())
		}

		// Moderate drop (~50%) with just 3 sched_switch events.
		feedLaunchKernels(d, pid, 50)
		feedSchedSwitches(d, pid, preemptor, 3)
		d.checkAt(clk.tick())

		if sink.count() != 1 {
			t.Fatalf("expected 1 emission with stricter thresholds, got %d", sink.count())
		}
	})

	t.Run("nil_sink_no_panic", func(t *testing.T) {
		cfg := DetectorConfig{
			ThroughputDropPct:    30,
			SchedSwitchThreshold: 5,
			CheckInterval:        time.Second,
		}
		d := NewDetector(cfg, nil)
		clk := newTestClock()

		// Build baseline and trigger straggler condition.
		for i := 0; i < minBaselineSamples; i++ {
			feedLaunchKernels(d, pid, 100)
			d.checkAt(clk.tick())
		}
		feedLaunchKernels(d, pid, 10)
		feedSchedSwitches(d, pid, preemptor, 20)
		d.checkAt(clk.tick()) // should not panic
	})

	t.Run("driver_launch_kernel_counted", func(t *testing.T) {
		sink := &testSink{}
		cfg := DetectorConfig{
			ThroughputDropPct:    30,
			SchedSwitchThreshold: 5,
			CheckInterval:        time.Second,
		}
		d := NewDetector(cfg, sink)
		clk := newTestClock()

		// Build baseline using Driver API cuLaunchKernel events.
		for i := 0; i < minBaselineSamples; i++ {
			for j := 0; j < 100; j++ {
				d.ProcessEvent(events.Event{
					Source: events.SourceDriver,
					Op:     uint8(events.DriverLaunchKernel),
					PID:    pid,
				})
			}
			d.checkAt(clk.tick())
		}

		// Drop throughput + sched_switch.
		for j := 0; j < 10; j++ {
			d.ProcessEvent(events.Event{
				Source: events.SourceDriver,
				Op:     uint8(events.DriverLaunchKernel),
				PID:    pid,
			})
		}
		feedSchedSwitches(d, pid, preemptor, 20)
		d.checkAt(clk.tick())

		if sink.count() != 1 {
			t.Fatalf("expected 1 emission from Driver API events, got %d", sink.count())
		}
	})

	t.Run("irrelevant_events_ignored", func(t *testing.T) {
		sink := &testSink{}
		cfg := DefaultConfig()
		d := NewDetector(cfg, sink)

		// Feed only non-launch, non-sched_switch events.
		for i := 0; i < 100; i++ {
			d.ProcessEvent(events.Event{
				Source: events.SourceCUDA,
				Op:     uint8(events.CUDAMalloc),
				PID:    pid,
			})
			d.ProcessEvent(events.Event{
				Source: events.SourceHost,
				Op:     uint8(events.HostPageAlloc),
				PID:    pid,
			})
		}

		// No PID state should be created for non-launch events.
		d.mu.Lock()
		_, exists := d.pids[pid]
		d.mu.Unlock()
		if exists {
			t.Error("expected no PID state for non-launch events")
		}
	})

	t.Run("default_config_values", func(t *testing.T) {
		cfg := DefaultConfig()
		if cfg.ThroughputDropPct != 30 {
			t.Errorf("expected default drop 30%%, got %g", cfg.ThroughputDropPct)
		}
		if cfg.SchedSwitchThreshold != 5 {
			t.Errorf("expected default threshold 5, got %d", cfg.SchedSwitchThreshold)
		}
		if cfg.CheckInterval != time.Second {
			t.Errorf("expected default interval 1s, got %v", cfg.CheckInterval)
		}
	})

	t.Run("invalid_config_uses_defaults", func(t *testing.T) {
		d := NewDetector(DetectorConfig{}, nil)
		c := d.Config()
		if c.ThroughputDropPct != 30 {
			t.Errorf("expected default drop 30%%, got %g", c.ThroughputDropPct)
		}
		if c.SchedSwitchThreshold != 5 {
			t.Errorf("expected default threshold 5, got %d", c.SchedSwitchThreshold)
		}
		if c.CheckInterval != time.Second {
			t.Errorf("expected default interval 1s, got %v", c.CheckInterval)
		}
	})

	t.Run("multiple_pids_independent", func(t *testing.T) {
		sink := &testSink{}
		cfg := DetectorConfig{
			ThroughputDropPct:    30,
			SchedSwitchThreshold: 5,
			CheckInterval:        time.Second,
		}
		d := NewDetector(cfg, sink)
		clk := newTestClock()

		pid1 := uint32(1000)
		pid2 := uint32(2000)

		// Build baseline for both PIDs.
		for i := 0; i < minBaselineSamples; i++ {
			feedLaunchKernels(d, pid1, 100)
			feedLaunchKernels(d, pid2, 100)
			d.checkAt(clk.tick())
		}

		// Only PID1 drops, PID2 stays healthy.
		feedLaunchKernels(d, pid1, 10)
		feedSchedSwitches(d, pid1, preemptor, 20)
		feedLaunchKernels(d, pid2, 100)
		d.checkAt(clk.tick())

		if sink.count() != 1 {
			t.Fatalf("expected 1 emission (only PID1), got %d", sink.count())
		}
		if sink.last().PID != pid1 {
			t.Errorf("expected emission for PID %d, got %d", pid1, sink.last().PID)
		}
	})
}
