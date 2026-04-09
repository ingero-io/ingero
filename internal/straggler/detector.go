// Package straggler detects CPU scheduling contention correlated with GPU
// throughput drops and emits StraggleState messages for remediation.
//
// Signal flow:
//
//	host_trace.bpf.c (sched_switch) ──► Detector ──┐
//	                                                ├──► Sink ──► UDS ──► Orchestrator
//	CUDA events (cudaLaunchKernel)  ──► Detector ──┘
//
// The detector maintains per-PID state: an exponential moving average of
// cudaLaunchKernel ops/sec (throughput baseline) and a sliding count of
// sched_switch events. When both signals exceed their thresholds
// simultaneously, a StraggleState is emitted.
package straggler

import (
	"context"
	"fmt"
	"log"
	"sync"
	"time"

	"github.com/ingero-io/ingero/pkg/events"
)

// StraggleState represents a detected straggler condition for a single PID.
// JSON field names are the cross-language contract with the orchestrator.
//
// v0.10: Comm carries the kernel-captured process name from bpf_get_current_comm()
// for human-readable orchestrator logs and PID-reuse detection. May be empty when
// the BPF probe could not capture comm or when the detector hasn't seen a non-empty
// comm for the PID yet (e.g., only sched_switch events with NULL comm so far).
type StraggleState struct {
	PID               uint32   `json:"pid"`
	Comm              string   `json:"comm,omitempty"`
	ThroughputDropPct float64  `json:"throughput_drop_pct"`
	SchedSwitchCount  uint32   `json:"sched_switch_count"`
	PreemptingPIDs    []uint32 `json:"preempting_pids"`
	TimestampNs       int64    `json:"timestamp_ns"`
	Sustained         bool     `json:"sustained"`
}

// Sink receives StraggleState emissions. Implemented by remediate.Server.
type Sink interface {
	SendStraggle(StraggleState) error
}

// DetectorConfig holds configurable thresholds for straggler detection.
type DetectorConfig struct {
	// ThroughputDropPct is the percentage drop from baseline that triggers
	// the throughput leg of detection. Default: 30 (meaning 30% drop).
	ThroughputDropPct float64

	// SchedSwitchThreshold is the minimum number of sched_switch events
	// per check interval needed to trigger the contention leg. Default: 5.
	SchedSwitchThreshold int

	// CheckInterval is how often the detector evaluates conditions.
	// Default: 1000ms.
	CheckInterval time.Duration
}

// DefaultConfig returns a DetectorConfig with sensible defaults.
func DefaultConfig() DetectorConfig {
	return DetectorConfig{
		ThroughputDropPct:    30,
		SchedSwitchThreshold: 5,
		CheckInterval:        time.Second,
	}
}

// emaAlpha controls how fast the baseline adapts to throughput changes.
// 0.2 means ~5 intervals to mostly converge to a new steady state.
const emaAlpha = 0.2

// minBaselineSamples is the number of check cycles before detection activates.
// Ensures the EMA has a meaningful baseline before flagging drops.
const minBaselineSamples = 3

// idleEvictIntervals is the number of consecutive zero-activity check intervals
// before a PID's state is evicted. Prevents unbounded memory growth from
// short-lived GPU processes.
const idleEvictIntervals = 30

// pidState tracks per-PID throughput and scheduling contention.
type pidState struct {
	// Throughput tracking (cudaLaunchKernel ops/sec via EMA).
	launchCount int64     // cumulative cudaLaunchKernel events
	prevCount   int64     // count at previous check
	prevTime    time.Time // time of previous check
	baseline    float64   // EMA of ops/sec
	samples     int       // how many check cycles have contributed to the EMA

	// Sched_switch tracking (reset each check interval).
	schedSwitches  uint32
	preemptingPIDs map[uint32]bool // PIDs seen preempting this PID

	// Idle tracking for eviction.
	idleIntervals int // consecutive intervals with zero launch events

	// Sustained re-emission state: true after initial correlated detection,
	// cleared when sched_switch drops below threshold.
	emitted bool

	// v0.10: most recent non-empty comm observed for this PID. Sourced from
	// CUDA/Driver event captures (sched_switch events use ctx->next_comm which
	// is also propagated). Empty string until first observation.
	comm string
}

// Detector watches for correlated throughput drops and CPU scheduling contention.
// Thread-safe: ProcessEvent is called from the event loop, Run drives periodic checks.
type Detector struct {
	mu     sync.Mutex
	config DetectorConfig
	pids   map[uint32]*pidState
	sink   Sink
}

// NewDetector creates a straggler detector.
// sink receives StraggleState emissions. Pass nil to disable emission (testing).
func NewDetector(config DetectorConfig, sink Sink) *Detector {
	if config.ThroughputDropPct <= 0 {
		config.ThroughputDropPct = DefaultConfig().ThroughputDropPct
	}
	if config.SchedSwitchThreshold <= 0 {
		config.SchedSwitchThreshold = DefaultConfig().SchedSwitchThreshold
	}
	if config.CheckInterval <= 0 {
		config.CheckInterval = DefaultConfig().CheckInterval
	}
	return &Detector{
		config: config,
		pids:   make(map[uint32]*pidState),
		sink:   sink,
	}
}

// ProcessEvent updates internal state from an incoming event.
// Called inline from the trace event loop for every event that passes the PID filter.
//
// Only CUDA/Driver cudaLaunchKernel and Host sched_switch events are relevant;
// all others are ignored with zero overhead (single switch on Source).
func (d *Detector) ProcessEvent(evt events.Event) {
	switch evt.Source {
	case events.SourceCUDA:
		if events.CUDAOp(evt.Op) != events.CUDALaunchKernel {
			return
		}
		d.mu.Lock()
		ps := d.getOrCreate(evt.PID)
		ps.launchCount++
		if evt.Comm != "" {
			ps.comm = evt.Comm
		}
		d.mu.Unlock()

	case events.SourceDriver:
		if events.DriverOp(evt.Op) != events.DriverLaunchKernel {
			return
		}
		d.mu.Lock()
		ps := d.getOrCreate(evt.PID)
		ps.launchCount++
		if evt.Comm != "" {
			ps.comm = evt.Comm
		}
		d.mu.Unlock()

	case events.SourceHost:
		if events.HostOp(evt.Op) != events.HostSchedSwitch {
			return
		}
		// sched_switch encoding from host_trace.bpf.c:
		//   evt.PID  = target PID (the process coming BACK on-CPU)
		//   evt.Comm = next_comm from tracepoint context (the resuming target's comm)
		//   evt.Args[1] = prev_pid (the process that was running before,
		//                  i.e., the most recent preemptor)
		//   evt.Duration = off-CPU duration
		//
		// We count this as one preemption event for the target PID and
		// record prev_pid as a preempting process.
		d.mu.Lock()
		ps, ok := d.pids[evt.PID]
		if ok {
			ps.schedSwitches++
			if evt.Comm != "" {
				ps.comm = evt.Comm
			}
			preemptorPID := uint32(evt.Args[1])
			if preemptorPID > 0 && preemptorPID != evt.PID {
				if ps.preemptingPIDs == nil {
					ps.preemptingPIDs = make(map[uint32]bool)
				}
				ps.preemptingPIDs[preemptorPID] = true
			}
		}
		d.mu.Unlock()
	}
}

// Run starts the periodic check loop. Blocks until ctx is cancelled.
// Launch in a goroutine: go detector.Run(ctx)
func (d *Detector) Run(ctx context.Context) {
	ticker := time.NewTicker(d.config.CheckInterval)
	defer ticker.Stop()
	for {
		select {
		case <-ctx.Done():
			return
		case <-ticker.C:
			d.check()
		}
	}
}

// check evaluates all tracked PIDs for straggler conditions and emits if detected.
func (d *Detector) check() {
	d.checkAt(time.Now())
}

// checkAt is the time-parameterized core of check(), enabling deterministic tests.
func (d *Detector) checkAt(now time.Time) {
	// Collect emissions under lock, emit outside lock (H1 fix).
	// This prevents the event loop from stalling on UDS writes.
	var emissions []StraggleState

	d.mu.Lock()

	for pid, ps := range d.pids {
		// Skip first check (no previous snapshot to compute rate from).
		if ps.prevTime.IsZero() {
			ps.prevCount = ps.launchCount
			ps.prevTime = now
			ps.schedSwitches = 0
			ps.preemptingPIDs = nil
			continue
		}

		elapsed := now.Sub(ps.prevTime).Seconds()
		if elapsed <= 0 {
			// L7 fix: reset counters even on clock backwards to avoid
			// double-sized delta on the next interval.
			ps.prevTime = now
			ps.prevCount = ps.launchCount
			ps.schedSwitches = 0
			ps.preemptingPIDs = nil
			continue
		}

		// Compute current ops/sec.
		delta := ps.launchCount - ps.prevCount
		currentOps := float64(delta) / elapsed

		// Snapshot and reset counters for next interval.
		schedCount := ps.schedSwitches
		preemptors := ps.preemptingPIDs
		ps.prevCount = ps.launchCount
		ps.prevTime = now
		ps.schedSwitches = 0
		ps.preemptingPIDs = nil

		// M12 fix: evict PIDs with no activity for idleEvictIntervals.
		if delta == 0 && schedCount == 0 {
			ps.idleIntervals++
			if ps.idleIntervals >= idleEvictIntervals {
				delete(d.pids, pid)
				continue
			}
		} else {
			ps.idleIntervals = 0
		}

		// Update EMA baseline. During baseline building (samples < minBaselineSamples),
		// always update. After baseline is established, update is deferred until after
		// detection to implement M1 fix (skip update during sustained straggler).
		if ps.samples == 0 {
			ps.baseline = currentOps
		} else if ps.samples < minBaselineSamples {
			ps.baseline = emaAlpha*currentOps + (1-emaAlpha)*ps.baseline
		}
		ps.samples++

		// Need enough baseline data before detection kicks in.
		if ps.samples < minBaselineSamples {
			continue
		}

		// Need a meaningful baseline (avoid false positives during idle).
		if ps.baseline <= 0 {
			continue
		}

		// Check throughput drop.
		dropThreshold := ps.baseline * (1 - d.config.ThroughputDropPct/100)
		throughputDrop := currentOps < dropThreshold

		// Check scheduling contention.
		contention := int(schedCount) >= d.config.SchedSwitchThreshold

		// Compute actual drop percentage.
		dropPct := (1 - currentOps/ps.baseline) * 100

		// Collect preempting PIDs.
		var preemptingPIDs []uint32
		for p := range preemptors {
			preemptingPIDs = append(preemptingPIDs, p)
		}

		// Both signals must fire for initial correlated detection.
		if throughputDrop && contention {
			ps.emitted = true
			// Straggler detected — do NOT update EMA baseline (M1 fix).
			state := StraggleState{
				PID:               pid,
				Comm:              ps.comm,
				ThroughputDropPct: dropPct,
				SchedSwitchCount:  schedCount,
				PreemptingPIDs:    preemptingPIDs,
				TimestampNs:       now.UnixNano(),
				Sustained:         false,
			}
			log.Printf("INFO: straggler: detected pid=%d comm=%q drop=%.1f%% sched_switches=%d preemptors=%v",
				pid, ps.comm, dropPct, schedCount, preemptingPIDs)
			emissions = append(emissions, state)
		} else if contention && ps.emitted {
			// Sustained re-emission: sched_switch still elevated after initial
			// correlation. Do NOT update EMA baseline (M1 fix preserved).
			state := StraggleState{
				PID:               pid,
				Comm:              ps.comm,
				ThroughputDropPct: dropPct,
				SchedSwitchCount:  schedCount,
				PreemptingPIDs:    preemptingPIDs,
				TimestampNs:       now.UnixNano(),
				Sustained:         true,
			}
			log.Printf("INFO: straggler: sustained pid=%d comm=%q drop=%.1f%% sched_switches=%d",
				pid, ps.comm, dropPct, schedCount)
			emissions = append(emissions, state)
		} else {
			// No detection and no sustained contention — update EMA baseline.
			if !contention {
				ps.emitted = false // Clear sustained state when contention ends
			}
			// M1 fix: update EMA only when NOT in straggler state.
			if ps.samples >= minBaselineSamples {
				ps.baseline = emaAlpha*currentOps + (1-emaAlpha)*ps.baseline
			}
			continue
		}
	}

	d.mu.Unlock()

	// H1 fix: emit outside the lock so UDS writes don't block ProcessEvent.
	if d.sink != nil {
		for _, state := range emissions {
			d.sink.SendStraggle(state)
		}
	}
}

// getOrCreate returns the pidState for pid, creating it if needed.
// Caller must hold d.mu.
func (d *Detector) getOrCreate(pid uint32) *pidState {
	ps, ok := d.pids[pid]
	if !ok {
		ps = &pidState{}
		d.pids[pid] = ps
	}
	return ps
}

// Config returns the detector's configuration (for testing/inspection).
func (d *Detector) Config() DetectorConfig {
	return d.config
}

// String implements fmt.Stringer for DetectorConfig.
func (c DetectorConfig) String() string {
	return fmt.Sprintf("drop=%g%% sched_threshold=%d interval=%v",
		c.ThroughputDropPct, c.SchedSwitchThreshold, c.CheckInterval)
}
