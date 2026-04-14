// Package correlate provides cross-layer correlation between host kernel events,
// system metrics, and CUDA latency statistics.
//
// v0.2: "WHY" engine — correlations (host event counts → CUDA tail latency).
// v0.3: "ROOT CAUSE" engine — causal chains (system + host + CUDA → timeline + fix).
//
// Call chain: correlate.Engine.RecordHost() called per host event →
//   Engine.SetSystemSnapshot() called each display tick →
//   Engine.SnapshotCausalChains() builds multi-layer root cause chains →
//   explain.go / watch.go renders chains as incident reports
//
// Algorithm (v0.3 — 2-layer causal chains + system context):
//  1. Maintain a sliding window of recent host events (last 10s).
//  2. Track latest system snapshot (CPU/mem/load from /proc).
//  3. At Snapshot time, for each CUDA op with anomalous tail (p99 > 3x p50):
//     a. Check system context: CPU > 90%? Memory > 95%? Swap? High load?
//     b. Check host window for sched_switch events (existing v0.2 logic).
//     c. Build timeline by sorting all correlated events by timestamp.
//     d. Generate explanation and recommendations.
//  4. OOM events always produce a chain.
package correlate

import (
	"fmt"
	"sort"
	"strings"
	"sync"
	"time"

	"github.com/ingero-io/ingero/internal/stats"
	"github.com/ingero-io/ingero/pkg/events"
)

// DefaultMaxAge is the maximum age of host events kept in the sliding window.
const DefaultMaxAge = 10 * time.Second

// DefaultSchedSwitchThreshold is the minimum number of sched_switch events
// in the window before we emit a correlation. Below this, preemption is normal.
const DefaultSchedSwitchThreshold = 5

// DefaultTailRatio is the p99/p50 ratio that triggers correlation analysis.
// A ratio > 3 means the tail latency is anomalously high.
const DefaultTailRatio = 3.0

// DefaultThroughputDropRatio is the current/peak rate below which a
// throughput-drop chain is emitted. 0.6 means >40% drop from peak.
const DefaultThroughputDropRatio = 0.6

// minThroughputPeak is the minimum peak ops/s before rate-based detection
// kicks in. Below this, event rates are too noisy for meaningful analysis.
const minThroughputPeak = 10.0

// minBaselineSnapshots is the number of snapshots needed before throughput-
// drop detection activates. This ensures the peak rate is a real baseline.
const minBaselineSnapshots = 5

// Correlation describes a detected relationship between host events and
// CUDA latency anomalies.
type Correlation struct {
	CUDAOp      string        // e.g., "cudaStreamSync"
	P99         time.Duration // observed p99 latency
	P50         time.Duration // observed p50 latency (baseline)
	TailRatio   float64       // p99/p50
	Cause       string        // human-readable cause description
	HostOpCount int           // number of correlated host events
	HostOp      string        // which host op (e.g., "sched_switch")
}

// String returns a human-readable one-line summary of the correlation.
func (c Correlation) String() string {
	return fmt.Sprintf("%s p99=%v (%.1fx p50) — %s", c.CUDAOp, c.P99, c.TailRatio, c.Cause)
}

// CausalChain represents a multi-layer root cause chain.
// Built from system context + host events + CUDA stats.
type CausalChain struct {
	ID              string       // content-based chain ID (e.g., "tail-medium-cuLaunchKernel")
	Severity        string       // "HIGH", "MEDIUM", "LOW"
	Summary         string       // one-line description
	RootCause       string       // human-readable root cause
	Timeline        []ChainEvent // ordered events across layers
	Explanation     string       // paragraph explaining the chain
	Recommendations []string     // actionable fixes
}

// ChainEvent is a single event in a causal chain timeline.
type ChainEvent struct {
	Timestamp time.Time
	Layer     string        // "SYSTEM", "HOST", "CUDA"
	Op        string        // operation name
	Detail    string        // human-readable detail
	Duration  time.Duration // 0 if not applicable
}

// SystemContext holds system-level metrics used for causal chain analysis.
// Mirrors stats.SystemSnapshot to avoid import cycle.
type SystemContext struct {
	CPUPercent float64
	MemUsedPct float64
	MemAvailMB int64
	SwapUsedMB int64
	LoadAvg1   float64
	Timestamp  time.Time
}

// PeakSystemContext computes per-metric worst-case values from a series of
// system snapshots. Different metrics may peak at different times (CPU at t=5,
// swap at t=30), so using the last snapshot would miss transient pressure.
// Returns nil if the slice is empty.
func PeakSystemContext(snapshots []SystemContext) *SystemContext {
	if len(snapshots) == 0 {
		return nil
	}
	peak := SystemContext{
		Timestamp:  snapshots[len(snapshots)-1].Timestamp,
		MemAvailMB: snapshots[0].MemAvailMB, // seed with first; lower is worse
	}
	for _, s := range snapshots {
		if s.CPUPercent > peak.CPUPercent {
			peak.CPUPercent = s.CPUPercent
		}
		if s.MemUsedPct > peak.MemUsedPct {
			peak.MemUsedPct = s.MemUsedPct
		}
		if s.MemAvailMB < peak.MemAvailMB {
			peak.MemAvailMB = s.MemAvailMB
		}
		if s.SwapUsedMB > peak.SwapUsedMB {
			peak.SwapUsedMB = s.SwapUsedMB
		}
		if s.LoadAvg1 > peak.LoadAvg1 {
			peak.LoadAvg1 = s.LoadAvg1
		}
	}
	return &peak
}

// Option configures a correlation Engine.
type Option func(*Engine)

// WithMaxAge sets the sliding window duration for host events.
// The default is DefaultMaxAge (10s), which suits live streaming (watch/demo).
// Use a longer duration for explain --since, or 0 to disable pruning entirely
// (required for historical replay where events have past timestamps).
func WithMaxAge(d time.Duration) Option {
	return func(e *Engine) {
		e.maxAge = d
	}
}

// cgroupStats tracks off-CPU scheduling statistics for a single cgroup.
// Used for noisy neighbor detection: if one cgroup's off-CPU time is much
// higher than peers, something else on the machine is stealing CPU.
type cgroupStats struct {
	offCPUDurations []time.Duration // individual sched_switch durations
	totalOffCPU     time.Duration
	eventCount      int64
	firstSeen       time.Time // timestamp of first sched_switch for this cgroup
}

// CGroupSchedStat is the exported per-cgroup scheduling stats for storage.
type CGroupSchedStat struct {
	CGroupID     uint64
	P99OffCPU    time.Duration
	TotalOffCPU  time.Duration
	EventCount   int64
	WindowStart  time.Time
	WindowEnd    time.Time
}

// opKey identifies a unique operation by source and op code.
type opKey struct {
	source events.Source
	op     uint8
}

// graphLaunchKey identifies a unique graph executable per PID.
type graphLaunchKey struct {
	pid        uint32
	execHandle uint64
}

// graphLaunchTracker tracks rolling launch rate for a graph executable.
type graphLaunchTracker struct {
	timestamps []time.Time // launch timestamps in the sliding window
	peakRate   float64     // peak launches/sec observed
	snapCount  int         // number of rate samples
}

// graphCaptureState tracks an in-flight graph capture for a PID.
type graphCaptureState struct {
	beginTime time.Time
	stream    uint64
	mode      uint32
}

// DefaultGraphCorrelationWindow is the time window for matching graph events
// with host/CUDA events for causal chain construction.
const DefaultGraphCorrelationWindow = 10 * time.Millisecond

// DefaultGraphFreqDropRatio is the launch rate below which a frequency
// anomaly is flagged. 0.5 = rate dropped more than 50% from peak.
const DefaultGraphFreqDropRatio = 0.5

// DefaultGraphFreqWindow is the sliding window for launch rate baseline.
const DefaultGraphFreqWindow = 10 * time.Second

// DefaultGraphNoLaunchTimeout is the window after instantiate within which
// a launch is expected. If no launch occurs, "never launched" is flagged.
const DefaultGraphNoLaunchTimeout = 30 * time.Second

// DefaultGraphCaptureTimeout is the maximum duration to wait for a
// cudaStreamEndCapture after a cudaStreamBeginCapture before flagging a
// potential cuBLAS lazy-init failure. 5 seconds is generous — a normal
// graph capture completes in milliseconds.
const DefaultGraphCaptureTimeout = 5 * time.Second

// minGraphCaptureDuration is the threshold below which a completed capture
// is suspiciously short. Combined with a non-zero RetCode, this suggests
// the capture failed immediately (e.g., due to cuBLAS lazy initialization).
const minGraphCaptureDuration = 1 * time.Millisecond

// Engine performs cross-layer correlation between host events and CUDA stats.
type Engine struct {
	mu         sync.RWMutex
	hostWindow []events.Event
	ioWindow   []events.Event  // block I/O events
	tcpWindow  []events.Event  // TCP retransmit events
	netWindow  []events.Event  // network socket events
	maxAge     time.Duration
	sysCtx     *SystemContext // latest system context, nil if not available
	node       string         // node identity for chain ID prefixing (v0.9)

	// Throughput-drop detection: track event rates across snapshots.
	latestTime   time.Time       // latest event timestamp (from AdvanceClock)
	prevOpCounts map[opKey]int64 // op counts at previous snapshot
	prevSnapTime time.Time       // event-time of previous snapshot
	peakRates    map[opKey]float64 // peak ops/sec observed per op
	snapCount    int             // total snapshots taken

	// Per-cgroup off-CPU tracking for noisy neighbor detection.
	// Keyed by cgroup_id, tracks sched_switch durations per cgroup.
	cgroupOffCPU  map[uint64]*cgroupStats
	targetCGroups map[uint64]bool // cgroup IDs of the target workload

	// CUDA Graph correlation state.
	graphWindow      []events.Event                       // graph events sliding window
	graphCaptures    map[uint32]*graphCaptureState         // in-flight captures per PID
	graphLaunchRates map[graphLaunchKey]*graphLaunchTracker // per-PID per-exec launch tracking
	graphInstantiations map[graphLaunchKey]time.Time        // instantiate without launch detection
}

// New creates a correlation engine with the default 10s sliding window.
// Pass WithMaxAge to override for longer collection windows or historical replay.
func New(opts ...Option) *Engine {
	e := &Engine{
		maxAge:              DefaultMaxAge,
		prevOpCounts:        make(map[opKey]int64),
		peakRates:           make(map[opKey]float64),
		cgroupOffCPU:        make(map[uint64]*cgroupStats),
		targetCGroups:       make(map[uint64]bool),
		graphCaptures:       make(map[uint32]*graphCaptureState),
		graphLaunchRates:    make(map[graphLaunchKey]*graphLaunchTracker),
		graphInstantiations: make(map[graphLaunchKey]time.Time),
	}
	for _, opt := range opts {
		opt(e)
	}
	return e
}

// SetNode sets the node identity used for chain ID prefixing.
// Must be called before any chains are generated.
func (e *Engine) SetNode(node string) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.node = node
}

// chainID returns a node-namespaced chain ID: "{node}:{descriptor}".
// If node is empty, returns the descriptor as-is for backward compatibility.
func (e *Engine) chainID(descriptor string) string {
	if e.node == "" {
		return descriptor
	}
	return e.node + ":" + descriptor
}

// RecordHost adds a host event to the sliding window.
// Old events beyond maxAge are pruned lazily.
//
// Per-cgroup off-CPU stats for noisy neighbor detection are tracked
// separately via RecordCGroupSchedSwitch(), which must be called for
// ALL sched_switch events (before PID filtering) so peer cgroup data
// is available.
func (e *Engine) RecordHost(evt events.Event) {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.hostWindow = append(e.hostWindow, evt)
	e.prune()
}

// AdvanceClock updates the engine's notion of "current time" from the event
// stream. In live mode, call with time.Now(). In replay mode, call with
// evt.Timestamp. Used for throughput-drop rate calculations where wall-clock
// is meaningless during replay.
func (e *Engine) AdvanceClock(t time.Time) {
	e.mu.Lock()
	if t.After(e.latestTime) {
		e.latestTime = t
	}
	e.mu.Unlock()
}

// RecordCGroupSchedSwitch tracks per-cgroup off-CPU stats for noisy neighbor
// detection. Must be called for ALL sched_switch events regardless of PID
// filter, because peer cgroup data is needed to detect scheduling contention.
//
// Call this BEFORE the PID filter in the event loop. RecordHost (called after
// the PID filter) only updates the host sliding window used for correlations.
func (e *Engine) RecordCGroupSchedSwitch(cgroupID uint64, duration time.Duration) {
	if cgroupID <= 1 || duration <= 0 {
		return
	}
	// Cap at 1s: voluntary sleeps (process blocked on I/O, futex, etc.) produce
	// multi-second durations that are not CPU contention. Real involuntary
	// preemption (noisy neighbor stealing CPU) never exceeds ~1s because the
	// Linux CFS scheduler's default timeslice is much shorter.
	const maxContention = time.Second
	if duration > maxContention {
		duration = maxContention
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	cs, ok := e.cgroupOffCPU[cgroupID]
	if !ok {
		cs = &cgroupStats{firstSeen: time.Now()}
		e.cgroupOffCPU[cgroupID] = cs
	}
	cs.offCPUDurations = append(cs.offCPUDurations, duration)
	cs.totalOffCPU += duration
	cs.eventCount++
}

// SetTargetCGroup registers a cgroup ID as belonging to the target workload.
// Used for noisy neighbor detection to distinguish target vs peer cgroups.
func (e *Engine) SetTargetCGroup(cgroupID uint64) {
	if cgroupID <= 1 {
		return // 0 = none, 1 = root cgroup
	}
	e.mu.Lock()
	defer e.mu.Unlock()
	e.targetCGroups[cgroupID] = true
}

// RecordEvent adds any event to the appropriate sliding window for correlation.
// This extends RecordHost to support I/O, TCP, and network events.
func (e *Engine) RecordEvent(evt events.Event) {
	e.mu.Lock()
	defer e.mu.Unlock()

	switch evt.Source {
	case events.SourceHost:
		e.hostWindow = append(e.hostWindow, evt)
	case events.SourceIO:
		e.ioWindow = append(e.ioWindow, evt)
	case events.SourceTCP:
		e.tcpWindow = append(e.tcpWindow, evt)
	case events.SourceNet:
		e.netWindow = append(e.netWindow, evt)
	case events.SourceCUDAGraph:
		e.graphWindow = append(e.graphWindow, evt)
		e.recordGraphState(evt)
	}
	e.prune()
}

// prune removes events older than maxAge from all windows.
// When maxAge == 0, pruning is disabled (unlimited window for historical replay).
// Must be called with e.mu held.
func (e *Engine) prune() {
	if e.maxAge == 0 {
		return
	}

	cutoff := time.Now().Add(-e.maxAge)
	e.hostWindow = pruneWindow(e.hostWindow, cutoff)
	e.ioWindow = pruneWindow(e.ioWindow, cutoff)
	e.tcpWindow = pruneWindow(e.tcpWindow, cutoff)
	e.netWindow = pruneWindow(e.netWindow, cutoff)
	e.graphWindow = pruneWindow(e.graphWindow, cutoff)
}

// pruneWindow removes events older than cutoff from a slice.
func pruneWindow(w []events.Event, cutoff time.Time) []events.Event {
	if len(w) == 0 {
		return w
	}
	i := 0
	for i < len(w) && w[i].Timestamp.Before(cutoff) {
		i++
	}
	if i > 0 {
		return w[i:]
	}
	return w
}

// SnapshotCorrelations analyzes CUDA stats against the host event window
// and returns any detected correlations.
//
// For each CUDA op where p99 > TailRatio * p50:
//   - Count sched_switch events for the given PID
//   - Count mm_page_alloc events and total bytes
//   - Check for OOM events
func (e *Engine) SnapshotCorrelations(cudaOps []stats.OpStats, pid uint32) []Correlation {
	e.mu.Lock()
	e.prune()
	window := make([]events.Event, len(e.hostWindow))
	copy(window, e.hostWindow)
	e.mu.Unlock()

	// Count host events by type for the given PID.
	schedSwitchCount := 0
	var totalOffCPU time.Duration
	pageAllocCount := 0
	var totalAllocBytes uint64
	oomCount := 0

	for _, evt := range window {
		if evt.Source != events.SourceHost {
			continue
		}

		switch events.HostOp(evt.Op) {
		case events.HostSchedSwitch:
			// pid=0 means "all processes" (no PID filter).
			if pid == 0 || evt.PID == pid || uint32(evt.Args[1]) == pid {
				schedSwitchCount++
				totalOffCPU += evt.Duration
			}
		case events.HostPageAlloc:
			if pid == 0 || evt.PID == pid {
				pageAllocCount++
				totalAllocBytes += evt.Args[0]
			}
		case events.HostOOMKill:
			oomCount++
		case events.HostMmPageAllocSummary:
			// Aggregated non-target mm_page_alloc events. Only counted
			// when pid=0 (global view) — target PIDs still emit raw
			// HostPageAlloc events counted above.
			if pid == 0 {
				pageAllocCount += int(evt.Args[0])
				totalAllocBytes += evt.Args[1]
			}
		case events.HostSchedSwitchSummary:
			// Aggregated non-target sched_switch transitions. Only
			// counted when pid=0 — target PIDs still emit raw
			// HostSchedSwitch events counted above.
			if pid == 0 {
				schedSwitchCount += int(evt.Args[0])
				totalOffCPU += time.Duration(evt.Args[1])
			}
		}
	}

	var correlations []Correlation

	// Check each CUDA op for anomalous tail latency.
	for _, op := range cudaOps {
		if op.Source != events.SourceCUDA && op.Source != events.SourceDriver {
			continue
		}
		if op.P50 == 0 || op.Count < 10 {
			continue
		}

		tailRatio := float64(op.P99) / float64(op.P50)
		if tailRatio < DefaultTailRatio {
			continue
		}

		// Correlate with sched_switch (CPU preemption).
		if schedSwitchCount >= DefaultSchedSwitchThreshold {
			correlations = append(correlations, Correlation{
				CUDAOp:      op.Op,
				P99:         op.P99,
				P50:         op.P50,
				TailRatio:   tailRatio,
				HostOpCount: schedSwitchCount,
				HostOp:      "sched_switch",
				Cause: fmt.Sprintf("correlated with %d sched_switch events (%v off-CPU)",
					schedSwitchCount, totalOffCPU.Round(time.Millisecond)),
			})
		}

		// Correlate with mm_page_alloc (memory pressure).
		if totalAllocBytes > 1<<30 { // > 1 GB
			correlations = append(correlations, Correlation{
				CUDAOp:      op.Op,
				P99:         op.P99,
				P50:         op.P50,
				TailRatio:   tailRatio,
				HostOpCount: pageAllocCount,
				HostOp:      "mm_page_alloc",
				Cause: fmt.Sprintf("correlated with %d page allocations (%.1f GB)",
					pageAllocCount, float64(totalAllocBytes)/(1<<30)),
			})
		}
	}

	// OOM events always generate a correlation (regardless of CUDA stats).
	if oomCount > 0 {
		correlations = append(correlations, Correlation{
			CUDAOp:      "all",
			HostOpCount: oomCount,
			HostOp:      "oom_kill",
			Cause:       fmt.Sprintf("OOM killer triggered %d time(s) — severe memory pressure", oomCount),
		})
	}

	return correlations
}

// HostEventCount returns the number of host events currently in the window.
// Uses RLock since this is a read-only operation.
func (e *Engine) HostEventCount() int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return len(e.hostWindow)
}

// SetSystemSnapshot updates the system context used for causal chain analysis.
// Called once per display tick from watch.go with the latest sysinfo snapshot.
func (e *Engine) SetSystemSnapshot(ctx *SystemContext) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.sysCtx = ctx
}

// SnapshotCausalChains analyzes CUDA stats against host events AND system context,
// producing formal causal chains with timelines, explanations, and recommendations.
func (e *Engine) SnapshotCausalChains(cudaOps []stats.OpStats, pid uint32) []CausalChain {
	e.mu.Lock()
	e.prune()
	window := make([]events.Event, len(e.hostWindow))
	copy(window, e.hostWindow)
	ioWindow := make([]events.Event, len(e.ioWindow))
	copy(ioWindow, e.ioWindow)
	tcpWindow := make([]events.Event, len(e.tcpWindow))
	copy(tcpWindow, e.tcpWindow)
	netWindow := make([]events.Event, len(e.netWindow))
	copy(netWindow, e.netWindow)
	graphWindow := make([]events.Event, len(e.graphWindow))
	copy(graphWindow, e.graphWindow)
	sysCtx := e.sysCtx
	e.mu.Unlock()

	// Count host events by type for the given PID.
	schedSwitchCount := 0
	var totalOffCPU time.Duration
	pageAllocCount := 0
	var totalAllocBytes uint64
	oomCount := 0
	podRestartCount := 0
	podEvictionCount := 0

	for _, evt := range window {
		if evt.Source != events.SourceHost {
			continue
		}
		switch events.HostOp(evt.Op) {
		case events.HostSchedSwitch:
			// pid=0 means "all processes" (no PID filter).
			if pid == 0 || evt.PID == pid || uint32(evt.Args[1]) == pid {
				schedSwitchCount++
				totalOffCPU += evt.Duration
			}
		case events.HostPageAlloc:
			if pid == 0 || evt.PID == pid {
				pageAllocCount++
				totalAllocBytes += evt.Args[0]
			}
		case events.HostOOMKill:
			oomCount++
		case events.HostPodRestart:
			podRestartCount++
		case events.HostPodEviction:
			podEvictionCount++
		case events.HostPodOOMKill:
			oomCount++ // K8s OOM kill treated same as kernel OOM
		case events.HostMmPageAllocSummary:
			// Aggregated non-target mm_page_alloc events — only
			// contribute to the global view (pid=0). See
			// SnapshotCorrelations for the same convention.
			if pid == 0 {
				pageAllocCount += int(evt.Args[0])
				totalAllocBytes += evt.Args[1]
			}
		case events.HostSchedSwitchSummary:
			// Aggregated non-target sched_switch transitions.
			if pid == 0 {
				schedSwitchCount += int(evt.Args[0])
				totalOffCPU += time.Duration(evt.Args[1])
			}
		}
	}

	// Count I/O events with read/write breakdown, max latency, and throughput.
	ioCount := len(ioWindow)
	var ioTotalDur, ioMaxLatency time.Duration
	var ioReadCount, ioWriteCount int
	var ioTotalBytes uint64
	for _, evt := range ioWindow {
		ioTotalDur += evt.Duration
		if evt.Duration > ioMaxLatency {
			ioMaxLatency = evt.Duration
		}
		ioTotalBytes += evt.Args[0] * 512 // Args[0] = nr_sector, 512 bytes/sector
		switch events.IOOp(evt.Op) {
		case events.IORead:
			ioReadCount++
		case events.IOWrite:
			ioWriteCount++
		}
	}

	// Count TCP retransmit events.
	tcpRetransmitCount := len(tcpWindow)

	// Count network events and total bytes transferred.
	netCount := len(netWindow)
	var netTotalBytes uint64
	for _, evt := range netWindow {
		netTotalBytes += evt.Args[1] // Args[1] = bytes transferred
	}

	infraCtx := &infraContext{
		ioCount:            ioCount,
		ioTotalDur:         ioTotalDur,
		ioReadCount:        ioReadCount,
		ioWriteCount:       ioWriteCount,
		ioMaxLatency:       ioMaxLatency,
		ioTotalBytes:       ioTotalBytes,
		tcpRetransmitCount: tcpRetransmitCount,
		netCount:           netCount,
		netTotalBytes:      netTotalBytes,
	}

	var chains []CausalChain

	for _, op := range cudaOps {
		if op.Source != events.SourceCUDA && op.Source != events.SourceDriver {
			continue
		}
		if op.P50 == 0 || op.Count < 10 {
			continue
		}

		tailRatio := float64(op.P99) / float64(op.P50)
		if tailRatio < DefaultTailRatio {
			continue
		}

		chain := e.buildChain(op, tailRatio, sysCtx, schedSwitchCount, totalOffCPU, pageAllocCount, totalAllocBytes, infraCtx)
		if chain != nil {
			chains = append(chains, *chain)
		}
	}

	// --- Throughput-drop detection ---
	// For CUDA/Driver ops that didn't trigger the tail-ratio gate, check if
	// their event rate dropped significantly from the observed peak. This
	// catches contention on fast GPUs where per-call latency stays low but
	// throughput tanks.
	now := e.latestTime
	if now.IsZero() {
		now = time.Now()
	}

	// Compute current rates and update peak rates (always, even during baseline).
	// opRates maps each op to its current rate this window.
	opRates := make(map[opKey]float64)
	if !e.prevSnapTime.IsZero() {
		elapsed := now.Sub(e.prevSnapTime).Seconds()
		if elapsed >= 0.5 {
			for _, op := range cudaOps {
				if op.Source != events.SourceCUDA && op.Source != events.SourceDriver {
					continue
				}
				key := opKey{op.Source, op.OpCode}
				prevCount, ok := e.prevOpCounts[key]
				if !ok {
					continue
				}
				delta := op.Count - prevCount
				if delta < 10 {
					continue
				}
				rate := float64(delta) / elapsed
				opRates[key] = rate
				if rate > e.peakRates[key] {
					e.peakRates[key] = rate
				}
			}
		}
	}

	// Only detect drops after enough baseline snapshots to establish a real peak.
	if e.snapCount >= minBaselineSnapshots && len(opRates) > 0 {
		// Collect ops that already have tail-ratio chains (skip them).
		tailChainOps := make(map[opKey]bool)
		for _, ch := range chains {
			if len(ch.Timeline) > 0 {
				last := ch.Timeline[len(ch.Timeline)-1]
				if last.Layer == "CUDA" || last.Layer == "DRIVER" {
					for _, op := range cudaOps {
						if op.Op == last.Op {
							tailChainOps[opKey{op.Source, op.OpCode}] = true
						}
					}
				}
			}
		}

		for _, op := range cudaOps {
			if op.Source != events.SourceCUDA && op.Source != events.SourceDriver {
				continue
			}
			key := opKey{op.Source, op.OpCode}
			if tailChainOps[key] {
				continue // already has a tail-ratio chain
			}
			currentRate, ok := opRates[key]
			if !ok {
				continue
			}

			peak := e.peakRates[key]
			if peak < minThroughputPeak {
				continue // rate too low to be meaningful
			}
			if currentRate < 1.0 {
				continue // workload stopped — not a real drop
			}

			ratio := currentRate / peak
			if ratio >= DefaultThroughputDropRatio {
				continue // drop < 40% — not significant
			}

			// Require corroborating host/infra evidence.
			hasEvidence := schedSwitchCount >= DefaultSchedSwitchThreshold ||
				infraCtx.ioCount >= 50 ||
				infraCtx.tcpRetransmitCount > 5

			if !hasEvidence {
				continue
			}

			chain := e.buildThroughputDropChain(op, peak, currentRate,
				sysCtx, schedSwitchCount, totalOffCPU, infraCtx)
			if chain != nil {
				chains = append(chains, *chain)
			}
		}
	}

	// Update throughput tracking state for next snapshot.
	e.prevOpCounts = make(map[opKey]int64)
	for _, op := range cudaOps {
		key := opKey{op.Source, op.OpCode}
		e.prevOpCounts[key] = op.Count
	}
	e.prevSnapTime = now
	e.snapCount++

	// OOM always produces a HIGH chain.
	if oomCount > 0 {
		chains = append(chains, CausalChain{
			ID:       e.chainID("oom"),
			Severity: "HIGH",
			Summary:  fmt.Sprintf("OOM killer triggered %d time(s)", oomCount),
			RootCause: "host memory exhaustion triggered OOM killer",
			Timeline: []ChainEvent{
				{Layer: "HOST", Op: "oom_kill", Detail: fmt.Sprintf("OOM killer invoked %d time(s)", oomCount)},
				{Layer: "CUDA", Op: "all", Detail: "all GPU operations affected by memory pressure"},
			},
			Explanation:     fmt.Sprintf("The Linux OOM killer was invoked %d time(s), indicating severe host memory exhaustion. This directly impacts CUDA operations that require host-side memory allocation.", oomCount),
			Recommendations: []string{"Reduce host memory usage", "Add swap space as a buffer", "Increase system RAM"},
		})
	}

	// TCP retransmit burst: always produces a chain when > 10 retransmits.
	if tcpRetransmitCount > 10 {
		severity := "MEDIUM"
		if tcpRetransmitCount > 100 {
			severity = "HIGH"
		}
		chains = append(chains, CausalChain{
			ID:        e.chainID(fmt.Sprintf("tcp-retransmit-%s", strings.ToLower(severity))),
			Severity:  severity,
			Summary:   fmt.Sprintf("%d TCP retransmits during trace window", tcpRetransmitCount),
			RootCause: "TCP retransmit burst indicates network congestion or packet loss",
			Timeline: []ChainEvent{
				{Layer: "NET", Op: "tcp_retransmit", Detail: fmt.Sprintf("%d retransmits", tcpRetransmitCount)},
				{Layer: "CUDA", Op: "all", Detail: "GPU operations may stall waiting for network data (NCCL, gRPC)"},
			},
			Explanation:     fmt.Sprintf("%d TCP retransmissions detected. This indicates network congestion, packet loss, or NIC issues. For distributed training (NCCL/DDP), retransmits cause all-reduce stalls. For inference serving, retransmits add latency to client responses.", tcpRetransmitCount),
			Recommendations: []string{"Check network switch errors and congestion", "Verify NIC link speed and errors (ethtool -S)", "For NCCL: consider NCCL_SOCKET_IFNAME and NCCL_IB_DISABLE tuning"},
		})
	}

	// Pod restart: always produces a chain.
	if podRestartCount > 0 {
		chains = append(chains, CausalChain{
			ID:        e.chainID("pod-restart"),
			Severity:  "HIGH",
			Summary:   fmt.Sprintf("K8s pod restart detected (%d restart(s))", podRestartCount),
			RootCause: "pod container restart concurrent with GPU workload",
			Timeline: []ChainEvent{
				{Layer: "HOST", Op: "pod_restart", Detail: fmt.Sprintf("%d container restart(s) detected", podRestartCount)},
				{Layer: "CUDA", Op: "all", Detail: "GPU operations may have been interrupted by pod restart"},
			},
			Explanation:     fmt.Sprintf("A K8s pod container restarted %d time(s) during the trace window. Container restarts cause process termination and re-initialization, resulting in GPU event gaps and lost in-flight work. Common causes: OOM kills, liveness probe failures, application crashes.", podRestartCount),
			Recommendations: []string{"Check pod events: kubectl describe pod <name>", "Review container logs: kubectl logs <pod> --previous", "Check for OOM: kubectl get events --field-selector reason=OOMKilling"},
		})
	}

	// Pod eviction: always produces a chain.
	if podEvictionCount > 0 {
		chains = append(chains, CausalChain{
			ID:        e.chainID("pod-eviction"),
			Severity:  "HIGH",
			Summary:   fmt.Sprintf("K8s pod eviction detected (%d eviction(s))", podEvictionCount),
			RootCause: "pod evicted by K8s scheduler during GPU workload",
			Timeline: []ChainEvent{
				{Layer: "HOST", Op: "pod_eviction", Detail: fmt.Sprintf("%d pod eviction(s) detected", podEvictionCount)},
				{Layer: "CUDA", Op: "all", Detail: "GPU workload terminated by eviction"},
			},
			Explanation:     fmt.Sprintf("A K8s pod was evicted %d time(s) during the trace window. Evictions are triggered by resource pressure (CPU, memory, disk) or preemption by higher-priority pods.", podEvictionCount),
			Recommendations: []string{"Check node resource pressure: kubectl describe node", "Review pod resource requests and limits", "Set PriorityClass for GPU workloads"},
		})
	}

	// Noisy neighbor detection: compare target cgroup's off-CPU p99 vs peer median.
	if noisyChain := e.detectNoisyNeighbor(); noisyChain != nil {
		chains = append(chains, *noisyChain)
	}

	// CUDA Graph correlation rules (FR55, FR56, FR59).
	graphChains := e.snapshotGraphChains(pid, window, graphWindow, cudaOps)
	chains = append(chains, graphChains...)

	return chains
}

// infraContext holds aggregated metrics from infrastructure event windows
// (I/O, TCP, network) for causal chain construction.
type infraContext struct {
	ioCount            int
	ioTotalDur         time.Duration
	ioReadCount        int
	ioWriteCount       int
	ioMaxLatency       time.Duration
	ioTotalBytes       uint64 // nr_sector * 512
	tcpRetransmitCount int
	netCount           int
	netTotalBytes      uint64
}

// buildChain constructs a causal chain for a single CUDA op with anomalous tail latency.
func (e *Engine) buildChain(
	op stats.OpStats,
	tailRatio float64,
	sysCtx *SystemContext,
	schedSwitchCount int,
	totalOffCPU time.Duration,
	pageAllocCount int,
	totalAllocBytes uint64,
	infra *infraContext,
) *CausalChain {
	var timeline []ChainEvent
	var causes []string
	var recommendations []string
	severity := "MEDIUM"

	// Layer 1: System context.
	if sysCtx != nil {
		if sysCtx.CPUPercent > 90 {
			timeline = append(timeline, ChainEvent{
				Layer:  "SYSTEM",
				Op:     "cpu",
				Detail: fmt.Sprintf("CPU %.0f%%", sysCtx.CPUPercent),
			})
			causes = append(causes, "high CPU utilization")
			severity = "HIGH"
		}
		if sysCtx.MemUsedPct > 95 {
			timeline = append(timeline, ChainEvent{
				Layer:  "SYSTEM",
				Op:     "memory",
				Detail: fmt.Sprintf("Memory %.0f%% (%d MB available)", sysCtx.MemUsedPct, sysCtx.MemAvailMB),
			})
			causes = append(causes, "memory pressure")
			severity = "HIGH"
		}
		if sysCtx.SwapUsedMB > 0 {
			timeline = append(timeline, ChainEvent{
				Layer:  "SYSTEM",
				Op:     "swap",
				Detail: fmt.Sprintf("Swap %d MB in use", sysCtx.SwapUsedMB),
			})
			causes = append(causes, "swap activity")
			severity = "HIGH"
			recommendations = append(recommendations, "Investigate memory consumers, consider adding RAM")
		}
		if sysCtx.LoadAvg1 > 10 {
			timeline = append(timeline, ChainEvent{
				Layer:  "SYSTEM",
				Op:     "load",
				Detail: fmt.Sprintf("Load average %.1f", sysCtx.LoadAvg1),
			})
			causes = append(causes, "high system load")
		}
	}

	// Layer 2: Host events.
	if schedSwitchCount >= DefaultSchedSwitchThreshold {
		timeline = append(timeline, ChainEvent{
			Layer:    "HOST",
			Op:       "sched_switch",
			Detail:   fmt.Sprintf("%d context switches (%v off-CPU)", schedSwitchCount, totalOffCPU.Round(time.Millisecond)),
			Duration: totalOffCPU,
		})
		causes = append(causes, fmt.Sprintf("%d sched_switch events", schedSwitchCount))
		recommendations = append(recommendations, "Pin training process to dedicated cores with taskset")
		recommendations = append(recommendations, "Add nice -n 19 to background jobs (logrotate, cron)")
	}
	if totalAllocBytes > 1<<30 {
		timeline = append(timeline, ChainEvent{
			Layer:  "HOST",
			Op:     "mm_page_alloc",
			Detail: fmt.Sprintf("%d allocations (%.1f GB)", pageAllocCount, float64(totalAllocBytes)/(1<<30)),
		})
		causes = append(causes, "large host memory allocations")
		recommendations = append(recommendations, "Identify the process allocating host memory during GPU work")
	}

	// Layer 2b: Infrastructure events (I/O, TCP, network).
	// Only include I/O when it's relevant to the CUDA stall — background
	// journald/flush noise (50+ ops with sub-ms latency) should not appear
	// in every chain. Require either significant volume (≥200 ops), total
	// duration ≥1s, or peak I/O latency ≥1% of the CUDA p99.
	cudaP99 := time.Duration(op.P99)
	if infra != nil {
		ioRelevant := infra.ioCount >= 200 ||
			infra.ioTotalDur >= time.Second ||
			(infra.ioMaxLatency > 0 && cudaP99 > 0 && infra.ioMaxLatency >= cudaP99/100)
		if ioRelevant {
			// Build detailed breakdown: reads vs writes, throughput, peak latency.
			detail := fmt.Sprintf("%d I/O ops", infra.ioCount)
			if infra.ioReadCount > 0 || infra.ioWriteCount > 0 {
				detail += fmt.Sprintf(" (%d reads, %d writes)", infra.ioReadCount, infra.ioWriteCount)
			}
			detail += fmt.Sprintf(", %v total", infra.ioTotalDur.Round(time.Millisecond))
			if infra.ioTotalBytes > 0 {
				detail += fmt.Sprintf(", %.1f MB", float64(infra.ioTotalBytes)/(1<<20))
			}
			if infra.ioMaxLatency > 0 {
				detail += fmt.Sprintf(", peak %v", infra.ioMaxLatency.Round(time.Microsecond))
			}
			timeline = append(timeline, ChainEvent{
				Layer:    "IO",
				Op:       "block_io",
				Detail:   detail,
				Duration: infra.ioTotalDur,
			})
			causes = append(causes, "heavy block I/O")
			if infra.ioMaxLatency > 50*time.Millisecond || infra.ioTotalDur > 2*time.Second {
				severity = "HIGH"
			}

			// Latency-based disk technology recommendations.
			if infra.ioMaxLatency > 20*time.Millisecond {
				recommendations = append(recommendations,
					fmt.Sprintf("Block I/O peak latency %v suggests spinning disk or network storage — NVMe SSD reduces this to <1ms",
						infra.ioMaxLatency.Round(time.Millisecond)))
			}
			if infra.ioWriteCount > infra.ioReadCount {
				recommendations = append(recommendations, "Write-heavy I/O during GPU work: check checkpoint saves, use async checkpointing or a separate fast volume")
			} else if infra.ioReadCount > infra.ioWriteCount {
				recommendations = append(recommendations, "Read-heavy I/O during GPU work: DataLoader bottleneck — pre-load data to /dev/shm, increase RAM for page cache, or use NVMe")
			} else {
				recommendations = append(recommendations, "Check for checkpoint writes, model loads, or DataLoader disk reads during GPU work")
			}
		}

		// TCP retransmit burst during GPU stall.
		if infra.tcpRetransmitCount > 5 {
			timeline = append(timeline, ChainEvent{
				Layer:  "NET",
				Op:     "tcp_retransmit",
				Detail: fmt.Sprintf("%d retransmits", infra.tcpRetransmitCount),
			})
			causes = append(causes, fmt.Sprintf("%d TCP retransmits", infra.tcpRetransmitCount))
			if infra.tcpRetransmitCount > 50 {
				severity = "HIGH"
			}
			recommendations = append(recommendations, "Check network health: switch errors, NIC stats (ethtool -S), MTU mismatches")
		}

		// GPU idle during network I/O.
		if infra.netCount > 100 && infra.netTotalBytes > 1<<20 { // >100 events, >1MB
			timeline = append(timeline, ChainEvent{
				Layer:  "NET",
				Op:     "socket_io",
				Detail: fmt.Sprintf("%d socket ops (%.1f MB)", infra.netCount, float64(infra.netTotalBytes)/(1<<20)),
			})
			causes = append(causes, "heavy network socket I/O")
			recommendations = append(recommendations, "Profile network calls — GPU may be idle waiting for HTTP/gRPC responses or NCCL transfers")
		}
	}

	// No causes found — not enough evidence for a chain.
	if len(causes) == 0 {
		return nil
	}

	// Layer 3: CUDA/Driver (the observed symptom).
	layer := "CUDA"
	if op.Source == events.SourceDriver {
		layer = "DRIVER"
	}
	timeline = append(timeline, ChainEvent{
		Layer:    layer,
		Op:       op.Op,
		Detail:   fmt.Sprintf("p99=%v (%.1fx p50=%v)", op.P99, tailRatio, op.P50),
		Duration: op.P99,
	})

	causeStr := causes[0]
	for i := 1; i < len(causes); i++ {
		causeStr += " + " + causes[i]
	}

	return &CausalChain{
		ID:       e.chainID(fmt.Sprintf("tail-%s-%s", strings.ToLower(severity), op.Op)),
		Severity: severity,
		Summary: fmt.Sprintf("%s p99=%v (%.1fx p50) — %s",
			op.Op, op.P99, tailRatio, causeStr),
		RootCause:       causeStr,
		Timeline:        timeline,
		Explanation:     buildExplanation(op, tailRatio, causes, sysCtx, infra),
		Recommendations: dedup(recommendations),
	}
}

// detectNoisyNeighbor checks if the target cgroup's off-CPU p99 is significantly
// higher than peer cgroups, indicating another workload is stealing CPU.
// Must be called WITHOUT e.mu held (acquires it internally).
func (e *Engine) detectNoisyNeighbor() *CausalChain {
	e.mu.Lock()
	defer e.mu.Unlock()

	if len(e.cgroupOffCPU) < 2 || len(e.targetCGroups) == 0 {
		return nil
	}

	// Compute p99 off-CPU for each cgroup.
	type cgroupP99 struct {
		cgroupID uint64
		p99      time.Duration
		count    int64
	}
	var targetP99s []cgroupP99
	var peerP99s []time.Duration

	for cgID, cs := range e.cgroupOffCPU {
		if cs.eventCount < 10 {
			continue // not enough data
		}
		p99 := durationPercentile(cs.offCPUDurations, 0.99)
		if e.targetCGroups[cgID] {
			targetP99s = append(targetP99s, cgroupP99{cgroupID: cgID, p99: p99, count: cs.eventCount})
		} else {
			peerP99s = append(peerP99s, p99)
		}
	}

	if len(targetP99s) == 0 || len(peerP99s) == 0 {
		return nil
	}

	// Compute peer median p99.
	sort.Slice(peerP99s, func(i, j int) bool { return peerP99s[i] < peerP99s[j] })
	peerMedian := peerP99s[len(peerP99s)/2]
	if peerMedian == 0 {
		return nil
	}

	// Check if any target cgroup's p99 > 2x peer median.
	for _, tp := range targetP99s {
		ratio := float64(tp.p99) / float64(peerMedian)
		if ratio < 2.0 {
			continue
		}

		severity := "MEDIUM"
		if ratio > 5.0 {
			severity = "HIGH"
		}
		return &CausalChain{
			ID:       e.chainID(fmt.Sprintf("noisy-neighbor-%s", strings.ToLower(severity))),
			Severity: severity,
			Summary:  fmt.Sprintf("noisy neighbor: target cgroup off-CPU p99=%v (%.1fx peer median %v)", tp.p99, ratio, peerMedian),
			RootCause: "another process/cgroup is consuming CPU, starving the GPU workload",
			Timeline: []ChainEvent{
				{Layer: "HOST", Op: "sched_switch", Detail: fmt.Sprintf("target cgroup %d: p99 off-CPU=%v (%d events)", tp.cgroupID, tp.p99, tp.count)},
				{Layer: "HOST", Op: "noisy_neighbor", Detail: fmt.Sprintf("peer cgroups median p99=%v (%.1fx lower)", peerMedian, ratio)},
				{Layer: "CUDA", Op: "all", Detail: "GPU operations delayed by CPU scheduling starvation"},
			},
			Explanation: fmt.Sprintf("The target workload's cgroup (ID %d) has off-CPU p99=%v, which is %.1fx higher than the peer median of %v. This means another process or container on this machine is consuming CPU cycles, causing the GPU workload to be preempted more than its neighbors.", tp.cgroupID, tp.p99, ratio, peerMedian),
			Recommendations: []string{
				"Identify the CPU-heavy neighbor process (top, htop, or 'ps aux --sort=-pcpu')",
				"Pin GPU workload to dedicated CPU cores with taskset or cpuset cgroup",
				"Consider CPU resource limits (cgroup v2 cpu.max) for non-GPU workloads",
			},
		}
	}

	return nil
}

// durationPercentile computes the pth percentile from a duration slice.
// Returns 0 if the slice is empty.
func durationPercentile(durations []time.Duration, pct float64) time.Duration {
	n := len(durations)
	if n == 0 {
		return 0
	}
	sorted := make([]time.Duration, n)
	copy(sorted, durations)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })

	idx := int(pct*float64(n)+0.5) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= n {
		idx = n - 1
	}
	return sorted[idx]
}

// SnapshotCGroupSchedStats returns per-cgroup off-CPU statistics for storage.
func (e *Engine) SnapshotCGroupSchedStats() []CGroupSchedStat {
	e.mu.RLock()
	defer e.mu.RUnlock()

	result := make([]CGroupSchedStat, 0, len(e.cgroupOffCPU))
	now := time.Now()
	for cgID, cs := range e.cgroupOffCPU {
		if cs.eventCount == 0 {
			continue
		}
		result = append(result, CGroupSchedStat{
			CGroupID:    cgID,
			P99OffCPU:   durationPercentile(cs.offCPUDurations, 0.99),
			TotalOffCPU: cs.totalOffCPU,
			EventCount:  cs.eventCount,
			WindowStart: cs.firstSeen,
			WindowEnd:   now,
		})
	}
	return result
}

// buildExplanation generates a human-readable paragraph explaining the chain.
func buildExplanation(op stats.OpStats, tailRatio float64, causes []string, sysCtx *SystemContext, infra *infraContext) string {
	explanation := fmt.Sprintf("%s tail latency is %.1fx higher than typical (p99=%v vs p50=%v). ", op.Op, tailRatio, op.P99, op.P50)

	if sysCtx != nil && sysCtx.CPUPercent > 90 {
		explanation += fmt.Sprintf("The system CPU is at %.0f%%, indicating heavy contention for CPU resources. ", sysCtx.CPUPercent)
	}
	if sysCtx != nil && sysCtx.SwapUsedMB > 0 {
		explanation += fmt.Sprintf("The system is using %d MB of swap, causing memory access latency spikes. ", sysCtx.SwapUsedMB)
	}

	// I/O-specific explanation with disk technology inference.
	if infra != nil && (infra.ioCount > 50 || infra.ioTotalDur > 500*time.Millisecond) {
		explanation += fmt.Sprintf("Block I/O activity (%d ops, %v total, %.1f MB) is concurrent with GPU stalls. ",
			infra.ioCount, infra.ioTotalDur.Round(time.Millisecond), float64(infra.ioTotalBytes)/(1<<20))
		if infra.ioMaxLatency > 20*time.Millisecond {
			explanation += fmt.Sprintf("Peak I/O latency of %v indicates spinning disk or network-attached storage; NVMe SSD typically achieves <1ms. ",
				infra.ioMaxLatency.Round(time.Millisecond))
		} else if infra.ioMaxLatency > 2*time.Millisecond {
			explanation += fmt.Sprintf("Peak I/O latency of %v suggests SATA SSD; NVMe would halve this. ",
				infra.ioMaxLatency.Round(time.Microsecond))
		}
		if infra.ioReadCount > infra.ioWriteCount*2 {
			explanation += "Read-dominant I/O pattern suggests DataLoader or model loading from disk. "
		} else if infra.ioWriteCount > infra.ioReadCount*2 {
			explanation += "Write-dominant I/O pattern suggests checkpoint saves or logging during GPU work. "
		}
	}

	// TCP-specific explanation.
	if infra != nil && infra.tcpRetransmitCount > 5 {
		explanation += fmt.Sprintf("%d TCP retransmissions indicate network congestion or packet loss. ", infra.tcpRetransmitCount)
		if infra.tcpRetransmitCount > 50 {
			explanation += "This volume of retransmits can add seconds of latency to distributed training (NCCL AllReduce) or inference serving. "
		}
	}

	// Network I/O explanation.
	if infra != nil && infra.netCount > 100 && infra.netTotalBytes > 1<<20 {
		explanation += fmt.Sprintf("Heavy network socket I/O (%d ops, %.1f MB) concurrent with GPU stalls — GPU may be idle waiting for HTTP/gRPC responses or NCCL data transfers. ",
			infra.netCount, float64(infra.netTotalBytes)/(1<<20))
	}

	for _, cause := range causes {
		if cause != "high CPU utilization" && cause != "swap activity" &&
			cause != "heavy block I/O" && cause != "heavy network socket I/O" &&
			!strings.Contains(cause, "TCP retransmit") {
			explanation += fmt.Sprintf("Contributing factor: %s. ", cause)
		}
	}

	return explanation
}

// buildThroughputDropChain creates a chain for a CUDA op whose event rate
// dropped significantly from its peak while host evidence confirms resource
// contention. This complements the tail-ratio gate for fast GPUs where
// per-call latency stays low but aggregate throughput drops.
func (e *Engine) buildThroughputDropChain(
	op stats.OpStats,
	peakRate, currentRate float64,
	sysCtx *SystemContext,
	schedSwitchCount int,
	totalOffCPU time.Duration,
	infra *infraContext,
) *CausalChain {
	dropPct := (1.0 - currentRate/peakRate) * 100
	var timeline []ChainEvent
	var causes []string
	var recommendations []string
	severity := "MEDIUM"
	if dropPct > 60 {
		severity = "HIGH"
	}

	// Layer 1: System context (same pattern as buildChain).
	if sysCtx != nil {
		if sysCtx.CPUPercent > 90 {
			timeline = append(timeline, ChainEvent{
				Layer: "SYSTEM", Op: "cpu",
				Detail: fmt.Sprintf("CPU %.0f%%", sysCtx.CPUPercent),
			})
			causes = append(causes, "high CPU utilization")
			severity = "HIGH"
		}
		if sysCtx.MemUsedPct > 95 {
			timeline = append(timeline, ChainEvent{
				Layer: "SYSTEM", Op: "memory",
				Detail: fmt.Sprintf("Memory %.0f%% used (%d MB available)", sysCtx.MemUsedPct, sysCtx.MemAvailMB),
			})
			causes = append(causes, "memory pressure")
		}
	}

	// Layer 2: Host evidence.
	if schedSwitchCount >= DefaultSchedSwitchThreshold {
		timeline = append(timeline, ChainEvent{
			Layer: "HOST", Op: "sched_switch",
			Detail:   fmt.Sprintf("%d context switches (%s off-CPU)", schedSwitchCount, totalOffCPU.Round(time.Millisecond)),
			Duration: totalOffCPU,
		})
		causes = append(causes, fmt.Sprintf("%d sched_switch events", schedSwitchCount))
		recommendations = append(recommendations,
			"Pin training process to dedicated cores with taskset",
			"Add nice -n 19 to background jobs (logrotate, cron)",
		)
	}
	if infra != nil && infra.ioCount >= 50 {
		timeline = append(timeline, ChainEvent{
			Layer: "INFRA", Op: "block_io",
			Detail: fmt.Sprintf("%d I/O ops (%s total)", infra.ioCount, infra.ioTotalDur.Round(time.Millisecond)),
		})
		causes = append(causes, fmt.Sprintf("%d block I/O operations", infra.ioCount))
	}
	if infra != nil && infra.tcpRetransmitCount > 5 {
		timeline = append(timeline, ChainEvent{
			Layer: "NET", Op: "tcp_retransmit",
			Detail: fmt.Sprintf("%d retransmits", infra.tcpRetransmitCount),
		})
		causes = append(causes, fmt.Sprintf("%d TCP retransmits", infra.tcpRetransmitCount))
		recommendations = append(recommendations, "Check network health: switch errors, NIC stats (ethtool -S), MTU mismatches")
	}

	if len(causes) == 0 {
		return nil
	}

	// Layer 3: The affected CUDA/Driver op (throughput drop).
	layer := "CUDA"
	if op.Source == events.SourceDriver {
		layer = "DRIVER"
	}
	timeline = append(timeline, ChainEvent{
		Layer:  layer,
		Op:     op.Op,
		Detail: fmt.Sprintf("throughput dropped %.0f%% (%.0f → %.0f ops/s)", dropPct, peakRate, currentRate),
	})

	summary := fmt.Sprintf("%s throughput dropped %.0f%% — %s", op.Op, dropPct, strings.Join(causes, " + "))

	return &CausalChain{
		ID:              e.chainID(fmt.Sprintf("drop-%s-%s", strings.ToLower(severity), op.Op)),
		Severity:        severity,
		Summary:         summary,
		RootCause:       strings.Join(causes, " + "),
		Timeline:        timeline,
		Explanation:     e.buildThroughputExplanation(op, peakRate, currentRate, dropPct, causes, sysCtx),
		Recommendations: dedup(recommendations),
	}
}

func (e *Engine) buildThroughputExplanation(op stats.OpStats, peak, current, dropPct float64, causes []string, sysCtx *SystemContext) string {
	var b strings.Builder
	fmt.Fprintf(&b, "%s throughput dropped %.0f%% from %.0f to %.0f ops/s. ", op.Op, dropPct, peak, current)
	fmt.Fprintf(&b, "Per-call latency (p50=%v, p99=%v) remained low, but aggregate throughput declined due to %s. ",
		op.P50, op.P99, strings.Join(causes, " and "))
	if sysCtx != nil && sysCtx.CPUPercent > 90 {
		fmt.Fprintf(&b, "The system CPU is at %.0f%%, causing the host-side CUDA dispatch path to be delayed between calls. ", sysCtx.CPUPercent)
	}
	fmt.Fprintf(&b, "This pattern is typical on fast GPUs where individual calls complete quickly but the CPU cannot dispatch them fast enough under contention.")
	return b.String()
}

// severityRank maps severity strings to numeric rank for comparison.
// Higher rank = more severe.
func severityRank(s string) int {
	switch s {
	case "HIGH":
		return 3
	case "MEDIUM":
		return 2
	case "LOW":
		return 1
	default:
		return 0
	}
}

// ReplayEventsForChains replays events chronologically through a stats
// collector and correlation engine, snapshotting chains at 1-second boundaries.
//
// This preserves temporal dynamics: the baseline period sets a low p50, and
// the anomaly period pushes p99 up, creating high tail ratios. A single-pass
// replay averages away the baseline→anomaly transition.
//
// Used by both `ingero explain` and the MCP `get_causal_chains` tool.
func ReplayEventsForChains(evts []events.Event, collector *stats.Collector, corr *Engine, pid uint32) []CausalChain {
	var nextWindow time.Time
	bestChains := make(map[string]CausalChain) // key: layer:op → highest-severity chain

	mergeChains := func(chains []CausalChain) {
		for _, ch := range chains {
			if len(ch.Timeline) == 0 {
				continue
			}
			last := ch.Timeline[len(ch.Timeline)-1]
			key := last.Layer + ":" + last.Op
			if existing, ok := bestChains[key]; !ok || severityRank(ch.Severity) > severityRank(existing.Severity) {
				bestChains[key] = ch
			}
		}
	}

	recordWindow := func() {
		snap := collector.Snapshot()
		mergeChains(corr.SnapshotCausalChains(snap.Ops, pid))
	}

	// Global collector with a window large enough to hold all events.
	// The rolling collector (1,000 samples) detects concentrated anomalies
	// (e.g., CPU stress burst). The global collector detects anomalies that
	// are spread across the full session (e.g., cudaDeviceSync with rare
	// tail spikes that never concentrate in a single 1-second window).
	globalWindow := len(evts)
	if globalWindow < 1000 {
		globalWindow = 1000
	}
	globalCollector := stats.New(stats.WithWindowSize(globalWindow))

	for _, evt := range evts {
		collector.Record(evt)
		globalCollector.Record(evt)
		corr.AdvanceClock(evt.Timestamp)
		switch evt.Source {
		case events.SourceHost:
			corr.RecordHost(evt)
		case events.SourceIO, events.SourceTCP, events.SourceNet, events.SourceCUDAGraph:
			corr.RecordEvent(evt)
		}

		if nextWindow.IsZero() {
			nextWindow = evt.Timestamp.Add(time.Second)
		}
		if evt.Timestamp.After(nextWindow) {
			nextWindow = evt.Timestamp.Add(time.Second)
			recordWindow()
		}
	}

	// Final rolling-window snapshot.
	recordWindow()

	// Global snapshot: catches session-wide anomalies missed by the
	// rolling window (e.g., operations whose tail latency is spread
	// across the entire trace rather than concentrated in a burst).
	globalSnap := globalCollector.Snapshot()
	mergeChains(corr.SnapshotCausalChains(globalSnap.Ops, pid))

	// Collect results.
	result := make([]CausalChain, 0, len(bestChains))
	for _, ch := range bestChains {
		result = append(result, ch)
	}
	return result
}

// recordGraphState updates internal graph tracking state for correlation.
// Must be called with e.mu held.
func (e *Engine) recordGraphState(evt events.Event) {
	pid := evt.PID
	op := events.CUDAGraphOp(evt.Op)

	switch op {
	case events.GraphBeginCapture:
		e.graphCaptures[pid] = &graphCaptureState{
			beginTime: evt.Timestamp,
			stream:    evt.StreamHandle,
			mode:      evt.CaptureMode,
		}
	case events.GraphEndCapture:
		delete(e.graphCaptures, pid)
	case events.GraphInstantiate:
		key := graphLaunchKey{pid: pid, execHandle: evt.ExecHandle}
		e.graphInstantiations[key] = evt.Timestamp
	case events.GraphLaunch:
		key := graphLaunchKey{pid: pid, execHandle: evt.ExecHandle}
		// Clear instantiate-without-launch tracking.
		delete(e.graphInstantiations, key)
		// Track launch rate.
		tracker, ok := e.graphLaunchRates[key]
		if !ok {
			tracker = &graphLaunchTracker{}
			e.graphLaunchRates[key] = tracker
		}
		tracker.timestamps = append(tracker.timestamps, evt.Timestamp)
		// Prune old timestamps outside the frequency window.
		cutoff := evt.Timestamp.Add(-DefaultGraphFreqWindow)
		i := 0
		for i < len(tracker.timestamps) && tracker.timestamps[i].Before(cutoff) {
			i++
		}
		if i > 0 {
			tracker.timestamps = tracker.timestamps[i:]
		}
		// Update peak rate.
		if len(tracker.timestamps) >= 2 {
			elapsed := tracker.timestamps[len(tracker.timestamps)-1].Sub(tracker.timestamps[0]).Seconds()
			if elapsed > 0.1 {
				rate := float64(len(tracker.timestamps)) / elapsed
				if rate > tracker.peakRate {
					tracker.peakRate = rate
				}
				tracker.snapCount++
			}
		}
	}
}

// SnapshotGraphChains builds causal chains for CUDA Graph events by
// correlating with host events and CUDA memory events in the current windows.
// Called from SnapshotCausalChains. Must be called with e.mu held.
func (e *Engine) snapshotGraphChains(pid uint32, hostWindow, graphWindow []events.Event, cudaOps []stats.OpStats) []CausalChain {
	var chains []CausalChain

	// Rule 1: Graph Capture + Memory Pressure (FR55).
	// Check if any graph capture overlapped with cudaMalloc failures.
	chains = append(chains, e.checkGraphCaptureOOM(pid, graphWindow, cudaOps)...)

	// Rule 2: Graph Launch + CPU Contention (FR56).
	chains = append(chains, e.checkGraphLaunchCPUContention(pid, graphWindow, hostWindow)...)

	// Rule 3: Graph Launch Frequency Anomaly.
	chains = append(chains, e.checkGraphFrequencyAnomaly(pid)...)

	// Rule 4: Capture Never Launched.
	chains = append(chains, e.checkGraphNeverLaunched(pid, graphWindow)...)

	// Rule 5: Graph Capture Warmup Failure (cuBLAS lazy init).
	chains = append(chains, e.checkGraphCaptureWarmup(pid, graphWindow)...)

	return chains
}

// checkGraphCaptureOOM detects OOM during graph capture (FR55).
func (e *Engine) checkGraphCaptureOOM(pid uint32, graphWindow []events.Event, cudaOps []stats.OpStats) []CausalChain {
	// Find graph captures (begin→end pairs).
	type captureSpan struct {
		begin, end time.Time
		stream     uint64
		mode       uint32
		duration   time.Duration
	}
	var captures []captureSpan
	beginTimes := make(map[uint32]events.Event) // tid → begin event

	for _, evt := range graphWindow {
		if pid != 0 && evt.PID != pid {
			continue
		}
		op := events.CUDAGraphOp(evt.Op)
		switch op {
		case events.GraphBeginCapture:
			beginTimes[evt.TID] = evt
		case events.GraphEndCapture:
			if begin, ok := beginTimes[evt.TID]; ok {
				captures = append(captures, captureSpan{
					begin:    begin.Timestamp,
					end:      evt.Timestamp,
					stream:   begin.StreamHandle,
					mode:     begin.CaptureMode,
					duration: evt.Duration,
				})
				delete(beginTimes, evt.TID)
			}
		}
	}

	if len(captures) == 0 {
		return nil
	}

	// Check for cudaMalloc failures in CUDA ops stats.
	var mallocFailures int
	for _, op := range cudaOps {
		if op.Source == events.SourceCUDA && op.OpCode == uint8(events.CUDAMalloc) {
			// Check if error rate is elevated (count vs errors not tracked in OpStats,
			// but we can check if any graph event had non-zero RetCode).
			break
		}
	}

	// Also check graph events themselves for failures during capture.
	for _, evt := range graphWindow {
		if pid != 0 && evt.PID != pid {
			continue
		}
		if events.CUDAGraphOp(evt.Op) == events.GraphEndCapture && evt.RetCode != 0 {
			mallocFailures++
		}
	}

	// Check system context for memory pressure during capture.
	memPressure := false
	if e.sysCtx != nil && e.sysCtx.MemUsedPct > 90 {
		memPressure = true
	}

	if mallocFailures == 0 && !memPressure {
		return nil
	}

	var timeline []ChainEvent
	for _, cap := range captures {
		modeName := "global"
		switch cap.mode {
		case 1:
			modeName = "thread_local"
		case 2:
			modeName = "relaxed"
		}
		timeline = append(timeline, ChainEvent{
			Timestamp: cap.begin,
			Layer:     "CUDA_GRAPH",
			Op:        "graphCapture",
			Detail:    fmt.Sprintf("capture (mode=%s, stream=0x%x, duration=%v)", modeName, cap.stream, cap.duration.Round(time.Microsecond)),
			Duration:  cap.duration,
		})
	}

	cause := "memory pressure during CUDA Graph capture"
	if mallocFailures > 0 {
		cause = fmt.Sprintf("graph capture failed (%d capture(s) returned error) — OOM during graph capture", mallocFailures)
		timeline = append(timeline, ChainEvent{
			Layer:  "CUDA",
			Op:     "cudaMalloc",
			Detail: fmt.Sprintf("%d graph capture failure(s)", mallocFailures),
		})
	}
	if memPressure {
		timeline = append(timeline, ChainEvent{
			Layer:  "SYSTEM",
			Op:     "memory",
			Detail: fmt.Sprintf("VRAM/memory pressure (%.0f%% used)", e.sysCtx.MemUsedPct),
		})
	}

	return []CausalChain{{
		ID:       e.chainID("graph-capture-oom"),
		Severity: "HIGH",
		Summary:  fmt.Sprintf("OOM during graph capture (%d capture(s))", len(captures)),
		RootCause: cause,
		Timeline:  timeline,
		Explanation: "CUDA Graph capture overlapped with memory pressure. When VRAM is near capacity, graph capture allocates temporary buffers that can trigger OOM. This is a common failure mode in vLLM with torch.compile(mode='reduce-overhead').",
		Recommendations: []string{
			"Reduce model batch size to free VRAM before graph capture",
			"Set PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to reduce fragmentation",
			"Monitor VRAM with ingero trace --remediate during graph warmup phase",
		},
	}}
}

// checkGraphLaunchCPUContention detects CPU scheduling interference during graph launch (FR56).
func (e *Engine) checkGraphLaunchCPUContention(pid uint32, graphWindow, hostWindow []events.Event) []CausalChain {
	// Find graph launches.
	var launches []events.Event
	for _, evt := range graphWindow {
		if pid != 0 && evt.PID != pid {
			continue
		}
		if events.CUDAGraphOp(evt.Op) == events.GraphLaunch {
			launches = append(launches, evt)
		}
	}

	if len(launches) == 0 {
		return nil
	}

	// Count sched_switch events near graph launches.
	schedSwitchNearLaunch := 0
	var totalOffCPU time.Duration
	window := DefaultGraphCorrelationWindow

	for _, launch := range launches {
		launchStart := launch.Timestamp.Add(-launch.Duration) // approximate entry time
		for _, host := range hostWindow {
			if host.Source != events.SourceHost {
				continue
			}
			if events.HostOp(host.Op) != events.HostSchedSwitch {
				continue
			}
			if pid != 0 && host.PID != pid && uint32(host.Args[1]) != pid {
				continue
			}
			// Check if sched_switch is within the correlation window of the launch.
			if host.Timestamp.After(launchStart.Add(-window)) && host.Timestamp.Before(launch.Timestamp.Add(window)) {
				schedSwitchNearLaunch++
				totalOffCPU += host.Duration
			}
		}
	}

	if schedSwitchNearLaunch < DefaultSchedSwitchThreshold {
		return nil
	}

	return []CausalChain{{
		ID:       e.chainID("graph-launch-cpu-contention"),
		Severity: "MEDIUM",
		Summary:  fmt.Sprintf("CPU contention delaying graph dispatch (%d launch(es), %d sched_switch)", len(launches), schedSwitchNearLaunch),
		RootCause: "CPU scheduling interference during CUDA Graph launch",
		Timeline: []ChainEvent{
			{Layer: "CUDA_GRAPH", Op: "graphLaunch", Detail: fmt.Sprintf("%d graph launch(es)", len(launches))},
			{Layer: "HOST", Op: "sched_switch", Detail: fmt.Sprintf("%d context switches (%v off-CPU) near graph launches", schedSwitchNearLaunch, totalOffCPU.Round(time.Millisecond)), Duration: totalOffCPU},
		},
		Explanation: fmt.Sprintf("CUDA Graph launches experienced CPU scheduling interference. %d sched_switch events occurred within %v of graph launches, causing %v total off-CPU time. Graph launches are lightweight host-side operations, but CPU contention delays dispatch to the GPU.", schedSwitchNearLaunch, window, totalOffCPU.Round(time.Millisecond)),
		Recommendations: []string{
			"Pin the inference process to dedicated CPU cores (taskset/cset)",
			"Reduce background CPU load during inference",
			"Add nice -n 19 to non-critical processes",
		},
	}}
}

// checkGraphFrequencyAnomaly detects graph launch rate drops (graph pool exhaustion).
func (e *Engine) checkGraphFrequencyAnomaly(pid uint32) []CausalChain {
	var chains []CausalChain

	for key, tracker := range e.graphLaunchRates {
		if pid != 0 && key.pid != pid {
			continue
		}
		if tracker.snapCount < 3 || tracker.peakRate < 5.0 {
			continue // not enough data or too low to be meaningful
		}
		if len(tracker.timestamps) < 2 {
			continue
		}

		elapsed := tracker.timestamps[len(tracker.timestamps)-1].Sub(tracker.timestamps[0]).Seconds()
		if elapsed < 0.5 {
			continue
		}
		currentRate := float64(len(tracker.timestamps)) / elapsed

		ratio := currentRate / tracker.peakRate
		if ratio >= DefaultGraphFreqDropRatio {
			continue // drop not significant enough
		}

		dropPct := (1 - ratio) * 100

		chains = append(chains, CausalChain{
			ID:       e.chainID(fmt.Sprintf("graph-freq-anomaly-0x%x", key.execHandle)),
			Severity: "MEDIUM",
			Summary:  fmt.Sprintf("graph launch rate dropped %.0f%% (exec 0x%x, PID %d)", dropPct, key.execHandle, key.pid),
			RootCause: "graph pool exhaustion — likely re-capture triggered by new batch size",
			Timeline: []ChainEvent{
				{Layer: "CUDA_GRAPH", Op: "graphLaunch", Detail: fmt.Sprintf("rate dropped from %.0f to %.0f launches/sec (exec 0x%x)", tracker.peakRate, currentRate, key.execHandle)},
			},
			Explanation: fmt.Sprintf("GraphLaunch rate for executable 0x%x dropped %.0f%% from peak (%.0f → %.0f launches/sec). In vLLM, this pattern indicates a new batch size arrived that doesn't match any pre-captured graph, forcing a re-capture cycle. During re-capture, the existing graph pool is not launched.", key.execHandle, dropPct, tracker.peakRate, currentRate),
			Recommendations: []string{
				"Pre-warm all expected batch sizes during model startup",
				"Set max_num_batched_tokens to limit batch size variability",
				"Monitor with ingero trace to identify which batch sizes trigger re-capture",
			},
		})
	}

	return chains
}

// checkGraphNeverLaunched detects graphs that were instantiated but never launched.
func (e *Engine) checkGraphNeverLaunched(pid uint32, graphWindow []events.Event) []CausalChain {
	var chains []CausalChain

	now := time.Now()
	if !e.latestTime.IsZero() {
		now = e.latestTime
	}

	for key, instantTime := range e.graphInstantiations {
		if pid != 0 && key.pid != pid {
			continue
		}
		if now.Sub(instantTime) < DefaultGraphNoLaunchTimeout {
			continue // still within expected window
		}
		chains = append(chains, CausalChain{
			ID:       e.chainID(fmt.Sprintf("graph-never-launched-0x%x", key.execHandle)),
			Severity: "LOW",
			Summary:  fmt.Sprintf("graph instantiated but never launched (exec 0x%x, PID %d)", key.execHandle, key.pid),
			RootCause: "graph instantiated but never launched — wasted VRAM",
			Timeline: []ChainEvent{
				{Layer: "CUDA_GRAPH", Op: "graphInstantiate", Detail: fmt.Sprintf("instantiated exec 0x%x at %s, no launch after %v", key.execHandle, instantTime.Format("15:04:05.000"), now.Sub(instantTime).Round(time.Second))},
			},
			Explanation: fmt.Sprintf("A CUDA Graph was instantiated (exec handle 0x%x) but no GraphLaunch was observed within %v. This wastes GPU memory — each instantiated graph holds device memory for its captured operations.", key.execHandle, DefaultGraphNoLaunchTimeout),
			Recommendations: []string{
				"Verify the graph capture path leads to actual execution",
				"Destroy unused graph executables to free VRAM",
			},
		})
	}

	return chains
}

// checkGraphCaptureWarmup detects CUDA Graph capture failures caused by
// cuBLAS lazy initialization. Two patterns are flagged:
//   - A BeginCapture has no matching EndCapture within DefaultGraphCaptureTimeout.
//   - An EndCapture arrived but the capture was abnormally short (< 1ms) with
//     a non-zero RetCode, suggesting the capture failed immediately.
func (e *Engine) checkGraphCaptureWarmup(pid uint32, graphWindow []events.Event) []CausalChain {
	now := time.Now()
	if !e.latestTime.IsZero() {
		now = e.latestTime
	}

	// Pattern A: in-flight captures that timed out (no EndCapture received).
	var timedOut []struct {
		pid    uint32
		stream uint64
		age    time.Duration
	}
	for capPID, state := range e.graphCaptures {
		if pid != 0 && capPID != pid {
			continue
		}
		age := now.Sub(state.beginTime)
		if age >= DefaultGraphCaptureTimeout {
			timedOut = append(timedOut, struct {
				pid    uint32
				stream uint64
				age    time.Duration
			}{capPID, state.stream, age})
		}
	}

	// Pattern B: completed captures with abnormally short duration AND error.
	type failedCapture struct {
		pid      uint32
		stream   uint64
		duration time.Duration
		retCode  int32
	}
	var shortFails []failedCapture
	beginTimes := make(map[uint32]events.Event) // TID → begin event

	for _, evt := range graphWindow {
		if pid != 0 && evt.PID != pid {
			continue
		}
		op := events.CUDAGraphOp(evt.Op)
		switch op {
		case events.GraphBeginCapture:
			beginTimes[evt.TID] = evt
		case events.GraphEndCapture:
			if _, ok := beginTimes[evt.TID]; ok {
				delete(beginTimes, evt.TID)
			}
			if evt.RetCode != 0 && evt.Duration < minGraphCaptureDuration {
				shortFails = append(shortFails, failedCapture{
					pid:      evt.PID,
					stream:   evt.StreamHandle,
					duration: evt.Duration,
					retCode:  evt.RetCode,
				})
			}
		}
	}

	if len(timedOut) == 0 && len(shortFails) == 0 {
		return nil
	}

	var timeline []ChainEvent

	for _, to := range timedOut {
		timeline = append(timeline, ChainEvent{
			Layer:  "CUDA_GRAPH",
			Op:     "graphBeginCapture",
			Detail: fmt.Sprintf("PID %d: BeginCapture on stream 0x%x with no EndCapture after %v", to.pid, to.stream, to.age.Round(time.Second)),
		})
	}
	for _, sf := range shortFails {
		timeline = append(timeline, ChainEvent{
			Layer:    "CUDA_GRAPH",
			Op:       "graphEndCapture",
			Detail:   fmt.Sprintf("PID %d: capture failed (retCode=%d, duration=%v) on stream 0x%x", sf.pid, sf.retCode, sf.duration.Round(time.Microsecond), sf.stream),
			Duration: sf.duration,
		})
	}

	timeline = append(timeline, ChainEvent{
		Layer:  "CUDA",
		Op:     "cuBLAS",
		Detail: "cuBLAS defers handle creation until first use; if first use is during capture, disallowed operations abort the capture",
	})

	return []CausalChain{{
		ID:       e.chainID("graph-capture-warmup"),
		Severity: "MEDIUM",
		Summary:  "CUDA Graph capture failure — possible cuBLAS lazy initialization",
		RootCause: "cuBLAS defers handle creation until first use. If first use occurs during graph capture, the capture fails because handle creation triggers disallowed operations.",
		Timeline:  timeline,
		Explanation: "CUDA Graph capture failure — possible cuBLAS lazy initialization. " +
			"Add 3+ warmup iterations before cudaStreamBeginCapture to ensure cuBLAS handles are initialized. " +
			"cuBLAS (and cuDNN) lazily create internal handles, memory pools, and kernel plans on first invocation. " +
			"These initialization steps call CUDA APIs (e.g., cudaMalloc, cudaEventCreate) that are disallowed during stream capture, " +
			"causing the capture to fail or produce an invalid graph.",
		Recommendations: []string{
			"Add 3+ warmup iterations before cudaStreamBeginCapture to ensure cuBLAS handles are initialized",
			"Call cublasCreate() and run a dummy GEMM before the first graph capture",
			"For PyTorch: use torch.cuda.make_graphed_callables() which handles warmup automatically",
			"Set CUBLAS_WORKSPACE_CONFIG=:4096:8 to pre-allocate cuBLAS workspaces",
		},
	}}
}

// dedup removes duplicate strings from a slice while preserving order.
func dedup(ss []string) []string {
	seen := make(map[string]bool)
	var result []string
	for _, s := range ss {
		if !seen[s] {
			seen[s] = true
			result = append(result, s)
		}
	}
	return result
}
