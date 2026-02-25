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
	"sync"
	"time"

	"github.com/ingero-io/ingero/pkg/events"
	"github.com/ingero-io/ingero/internal/stats"
)

// DefaultMaxAge is the maximum age of host events kept in the sliding window.
const DefaultMaxAge = 10 * time.Second

// DefaultSchedSwitchThreshold is the minimum number of sched_switch events
// in the window before we emit a correlation. Below this, preemption is normal.
const DefaultSchedSwitchThreshold = 5

// DefaultTailRatio is the p99/p50 ratio that triggers correlation analysis.
// A ratio > 3 means the tail latency is anomalously high.
const DefaultTailRatio = 3.0

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
	ID              string       // unique chain ID (e.g., "chain-001")
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

// Engine performs cross-layer correlation between host events and CUDA stats.
type Engine struct {
	mu         sync.RWMutex
	hostWindow []events.Event
	maxAge     time.Duration
	sysCtx     *SystemContext // latest system context, nil if not available
	chainSeq   int           // sequence counter for chain IDs
}

// New creates a correlation engine with the default 10s sliding window.
// Pass WithMaxAge to override for longer collection windows or historical replay.
func New(opts ...Option) *Engine {
	e := &Engine{
		maxAge: DefaultMaxAge,
	}
	for _, opt := range opts {
		opt(e)
	}
	return e
}

// RecordHost adds a host event to the sliding window.
// Old events beyond maxAge are pruned lazily.
func (e *Engine) RecordHost(evt events.Event) {
	e.mu.Lock()
	defer e.mu.Unlock()

	e.hostWindow = append(e.hostWindow, evt)
	e.prune()
}

// prune removes events older than maxAge from the window.
// When maxAge == 0, pruning is disabled (unlimited window for historical replay).
// Must be called with e.mu held.
func (e *Engine) prune() {
	if e.maxAge == 0 || len(e.hostWindow) == 0 {
		return
	}

	cutoff := time.Now().Add(-e.maxAge)
	i := 0
	for i < len(e.hostWindow) && e.hostWindow[i].Timestamp.Before(cutoff) {
		i++
	}
	if i > 0 {
		e.hostWindow = e.hostWindow[i:]
	}
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
	sysCtx := e.sysCtx
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
		}
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

		chain := e.buildChain(op, tailRatio, sysCtx, schedSwitchCount, totalOffCPU, pageAllocCount, totalAllocBytes)
		if chain != nil {
			chains = append(chains, *chain)
		}
	}

	// OOM always produces a HIGH chain.
	if oomCount > 0 {
		e.chainSeq++
		chains = append(chains, CausalChain{
			ID:       fmt.Sprintf("chain-%03d", e.chainSeq),
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

	return chains
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

	e.chainSeq++
	causeStr := causes[0]
	for i := 1; i < len(causes); i++ {
		causeStr += " + " + causes[i]
	}

	return &CausalChain{
		ID:       fmt.Sprintf("chain-%03d", e.chainSeq),
		Severity: severity,
		Summary: fmt.Sprintf("%s p99=%v (%.1fx p50) — %s",
			op.Op, op.P99, tailRatio, causeStr),
		RootCause:       causeStr,
		Timeline:        timeline,
		Explanation:     buildExplanation(op, tailRatio, causes, sysCtx),
		Recommendations: dedup(recommendations),
	}
}

// buildExplanation generates a human-readable paragraph explaining the chain.
func buildExplanation(op stats.OpStats, tailRatio float64, causes []string, sysCtx *SystemContext) string {
	explanation := fmt.Sprintf("%s tail latency is %.1fx higher than typical (p99=%v vs p50=%v). ", op.Op, tailRatio, op.P99, op.P50)

	if sysCtx != nil && sysCtx.CPUPercent > 90 {
		explanation += fmt.Sprintf("The system CPU is at %.0f%%, indicating heavy contention for CPU resources. ", sysCtx.CPUPercent)
	}
	if sysCtx != nil && sysCtx.SwapUsedMB > 0 {
		explanation += fmt.Sprintf("The system is using %d MB of swap, causing memory access latency spikes. ", sysCtx.SwapUsedMB)
	}

	for _, cause := range causes {
		if cause != "high CPU utilization" && cause != "swap activity" {
			explanation += fmt.Sprintf("Contributing factor: %s. ", cause)
		}
	}

	return explanation
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
		if evt.Source == events.SourceHost {
			corr.RecordHost(evt)
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
