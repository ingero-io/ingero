// Package stats provides rolling latency statistics, time-fraction breakdown,
// periodic spike detection, and anomaly flagging for events from all sources.
package stats

import (
	"fmt"
	"math"
	"sort"
	"sync"
	"time"

	"github.com/ingero-io/ingero/pkg/events"
)

// ---------------------------------------------------------------------------
// Configuration defaults
// ---------------------------------------------------------------------------

const (
	// DefaultWindowSize is the number of latency samples to keep per operation.
	// With 1000 samples and 4 CUDA ops, that's 4000 samples total (~32KB).
	// Sorting 1000 elements for percentile computation takes ~10µs — negligible.
	DefaultWindowSize = 1000

	// DefaultAnomalyThreshold is the multiplier above the median (p50) that
	// triggers an anomaly flag. Median is robust to outliers (unlike mean).
	DefaultAnomalyThreshold = 3.0

	// DefaultSpikeMinCount is the minimum number of anomalies needed before
	// we attempt periodic pattern detection. Below this, we don't have enough
	// data points to detect a pattern.
	DefaultSpikeMinCount = 3

	// DefaultSpikeTolerance is the allowed variance in inter-spike intervals.
	// 0.3 means intervals can vary by ±30% and still be considered periodic.
	// "Every ~200 events" means intervals between 140 and 260 count as matching.
	DefaultSpikeTolerance = 0.3
)

// ---------------------------------------------------------------------------
// Functional Options
// ---------------------------------------------------------------------------

// Option configures a Collector.
type Option func(*Collector)

// WithWindowSize sets the number of latency samples kept per operation.
func WithWindowSize(n int) Option {
	return func(c *Collector) { c.windowSize = n }
}

// WithAnomalyThreshold sets the multiplier above p50 that triggers anomaly flags.
func WithAnomalyThreshold(multiplier float64) Option {
	return func(c *Collector) { c.anomalyThreshold = multiplier }
}

// WithStartTime overrides the wall-clock start time (useful for testing).
func WithStartTime(t time.Time) Option {
	return func(c *Collector) { c.startTime = t }
}

// ---------------------------------------------------------------------------
// Core types
// ---------------------------------------------------------------------------

// opKey uniquely identifies an operation by combining Source and Op code.
// This prevents collisions between CUDA and Host ops that share the same
// numeric Op value (e.g., CUDA op 1 = cudaMalloc, Host op 1 = sched_switch).
type opKey struct {
	Source events.Source
	Op     uint8
}

// Collector aggregates events from all sources and computes rolling statistics.
// Thread-safe via RWMutex: Record() takes write lock, Snapshot() takes read lock.
type Collector struct {
	mu sync.RWMutex

	windows   map[opKey]*opWindow // per-operation stats, keyed by Source+Op
	startTime time.Time           // when collection started (for time-fraction)

	// Aggregate counters
	totalEvents   uint64
	anomalyEvents uint64

	// Configuration
	windowSize       int
	anomalyThreshold float64
}

// opWindow tracks rolling statistics for a single operation type.
// Uses a circular buffer of recent latencies for bounded memory + O(1) insertion.
type opWindow struct {
	samples []time.Duration
	pos     int  // next write position
	full    bool // has the buffer wrapped around at least once?

	// All-time counters (not bounded by window size).
	count    int64
	totalDur time.Duration
	minDur   time.Duration
	maxDur   time.Duration

	// Anomaly tracking.
	// cachedP50 is recomputed each time Snapshot() is called and used by
	// Record() for fast O(1) anomaly checking between snapshots.
	cachedP50    time.Duration
	anomalyCount int64

	// Spike detection: event-count positions at which anomalies occurred.
	// Used to detect periodic patterns like "spike every ~200 events".
	// We cap this at 1000 entries to bound memory.
	spikePositions []int64
}

// maxSpikePositions caps the spike history to prevent unbounded growth.
// 1000 spike positions is enough to detect patterns in long-running traces.
const maxSpikePositions = 1000

// ---------------------------------------------------------------------------
// Constructor
// ---------------------------------------------------------------------------

// New creates a stats Collector with the given options.
func New(opts ...Option) *Collector {
	c := &Collector{
		windows:          make(map[opKey]*opWindow),
		startTime:        time.Now(),
		windowSize:       DefaultWindowSize,
		anomalyThreshold: DefaultAnomalyThreshold,
	}
	for _, opt := range opts {
		opt(c)
	}
	return c
}

// ---------------------------------------------------------------------------
// Core methods
// ---------------------------------------------------------------------------

// Record adds an event to the collector. Thread-safe, O(1) amortized.
func (c *Collector) Record(evt events.Event) {
	c.mu.Lock()
	defer c.mu.Unlock()

	c.totalEvents++

	key := opKey{Source: evt.Source, Op: evt.Op}
	w, ok := c.windows[key]
	if !ok {
		w = &opWindow{
			samples: make([]time.Duration, c.windowSize),
			minDur:  evt.Duration,
			maxDur:  evt.Duration,
		}
		c.windows[key] = w
	}

	// Write to circular buffer.
	w.samples[w.pos] = evt.Duration
	w.pos = (w.pos + 1) % c.windowSize
	if w.pos == 0 {
		w.full = true
	}

	// Update all-time counters.
	w.count++
	w.totalDur += evt.Duration
	if evt.Duration < w.minDur {
		w.minDur = evt.Duration
	}
	if evt.Duration > w.maxDur {
		w.maxDur = evt.Duration
	}

	// Anomaly detection using cached p50 from last Snapshot() (~1s stale).
	// Keeps Record() O(1) instead of O(n log n) from recomputing percentile.
	if w.cachedP50 > 0 && w.count > 10 {
		threshold := time.Duration(float64(w.cachedP50) * c.anomalyThreshold)
		if evt.Duration > threshold {
			w.anomalyCount++
			c.anomalyEvents++

			// Record spike position for periodic pattern detection.
			if len(w.spikePositions) < maxSpikePositions {
				w.spikePositions = append(w.spikePositions, w.count)
			}
		}
	}
}

// IsAnomaly checks if an event's duration is anomalous for its operation.
// Returns false if there isn't enough data for a meaningful baseline.
//
// This is called per-event in JSON mode to add an "anomaly" field.
// Uses the cached p50 from the last Snapshot() — O(1).
func (c *Collector) IsAnomaly(evt events.Event) bool {
	c.mu.RLock()
	defer c.mu.RUnlock()

	key := opKey{Source: evt.Source, Op: evt.Op}
	w, ok := c.windows[key]
	if !ok || w.count <= 10 || w.cachedP50 == 0 {
		return false
	}

	threshold := time.Duration(float64(w.cachedP50) * c.anomalyThreshold)
	return evt.Duration > threshold
}

// Snapshot returns a point-in-time copy of all statistics.
// Also updates the cached p50 used by Record() for anomaly detection.
func (c *Collector) Snapshot() *Snapshot {
	c.mu.Lock() // Write lock because we update cachedP50
	defer c.mu.Unlock()

	wallClock := time.Since(c.startTime)

	snap := &Snapshot{
		WallClock:     wallClock,
		TotalEvents:   c.totalEvents,
		AnomalyEvents: c.anomalyEvents,
	}

	for key, w := range c.windows {
		p50 := computePercentile(w.samples, w.pos, w.full, 0.50)
		p95 := computePercentile(w.samples, w.pos, w.full, 0.95)
		p99 := computePercentile(w.samples, w.pos, w.full, 0.99)

		// Update cached p50 for anomaly detection in Record().
		w.cachedP50 = p50

		var timeFrac float64
		if wallClock > 0 {
			timeFrac = float64(w.totalDur) / float64(wallClock)
		}

		// Resolve op name via Event.OpName() which dispatches by Source.
		opName := events.Event{Source: key.Source, Op: key.Op}.OpName()

		snap.Ops = append(snap.Ops, OpStats{
			Op:           opName,
			OpCode:       key.Op,
			Source:        key.Source,
			Count:        w.count,
			P50:          p50,
			P95:          p95,
			P99:          p99,
			Min:          w.minDur,
			Max:          w.maxDur,
			TimeFraction: timeFrac,
			TotalDur:     w.totalDur,
			AnomalyCount: w.anomalyCount,
			SpikePattern: detectSpikePattern(w.spikePositions),
		})
	}

	// Sort by time-fraction descending — the biggest time consumer at top.
	// This answers the question: "What is my GPU spending the most time on?"
	sort.Slice(snap.Ops, func(i, j int) bool {
		return snap.Ops[i].TimeFraction > snap.Ops[j].TimeFraction
	})

	return snap
}

// TotalEvents returns the total number of events recorded (thread-safe).
func (c *Collector) TotalEvents() uint64 {
	c.mu.RLock()
	defer c.mu.RUnlock()
	return c.totalEvents
}

// ---------------------------------------------------------------------------
// Snapshot types (read-only copies for display/serialization)
// ---------------------------------------------------------------------------

// Snapshot is a point-in-time copy of all statistics.
// It's safe to use without any locking — it's an independent copy.
type Snapshot struct {
	WallClock     time.Duration // time since collection started
	TotalEvents   uint64        // total events recorded
	AnomalyEvents uint64        // total events flagged as anomalous
	Ops           []OpStats     // per-operation stats, sorted by TimeFraction desc

	// System is the latest system context (CPU/mem/load) at snapshot time.
	// Nil if sysinfo collector is not running.
	System *SystemSnapshot

	// TraceDB is the latest trace-DB size + prune counters at snapshot
	// time. Nil when the snapshot source has no access to a Store (e.g.
	// non-trace commands, or the minimal trace setup before the store
	// is wired in). Populated via Snapshot.WithTraceDB by the caller to
	// avoid an import cycle stats -> store.
	TraceDB *TraceDBSnapshot

	// RingbufOverflows is the cumulative eBPF-ringbuf + in-process channel
	// drop count summed across every attached tracer (cuda, host, driver,
	// io, tcp, net, host-critical). A non-zero, fast-climbing value means
	// the kernel is producing events faster than userspace drains them —
	// increase ring sizes or reduce instrumentation scope. Zero when no
	// tracer is attached.
	RingbufOverflows uint64
}

// TraceDBSnapshot mirrors store.Stats without importing the store
// package (stats is imported by store transitively through export).
// The caller (watch.go) copies store.ReadStats() into this struct
// before calling prom.UpdateSnapshot.
type TraceDBSnapshot struct {
	DiskBytes  int64
	PrunedRows uint64
}

// SystemSnapshot holds point-in-time CPU/memory/load metrics.
// Embedded here to avoid an import cycle (stats cannot import sysinfo).
// Populated by the caller (watch.go) from sysinfo.Collector.Snapshot().
type SystemSnapshot struct {
	CPUPercent float64
	MemUsedPct float64
	MemAvailMB int64
	MemTotalMB int64
	SwapUsedMB int64
	LoadAvg1   float64
	LoadAvg5   float64
	PageFaults int64
}

// OpStats contains statistics for a single operation type (CUDA or Host).
type OpStats struct {
	Op     string        // human-readable name (e.g., "cudaMalloc", "sched_switch")
	OpCode uint8         // raw op code
	Source events.Source // which layer produced this op (cuda, host, nvidia)

	Count int64 // total events for this operation

	// Percentile latencies from the rolling window.
	P50 time.Duration
	P95 time.Duration
	P99 time.Duration
	Min time.Duration
	Max time.Duration

	// TimeFraction is totalDur / wallClock. May sum to > 1.0 (concurrent streams)
	// or < 1.0 (idle time).
	TimeFraction float64
	TotalDur     time.Duration

	// Anomaly tracking.
	AnomalyCount int64
	SpikePattern string // e.g., "every ~200 events" or "" if no pattern
}

// ---------------------------------------------------------------------------
// Percentile computation
// ---------------------------------------------------------------------------

// computePercentile computes a percentile from a circular buffer using
// nearest-rank method. Copies + sorts the active portion (~10µs for n=1000).
func computePercentile(samples []time.Duration, pos int, full bool, pct float64) time.Duration {
	n := len(samples)
	if full {
		// Buffer has wrapped — all elements are valid.
	} else {
		// Buffer hasn't wrapped — only samples[0:pos] are valid.
		n = pos
	}

	if n == 0 {
		return 0
	}

	// Copy the active portion so we don't corrupt the circular buffer order.
	sorted := make([]time.Duration, n)
	if full {
		copy(sorted, samples)
	} else {
		copy(sorted, samples[:pos])
	}

	sort.Slice(sorted, func(i, j int) bool {
		return sorted[i] < sorted[j]
	})

	// Nearest-rank percentile: index = ceil(pct * n) - 1
	//
	// Examples with n=100:
	//   p50: ceil(0.50 * 100) - 1 = 49  (50th element, 0-indexed)
	//   p95: ceil(0.95 * 100) - 1 = 94  (95th element)
	//   p99: ceil(0.99 * 100) - 1 = 98  (99th element)
	idx := int(math.Ceil(pct*float64(n))) - 1
	if idx < 0 {
		idx = 0
	}
	if idx >= n {
		idx = n - 1
	}

	return sorted[idx]
}

// ---------------------------------------------------------------------------
// Periodic spike detection
// ---------------------------------------------------------------------------

// detectSpikePattern checks if anomaly positions show periodic spacing.
// Uses median interval + 60% consistency threshold within ±30% tolerance.
func detectSpikePattern(positions []int64) string {
	if len(positions) < DefaultSpikeMinCount {
		return ""
	}

	// Compute intervals between consecutive spikes.
	intervals := make([]int64, len(positions)-1)
	for i := 1; i < len(positions); i++ {
		intervals[i-1] = positions[i] - positions[i-1]
	}

	if len(intervals) == 0 {
		return ""
	}

	// Find median interval.
	sorted := make([]int64, len(intervals))
	copy(sorted, intervals)
	sort.Slice(sorted, func(i, j int) bool { return sorted[i] < sorted[j] })
	medianInterval := sorted[len(sorted)/2]

	if medianInterval <= 0 {
		return ""
	}

	// Check if intervals are consistent (within tolerance of median).
	tolerance := float64(medianInterval) * DefaultSpikeTolerance
	consistent := 0
	for _, iv := range intervals {
		if math.Abs(float64(iv)-float64(medianInterval)) <= tolerance {
			consistent++
		}
	}

	// Require >60% of intervals to be consistent for a pattern.
	if float64(consistent)/float64(len(intervals)) >= 0.6 {
		return fmt.Sprintf("every ~%d events", medianInterval)
	}

	return ""
}
