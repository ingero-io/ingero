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

	// NCCLDataPoints is a per-snapshot drained buffer of NCCL collective
	// events captured by ncclprobe between snapshots. Populated by the
	// caller (cli/trace.go) before passing the snapshot to OTLP/Prometheus
	// emitters; nil when --nccl is off. Each element becomes one
	// nccl.collective.duration_ms + nccl.collective.bytes data point.
	NCCLDataPoints []NCCLDataPoint

	// ThrottleReadings carries the latest NVML clock-throttle reason
	// decode for every visible GPU. v0.12.10 (W2-poller). Populated by
	// the cli/trace.go onSnapshot closure from the throttle poller's
	// last-value-wins buffer; nil when nvidia-smi is not on PATH or the
	// poller has not yet completed its first read. Each element becomes
	// four gauge data points (power/thermal/sw/hw active) labelled with
	// gpu.uuid in the OTLP exporter.
	ThrottleReadings []ThrottleReading

	// NCCLProcessReadings carries the latest snapshot of NCCL-loaded
	// processes from the libnccl discovery scanner (v0.14 item A).
	// Each element becomes one gpu.nccl.process_loaded gauge=1 data
	// point with pid/comm/libnccl_path/libnccl_version labels; the
	// length of the slice becomes the gpu.nccl.processes_total gauge.
	// Nil when --libnccl-discovery-interval is 0 or the scanner has
	// not yet completed its first pass.
	NCCLProcessReadings []NCCLProcessReading

	// MemFragReadings carries the latest NVML-poll memory-usage
	// snapshot per GPU (v0.14 item D, W1 baseline). Each element
	// becomes four gauges (used/free/total/fragmentation_estimate)
	// labelled with gpu.uuid. Nil when --memfrag-poll-interval is 0
	// or the poller has not yet completed its first read.
	MemFragReadings []MemFragReading

	// MemFragProcessReadings carries per-process GPU memory usage
	// from `nvidia-smi --query-compute-apps`. Each element becomes
	// one gpu.memory.process.allocated_bytes gauge labelled with
	// gpu.uuid + pid. Nil under the same conditions as
	// MemFragReadings.
	MemFragProcessReadings []MemFragProcessReading

	// MemcpyDirReadings carries per-direction memcpy aggregates
	// (v0.14 item C). Each element becomes one
	// gpu.memcpy.bytes_total counter and one
	// gpu.memcpy.duration_ms gauge labelled with `direction`.
	// Empty when no memcpy events have been seen this run.
	MemcpyDirReadings []MemcpyDirStats

	// NCCLCollectiveCounters carries running counts of NCCL
	// collective events grouped by op_type. Populated by
	// snapshotNCCLCollectiveCounters in cli/nccl_counters.go and
	// emitted by the Prometheus exporter as
	// gpu_nccl_collective_count{op_type} +
	// gpu_nccl_collective_bytes_total{op_type} +
	// gpu_nccl_collective_barrier_events{op_type}. v0.15 F2:
	// per-event NCCLDataPoints are OTLP-only because per-event
	// gauges do not fit Prometheus pull; the running counters here
	// are the pull-friendly equivalent.
	NCCLCollectiveCounters []NCCLCollectiveCounter
	// v0.15 item K: per-cmd memfrag IOCTL counters.
	MemfragIOCTLCounters []MemfragIOCTLCounter
	// v0.15 item L: throttle event-edge counters.
	ThrottleEvents ThrottleEventCounters
	// v0.15 item M: per-PID kernel-launch aggregates.
	KernelLaunches []KernelLaunchSnapshot

	// v0.16.3 inference exporter surface. Populated by the
	// cli/trace.go onSnapshot callback from infer.Engine when the
	// --inference umbrella is engaged; nil/empty otherwise. The OTLP
	// + Prometheus exporters fan these out as ingero.infer.* metrics
	// so operators get the same workload-key view they already get on
	// the UDS socket.
	InferWorkloads []InferWorkloadStats
	InferStats     InferEngineStats
	InferSampler   InferSamplerState
}

// InferWorkloadStats is the per-workload exporter view of one entry
// in the infer.Engine LRU. v0.16.3 surface for ingero.infer.* metric
// emission. Plain types (no time.Duration / no internal/infer types)
// so the export package can read this without an import cycle.
//
// Phase is the v0.16.1 classifier verdict ("prefill" | "decode" |
// "mixed" | "unknown" | "" when classifier disabled); used as a data
// point attribute. Histogram is a frozen view of the per-workload
// step-duration histogram (the same Histogram type as memcpy /
// kernel-launch elsewhere in this package).
type InferWorkloadStats struct {
	CGroupHash   string
	PID          uint32
	StreamHandle uint64
	Phase        string
	// KernelFingerprint is non-zero only when the engine runs with
	// --inference-fingerprint-key; the OTLP / Prometheus exporters
	// emit it as a data point attribute so per-sequence baselines
	// don't collapse into a single series. v0.16.5b.
	KernelFingerprint uint64
	MeanNs            float64
	P95Ns             float64
	Samples           int
	Histogram         HistogramSnapshot
}

// InferEngineStats is the engine-level exporter view (v0.16.3).
// WorkloadsTracked feeds ingero.infer.workloads_tracked; OutliersTotal
// feeds ingero.infer.outlier_total per bucket; ThrottleAtOutlier feeds
// ingero.infer.throttle_active_total per bucket.
//
// KVCacheAllocAgeHist feeds ingero.infer.kvcache.alloc_age_ms - the
// cumulative distribution of
// live alloc ages sampled at decode-phase outliers. HasObservation is
// false when no decode outliers have fired or the KVCacheTracker is
// unconfigured; the exporter still emits the metric (zero count) so
// the series is wired even before the first observation.
type InferEngineStats struct {
	WorkloadsTracked    int
	OutliersTotal       map[string]uint64 // bucket ("1.5x"|"2x"|"3x") -> cumulative
	ThrottleAtOutlier   map[string]uint64 // bucket -> cumulative outliers seen with non-zero throttle
	KVCacheAllocAgeHist HistogramSnapshot
}

// InferSamplerState is the engine's sampler-degradation state for
// ingero.infer.sampler.* metric emission (v0.16.3). Cause is a single
// human-friendly string ("3x:cgroup=<hash>,pid=<n>,phase=<p>") that
// becomes the AttrInferSamplerCause attribute on the gauge. Empty
// when no degradation has fired yet.
type InferSamplerState struct {
	Degraded          bool
	DegradationsTotal uint64
	LastCause         string
}

// ThrottleReading is one decoded NVML clock-throttle sample for one GPU,
// plain-field for OTLP emission. Mirrors internal/nvml.Reading + buckets
// without importing the nvml package (would create an import cycle).
//
// The four `*Active` flags map 1:1 to OTel gauge metrics:
//
//	PowerActive   -> gpu.throttle.power_active
//	ThermalActive -> gpu.throttle.thermal_active
//	SWActive      -> gpu.throttle.sw_active
//	HWActive      -> gpu.throttle.hw_active
//
// Each metric's value is 1 when the flag is true, 0 otherwise.
type ThrottleReading struct {
	UUID          string
	Bitmask       uint64
	PowerActive   bool
	ThermalActive bool
	SWActive      bool
	HWActive      bool
}

// NCCLDataPoint is one captured NCCL collective ready for OTLP emission.
// Plain fields (no time.Duration) so the export package can emit without
// importing ncclprobe (avoids import cycle since ncclprobe needs events).
type NCCLDataPoint struct {
	TimestampUnixNano int64
	OpType            string // "ncclAllReduce", "ncclAllGather", etc.
	CommIDHash        string // 16-hex-char string
	Rank              uint32
	NRanks            uint32
	Datatype          uint32
	ReduceOp          uint32
	DurationMs        float64
	CountBytes        uint64
	ReturnCode        int32

	// IsBarrier flips the OTLP encoder from emitting
	// `nccl.collective.duration_ms` to `nccl.collective.barrier_wait_ms`.
	// v0.12.1 hardening: replaces a stringly-typed "barrier_wait:" prefix
	// on OpType that was vulnerable to in-band sentinel collisions if a
	// future processor injected an op_type containing the prefix.
	IsBarrier bool

	// PeerRank is non-zero only for ncclSend / ncclRecv point-to-point
	// primitives (PARM4 of those calls). Zero for collectives. v0.12.2:
	// enables topology-mapping for pipeline-parallel workloads.
	PeerRank uint32
}

// NCCLProcessReading is one discovered NCCL-loaded process surfaced
// by the libnccl discovery scanner (v0.14 item A). Plain fields so
// the export package emits without importing ncclprobe (avoids an
// import cycle).
//
// Each reading produces one gpu.nccl.process_loaded gauge=1 data
// point with the four-label set (pid, comm, libnccl_path,
// libnccl_version); the slice length feeds gpu.nccl.processes_total.
type NCCLProcessReading struct {
	PID        uint32
	Comm       string
	LibPath    string
	LibVersion string
}

// MemFragReading is one polled NVML memory snapshot per GPU
// (v0.14 item D). The fragmentation estimate is a coarse heuristic
// computed from used/free/total: it is NOT the IOCTL-level memfrag
// tracking that v0.15 W1 ships; the comment in the OTLP encoder
// surfaces this caveat in the metric description.
type MemFragReading struct {
	UUID                  string
	UsedBytes             int64
	FreeBytes             int64
	TotalBytes            int64
	FragmentationEstimate float64 // [0,1]: 0 = unfragmented, 1 = fully fragmented
}

// MemcpyDirStats is one per-direction aggregate of CUDA memcpy
// events (v0.14 item C, v0.15 item C).
//
// BytesTotal is monotonically growing across the agent process
// lifetime; the OTLP exporter emits it as a cumulative Sum metric.
//
// DurationHistogram is the per-event histogram of memcpy duration in
// milliseconds, reset each drain. Replaces the v0.14 per-window
// AverageDurationMs gauge with a real per-event histogram (v0.15
// item B + C). HasObservation=false on direction-with-only-Bytes
// rows (rare; normally every direction has duration too).
//
// EventsInWindow mirrors DurationHistogram.Count and is preserved
// for callers that want a count without inspecting the snapshot.
type MemcpyDirStats struct {
	Direction         string
	BytesTotal        int64
	DurationHistogram HistogramSnapshot
	EventsInWindow    int64
}

// MemfragIOCTLCounter is a per-IOCTL-cmd running tally fed by the
// v0.15 W1 memfrag kprobe (internal/ebpf/memfrag). Cumulative
// across the agent process lifetime; the OTLP exporter emits as a
// Sum metric with cmd as a label.
type MemfragIOCTLCounter struct {
	Cmd   uint32
	Count int64
}

// ThrottleEventCounters carries per-bucket cumulative event counts
// from the v0.15 L edge detector. Mirrors
// nvml.ThrottleEventCounters at the stats-shape layer so the
// exporter doesn't need to import internal/nvml.
type ThrottleEventCounters struct {
	PowerEvents   int64
	ThermalEvents int64
	SWEvents      int64
	HWEvents      int64
}

// KernelLaunchSnapshot is the per-PID aggregate from v0.15 M
// (cuLaunchKernel uprobe). Count is cumulative; histograms are
// per-window (reset each snapshot is the responsibility of the
// caller, the same shape as MemcpyDirStats.DurationHistogram).
type KernelLaunchSnapshot struct {
	PID                 uint32
	Count               int64
	ThreadsPerBlockHist HistogramSnapshot
	GridBlocksHist      HistogramSnapshot
}

// MemFragProcessReading is one per-process GPU-memory data point
// from `nvidia-smi --query-compute-apps`. UUID is the GPU UUID the
// process is using; PID is the host PID.
type MemFragProcessReading struct {
	UUID      string
	PID       uint32
	UsedBytes int64
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

// NCCLCollectiveCounter is a running tally of one NCCL op type's
// collective event count + bytes total since agent process start.
// v0.15 F2 surface for the Prometheus pull exporter; OTLP keeps the
// per-event gauge view via NCCLDataPoints.
//
// BarrierEvents is non-zero only on entries created from
// IsBarrier=true data points; the standard collective entries leave
// it at 0. A given OpType can appear in TWO entries when both
// regular collectives and barrier-wait events have been seen for
// it (regular collectives populate Count + BytesTotal, barriers
// populate BarrierEvents). The Prometheus exporter sums or emits
// independently as appropriate.
type NCCLCollectiveCounter struct {
	OpType        string
	Count         int64 // collective events seen
	BytesTotal    int64 // sum of CountBytes across collective events
	BarrierEvents int64 // barrier-wait correlations
}
