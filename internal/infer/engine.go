package infer

import (
	"crypto/rand"
	"encoding/hex"
	"fmt"
	"log/slog"
	"sync"
	"time"

	"github.com/ingero-io/ingero/internal/correlate"
	"github.com/ingero-io/ingero/internal/infer/kvcache"
	"github.com/ingero-io/ingero/internal/sampling"
	"github.com/ingero-io/ingero/internal/stats"
	"github.com/ingero-io/ingero/pkg/events"
)

// OutlierBucket labels how far a step exceeded its workload's
// baseline p95. Buckets are mutually exclusive — a step that crosses
// 3x is reported as "3x" only, not also as "1.5x" and "2x". SLO-style
// "exceeded 1.5x or higher" math happens at PromQL/Grafana time over
// the cumulative counter sums.
type OutlierBucket string

const (
	BucketNone   OutlierBucket = ""
	Bucket1_5x   OutlierBucket = "1.5x"
	Bucket2x     OutlierBucket = "2x"
	Bucket3x     OutlierBucket = "3x"
)

// Defaults used when Config fields are zero.
const (
	defaultWarmupSamples         = 30
	defaultOutlierThresholdRatio = 3.0
	defaultMaxWorkloads          = 1024
	defaultSeverityTTL           = 30 * time.Second
	defaultLogMinInterval        = 60 * time.Second
	defaultMaxStepDuration       = 60 * time.Second
	defaultOutlierQueueCap       = 4096
	defaultKVCacheTopN           = 5
)

// Config tunes Engine behavior. Zero values resolve to the defaults
// above so callers can pass an empty Config and get a working engine.
type Config struct {
	// WarmupSamples is the number of healthy steps a workload must
	// observe before its baseline is used for classification. Below
	// the threshold, every Update folds in raw — no outliers fired.
	WarmupSamples int

	// OutlierThresholdRatio is the multiplier applied to baseline p95
	// for the LARGEST bucket. Smaller buckets (1.5x, 2x) are fixed at
	// their literal ratios; this knob only widens or narrows the
	// outermost bucket boundary.
	OutlierThresholdRatio float64

	// PauseOnSeverity is the lowest severity (HIGH | MEDIUM | LOW)
	// at which baseline updates are paused for a PID. Empty disables
	// the gate entirely. Default HIGH.
	PauseOnSeverity string

	// MaxWorkloads bounds the per-workload map. Default 1024.
	MaxWorkloads int

	// SeverityTTL is how long a chain-severity entry stays active
	// after the producer last reported it. Default 30s, matching the
	// sampler cooldown.
	SeverityTTL time.Duration

	// LogMinInterval rate-limits per-workload INFO log lines on
	// outlier emission. Default 60s.
	LogMinInterval time.Duration

	// SamplerDegradeOn is the smallest bucket that triggers
	// sampler.SetDegraded(true) on the attached store sampler. Empty
	// disables the feedback loop entirely. Allowed values: "1.5x",
	// "2x", "3x". Default "3x".
	SamplerDegradeOn OutlierBucket

	// Sampler is the attached store sampler. Nil-safe — when nil,
	// the SamplerDegradeOn knob is silently ignored.
	Sampler *sampling.Sampler

	// MaxStepDuration is the upper bound on a sane step duration.
	// Anything longer is treated as a process restart, clock skew, or
	// idle-loop sync and ignored. Default 60s.
	MaxStepDuration time.Duration

	// OutlierQueueCap bounds the queue drained on each snapshot tick.
	// On overflow, oldest is dropped — same shape as the NCCL drain
	// buffer in internal/cli/trace.go. Default 4096.
	OutlierQueueCap int

	// PhaseClassifierEnabled toggles phase-aware baselines. When
	// true (default), each step's observable signals are read at
	// sync-event time and ClassifyPhase decides which sub-baseline
	// (per (cgroup, pid, stream, phase)) to update. When false,
	// every step uses Phase="" and the engine reverts to v0.16.0
	// single-baseline-per-stream behavior.
	PhaseClassifierEnabled bool

	// FingerprintKeyEnabled adds a per-step KernelFingerprint as the
	// fifth WorkloadKey dimension (v0.16.5b). Off by default: most
	// inference deployments serve a single model per (pid, stream)
	// so the extra dimension just inflates the LRU. Engage it when
	// the same process serves multiple models or model versions on
	// the same stream and you want independent baselines per
	// kernel-launch sequence.
	FingerprintKeyEnabled bool

	// KVCacheTracker is the per-PID allocation tracker consulted on
	// decode-phase outliers to surface KV-cache age context.
	// Nil-safe: when unset (operator did not engage the feature, or
	// memtrack-style allocations weren't visible), the engine simply
	// omits the alloc-age fields from outlier events.
	KVCacheTracker *kvcache.Tracker

	// KVCacheTopN is the number of oldest live allocations the
	// engine attaches to a decode-phase outlier event. Zero or
	// negative resolves to defaultKVCacheTopN. The cap exists to
	// keep the UDS envelope and OTLP attribute sets bounded - a
	// fragmented KV cache might have thousands of live blocks; we
	// only need the top of the age distribution to identify the
	// stale-cache pattern.
	KVCacheTopN int

	// PhaseConfig holds the classifier thresholds. Empty fields
	// resolve to LLM-tuned defaults via PhaseConfig.Resolved.
	PhaseConfig PhaseConfig

	// ObservableTTL bounds memory growth in the observable counter
	// store. Entries idle longer than this duration are pruned on
	// each Engine.Stats() call. Default 5 minutes.
	ObservableTTL time.Duration
}

// resolved fills in defaults; called once at New.
func (c Config) resolved() Config {
	if c.WarmupSamples <= 0 {
		c.WarmupSamples = defaultWarmupSamples
	}
	if c.OutlierThresholdRatio <= 0 {
		c.OutlierThresholdRatio = defaultOutlierThresholdRatio
	}
	if c.MaxWorkloads <= 0 {
		c.MaxWorkloads = defaultMaxWorkloads
	}
	if c.SeverityTTL <= 0 {
		c.SeverityTTL = defaultSeverityTTL
	}
	if c.LogMinInterval <= 0 {
		c.LogMinInterval = defaultLogMinInterval
	}
	if c.MaxStepDuration <= 0 {
		c.MaxStepDuration = defaultMaxStepDuration
	}
	if c.OutlierQueueCap <= 0 {
		c.OutlierQueueCap = defaultOutlierQueueCap
	}
	if c.SamplerDegradeOn == BucketNone {
		c.SamplerDegradeOn = Bucket3x
	}
	if c.ObservableTTL <= 0 {
		c.ObservableTTL = 5 * time.Minute
	}
	if c.KVCacheTopN <= 0 {
		c.KVCacheTopN = defaultKVCacheTopN
	}
	c.PhaseConfig = c.PhaseConfig.Resolved()
	return c
}

// OutlierEvent is one classified step that exceeded its baseline.
// Drained from Engine on each snapshot tick and emitted to OTLP +
// Prometheus + (when --remediate is on) the UDS socket.
//
// MemfragEvents and ThrottleReasons are v0.16.3 contextual fields:
// MemfragEvents is the count of NVIDIA memfrag IOCTL events observed
// during the step (KV cache pressure indicator); ThrottleReasons is
// the OR-fold of NVML clock-throttle bitmaps observed during the step
// (non-zero means a thermal/power slowdown coincided with the step).
// MinSMClockMHz is reserved for a future v0.16.x extension that pulls
// SM clock from the existing throttle poller; populated as 0 today.
type OutlierEvent struct {
	Key            WorkloadKey
	StepDurationNs int64
	BaselineP95Ns  int64
	BaselineMeanNs int64
	Bucket         OutlierBucket
	At             time.Time
	EventID        string // UUIDv4-shaped, opaque, for cross-channel correlation

	MemfragEvents    uint32
	ThrottleReasons  uint64
	MinSMClockMHz    uint32

	// KVCacheTopAllocAgesMs is the per-decode-outlier alloc-age
	// context. Populated only when phase=decode AND a KVCacheTracker
	// is configured AND at least one live allocation exists. Sorted
	// oldest-first so consumers can read [0] for the "stalest cache
	// block" age. Unit is milliseconds.
	KVCacheTopAllocAgesMs []uint64
}

// SamplerDegradedEvent fires when the engine flips the store sampler
// from healthy to degraded (admit 100%) on a fresh outlier. v0.16.3
// adds this as a sibling to OutlierEvent so operators see the cause
// + cooldown end on the UDS socket and as OTLP metrics, instead of
// only learning "the sampler degraded again" by inspecting per-event
// JSON. CooldownEnd is when the sampler will return to healthy
// admission absent another trigger.
type SamplerDegradedEvent struct {
	Key         WorkloadKey
	Bucket      OutlierBucket
	At          time.Time
	CooldownEnd time.Time
	Cause       string // human-friendly summary, also used as AttrInferSamplerCause
}

// EngineStats is a snapshot of cumulative engine telemetry. Returned
// from Stats(); callers (the OTLP exporter) emit it as gauges.
type EngineStats struct {
	WorkloadsTracked int
	Evictions        uint64
	OutliersTotal    map[OutlierBucket]uint64
	QueueDropped     uint64
	// PhaseDistribution counts how many steps have been classified
	// into each phase. Cumulative across the engine's lifetime.
	// Useful for verifying the phase classifier is producing a
	// reasonable distribution (e.g. for vLLM serving, expect
	// roughly 1 prefill per N decodes).
	PhaseDistribution map[Phase]uint64

	// v0.16.3: sampler-degradation state surface.
	// SamplerDegraded is true while the store sampler is in degraded
	// (100% admit) state; the field flips back to false once the
	// configured cooldown elapses. SamplerDegradationsTotal counts
	// every flip-to-degraded transition since engine start (lossless
	// edge counter so back-to-back degrades are visible). LastCause
	// is a human-friendly summary of the most recent flip
	// ("3x:cgroup=<hash>,pid=<n>,phase=<p>"). SamplerDegradedUntil is
	// the wall-clock time the cooldown ends; zero when no degradation
	// has fired yet.
	SamplerDegraded         bool
	SamplerDegradationsTotal uint64
	LastDegradationCause    string
	SamplerDegradedUntil    time.Time

	// v0.16.3: ThrottleAtOutlier is the cumulative count of outliers
	// whose ThrottleReasons was non-zero, broken down by bucket. Lets
	// dashboards distinguish "outliers caused by thermal slowdown"
	// from background outliers without joining gauge time series.
	ThrottleAtOutlier map[OutlierBucket]uint64
}

// Engine is the per-workload step-duration baseline + outlier
// classifier. Concurrent-safe; producers (the sync-event hot path,
// the snapshot loop) and consumers (the snapshot drain) share a
// single mutex.
//
// Lifecycle: created once in cli.traceRunE when --inference is set;
// fed events via OnSyncEvent and chains via OnChainSnapshot; drained
// once per snapshot tick via Drain.
type Engine struct {
	cfg Config
	log *slog.Logger

	mu       sync.Mutex
	wmap     *workloadMap
	severity *severityGate

	// lastSync is keyed by the observable tuple (cgroup, pid,
	// stream) WITHOUT phase — we need the step duration before we
	// can classify phase, so we look up lastSync by the un-phased
	// key, compute step, then build the full WorkloadKey for the
	// baseliner lookup.
	lastSync map[observableKey]time.Time

	// observables holds the kernel-launch / memcpy / NCCL counters
	// per (cgroup, pid, stream), accumulated between consecutive
	// syncs and read+reset at each sync. Phase classifier feeds on
	// the snapshot returned by ResetAndRead.
	observables *stepObservables

	queue []OutlierEvent

	// per-workload INFO log rate-limit. Mirrors the maybeLogAnomaly
	// pattern in internal/health/loop.go:506-521. Keyed by the
	// post-classification WorkloadKey so rate-limiting is per-phase.
	lastLogAt map[WorkloadKey]time.Time

	// Cumulative outlier counters per bucket. Engine-owned; emitter
	// reads via Stats() and projects onto the OTLP cumulative counter.
	outliers map[OutlierBucket]uint64

	// phaseCounts tracks the cumulative classification distribution
	// for telemetry. Lock-protected by mu (updated on the sync
	// hot path).
	phaseCounts map[Phase]uint64

	// Cumulative count of events dropped because the queue was full
	// when the producer tried to append. Surfaced via Stats().
	queueDropped uint64

	// pauseRank is the cached rank threshold parsed once from cfg so
	// the hot path doesn't re-parse a string per sync event.
	pauseRank severityRank

	// v0.16.3 sampler-degradation observability state. Mirrors the
	// fields documented on EngineStats; lock-protected by mu.
	samplerDegradedUntil    time.Time
	samplerDegradationsCnt  uint64
	samplerLastCause        string
	throttleAtOutlier       map[OutlierBucket]uint64

	// KV-cache alloc-age histogram. Engine-level cumulative;
	// observations folded in on each decode-phase outlier. The
	// stats.Histogram has its own internal lock so we don't need to
	// hold e.mu when calling Observe.
	kvCacheAllocAgeHist *stats.Histogram

	// v0.16.3 sampler-degraded event queue. Drained by Engine.DrainSampler
	// at the same cadence as DrainOutliers. Bounded by samplerQueueCap
	// so a flapping degrade loop does not grow unbounded.
	samplerQueue []SamplerDegradedEvent

	// v0.16.3 throttle reader hook. Set by the agent (cli/trace.go)
	// after Engine.New so OnSyncEvent can read the latest aggregated
	// throttle bitmap when an outlier fires. Nil-safe: when unset
	// (tests, or operators without the throttle poller running) the
	// engine falls back to the per-step OR-fold from observables.
	throttleReader func() uint64
}

// samplerCooldownDuration is the duration the sampler stays in
// "degraded" state after a triggering outlier. Mirrors
// sampling.DefaultCooldownDuration so an operator who never tunes the
// sampler config sees the same value here as on the sampler itself.
// Used only by the observability fields (samplerDegradedUntil); the
// underlying sampling.Sampler owns its own decay clock.
const samplerCooldownDuration = 30 * time.Second

// samplerQueueCap bounds the in-engine queue of SamplerDegradedEvent
// drained on each snapshot tick. Mirrors OutlierQueueCap but smaller
// because sampler flips are rare events (one per cooldown window in
// the worst case).
const samplerQueueCap = 256

// New constructs an Engine with cfg's defaults filled in. Returns
// nil if cfg.SamplerDegradeOn is set to a value that's not one of
// the allowed bucket strings — caller should validate at flag-parse
// time, but we re-check here for defense-in-depth.
func New(cfg Config, log *slog.Logger) *Engine {
	if log == nil {
		log = slog.Default()
	}
	cfg = cfg.resolved()
	switch cfg.SamplerDegradeOn {
	case BucketNone, Bucket1_5x, Bucket2x, Bucket3x:
		// valid
	default:
		log.Warn("infer: unknown SamplerDegradeOn, falling back to 3x",
			"value", string(cfg.SamplerDegradeOn))
		cfg.SamplerDegradeOn = Bucket3x
	}
	return &Engine{
		cfg:                 cfg,
		log:                 log,
		wmap:                newWorkloadMap(cfg.MaxWorkloads),
		severity:            newSeverityGate(cfg.SeverityTTL),
		lastSync:            make(map[observableKey]time.Time, cfg.MaxWorkloads),
		observables:         newStepObservables(),
		lastLogAt:           make(map[WorkloadKey]time.Time),
		outliers:            make(map[OutlierBucket]uint64, 4),
		phaseCounts:         make(map[Phase]uint64, 4),
		throttleAtOutlier:   make(map[OutlierBucket]uint64, 4),
		kvCacheAllocAgeHist: stats.NewHistogram(stats.DefaultInferKVCacheAllocAgeBoundsMs),
		pauseRank:           parseSeverity(cfg.PauseOnSeverity),
	}
}

// SetThrottleReader installs a callback the engine consults when an
// outlier fires, to pull the latest aggregated NVML throttle-reasons
// bitmap. The agent (cli/trace.go) sets this after constructing the
// engine; nil-safe both before and after - when unset, OutlierEvent
// falls back to the per-step OR-fold accumulated via RecordThrottle
// (which today is empty because the agent doesn't yet thread per-PID
// throttle into observables).
//
// Concurrency: callers must call this once at startup before the
// event hot path goes live. The function is not lock-protected.
func (e *Engine) SetThrottleReader(fn func() uint64) {
	e.throttleReader = fn
}

// OnLaunchEvent records a cudaLaunchKernel event between syncs.
//
// v0.16.5a: the BPF probe now captures the cudaStream_t arg into
// evt.Args[1] (was 0 on v0.16.4 and earlier). Observable counters
// bucket by the actual (cgroup, pid, stream) tuple - apples-to-apples
// with the sync event's lookup. Pre-v0.16.5a BPF builds left Args[1]
// at 0; that maps to the same single (pid, 0) bucket the v0.16.4
// engine used, so an old probe + new agent degrade gracefully to the
// v0.16.4 behavior without misattribution.
//
// Hot path; no allocations beyond map insert on first event for a
// new key. No-op when the engine's phase classifier is disabled.
func (e *Engine) OnLaunchEvent(evt events.Event, cgroupHash string, kernelDuration time.Duration) {
	if !e.cfg.PhaseClassifierEnabled {
		return
	}
	if evt.Source != events.SourceCUDA {
		return
	}
	switch events.CUDAOp(evt.Op) {
	case events.CUDALaunchKernel:
		// allowed
	default:
		return
	}
	// Args[0] is the kernel function pointer (per cuda_trace.bpf.c).
	// Feeds the v0.16.5b KernelFingerprint fold inside AddLaunch.
	e.observables.AddLaunch(observableKey{
		CGroupHash:   cgroupHash,
		PID:          evt.PID,
		StreamHandle: evt.Args[1],
	}, evt.Args[0], kernelDuration, evt.Timestamp)
}

// OnMemcpyEvent records a cudaMemcpy / cudaMemcpyAsync event. bytes
// is taken from the BPF event's Args[0] (per cuda_trace.bpf.c).
//
// Memcpy still aggregates at PID level (StreamHandle=0) because the
// BPF probe layout has only two arg slots and they're already used
// for byte count + direction; capturing the stream too needs a
// re-shape that's deferred. In practice, memcpy_bytes is a much
// weaker phase signal than launch_count anyway, so the per-stream
// gap here barely affects classification accuracy.
func (e *Engine) OnMemcpyEvent(evt events.Event, cgroupHash string, bytes int64) {
	if !e.cfg.PhaseClassifierEnabled {
		return
	}
	if evt.Source != events.SourceCUDA {
		return
	}
	switch events.CUDAOp(evt.Op) {
	case events.CUDAMemcpy, events.CUDAMemcpyAsync:
		// allowed
	default:
		return
	}
	e.observables.AddMemcpy(observableKey{
		CGroupHash:   cgroupHash,
		PID:          evt.PID,
		StreamHandle: 0, // PID-level aggregation; memcpy stream not yet captured
	}, bytes, evt.Timestamp)
}

// OnNCCLEvent records participation in an NCCL collective during
// a step. The classifier treats any non-zero NCCL count as a strong
// prefill signal (rule 1).
//
// v0.16.5a: NCCL events already carry stream_handle (the ncclprobe
// has captured it since v0.12). The engine now buckets per-stream so
// a tensor-parallel allreduce on stream X correctly classifies the
// X-stream sync as PhasePrefill without polluting the Y-stream
// baseline.
func (e *Engine) OnNCCLEvent(pid uint32, cgroupHash string, streamHandle uint64, at time.Time) {
	if !e.cfg.PhaseClassifierEnabled {
		return
	}
	e.observables.AddNCCL(observableKey{
		CGroupHash:   cgroupHash,
		PID:          pid,
		StreamHandle: streamHandle,
	}, at)
}

// OnMallocEvent records a successful cudaMalloc / cudaMallocManaged.
// Feeds the KV-cache lineage tracker so the engine can attach top-N
// alloc ages to decode-phase outlier events. Nil-safe: no-op when
// the operator did not configure a tracker.
func (e *Engine) OnMallocEvent(pid uint32, devPtr, size uint64, at time.Time) {
	if e.cfg.KVCacheTracker == nil {
		return
	}
	e.cfg.KVCacheTracker.OnMalloc(pid, devPtr, size, at)
}

// OnFreeEvent records a cudaFree. Nil-safe.
func (e *Engine) OnFreeEvent(pid uint32, devPtr uint64, at time.Time) {
	if e.cfg.KVCacheTracker == nil {
		return
	}
	e.cfg.KVCacheTracker.OnFree(pid, devPtr, at)
}

// OnProcessExit clears the KV-cache lineage state for a PID. Wired
// to host_trace's sched_process_exit so dead processes don't leak
// allocation tracking memory across long agent runs.
func (e *Engine) OnProcessExit(pid uint32) {
	if e.cfg.KVCacheTracker == nil {
		return
	}
	e.cfg.KVCacheTracker.OnProcessExit(pid)
}

// OnMemfragEvent records one NVIDIA closed-driver IOCTL event for
// the given PID (typically a KV-cache eviction or fragmenting
// allocation under VRAM pressure). Bumping the per-step memfrag
// count feeds the v0.16.3 phase rule that classifies memfrag-pressure
// steps with low launch density as decode (KV-cache pressure is
// decode-shape).
//
// Wired from cli/trace.go's memfrag tracer ringbuf consumer when
// --inference is engaged AND the experimental memfrag kprobe is
// loaded. No-op when the phase classifier is disabled.
func (e *Engine) OnMemfragEvent(pid uint32, cgroupHash string, at time.Time) {
	if !e.cfg.PhaseClassifierEnabled {
		return
	}
	e.observables.AddMemfrag(observableKey{
		CGroupHash:   cgroupHash,
		PID:          pid,
		StreamHandle: 0, // PID-level aggregation; see OnLaunchEvent
	}, at)
}

// OnSyncEvent processes one cudaStreamSynchronize / cudaDeviceSync /
// driver ctxSync event. cgroupHash is the resolved cgroup_path_hash
// for the event's PID (caller resolves it via the existing PodCache
// or cgroup cache). Empty cgroupHash is allowed; it produces a
// "unattributable" workload bucket but the baseline still tracks.
//
// The hot path:
//  1. Lookup last sync for the (cgroup, pid, stream) observable
//     tuple; compute step duration.
//  2. ResetAndRead the observable counters accumulated since the
//     prior sync.
//  3. ClassifyPhase from the observables (when enabled). Build the
//     full WorkloadKey including phase.
//  4. Gate on severity. Look up the per-(workload, phase) baseliner.
//  5. Update baseline, or classify outlier, or both.
func (e *Engine) OnSyncEvent(evt events.Event, cgroupHash string) {
	if !isSyncEvent(evt) {
		return
	}
	// streamKey carries the actual sync stream pointer for the per-
	// (cgroup, pid, stream) lastSync map and the eventual WorkloadKey.
	// Keeps separate baselines for separate streams (vLLM/TGI prefill
	// vs decode streams stay distinct).
	streamKey := observableKey{
		CGroupHash:   cgroupHash,
		PID:          evt.PID,
		StreamHandle: evt.Args[0],
	}
	// pidKey is the PID-level fallback bucket. v0.16.5a moved
	// launches and NCCL onto streamKey (BPF now captures the stream
	// for cudaLaunchKernel; ncclprobe has carried the stream since
	// v0.12). Memcpy and memfrag still bucket at PID level - the
	// memcpy BPF arg slots are full and memfrag is a process-wide
	// signal by nature - so this second bucket is read in parallel
	// and its counts merged into the per-step observable view.
	pidKey := observableKey{
		CGroupHash:   cgroupHash,
		PID:          evt.PID,
		StreamHandle: 0,
	}
	now := evt.Timestamp

	e.mu.Lock()
	prev, hadPrev := e.lastSync[streamKey]
	e.lastSync[streamKey] = now
	e.mu.Unlock()

	// Always reset observables on a sync, even on the first sync —
	// otherwise launch/memcpy/NCCL events that arrived before the
	// first sync would leak into the second step's classification.
	// streamObs carries the per-stream counters (launches, NCCL);
	// pidObs carries the PID-level remainder (memcpy, memfrag,
	// throttle). When an old BPF leaves stream=0 on launches, the two
	// buckets coincide and the classifier sees v0.16.4-equivalent
	// inputs without misattribution.
	streamObs := e.observables.ResetAndRead(streamKey, now)
	var obs observableCounters
	if streamKey != pidKey {
		pidObs := e.observables.ResetAndRead(pidKey, now)
		obs = observableCounters{
			LaunchCount:        streamObs.LaunchCount,
			TotalKernelNs:      streamObs.TotalKernelNs,
			NCCLCount:          streamObs.NCCLCount,
			MemcpyBytes:        pidObs.MemcpyBytes,
			MemfragCount:       pidObs.MemfragCount,
			MaxThrottleReasons: streamObs.MaxThrottleReasons | pidObs.MaxThrottleReasons,
			// v0.16.5b: fingerprint is per-stream (kernel launches
			// belong to a stream), so it travels with streamObs.
			KernelFingerprint: streamObs.KernelFingerprint,
		}
	} else {
		// stream==0 sentinel (default stream OR pre-v0.16.5a BPF):
		// pidObs and streamObs are the same bucket - reading twice
		// would double-reset and zero out the counters before they
		// were used. Fall back to the single-bucket read.
		obs = streamObs
	}

	if !hadPrev {
		return
	}
	if now.Before(prev) {
		// Clock skew, process restart, or out-of-order delivery.
		// Reset baseline, do not emit.
		return
	}
	step := now.Sub(prev)
	if step <= 0 || step > e.cfg.MaxStepDuration {
		return
	}

	// Severity gate. A HIGH chain on this PID pauses baseline updates
	// AND classification — we don't want to fold a degraded step
	// into the baseline OR fire an outlier alarm during a window the
	// operator already knows is anomalous.
	if e.severity.IsAtLeast(evt.PID, e.pauseRank, now) {
		return
	}

	// Classify the step into a phase. When the classifier is
	// disabled, every step lands in the empty-string phase bucket
	// (preserving v0.16.0 single-baseline-per-stream behavior).
	phase := Phase("")
	if e.cfg.PhaseClassifierEnabled {
		phase = ClassifyPhase(
			step,
			obs.LaunchCount,
			obs.TotalKernelNs,
			obs.MemcpyBytes,
			obs.NCCLCount,
			int(obs.MemfragCount),
			e.cfg.PhaseConfig,
		)
	}

	// v0.16.5b: KernelFingerprint becomes part of the WorkloadKey
	// only when the feature is engaged. Off by default to preserve
	// v0.16.4 LRU footprint; on, distinct kernel-launch sequences
	// produce independent baselines (multi-model-per-process serving).
	var fingerprint uint64
	if e.cfg.FingerprintKeyEnabled {
		fingerprint = obs.KernelFingerprint
	}
	key := WorkloadKey{
		CGroupHash:        streamKey.CGroupHash,
		PID:               streamKey.PID,
		StreamHandle:      streamKey.StreamHandle,
		Phase:             phase,
		KernelFingerprint: fingerprint,
	}

	e.mu.Lock()
	if e.cfg.PhaseClassifierEnabled {
		e.phaseCounts[phase]++
	}
	e.mu.Unlock()

	bl := e.wmap.GetOrCreate(key, now)

	if !bl.Warmed(e.cfg.WarmupSamples) {
		bl.Update(float64(step.Nanoseconds()))
		return
	}

	// Classify against p95.
	p95 := bl.P95()
	if p95 <= 0 {
		// Baseliner finished P² fill but somehow returned 0 — defense
		// in depth, treat as warming and fold the step.
		bl.Update(float64(step.Nanoseconds()))
		return
	}
	ratio := float64(step.Nanoseconds()) / p95
	bucket := classify(ratio, e.cfg.OutlierThresholdRatio)
	if bucket == BucketNone {
		// Healthy step — fold into baseline.
		bl.Update(float64(step.Nanoseconds()))
		return
	}

	// Outlier: do NOT fold (preserves baseline cleanliness during
	// extended anomaly windows). Enqueue, count, log (rate-limited),
	// optionally degrade the sampler.
	//
	// Phase=unknown steps fire outliers normally (so operators see
	// the anomaly) but DO NOT trigger sampler degradation: we lack
	// the workload context to know whether a slowdown is meaningful,
	// so flipping the sampler to 100% would cause unnecessary
	// storage pressure on novel patterns.
	//
	// v0.16.3: attach memfrag + throttle context. Throttle reasons
	// come from either (a) the explicit reader hook installed by the
	// agent (current state of the NVML poller, aggregated across
	// GPUs) or (b) the per-step OR-fold accumulated via
	// RecordThrottle. (a) is preferred because it covers the
	// whole-step interval even when no per-PID throttle plumbing is
	// wired; (b) is the eventual per-step path once the closed-driver
	// throttle kprobe lands.
	throttleReasons := obs.MaxThrottleReasons
	if e.throttleReader != nil {
		if r := e.throttleReader(); r != 0 {
			throttleReasons |= r
		}
	}
	// KV-cache lineage. Only attach for decode-phase outliers (and
	// the unclassified case when phase classifier is off, so the
	// classifier-disabled mode still gets the data); prefill /
	// unknown / mixed outliers don't get the alloc-age hit because
	// the signal isn't actionable there. The tracker is also nil
	// when --inference-kvcache-lineage is off.
	var kvAges []uint64
	if e.cfg.KVCacheTracker != nil {
		if !e.cfg.PhaseClassifierEnabled || phase == PhaseDecode {
			kvAges = e.cfg.KVCacheTracker.TopAllocAgesMs(evt.PID, e.cfg.KVCacheTopN, now)
			// Fold each age into the engine-level histogram so the
			// OTLP/Prom exporters surface the cumulative distribution
			// across decode outliers. Histogram.Observe takes its
			// own lock; no need to hold e.mu here.
			for _, a := range kvAges {
				e.kvCacheAllocAgeHist.Observe(float64(a))
			}
		}
	}
	ev := OutlierEvent{
		Key:                   key,
		StepDurationNs:        step.Nanoseconds(),
		BaselineP95Ns:         int64(p95),
		BaselineMeanNs:        int64(bl.Mean()),
		Bucket:                bucket,
		At:                    now,
		EventID:               newEventID(),
		MemfragEvents:         obs.MemfragCount,
		ThrottleReasons:       throttleReasons,
		MinSMClockMHz:         0, // reserved; SM clock plumbing deferred (see Batch 1 commit)
		KVCacheTopAllocAgesMs: kvAges,
	}
	e.enqueueOutlier(ev)
	e.maybeLogOutlier(ev)
	if !e.cfg.PhaseClassifierEnabled || phase.IsClassified() {
		e.maybeDegradeSampler(bucket, key, now)
	}
}

// OnChainSnapshot updates the per-PID severity gate from the latest
// causal chains for one PID. The Engine inspects the maximum severity
// across the chain set and stores that. now is the snapshot time.
//
// Caller pattern: in the snapshot loop where SnapshotCausalChains is
// invoked per-PID, immediately call OnChainSnapshot with the same PID
// and the returned chain slice. Empty slice clears the PID's entry
// (the producer is reporting "no active chains" for that PID).
func (e *Engine) OnChainSnapshot(chains []correlate.CausalChain, pid uint32, now time.Time) {
	maxRank := sevNone
	for _, c := range chains {
		if r := parseSeverity(c.Severity); r > maxRank {
			maxRank = r
		}
	}
	e.severity.Set(pid, maxRank, now)
}

// Drain returns and clears the queued outlier events. Called once per
// snapshot tick from internal/cli/trace.go's onSnapshot callback.
// Caller emits each event through the OTLP exporter, the Prometheus
// aggregator, and (when --remediate is on) the UDS server.
func (e *Engine) Drain() []OutlierEvent {
	e.mu.Lock()
	defer e.mu.Unlock()
	if len(e.queue) == 0 {
		return nil
	}
	out := e.queue
	e.queue = nil
	return out
}

// DrainSampler returns and clears the queued sampler-degraded events.
// v0.16.3: parallel to Drain, called from the same onSnapshot callback
// so the agent can emit one UDS message per flip-to-degraded transition.
func (e *Engine) DrainSampler() []SamplerDegradedEvent {
	e.mu.Lock()
	defer e.mu.Unlock()
	if len(e.samplerQueue) == 0 {
		return nil
	}
	out := e.samplerQueue
	e.samplerQueue = nil
	return out
}

// Snapshot returns the current baseline state for every tracked
// workload. Called once per snapshot tick to emit baseline mean +
// p95 gauges. Holds the workload map mutex internally.
func (e *Engine) Snapshot() []workloadSnapshot {
	return e.wmap.Snapshot(e.cfg.WarmupSamples)
}

// SnapshotForExport assembles the v0.16.3 exporter view of every
// tracked workload plus the engine-level aggregates. Returns:
//
//   - per-workload stats (mean, p95, sample count, full histogram
//     snapshot, plus the workload-key fields used as data point
//     attributes by the OTLP/Prometheus exporters);
//   - engine-level aggregates (workloads tracked, cumulative outlier
//     counts per bucket, throttle-at-outlier counts per bucket);
//   - sampler state (degraded flag, cumulative degradation count, the
//     human-friendly cause of the most recent flip).
//
// Plain stats types (no internal/infer types in the return) so the
// export package can encode without an import cycle.
func (e *Engine) SnapshotForExport() ([]stats.InferWorkloadStats, stats.InferEngineStats, stats.InferSamplerState) {
	rows := e.wmap.SnapshotForExport(e.cfg.WarmupSamples)

	// kvHist: snapshot the histogram outside e.mu (it has its own
	// internal lock).
	var kvHist stats.HistogramSnapshot
	if e.kvCacheAllocAgeHist != nil {
		kvHist = e.kvCacheAllocAgeHist.Snapshot()
	}

	e.mu.Lock()
	es := stats.InferEngineStats{
		WorkloadsTracked:    e.wmap.Len(),
		OutliersTotal:       make(map[string]uint64, len(e.outliers)),
		ThrottleAtOutlier:   make(map[string]uint64, len(e.throttleAtOutlier)),
		KVCacheAllocAgeHist: kvHist,
	}
	for k, v := range e.outliers {
		if k == BucketNone {
			continue
		}
		es.OutliersTotal[string(k)] = v
	}
	for k, v := range e.throttleAtOutlier {
		if k == BucketNone {
			continue
		}
		es.ThrottleAtOutlier[string(k)] = v
	}
	now := time.Now()
	ss := stats.InferSamplerState{
		Degraded:          !e.samplerDegradedUntil.IsZero() && now.Before(e.samplerDegradedUntil),
		DegradationsTotal: e.samplerDegradationsCnt,
		LastCause:         e.samplerLastCause,
	}
	e.mu.Unlock()

	return rows, es, ss
}

// Stats returns cumulative engine telemetry. Engine clears nothing
// — values are monotonic. The returned maps are copies; the caller
// may mutate them freely.
//
// Side effect: prunes stale observable counters (entries idle longer
// than cfg.ObservableTTL). The snapshot loop is the natural cadence
// for this — once per snapshot tick is plenty.
func (e *Engine) Stats() EngineStats {
	e.mu.Lock()
	now := time.Now()
	out := EngineStats{
		WorkloadsTracked:         e.wmap.Len(),
		QueueDropped:             e.queueDropped,
		OutliersTotal:            make(map[OutlierBucket]uint64, len(e.outliers)),
		PhaseDistribution:        make(map[Phase]uint64, len(e.phaseCounts)),
		SamplerDegraded:          !e.samplerDegradedUntil.IsZero() && now.Before(e.samplerDegradedUntil),
		SamplerDegradationsTotal: e.samplerDegradationsCnt,
		LastDegradationCause:     e.samplerLastCause,
		SamplerDegradedUntil:     e.samplerDegradedUntil,
		ThrottleAtOutlier:        make(map[OutlierBucket]uint64, len(e.throttleAtOutlier)),
	}
	for k, v := range e.outliers {
		out.OutliersTotal[k] = v
	}
	for k, v := range e.phaseCounts {
		out.PhaseDistribution[k] = v
	}
	for k, v := range e.throttleAtOutlier {
		out.ThrottleAtOutlier[k] = v
	}
	e.mu.Unlock()

	// Eviction tracking lives on workloadMap; surface it once per
	// stats fetch. The caller is the snapshot loop, which is the
	// natural cadence for "did we evict in this cycle?" warnings.
	if e.wmap.EvictedSinceLastClear() {
		e.log.Warn("infer: workload LRU evicting; consider increasing capacity",
			"capacity", e.cfg.MaxWorkloads)
		e.wmap.ClearEvictionFlag()
	}

	// Prune observable counters that have not seen activity within
	// the TTL. Bounds memory in the face of LRU evictions on the
	// workload map. observables.PruneStale takes its own lock; do
	// not hold e.mu during this call to avoid contention with the
	// hot path.
	if e.observables != nil {
		e.observables.PruneStale(time.Now(), e.cfg.ObservableTTL)
	}
	return out
}

// classify picks the outlier bucket for a duration:p95 ratio.
// outermost is the configurable threshold for the largest bucket
// (default 3.0). 2x and 1.5x are fixed.
func classify(ratio, outermost float64) OutlierBucket {
	switch {
	case ratio >= outermost:
		return Bucket3x
	case ratio >= 2.0:
		return Bucket2x
	case ratio >= 1.5:
		return Bucket1_5x
	default:
		return BucketNone
	}
}

// enqueueOutlier appends with overflow protection. Drops the oldest
// event when at cap so the queue cannot grow unbounded if the snapshot
// drain stalls. Mirrors the ncclBufferAdd pattern in trace.go.
func (e *Engine) enqueueOutlier(ev OutlierEvent) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.outliers[ev.Bucket]++
	if ev.ThrottleReasons != 0 {
		e.throttleAtOutlier[ev.Bucket]++
	}
	if len(e.queue) >= e.cfg.OutlierQueueCap {
		// Drop oldest (front-of-slice) so the most recent outlier is
		// always retained — those are the ones an operator most wants
		// to see when they next look at the agent.
		copy(e.queue, e.queue[1:])
		e.queue[len(e.queue)-1] = ev
		e.queueDropped++
		return
	}
	e.queue = append(e.queue, ev)
}

// maybeLogOutlier emits one INFO line per workload per
// LogMinInterval. Avoids flooding the agent log when a workload is
// stuck in a sustained outlier window. Pattern parallels
// internal/health/loop.go:506-521.
func (e *Engine) maybeLogOutlier(ev OutlierEvent) {
	e.mu.Lock()
	last, ok := e.lastLogAt[ev.Key]
	if ok && ev.At.Sub(last) < e.cfg.LogMinInterval {
		e.mu.Unlock()
		return
	}
	e.lastLogAt[ev.Key] = ev.At
	e.mu.Unlock()
	e.log.Info("infer: outlier",
		"cgroup", ev.Key.CGroupHash,
		"pid", ev.Key.PID,
		"stream", ev.Key.StreamHandle,
		"phase", string(ev.Key.Phase),
		"bucket", string(ev.Bucket),
		"step_ns", ev.StepDurationNs,
		"baseline_p95_ns", ev.BaselineP95Ns,
		"event_id", ev.EventID,
	)
}

// maybeDegradeSampler bumps the attached store sampler to admit 100%
// when the bucket meets the configured threshold. The sampler's own
// 30s cooldown handles decay back to the healthy admit rate; we never
// need to call SetDegraded(false).
//
// v0.16.3: also surfaces a SamplerDegradedEvent on the engine's
// dedicated queue (drained by DrainSampler at snapshot tick) and
// updates the observability counters consumed by Stats(). Cause is a
// human-friendly summary that becomes the AttrInferSamplerCause data
// point attribute on ingero.infer.sampler.* metrics. The function
// short-circuits when the bucket is below the configured threshold,
// preserving the previous behavior for unrelated buckets.
func (e *Engine) maybeDegradeSampler(bucket OutlierBucket, key WorkloadKey, now time.Time) {
	if e.cfg.Sampler == nil || e.cfg.SamplerDegradeOn == BucketNone {
		return
	}
	// Bucket order: 1.5x < 2x < 3x. Trigger when the observed bucket
	// is at-or-above the configured threshold.
	if bucketRank(bucket) < bucketRank(e.cfg.SamplerDegradeOn) {
		return
	}
	e.cfg.Sampler.SetDegraded(true)

	cooldownEnd := now.Add(samplerCooldownDuration)
	cause := fmt.Sprintf("%s:cgroup=%s,pid=%d,phase=%s",
		string(bucket), key.CGroupHash, key.PID, string(key.Phase))

	e.mu.Lock()
	e.samplerDegradationsCnt++
	e.samplerDegradedUntil = cooldownEnd
	e.samplerLastCause = cause
	if len(e.samplerQueue) < samplerQueueCap {
		e.samplerQueue = append(e.samplerQueue, SamplerDegradedEvent{
			Key:         key,
			Bucket:      bucket,
			At:          now,
			CooldownEnd: cooldownEnd,
			Cause:       cause,
		})
	}
	// Always log: sampler flips are infrequent enough not to need
	// rate-limiting, and operators want to see every flip for incident
	// timelines. Pattern parallels stragglerDetector logging in
	// internal/cli/trace.go.
	e.mu.Unlock()
	e.log.Info("infer: sampler degraded",
		"cause", cause,
		"bucket", string(bucket),
		"cooldown_end", cooldownEnd.Format(time.RFC3339Nano),
	)
}

// bucketRank totals a bucket so we can compare with >= without
// stringly typed switches sprinkled across the engine.
func bucketRank(b OutlierBucket) int {
	switch b {
	case Bucket3x:
		return 3
	case Bucket2x:
		return 2
	case Bucket1_5x:
		return 1
	default:
		return 0
	}
}

// isSyncEvent recognizes the sync-class events that the engine
// observes as step boundaries. CUDA runtime sync (StreamSync,
// DeviceSync) plus driver context sync (DriverCtxSync) all count as
// the same event class for our purposes — each marks the end of one
// burst of GPU work.
func isSyncEvent(evt events.Event) bool {
	switch evt.Source {
	case events.SourceCUDA:
		switch events.CUDAOp(evt.Op) {
		case events.CUDAStreamSync, events.CUDADeviceSync:
			return true
		}
	case events.SourceDriver:
		if events.DriverOp(evt.Op) == events.DriverCtxSync {
			return true
		}
	}
	return false
}

// newEventID generates a UUIDv4-shaped opaque ID for outlier-event
// cross-channel correlation. We don't pull in github.com/google/uuid
// because the rest of the agent uses a similar approach via
// crypto/rand for the existing alerter dedup_key generator.
func newEventID() string {
	var b [16]byte
	if _, err := rand.Read(b[:]); err != nil {
		// Fallback: use a fixed sentinel rather than fail the
		// outlier emission. The event is still meaningful without
		// a globally unique ID; cross-channel correlation just
		// becomes harder for these specific events.
		return "00000000-0000-0000-0000-000000000000"
	}
	// Set version (4) and variant (10xx) bits per RFC 4122.
	b[6] = (b[6] & 0x0f) | 0x40
	b[8] = (b[8] & 0x3f) | 0x80
	const dash = "-"
	var hexBuf [32]byte
	hex.Encode(hexBuf[:], b[:])
	return string(hexBuf[0:8]) + dash +
		string(hexBuf[8:12]) + dash +
		string(hexBuf[12:16]) + dash +
		string(hexBuf[16:20]) + dash +
		string(hexBuf[20:32])
}
