package infer

import (
	"crypto/rand"
	"encoding/hex"
	"log/slog"
	"sync"
	"time"

	"github.com/ingero-io/ingero/internal/correlate"
	"github.com/ingero-io/ingero/internal/sampling"
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
	c.PhaseConfig = c.PhaseConfig.Resolved()
	return c
}

// OutlierEvent is one classified step that exceeded its baseline.
// Drained from Engine on each snapshot tick and emitted to OTLP +
// Prometheus + (when --remediate is on) the UDS socket.
type OutlierEvent struct {
	Key            WorkloadKey
	StepDurationNs int64
	BaselineP95Ns  int64
	BaselineMeanNs int64
	Bucket         OutlierBucket
	At             time.Time
	EventID        string // UUIDv4-shaped, opaque, for cross-channel correlation
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
}

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
		cfg:         cfg,
		log:         log,
		wmap:        newWorkloadMap(cfg.MaxWorkloads),
		severity:    newSeverityGate(cfg.SeverityTTL),
		lastSync:    make(map[observableKey]time.Time, cfg.MaxWorkloads),
		observables: newStepObservables(),
		lastLogAt:   make(map[WorkloadKey]time.Time),
		outliers:    make(map[OutlierBucket]uint64, 4),
		phaseCounts: make(map[Phase]uint64, 4),
		pauseRank:   parseSeverity(cfg.PauseOnSeverity),
	}
}

// OnLaunchEvent records a cudaLaunchKernel event between syncs.
// Observable counters accumulate at the (cgroup, pid) level —
// stream=0 sentinel — because today's BPF probe puts the kernel
// function pointer in Args[0], NOT the stream handle (per
// bpf/cuda_trace.bpf.c:368). cudaMemcpy similarly carries byte
// count (Args[0]) and direction (Args[1]) but no stream. Until a
// BPF extension threads the stream through these events, phase
// classification is per-PID rather than per-stream — a documented
// v0.16.x limitation that's acceptable for the common case
// (single inference engine per pod) and degraded but functional
// for multi-stream serving.
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
	e.observables.AddLaunch(observableKey{
		CGroupHash:   cgroupHash,
		PID:          evt.PID,
		StreamHandle: 0, // PID-level aggregation; see func comment
	}, kernelDuration, evt.Timestamp)
}

// OnMemcpyEvent records a cudaMemcpy / cudaMemcpyAsync event. bytes
// is taken from the BPF event's Args[0] (per cuda_trace.bpf.c). No
// direction differentiation — the phase classifier only checks
// total bytes moved per step. Aggregates at PID level for the same
// reason as OnLaunchEvent (no stream in the BPF event today).
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
		StreamHandle: 0, // PID-level aggregation; see OnLaunchEvent
	}, bytes, evt.Timestamp)
}

// OnNCCLEvent records participation in an NCCL collective during
// a step. The classifier treats any non-zero NCCL count as a strong
// prefill signal (rule 1).
//
// streamHandle parameter is accepted but currently dropped — see
// OnLaunchEvent comment for the rationale (PID-level observable
// aggregation in v0.16.x). When the BPF probe extension lands, this
// reverts to per-stream attribution.
func (e *Engine) OnNCCLEvent(pid uint32, cgroupHash string, streamHandle uint64, at time.Time) {
	_ = streamHandle
	if !e.cfg.PhaseClassifierEnabled {
		return
	}
	e.observables.AddNCCL(observableKey{
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
	// pidKey is the PID-level observable bucket — that's where
	// OnLaunchEvent / OnMemcpyEvent / OnNCCLEvent accumulate today,
	// since the BPF probes for those events don't carry the stream.
	// When the BPF probe is extended (v0.16.x followup), this folds
	// back into streamKey.
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
	// Read from the PID-level bucket where launches accumulated.
	obs := e.observables.ResetAndRead(pidKey, now)

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
			e.cfg.PhaseConfig,
		)
	}

	key := WorkloadKey{
		CGroupHash:   streamKey.CGroupHash,
		PID:          streamKey.PID,
		StreamHandle: streamKey.StreamHandle,
		Phase:        phase,
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
	ev := OutlierEvent{
		Key:            key,
		StepDurationNs: step.Nanoseconds(),
		BaselineP95Ns:  int64(p95),
		BaselineMeanNs: int64(bl.Mean()),
		Bucket:         bucket,
		At:             now,
		EventID:        newEventID(),
	}
	e.enqueueOutlier(ev)
	e.maybeLogOutlier(ev)
	if !e.cfg.PhaseClassifierEnabled || phase.IsClassified() {
		e.maybeDegradeSampler(bucket)
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

// Snapshot returns the current baseline state for every tracked
// workload. Called once per snapshot tick to emit baseline mean +
// p95 gauges. Holds the workload map mutex internally.
func (e *Engine) Snapshot() []workloadSnapshot {
	return e.wmap.Snapshot(e.cfg.WarmupSamples)
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
	out := EngineStats{
		WorkloadsTracked:  e.wmap.Len(),
		QueueDropped:      e.queueDropped,
		OutliersTotal:     make(map[OutlierBucket]uint64, len(e.outliers)),
		PhaseDistribution: make(map[Phase]uint64, len(e.phaseCounts)),
	}
	for k, v := range e.outliers {
		out.OutliersTotal[k] = v
	}
	for k, v := range e.phaseCounts {
		out.PhaseDistribution[k] = v
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
func (e *Engine) maybeDegradeSampler(bucket OutlierBucket) {
	if e.cfg.Sampler == nil || e.cfg.SamplerDegradeOn == BucketNone {
		return
	}
	// Bucket order: 1.5x < 2x < 3x. Trigger when the observed bucket
	// is at-or-above the configured threshold.
	if bucketRank(bucket) >= bucketRank(e.cfg.SamplerDegradeOn) {
		e.cfg.Sampler.SetDegraded(true)
	}
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
