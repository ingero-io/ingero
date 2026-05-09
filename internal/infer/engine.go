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
	lastSync map[WorkloadKey]time.Time
	queue    []OutlierEvent

	// per-workload INFO log rate-limit. Mirrors the maybeLogAnomaly
	// pattern in internal/health/loop.go:506-521.
	lastLogAt map[WorkloadKey]time.Time

	// Cumulative outlier counters per bucket. Engine-owned; emitter
	// reads via Stats() and projects onto the OTLP cumulative counter.
	outliers map[OutlierBucket]uint64

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
		cfg:       cfg,
		log:       log,
		wmap:      newWorkloadMap(cfg.MaxWorkloads),
		severity:  newSeverityGate(cfg.SeverityTTL),
		lastSync:  make(map[WorkloadKey]time.Time, cfg.MaxWorkloads),
		lastLogAt: make(map[WorkloadKey]time.Time),
		outliers:  make(map[OutlierBucket]uint64, 3),
		pauseRank: parseSeverity(cfg.PauseOnSeverity),
	}
}

// OnSyncEvent processes one cudaStreamSynchronize / cudaDeviceSync /
// driver ctxSync event. cgroupHash is the resolved cgroup_path_hash
// for the event's PID (caller resolves it via the existing PodCache
// or cgroup cache). Empty cgroupHash is allowed; it produces a
// "unattributable" workload bucket but the baseline still tracks.
//
// The hot path: lookup last sync for this key; compute delta; gate
// on severity; classify or warmup-update; enqueue outlier if any.
func (e *Engine) OnSyncEvent(evt events.Event, cgroupHash string) {
	if !isSyncEvent(evt) {
		return
	}
	key := WorkloadKey{
		CGroupHash:   cgroupHash,
		PID:          evt.PID,
		StreamHandle: evt.Args[0],
	}
	now := evt.Timestamp

	e.mu.Lock()
	prev, hadPrev := e.lastSync[key]
	e.lastSync[key] = now
	e.mu.Unlock()
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
	e.maybeDegradeSampler(bucket)
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
// — values are monotonic. The returned map is a copy; the caller may
// mutate it freely.
func (e *Engine) Stats() EngineStats {
	e.mu.Lock()
	out := EngineStats{
		WorkloadsTracked: e.wmap.Len(),
		QueueDropped:     e.queueDropped,
		OutliersTotal:    make(map[OutlierBucket]uint64, len(e.outliers)),
	}
	for k, v := range e.outliers {
		out.OutliersTotal[k] = v
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
