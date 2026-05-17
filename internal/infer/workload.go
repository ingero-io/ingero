package infer

import (
	"container/list"
	"sync"
	"time"

	"github.com/ingero-io/ingero/internal/inferp99"
	"github.com/ingero-io/ingero/internal/stats"
)

// WorkloadKey identifies one inference workload bucket. v0.16.0 keyed
// on (cgroup_path_hash, pid, stream_handle); v0.16.1 added the Phase
// dimension so each (cgroup, pid, stream) workload has up to four
// independent baselines (one per phase classification).
//
// Why phase is part of the key (not a label): a single P² baseline
// that absorbs both prefill (~200ms) and decode (~5ms) steps from a
// vLLM continuous-batching stream produces false negatives — a slow
// decode under the prefill tail goes undetected. Splitting baselines
// by phase makes each baseline unimodal and enables apples-to-apples
// outlier comparison.
//
// Stream handle is the BPF event Args[0] for sync events; in real
// serving frameworks (PyTorch, vLLM, TGI, SGLang) distinct streams
// typically correspond to distinct logical work paths.
//
// Future revisions may add a kernel-fingerprint dimension (rolling
// hash of kernel-name set per step) to cover the "two models in one
// process on one stream on one phase" case; deferred from v0.16
// because phase-awareness already eliminates the dominant
// false-negative class.
type WorkloadKey struct {
	CGroupHash   string
	PID          uint32
	StreamHandle uint64
	Phase        Phase
	// KernelFingerprint distinguishes workloads that share
	// (cgroup, pid, stream, phase) but launch different kernel
	// sequences - e.g. a serverless host running two models on the
	// same engine, or a multi-tenant Triton with the same PID
	// servicing different model deployments. v0.16.5b. Zero unless
	// the operator engages --inference-fingerprint-key, in which
	// case the engine reads the per-step KernelFingerprint accumulator
	// from observables at sync-event time and includes it here.
	KernelFingerprint uint64
}

// workloadEntry is the LRU map's value cell. The list element pointer
// is cached so MoveToFront / Remove operations are O(1) without a
// linear search.
//
// p99 is the rolling-p99 SLO-breach tracker introduced alongside the
// existing baseliner. Both are fed by the same sync hot path (one
// step duration in, both trackers updated); the engine reads bl for
// per-step outlier classification and the SLO watcher periodically
// reads p99 for sustained-breach detection. Allocated lazily in
// GetOrCreate and never replaced for the entry's lifetime.
type workloadEntry struct {
	key      WorkloadKey
	bl       *WorkloadBaseliner
	p99      *inferp99.Tracker
	lastSync time.Time
	elem     *list.Element
}

// workloadMap is a bounded LRU keyed by WorkloadKey. Eviction policy:
// least-recently-updated (last sync timestamp), not least-recently-
// looked-up — a workload that emits many syncs is the one whose
// baseline we most want to keep, regardless of how often we look up
// adjacent ones. The map's internal list uses MoveToFront on Update
// (signaling the entry's freshness) but NOT on Get-without-update,
// matching that policy.
//
// Shape mirrors internal/health/cgroup_cache.go (container/list +
// map + mutex + eviction-flag). Capacity defaults to 1024.
type workloadMap struct {
	mu       sync.Mutex
	entries  map[WorkloadKey]*workloadEntry
	order    *list.List // front = most recently updated; back = oldest
	capacity int

	// evictedThisCycle gates a once-per-tick WARN log so that a churn
	// of short-lived workloads does not flood the agent's log. The
	// owning Engine clears this on each snapshot tick.
	evictedThisCycle bool
}

func newWorkloadMap(capacity int) *workloadMap {
	if capacity <= 0 {
		capacity = 1024
	}
	return &workloadMap{
		entries:  make(map[WorkloadKey]*workloadEntry, capacity),
		order:    list.New(),
		capacity: capacity,
	}
}

// GetOrCreate returns the existing baseliner for key, or constructs a
// fresh one and inserts it under the LRU bound. The lastSync field is
// the parameter `now` because GetOrCreate is called from the sync
// event hot path — by definition we just observed a sync for this
// key and are about to record its duration.
func (m *workloadMap) GetOrCreate(key WorkloadKey, now time.Time) *WorkloadBaseliner {
	return m.getOrCreateEntry(key, now).bl
}

// GetOrCreateWithP99 is the dual-tracker form of GetOrCreate. Returns
// both the per-step baseliner (existing InferenceOutlier classifier)
// and the rolling-p99 SLO-breach tracker (new InferenceSloBreach
// path). Both are populated lazily in one LRU touch so the sync hot
// path stays at O(1) per event.
func (m *workloadMap) GetOrCreateWithP99(key WorkloadKey, now time.Time) (*WorkloadBaseliner, *inferp99.Tracker) {
	e := m.getOrCreateEntry(key, now)
	return e.bl, e.p99
}

// getOrCreateEntry is the shared inner path. Holds the lock; performs
// LRU eviction; lazily allocates both trackers.
func (m *workloadMap) getOrCreateEntry(key WorkloadKey, now time.Time) *workloadEntry {
	m.mu.Lock()
	defer m.mu.Unlock()
	if e, ok := m.entries[key]; ok {
		e.lastSync = now
		m.order.MoveToFront(e.elem)
		return e
	}
	for m.order.Len() >= m.capacity {
		oldest := m.order.Back()
		if oldest == nil {
			break
		}
		oldEntry := oldest.Value.(*workloadEntry)
		m.order.Remove(oldest)
		delete(m.entries, oldEntry.key)
		m.evictedThisCycle = true
	}
	entry := &workloadEntry{
		key:      key,
		bl:       NewWorkloadBaseliner(),
		p99:      inferp99.NewTracker(inferp99.DefaultConfig()),
		lastSync: now,
	}
	entry.elem = m.order.PushFront(entry)
	m.entries[key] = entry
	return entry
}

// Get returns the baseliner for key without bumping its position in
// the LRU order. Used by the Engine's snapshot path so reading every
// workload's stats does not change eviction order.
func (m *workloadMap) Get(key WorkloadKey) (*WorkloadBaseliner, bool) {
	m.mu.Lock()
	defer m.mu.Unlock()
	e, ok := m.entries[key]
	if !ok {
		return nil, false
	}
	return e.bl, true
}

// Snapshot returns the (key, mean, p95, samples) tuple for every live
// workload. Holds the mutex for the duration of the copy; capped at
// `cap` entries so the worst case is bounded. Caller owns the slice.
type workloadSnapshot struct {
	Key     WorkloadKey
	Mean    float64
	P95     float64
	Samples int
	Warmed  bool
}

func (m *workloadMap) Snapshot(warmupSamples int) []workloadSnapshot {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]workloadSnapshot, 0, len(m.entries))
	for k, e := range m.entries {
		out = append(out, workloadSnapshot{
			Key:     k,
			Mean:    e.bl.Mean(),
			P95:     e.bl.P95(),
			Samples: e.bl.Samples(),
			Warmed:  e.bl.Warmed(warmupSamples),
		})
	}
	return out
}

// SnapshotForExport returns the v0.16.3 exporter-shape view: each
// tracked workload's mean / p95 / sample count plus the histogram
// snapshot. Plain stats types so the export package can read this
// without an internal/infer import (which would cycle through
// internal/cli on the way back).
//
// Skips not-yet-warmed workloads to avoid emitting noisy data points
// for entries with too-few samples; warmed-only matches the gauge
// emission path's existing convention. Caller owns the returned slice.
func (m *workloadMap) SnapshotForExport(warmupSamples int) []stats.InferWorkloadStats {
	m.mu.Lock()
	defer m.mu.Unlock()
	out := make([]stats.InferWorkloadStats, 0, len(m.entries))
	for k, e := range m.entries {
		if !e.bl.Warmed(warmupSamples) {
			continue
		}
		out = append(out, stats.InferWorkloadStats{
			CGroupHash:        k.CGroupHash,
			PID:               k.PID,
			StreamHandle:      k.StreamHandle,
			Phase:             string(k.Phase),
			KernelFingerprint: k.KernelFingerprint,
			MeanNs:            e.bl.Mean(),
			P95Ns:             e.bl.P95(),
			Samples:           e.bl.Samples(),
			Histogram:         e.bl.HistogramSnapshot(),
		})
	}
	return out
}

// Len returns the number of cached entries.
func (m *workloadMap) Len() int {
	m.mu.Lock()
	defer m.mu.Unlock()
	return len(m.entries)
}

// SloBreach is the workload-keyed breach returned from
// DrainSloBreaches. Wraps the metric-only inferp99.Breach with the
// workload identity (PID is the only field the EE-side wire requires
// today; the rest are carried for log context and future wire
// extensions).
type SloBreach struct {
	Key    WorkloadKey
	Breach inferp99.Breach
}

// DrainSloBreaches walks every workload entry, calls each tracker's
// CheckAt(now), and returns the breaches that fired this tick. The
// per-tracker state machine handles sustain + suppression internally
// so two consecutive Drain calls in the same suppression window
// return one breach for the workload, not two.
//
// Called from the SLO watcher goroutine on its 5s cadence.
func (m *workloadMap) DrainSloBreaches(now time.Time) []SloBreach {
	m.mu.Lock()
	defer m.mu.Unlock()
	var out []SloBreach
	for k, e := range m.entries {
		if e.p99 == nil {
			continue
		}
		if b, ok := e.p99.CheckAt(now); ok {
			out = append(out, SloBreach{Key: k, Breach: b})
		}
	}
	return out
}

// EvictedSinceLastClear returns true if at least one LRU eviction has
// happened since the last ClearEvictionFlag call. Mirrors the
// evictedThisCycle pattern in internal/health/cgroup_cache.go.
func (m *workloadMap) EvictedSinceLastClear() bool {
	m.mu.Lock()
	defer m.mu.Unlock()
	return m.evictedThisCycle
}

// ClearEvictionFlag resets the per-cycle eviction flag.
func (m *workloadMap) ClearEvictionFlag() {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.evictedThisCycle = false
}
