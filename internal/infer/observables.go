package infer

import (
	"sync"
	"time"
)

// observableKey is the (cgroup, pid, stream) tuple we accumulate
// per-step counters against. Phase is intentionally NOT part of the
// key here — we collect observables FIRST, classify phase from them
// LATER. The full WorkloadKey (with phase) is computed at sync-event
// time when we look up the baseliner.
type observableKey struct {
	CGroupHash   string
	PID          uint32
	StreamHandle uint64
}

// stepObservables accumulates the four signals the phase classifier
// reads from at each step boundary. Counters are reset to zero on
// each ResetAndRead. Single-instance, mutex-guarded; the surrounding
// Engine drops to it on every kernel-launch / memcpy / NCCL event.
//
// Memory profile: bounded by the number of distinct
// (cgroup, pid, stream) keys observed, which equals the workload
// LRU cap (default 1024) plus the brief window between key-eviction
// and the next sync's read. Reset is O(1) per key (just zero the
// fields); we don't delete entries on reset because the next step
// for the same key is far more likely than the entry being stale.
//
// Eviction (rare): when the workload LRU evicts a key, we leak its
// observable entry until either (a) the same key reappears (and we
// reuse the entry) or (b) PruneStale runs. PruneStale is called by
// the Engine on each Snapshot tick.
type stepObservables struct {
	mu  sync.Mutex
	per map[observableKey]*observableCounters
}

// observableCounters holds the running per-step totals plus a
// last-update timestamp for stale eviction.
//
// MemfragCount is the count of NVIDIA closed-driver IOCTL events
// observed via the v0.15 W1 memfrag kprobe between consecutive syncs.
// Non-zero values are a strong "decode-pressure" signal (KV cache
// eviction storm under VRAM pressure) and feed the v0.16.3 phase rule.
//
// MaxThrottleReasons is the OR-fold of NVML clock-throttle bitmaps
// observed during the step; non-zero means at least one throttle
// reason fired. v0.16.3 surface for thermal-context-on-outlier so a
// step slowdown caused by HW_SLOWDOWN is visibly thermal in the JSON
// envelope.
//
// KernelFingerprint is an FNV-1a-style fold over the kernel function
// pointers launched during the step. v0.16.5b: when the agent runs
// with --inference-fingerprint-key, this becomes part of the
// WorkloadKey so the engine produces independent baselines for
// distinct kernel-launch sequences (e.g. two models served by the
// same process on the same stream). Order-dependent and stable across
// repeated identical sequences. Zero when no launches were observed
// in the step.
type observableCounters struct {
	LaunchCount        int
	TotalKernelNs      time.Duration
	MemcpyBytes        int64
	NCCLCount          int
	MemfragCount       uint32
	MaxThrottleReasons uint64
	KernelFingerprint  uint64
	LastUpdate         time.Time
}

// FNV-1a 64-bit constants. Used to fold kernel function pointers into
// KernelFingerprint. Order-dependent (kernel-launch order matters);
// duplicate kernels do NOT cancel (unlike a plain XOR-fold).
const (
	fnv1aOffset64 uint64 = 0xcbf29ce484222325
	fnv1aPrime64  uint64 = 0x100000001b3
)

// newStepObservables constructs an empty store. The Engine owns one
// of these and drives all updates.
func newStepObservables() *stepObservables {
	return &stepObservables{
		per: make(map[observableKey]*observableCounters),
	}
}

// AddLaunch increments the kernel-launch count for the workload key,
// accumulates the kernel duration, and folds the kernel function
// pointer into the per-step KernelFingerprint (v0.16.5b). funcPtr=0
// is allowed and skips the fingerprint fold so legacy / synthetic
// callers without a kernel-pointer source don't poison the hash.
//
// Called from the Engine's OnLaunchEvent on the trace event-loop hot
// path.
func (s *stepObservables) AddLaunch(key observableKey, funcPtr uint64, kernelDuration time.Duration, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	c := s.getOrCreateLocked(key)
	if c.LaunchCount == 0 {
		c.KernelFingerprint = fnv1aOffset64
	}
	if funcPtr != 0 {
		c.KernelFingerprint ^= funcPtr
		c.KernelFingerprint *= fnv1aPrime64
	}
	c.LaunchCount++
	c.TotalKernelNs += kernelDuration
	c.LastUpdate = now
}

// AddMemcpy accumulates memcpy bytes for the workload key. Direction
// (h2d / d2h / etc.) is intentionally not differentiated here — the
// phase classifier only reads total bytes per step.
func (s *stepObservables) AddMemcpy(key observableKey, bytes int64, now time.Time) {
	if bytes <= 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	c := s.getOrCreateLocked(key)
	c.MemcpyBytes += bytes
	c.LastUpdate = now
}

// AddNCCL increments the NCCL collective count for the workload key.
// Bumping by one on each collective is intentional — the phase
// classifier only checks ncclCount > 0.
func (s *stepObservables) AddNCCL(key observableKey, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	c := s.getOrCreateLocked(key)
	c.NCCLCount++
	c.LastUpdate = now
}

// AddMemfrag increments the memfrag IOCTL event count for the
// workload key. The phase classifier reads memfragCount to
// distinguish KV-cache-pressure decode steps (memfrag>0 + few
// launches) from compute-bound work.
func (s *stepObservables) AddMemfrag(key observableKey, now time.Time) {
	s.mu.Lock()
	defer s.mu.Unlock()
	c := s.getOrCreateLocked(key)
	// Saturating add: never wrap. A sustained memfrag storm pegs at
	// 4_294_967_295 rather than rolling back to 0, which would silently
	// reset the phase classifier's "high pressure" rule mid-storm and
	// surface as a misleading 0 in operator telemetry.
	if c.MemfragCount < ^uint32(0) {
		c.MemfragCount++
	}
	c.LastUpdate = now
}

// RecordThrottle OR-folds an NVML throttle-reasons bitmap into the
// per-step max. v0.16.3 thermal-context surface; the engine writes
// here on each NVML poll tick (latest-wins per step) and reads back
// at sync-event time to attach the bitmap to outlier events.
//
// The OR-fold preserves any reason that fired anywhere in the step,
// which is what an operator wants when asking "did this slow step
// coincide with a throttle?" — not "what was the last throttle
// reading." A zero reasons value is a no-op so the throttle path can
// call this on every poll tick without churn when no throttle is
// active.
func (s *stepObservables) RecordThrottle(key observableKey, reasons uint64, now time.Time) {
	if reasons == 0 {
		return
	}
	s.mu.Lock()
	defer s.mu.Unlock()
	c := s.getOrCreateLocked(key)
	c.MaxThrottleReasons |= reasons
	c.LastUpdate = now
}

// ResetAndRead atomically returns the current counters for the
// workload key and zeroes them in place. Called from the Engine's
// OnSyncEvent after computing step duration — the returned snapshot
// is fed to ClassifyPhase, then the next step's observables start
// accumulating from zero. The entry is NOT deleted (next step is
// expected, deletion would just churn the map).
//
// Returns the zero observableCounters (all fields 0) when the key
// has never been seen — this is the "first sync after process
// start" case. Caller treats absence as a no-op step.
func (s *stepObservables) ResetAndRead(key observableKey, now time.Time) observableCounters {
	s.mu.Lock()
	defer s.mu.Unlock()
	c, ok := s.per[key]
	if !ok {
		// Materialize a zero entry so future events for this key
		// have somewhere to land. Cheap (~32 bytes).
		c = &observableCounters{LastUpdate: now}
		s.per[key] = c
		return observableCounters{}
	}
	out := *c
	c.LaunchCount = 0
	c.TotalKernelNs = 0
	c.MemcpyBytes = 0
	c.NCCLCount = 0
	c.MemfragCount = 0
	c.MaxThrottleReasons = 0
	c.KernelFingerprint = 0
	c.LastUpdate = now
	return out
}

// PruneStale deletes entries that have not been updated within the
// given duration. Called periodically (via Engine.Stats) to bound
// memory in the face of LRU evictions on the workload map. ttl
// should be longer than the longest expected idle gap between syncs
// for a real workload — default 5 minutes is safe.
func (s *stepObservables) PruneStale(now time.Time, ttl time.Duration) int {
	s.mu.Lock()
	defer s.mu.Unlock()
	pruned := 0
	for k, c := range s.per {
		if now.Sub(c.LastUpdate) > ttl {
			delete(s.per, k)
			pruned++
		}
	}
	return pruned
}

// Len returns the current count of tracked workload keys. Test-only.
func (s *stepObservables) Len() int {
	s.mu.Lock()
	defer s.mu.Unlock()
	return len(s.per)
}

// Peek returns a copy of the counters for a key without resetting.
// Test helper; the production hot path uses ResetAndRead.
func (s *stepObservables) Peek(key observableKey) observableCounters {
	s.mu.Lock()
	defer s.mu.Unlock()
	if c, ok := s.per[key]; ok {
		return *c
	}
	return observableCounters{}
}

func (s *stepObservables) getOrCreateLocked(key observableKey) *observableCounters {
	if c, ok := s.per[key]; ok {
		return c
	}
	c := &observableCounters{}
	s.per[key] = c
	return c
}
