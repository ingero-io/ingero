// Package kvcache tracks live cudaMalloc allocations per inference
// process so the outlier path can attach KV-cache age context to
// decode-phase outliers.
//
// Concept: when a decode step slows down, the proximate cause is
// often KV-cache fragmentation - the engine evicting / re-allocating
// blocks under VRAM pressure. The age distribution of live
// allocations at outlier time tells you "old caches stuck around"
// (eviction wasn't keeping up) vs "everything is freshly allocated"
// (cold-start cost). Pairing every cudaMalloc with its eventual
// cudaFree gives that signal.
//
// Producer: internal/cli/trace.go's event loop calls OnMalloc /
// OnFree on every cudaMalloc / cudaFree (and the Managed variants).
// Consumer: internal/infer.Engine on a decode-phase outlier reads
// TopAllocAges(pid) and attaches the result to OutlierEvent.
//
// The tracker is engine-agnostic: any process that issues
// cudaMalloc / cudaFree will be tracked, not just inference engines.
// The Engine gates the consumption to "phase=decode outliers" so we
// don't pay the lookup cost on prefill or unknown.
package kvcache

import (
	"container/list"
	"sort"
	"sync"
	"time"
)

// DefaultMaxAllocsPerPID bounds the live-allocation map per PID. Real
// inference engines (vLLM, TGI) hold thousands of KV blocks live at
// peak; 8192 keeps full coverage for typical 70B-and-below serving
// without unbounded growth. Older entries are LRU-evicted on insert
// so the cap is hard.
const DefaultMaxAllocsPerPID = 8192

// allocation is one tracked malloc. Order in the LRU list is keyed
// off insertion time (oldest at the back), not access time, because
// the metric we want (oldest live allocation) is creation-time
// ordered, not lookup-time.
type allocation struct {
	pid     uint32
	devPtr  uint64
	size    uint64
	allocAt time.Time
	elem    *list.Element
}

// pidState owns one PID's live-allocation map. Internal-only;
// Tracker fans events out to the right pidState by pid.
type pidState struct {
	allocs map[uint64]*allocation // devPtr -> alloc
	order  *list.List             // front = newest, back = oldest
}

// Tracker is the cross-PID lineage store. Concurrency: a single
// mutex guards the per-PID map; the per-PID list/map are accessed
// only inside that lock. Producers (OnMalloc / OnFree) and
// consumers (TopAllocAges) all serialize on the same mutex - cheap
// because malloc/free events on the inference path are sub-1k Hz
// even at peak.
type Tracker struct {
	mu sync.Mutex

	// maxAllocsPerPID is the cap copied from Config.MaxAllocsPerPID
	// at construction time so the hot path doesn't re-read config.
	maxAllocsPerPID int

	pids map[uint32]*pidState
}

// Config tunes the Tracker. Zero-value MaxAllocsPerPID resolves to
// DefaultMaxAllocsPerPID.
type Config struct {
	MaxAllocsPerPID int
}

// New constructs an empty Tracker.
func New(cfg Config) *Tracker {
	if cfg.MaxAllocsPerPID <= 0 {
		cfg.MaxAllocsPerPID = DefaultMaxAllocsPerPID
	}
	return &Tracker{
		maxAllocsPerPID: cfg.MaxAllocsPerPID,
		pids:            make(map[uint32]*pidState),
	}
}

// OnMalloc records a successful cudaMalloc / cudaMallocManaged.
// devPtr=0 is silently dropped (cudaMalloc(NULL,...) on failure path,
// or returned-zero pointer indicating allocation failed). size is
// recorded for future use (a follow-up may surface "oldest big alloc"
// rather than "oldest alloc" but the current emit path only sends ages).
func (t *Tracker) OnMalloc(pid uint32, devPtr, size uint64, at time.Time) {
	if devPtr == 0 {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	ps := t.getOrCreatePIDLocked(pid)

	// Replace any stale entry on the same devPtr (driver could reuse
	// a freed pointer; if our cudaFree event was dropped we'd
	// otherwise leak the old entry).
	if existing, ok := ps.allocs[devPtr]; ok {
		ps.order.Remove(existing.elem)
		delete(ps.allocs, devPtr)
	}

	a := &allocation{
		pid:     pid,
		devPtr:  devPtr,
		size:    size,
		allocAt: at,
	}
	a.elem = ps.order.PushFront(a)
	ps.allocs[devPtr] = a

	// Cap. Evict from the back (oldest) until we're at-or-below the
	// limit. In steady state this is a one-element eviction; under a
	// burst it amortizes O(1).
	for ps.order.Len() > t.maxAllocsPerPID {
		oldest := ps.order.Back()
		if oldest == nil {
			break
		}
		victim := oldest.Value.(*allocation)
		ps.order.Remove(oldest)
		delete(ps.allocs, victim.devPtr)
	}
}

// OnFree records a cudaFree. Missing entry (cudaFree without a
// matching tracked cudaMalloc) is silently ignored - happens
// regularly because (a) startup-time allocations may have been
// LRU-evicted and (b) the agent may have started after the workload.
func (t *Tracker) OnFree(pid uint32, devPtr uint64, at time.Time) {
	_ = at
	if devPtr == 0 {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()

	ps, ok := t.pids[pid]
	if !ok {
		return
	}
	a, ok := ps.allocs[devPtr]
	if !ok {
		return
	}
	ps.order.Remove(a.elem)
	delete(ps.allocs, devPtr)
}

// OnProcessExit clears all tracked allocations for the PID. Called
// when the agent observes the process leaving (sched_process_exit
// or trace's PID-death detector). Bounds memory across long-running
// agent sessions where many short-lived inference processes have
// come and gone.
func (t *Tracker) OnProcessExit(pid uint32) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.pids, pid)
}

// TopAllocAgesMs returns up to n alloc ages in milliseconds, sorted
// oldest-first, for the given PID. Returns nil for an unknown PID
// or one with no live allocations. Cap at n is hard; older entries
// past n are dropped.
//
// Computed at call time (now - alloc_at) so the consumer chooses the
// reference timestamp - the engine uses the outlier event's
// timestamp to keep the age semantically anchored to the slow step.
func (t *Tracker) TopAllocAgesMs(pid uint32, n int, now time.Time) []uint64 {
	if n <= 0 {
		return nil
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	ps, ok := t.pids[pid]
	if !ok || ps.order.Len() == 0 {
		return nil
	}

	// Walk the LRU from the back (oldest) forward, collecting up to
	// n ages.
	out := make([]uint64, 0, n)
	for e := ps.order.Back(); e != nil && len(out) < n; e = e.Prev() {
		a := e.Value.(*allocation)
		age := now.Sub(a.allocAt)
		if age < 0 {
			age = 0
		}
		out = append(out, uint64(age/time.Millisecond))
	}

	// Defensive sort: list order should already be oldest-first
	// (oldest at back, walking back→front), but a future change to
	// the LRU policy might break that invariant. Cheap (n is
	// typically <= 5).
	sort.Slice(out, func(i, j int) bool { return out[i] > out[j] })
	return out
}

// LiveAllocations returns the current live count for a PID. Test
// helper; the production hot path uses TopAllocAgesMs.
func (t *Tracker) LiveAllocations(pid uint32) int {
	t.mu.Lock()
	defer t.mu.Unlock()
	ps, ok := t.pids[pid]
	if !ok {
		return 0
	}
	return ps.order.Len()
}

func (t *Tracker) getOrCreatePIDLocked(pid uint32) *pidState {
	if ps, ok := t.pids[pid]; ok {
		return ps
	}
	ps := &pidState{
		allocs: make(map[uint64]*allocation),
		order:  list.New(),
	}
	t.pids[pid] = ps
	return ps
}
