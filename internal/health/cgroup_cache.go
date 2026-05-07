package health

import (
	"container/list"
	"crypto/sha256"
	"encoding/hex"
	"sync"

	"github.com/ingero-io/ingero/internal/cgroup"
)

// defaultCacheCapacity bounds the LRU at a value that comfortably covers
// the active PID set on a typical inference node (~10-100 PIDs) with
// headroom for short-lived helpers. Stale entries cost a few dozen bytes
// each and are evicted on demand; PID reuse within the cache lifetime is
// rare on a 10s push window.
const defaultCacheCapacity = 1024

// hashCGroupPath computes the SHA256-truncated-16 hex digest of a cgroup
// path. The agent's own emitter uses the same encoding for
// contract.AttrCgroupPathHash so per-cgroup metrics emitted by the
// per-cgroup collector and the legacy health_score data point share the
// hash space.
//
// CONFIDENTIALITY STATUS (v0.15 item D, folds v0.14 R3 ★3):
// This hash is a STABILITY tag for joining metric streams across the
// per-cgroup collector and the health_score push, NOT a confidentiality
// shield. SHA-256 of a known-shape input is reversible by an adversary
// with the cgroup path catalog (e.g., container-name conventions on a
// k8s node). If the cgroup path itself is sensitive in your deployment,
// disable cgroup tagging at the collector OR pass a per-cluster salt
// upstream of this function. We deliberately do NOT add a salt at this
// layer because (1) operators relying on the hash for stability would
// silently lose join keys across agent restarts unless the salt is
// persisted, and (2) the threat model treats cgroup paths as
// non-confidential by default.
func hashCGroupPath(path string) string {
	if path == "" {
		return ""
	}
	sum := sha256.Sum256([]byte(path))
	return hex.EncodeToString(sum[:])[:16]
}

type cacheEntry struct {
	pid  uint32
	hash string
	elem *list.Element
}

// cgroupCache is a bounded LRU PID -> cgroup_path_hash cache. The
// resolver runs once per cache miss; subsequent Get/Resolve for the
// same PID returns the cached hash without reading /proc. Empty hashes
// (resolution failure or root cgroup) are cached too so repeated misses
// don't re-stat /proc every push.
type cgroupCache struct {
	mu        sync.Mutex
	entries   map[uint32]*cacheEntry
	order     *list.List
	cap       int
	resolveFn func(pid uint32) (string, error)

	// evictedThisCycle gates the once-per-cycle WARN log so a flood of
	// short-lived PIDs doesn't spam the agent log. Reset by
	// ClearEvictionFlag, which the loop calls once per push tick.
	evictedThisCycle bool
}

// newCGroupCache constructs an empty LRU. capacity <= 0 falls back to
// defaultCacheCapacity. The default resolver is cgroup.ReadCGroupPath;
// tests inject a stub via the resolveFn field.
func newCGroupCache(capacity int) *cgroupCache {
	if capacity <= 0 {
		capacity = defaultCacheCapacity
	}
	return &cgroupCache{
		entries:   make(map[uint32]*cacheEntry, capacity),
		order:     list.New(),
		cap:       capacity,
		resolveFn: cgroup.ReadCGroupPath,
	}
}

// Get returns the cached hash for pid plus a cached flag. Cached hits
// move the PID to the front of the LRU. Misses do not allocate.
func (c *cgroupCache) Get(pid uint32) (string, bool) {
	c.mu.Lock()
	defer c.mu.Unlock()
	e, ok := c.entries[pid]
	if !ok {
		return "", false
	}
	c.order.MoveToFront(e.elem)
	return e.hash, true
}

// Resolve returns the cached hash for pid, computing it on a miss by
// calling the configured resolver. Resolver errors and empty paths are
// cached as the empty string so the same PID does not re-stat /proc on
// every push tick.
func (c *cgroupCache) Resolve(pid uint32) string {
	c.mu.Lock()
	if e, ok := c.entries[pid]; ok {
		c.order.MoveToFront(e.elem)
		hash := e.hash
		c.mu.Unlock()
		return hash
	}
	resolveFn := c.resolveFn
	c.mu.Unlock()

	var hash string
	if resolveFn != nil {
		path, err := resolveFn(pid)
		if err == nil {
			hash = hashCGroupPath(path)
		}
	}

	c.mu.Lock()
	defer c.mu.Unlock()
	// Re-check after dropping the lock for the syscall: another
	// goroutine may have populated the entry while we were resolving.
	if e, ok := c.entries[pid]; ok {
		c.order.MoveToFront(e.elem)
		return e.hash
	}
	for c.order.Len() >= c.cap {
		oldest := c.order.Back()
		if oldest == nil {
			break
		}
		oldEntry := oldest.Value.(*cacheEntry)
		c.order.Remove(oldest)
		delete(c.entries, oldEntry.pid)
		c.evictedThisCycle = true
	}
	entry := &cacheEntry{pid: pid, hash: hash}
	entry.elem = c.order.PushFront(entry)
	c.entries[pid] = entry
	return hash
}

// EvictedSinceLastClear returns whether at least one LRU eviction has
// happened since the last ClearEvictionFlag call. The loop reads this
// once per push tick to gate a single WARN log without flooding when a
// node is churning through PIDs.
func (c *cgroupCache) EvictedSinceLastClear() bool {
	c.mu.Lock()
	defer c.mu.Unlock()
	return c.evictedThisCycle
}

// ClearEvictionFlag resets the per-cycle eviction warning flag. Called
// by the loop after it has logged (or chosen not to log) the current
// cycle's eviction state.
func (c *cgroupCache) ClearEvictionFlag() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.evictedThisCycle = false
}

// Len returns the number of cached entries. Test-only helper.
func (c *cgroupCache) Len() int {
	c.mu.Lock()
	defer c.mu.Unlock()
	return len(c.entries)
}
