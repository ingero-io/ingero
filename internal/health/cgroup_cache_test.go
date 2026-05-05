package health

import (
	"errors"
	"fmt"
	"sync"
	"sync/atomic"
	"testing"
)

// TestCGroupCache_DefaultCapacity verifies that a non-positive capacity
// argument falls back to the package default. The constructor must not
// produce a degenerate zero-cap cache that evicts every insert.
func TestCGroupCache_DefaultCapacity(t *testing.T) {
	c := newCGroupCache(0)
	if c.cap != defaultCacheCapacity {
		t.Fatalf("cap = %d, want %d", c.cap, defaultCacheCapacity)
	}
	c2 := newCGroupCache(-5)
	if c2.cap != defaultCacheCapacity {
		t.Fatalf("cap (negative) = %d, want %d", c2.cap, defaultCacheCapacity)
	}
}

// TestCGroupCache_HitCachesResolver verifies that a second Resolve for
// the same PID does NOT re-call the resolver. This guards the hot-path
// invariant: the cache exists precisely so the agent does not hit /proc
// once per event when the same handful of PIDs dominate the window.
func TestCGroupCache_HitCachesResolver(t *testing.T) {
	var calls atomic.Int32
	c := newCGroupCache(16)
	c.resolveFn = func(pid uint32) (string, error) {
		calls.Add(1)
		return fmt.Sprintf("/cg/%d", pid), nil
	}

	first := c.Resolve(42)
	second := c.Resolve(42)
	if first != second {
		t.Fatalf("hit returned different hash: %q vs %q", first, second)
	}
	if got := calls.Load(); got != 1 {
		t.Fatalf("resolver called %d times, want 1 (cache miss should run it once)", got)
	}

	got, ok := c.Get(42)
	if !ok {
		t.Fatal("Get reported cache miss after Resolve")
	}
	if got != first {
		t.Fatalf("Get returned %q, want %q", got, first)
	}
}

// TestCGroupCache_MissResolves verifies the hash returned on a miss is
// the SHA256-truncated-16 of the resolver path.
func TestCGroupCache_MissResolves(t *testing.T) {
	c := newCGroupCache(8)
	c.resolveFn = func(pid uint32) (string, error) {
		return "/kubepods.slice/pod-abc", nil
	}
	got := c.Resolve(7)
	want := hashCGroupPath("/kubepods.slice/pod-abc")
	if got != want {
		t.Fatalf("Resolve hash = %q, want %q", got, want)
	}
	if len(got) != 16 {
		t.Fatalf("hash length = %d, want 16", len(got))
	}
}

// TestCGroupCache_LRUEvictsOldest verifies the LRU ordering: with cap=2
// after resolving PIDs 1, 2, 3, PID 1 must be evicted (least recently
// used), and the eviction flag must be set.
func TestCGroupCache_LRUEvictsOldest(t *testing.T) {
	c := newCGroupCache(2)
	c.resolveFn = func(pid uint32) (string, error) {
		return fmt.Sprintf("/cg/%d", pid), nil
	}
	c.Resolve(1)
	c.Resolve(2)
	c.Resolve(3)

	if _, ok := c.Get(1); ok {
		t.Fatal("PID 1 should have been evicted")
	}
	if _, ok := c.Get(2); !ok {
		t.Fatal("PID 2 should still be cached")
	}
	if _, ok := c.Get(3); !ok {
		t.Fatal("PID 3 should still be cached")
	}
	if !c.EvictedSinceLastClear() {
		t.Fatal("eviction flag not set after over-cap insert")
	}
	c.ClearEvictionFlag()
	if c.EvictedSinceLastClear() {
		t.Fatal("eviction flag not cleared")
	}
}

// TestCGroupCache_LRUMoveToFront ensures Get on the oldest entry promotes
// it, so the next eviction takes the next-oldest instead. Without this
// promotion the LRU degrades to FIFO and frequently-touched PIDs would
// thrash.
func TestCGroupCache_LRUMoveToFront(t *testing.T) {
	c := newCGroupCache(2)
	c.resolveFn = func(pid uint32) (string, error) {
		return fmt.Sprintf("/cg/%d", pid), nil
	}
	c.Resolve(1)
	c.Resolve(2)
	if _, ok := c.Get(1); !ok {
		t.Fatal("PID 1 should be cached before promotion")
	}
	c.Resolve(3)

	if _, ok := c.Get(2); ok {
		t.Fatal("PID 2 (oldest after Get(1) promoted 1) should have been evicted")
	}
	if _, ok := c.Get(1); !ok {
		t.Fatal("PID 1 should still be cached (Get promoted it)")
	}
}

// TestCGroupCache_CachesEmptyOnError verifies that a resolver error
// caches the empty string for that PID. A subsequent Resolve must NOT
// re-invoke the resolver.
func TestCGroupCache_CachesEmptyOnError(t *testing.T) {
	var calls atomic.Int32
	c := newCGroupCache(8)
	c.resolveFn = func(pid uint32) (string, error) {
		calls.Add(1)
		return "", errors.New("simulated /proc unavailable")
	}
	first := c.Resolve(99)
	if first != "" {
		t.Fatalf("error path returned %q, want empty", first)
	}
	c.Resolve(99)
	if got := calls.Load(); got != 1 {
		t.Fatalf("resolver called %d times on cached error, want 1", got)
	}
	if _, ok := c.Get(99); !ok {
		t.Fatal("error result not cached")
	}
}

// TestCGroupCache_CachesEmptyOnEmptyPath verifies that a resolver
// returning an empty path (root cgroup / host process) caches the
// empty string and skips re-resolution.
func TestCGroupCache_CachesEmptyOnEmptyPath(t *testing.T) {
	var calls atomic.Int32
	c := newCGroupCache(8)
	c.resolveFn = func(pid uint32) (string, error) {
		calls.Add(1)
		return "", nil
	}
	c.Resolve(7)
	c.Resolve(7)
	if got := calls.Load(); got != 1 {
		t.Fatalf("resolver called %d times, want 1", got)
	}
	got, ok := c.Get(7)
	if !ok {
		t.Fatal("empty-path result not cached")
	}
	if got != "" {
		t.Fatalf("empty-path hash = %q, want empty", got)
	}
}

// TestCGroupCache_ConcurrentGet stresses the cache under concurrent
// access. Run with -race to detect data races on the LRU list, the
// entries map, and the eviction flag.
func TestCGroupCache_ConcurrentGet(t *testing.T) {
	c := newCGroupCache(64)
	c.resolveFn = func(pid uint32) (string, error) {
		return fmt.Sprintf("/cg/%d", pid), nil
	}
	const goroutines = 100
	const iters = 50
	var wg sync.WaitGroup
	wg.Add(goroutines)
	for g := 0; g < goroutines; g++ {
		go func(seed int) {
			defer wg.Done()
			for i := 0; i < iters; i++ {
				pid := uint32((seed*iters + i) % 200)
				_ = c.Resolve(pid)
				_, _ = c.Get(pid)
			}
		}(g)
	}
	wg.Wait()
}
