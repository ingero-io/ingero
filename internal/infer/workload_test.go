package infer

import (
	"sync"
	"testing"
	"time"
)

func TestWorkloadMap_GetOrCreate_Reuses(t *testing.T) {
	m := newWorkloadMap(8)
	k := WorkloadKey{CGroupHash: "abc", PID: 1, StreamHandle: 0xff}
	now := time.Now()
	a := m.GetOrCreate(k, now)
	b := m.GetOrCreate(k, now)
	if a != b {
		t.Error("GetOrCreate should return the same baseliner pointer for the same key")
	}
}

func TestWorkloadMap_LRUEvictsLeastRecentlyUpdated(t *testing.T) {
	m := newWorkloadMap(2)
	t0 := time.Now()
	k1 := WorkloadKey{CGroupHash: "a", PID: 1}
	k2 := WorkloadKey{CGroupHash: "a", PID: 2}
	k3 := WorkloadKey{CGroupHash: "a", PID: 3}
	m.GetOrCreate(k1, t0)
	m.GetOrCreate(k2, t0.Add(time.Second))
	if m.Len() != 2 {
		t.Fatalf("Len after 2 inserts = %d, want 2", m.Len())
	}
	// Refresh k1 so k2 becomes the least-recently-updated.
	m.GetOrCreate(k1, t0.Add(2*time.Second))
	// Insert k3 — should evict k2.
	m.GetOrCreate(k3, t0.Add(3*time.Second))

	if _, ok := m.Get(k1); !ok {
		t.Error("k1 should still be present (was refreshed)")
	}
	if _, ok := m.Get(k2); ok {
		t.Error("k2 should have been evicted")
	}
	if _, ok := m.Get(k3); !ok {
		t.Error("k3 should be present (just inserted)")
	}
	if !m.EvictedSinceLastClear() {
		t.Error("eviction flag should be set after evicting k2")
	}
}

func TestWorkloadMap_GetDoesNotBumpOrder(t *testing.T) {
	m := newWorkloadMap(2)
	t0 := time.Now()
	k1 := WorkloadKey{CGroupHash: "a", PID: 1}
	k2 := WorkloadKey{CGroupHash: "a", PID: 2}
	k3 := WorkloadKey{CGroupHash: "a", PID: 3}
	m.GetOrCreate(k1, t0)
	m.GetOrCreate(k2, t0.Add(time.Second))
	// Bare Get on k1 should NOT change LRU order.
	_, _ = m.Get(k1)
	// Inserting k3 should still evict k1 (not k2), because lastSync
	// on k1 is older than k2 and Get did not refresh it.
	m.GetOrCreate(k3, t0.Add(2*time.Second))
	if _, ok := m.Get(k1); ok {
		t.Error("k1 should have been evicted; Get should not bump order")
	}
}

func TestWorkloadMap_ClearEvictionFlag(t *testing.T) {
	m := newWorkloadMap(1)
	t0 := time.Now()
	k1 := WorkloadKey{PID: 1}
	k2 := WorkloadKey{PID: 2}
	m.GetOrCreate(k1, t0)
	m.GetOrCreate(k2, t0)
	if !m.EvictedSinceLastClear() {
		t.Fatal("eviction flag should be set")
	}
	m.ClearEvictionFlag()
	if m.EvictedSinceLastClear() {
		t.Error("eviction flag should be clear after ClearEvictionFlag")
	}
}

func TestWorkloadMap_SnapshotReturnsAllEntries(t *testing.T) {
	m := newWorkloadMap(8)
	now := time.Now()
	for pid := uint32(1); pid <= 5; pid++ {
		bl := m.GetOrCreate(WorkloadKey{PID: pid}, now)
		// Drive enough samples for warmup so Warmed reports correctly.
		for i := 0; i < 10; i++ {
			bl.Update(1_000_000)
		}
	}
	snap := m.Snapshot(5)
	if len(snap) != 5 {
		t.Errorf("snapshot len=%d, want 5", len(snap))
	}
	for _, s := range snap {
		if !s.Warmed {
			t.Errorf("snapshot for PID %d not warmed (samples=%d)", s.Key.PID, s.Samples)
		}
	}
}

func TestWorkloadMap_ConcurrentGetOrCreate(t *testing.T) {
	m := newWorkloadMap(64)
	var wg sync.WaitGroup
	now := time.Now()
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(w int) {
			defer wg.Done()
			for i := 0; i < 100; i++ {
				k := WorkloadKey{PID: uint32(w*100 + i)}
				m.GetOrCreate(k, now)
			}
		}(w)
	}
	wg.Wait()
	// 8 goroutines * 100 unique keys = 800 keys, but cap is 64 so we
	// should see exactly 64 with the rest evicted.
	if m.Len() != 64 {
		t.Errorf("Len after concurrent inserts = %d, want 64 (cap)", m.Len())
	}
}
