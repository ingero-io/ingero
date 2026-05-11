package kvcache

import (
	"testing"
	"time"
)

func TestTracker_MallocFreePair(t *testing.T) {
	tr := New(Config{})
	t0 := time.Now()
	tr.OnMalloc(7, 0xdead0001, 1024, t0)
	if got := tr.LiveAllocations(7); got != 1 {
		t.Errorf("after one malloc: live=%d, want 1", got)
	}
	tr.OnFree(7, 0xdead0001)
	if got := tr.LiveAllocations(7); got != 0 {
		t.Errorf("after matching free: live=%d, want 0", got)
	}
}

func TestTracker_FreeWithoutMallocIsNoop(t *testing.T) {
	tr := New(Config{})
	// Common production case: agent attached after the workload had
	// already allocated. cudaFree without a tracked malloc should
	// silently drop, not panic / not leak state.
	tr.OnFree(7, 0xdead0002)
	if got := tr.LiveAllocations(7); got != 0 {
		t.Errorf("free without malloc should be no-op, live=%d", got)
	}
}

func TestTracker_DevPtrZeroDropped(t *testing.T) {
	tr := New(Config{})
	t0 := time.Now()
	tr.OnMalloc(7, 0, 1024, t0)
	tr.OnFree(7, 0)
	if got := tr.LiveAllocations(7); got != 0 {
		t.Errorf("devPtr=0 should drop, live=%d", got)
	}
}

func TestTracker_DuplicateMallocReplaces(t *testing.T) {
	// If our cudaFree event was dropped (BPF ringbuf overflow), a
	// later cudaMalloc on the same devPtr should replace the stale
	// entry rather than leaking. The "stale" alloc is the older one
	// here.
	tr := New(Config{})
	t0 := time.Now()
	tr.OnMalloc(7, 0xdead0003, 100, t0)
	tr.OnMalloc(7, 0xdead0003, 200, t0.Add(50*time.Millisecond))

	if got := tr.LiveAllocations(7); got != 1 {
		t.Errorf("dup malloc should replace, live=%d, want 1", got)
	}
	ages := tr.TopAllocAgesMs(7, 1, t0.Add(60*time.Millisecond))
	if len(ages) != 1 || ages[0] != 10 {
		t.Errorf("expected age 10ms (60-50), got %v", ages)
	}
}

func TestTracker_TopAllocAgesOldestFirst(t *testing.T) {
	tr := New(Config{})
	t0 := time.Now()
	// Three allocs at increasing times.
	tr.OnMalloc(1, 0x1, 10, t0)                       // age 100ms at t=t0+100ms
	tr.OnMalloc(1, 0x2, 20, t0.Add(40*time.Millisecond))  // age 60ms
	tr.OnMalloc(1, 0x3, 30, t0.Add(80*time.Millisecond))  // age 20ms

	ages := tr.TopAllocAgesMs(1, 3, t0.Add(100*time.Millisecond))
	want := []uint64{100, 60, 20}
	if len(ages) != 3 {
		t.Fatalf("got %d ages, want 3 (got=%v)", len(ages), ages)
	}
	for i, w := range want {
		if ages[i] != w {
			t.Errorf("ages[%d] = %d, want %d (full=%v)", i, ages[i], w, ages)
		}
	}
}

func TestTracker_TopAllocAgesCappedAtN(t *testing.T) {
	tr := New(Config{})
	t0 := time.Now()
	for i := 0; i < 50; i++ {
		tr.OnMalloc(1, uint64(0x1000+i), 1, t0.Add(time.Duration(i)*time.Millisecond))
	}
	ages := tr.TopAllocAgesMs(1, 5, t0.Add(time.Second))
	if len(ages) != 5 {
		t.Errorf("Top(5) returned %d ages, want 5", len(ages))
	}
}

func TestTracker_LRUEvictsOldestOnOverflow(t *testing.T) {
	tr := New(Config{MaxAllocsPerPID: 3})
	t0 := time.Now()
	for i := 0; i < 5; i++ {
		tr.OnMalloc(1, uint64(0x1000+i), 1, t0.Add(time.Duration(i)*time.Millisecond))
	}
	if got := tr.LiveAllocations(1); got != 3 {
		t.Errorf("LRU cap = 3 but got %d live allocs", got)
	}
	// Oldest two (devPtr 0x1000, 0x1001) should be gone; surviving
	// ages should be [0x1002, 0x1003, 0x1004] = 2ms, 3ms, 4ms ago at t+5ms.
	ages := tr.TopAllocAgesMs(1, 5, t0.Add(5*time.Millisecond))
	if len(ages) != 3 {
		t.Fatalf("got %d ages, want 3", len(ages))
	}
	if ages[0] != 3 { // 0x1002 was at +2ms, age 3ms
		t.Errorf("ages[0]=%d, want 3 (oldest after eviction)", ages[0])
	}
}

func TestTracker_PIDIsolation(t *testing.T) {
	tr := New(Config{})
	t0 := time.Now()
	tr.OnMalloc(1, 0xa, 10, t0)
	tr.OnMalloc(2, 0xa, 20, t0)
	if got := tr.LiveAllocations(1); got != 1 {
		t.Errorf("PID 1 live=%d, want 1", got)
	}
	if got := tr.LiveAllocations(2); got != 1 {
		t.Errorf("PID 2 live=%d, want 1", got)
	}
	tr.OnFree(1, 0xa)
	if got := tr.LiveAllocations(1); got != 0 {
		t.Errorf("PID 1 free isolated, live=%d", got)
	}
	if got := tr.LiveAllocations(2); got != 1 {
		t.Errorf("PID 2 still has its alloc, live=%d", got)
	}
}

func TestTracker_OnProcessExitClearsState(t *testing.T) {
	tr := New(Config{})
	t0 := time.Now()
	for i := 0; i < 10; i++ {
		tr.OnMalloc(99, uint64(0x100+i), 1, t0)
	}
	tr.OnProcessExit(99)
	if got := tr.LiveAllocations(99); got != 0 {
		t.Errorf("after OnProcessExit, live=%d, want 0", got)
	}
	ages := tr.TopAllocAgesMs(99, 5, t0.Add(time.Second))
	if ages != nil {
		t.Errorf("after OnProcessExit, TopAllocAgesMs = %v, want nil", ages)
	}
}

func TestTracker_TopAllocAgesUnknownPID(t *testing.T) {
	tr := New(Config{})
	if got := tr.TopAllocAgesMs(42, 5, time.Now()); got != nil {
		t.Errorf("unknown PID should return nil, got %v", got)
	}
}

func TestTracker_TopAllocAgesNonPositiveN(t *testing.T) {
	tr := New(Config{})
	t0 := time.Now()
	tr.OnMalloc(1, 0x1, 1, t0)
	if got := tr.TopAllocAgesMs(1, 0, t0.Add(time.Second)); got != nil {
		t.Errorf("n=0 should return nil, got %v", got)
	}
	if got := tr.TopAllocAgesMs(1, -1, t0.Add(time.Second)); got != nil {
		t.Errorf("n=-1 should return nil, got %v", got)
	}
}
