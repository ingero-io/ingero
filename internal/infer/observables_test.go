package infer

import (
	"sync"
	"testing"
	"time"
)

func TestStepObservables_FirstResetReturnsZero(t *testing.T) {
	s := newStepObservables()
	k := observableKey{CGroupHash: "abc", PID: 1, StreamHandle: 0xff}
	got := s.ResetAndRead(k, time.Now())
	zero := observableCounters{}
	got.LastUpdate = time.Time{} // exclude LastUpdate from comparison
	if got != zero {
		t.Errorf("first ResetAndRead should return zero, got %+v", got)
	}
}

func TestStepObservables_AccumulatesAndResets(t *testing.T) {
	s := newStepObservables()
	k := observableKey{CGroupHash: "abc", PID: 1, StreamHandle: 0xff}
	now := time.Now()

	s.AddLaunch(k, 100*time.Microsecond, now)
	s.AddLaunch(k, 200*time.Microsecond, now)
	s.AddMemcpy(k, 1024, now)
	s.AddMemcpy(k, 2048, now)
	s.AddNCCL(k, now)

	got := s.ResetAndRead(k, now)
	if got.LaunchCount != 2 {
		t.Errorf("LaunchCount = %d, want 2", got.LaunchCount)
	}
	if got.TotalKernelNs != 300*time.Microsecond {
		t.Errorf("TotalKernelNs = %v, want 300us", got.TotalKernelNs)
	}
	if got.MemcpyBytes != 3072 {
		t.Errorf("MemcpyBytes = %d, want 3072", got.MemcpyBytes)
	}
	if got.NCCLCount != 1 {
		t.Errorf("NCCLCount = %d, want 1", got.NCCLCount)
	}

	// After reset, fresh read should return zero on the same key.
	got2 := s.ResetAndRead(k, now)
	got2.LastUpdate = time.Time{}
	if got2 != (observableCounters{}) {
		t.Errorf("second ResetAndRead should return zero, got %+v", got2)
	}
}

func TestStepObservables_PerWorkloadIsolation(t *testing.T) {
	s := newStepObservables()
	now := time.Now()
	a := observableKey{CGroupHash: "a", PID: 1}
	b := observableKey{CGroupHash: "b", PID: 2}

	s.AddLaunch(a, 100*time.Microsecond, now)
	s.AddMemcpy(b, 4096, now)

	got := s.Peek(a)
	if got.LaunchCount != 1 || got.MemcpyBytes != 0 {
		t.Errorf("a: %+v, want LaunchCount=1, MemcpyBytes=0", got)
	}
	got = s.Peek(b)
	if got.LaunchCount != 0 || got.MemcpyBytes != 4096 {
		t.Errorf("b: %+v, want LaunchCount=0, MemcpyBytes=4096", got)
	}
}

func TestStepObservables_RejectsZeroOrNegativeMemcpy(t *testing.T) {
	s := newStepObservables()
	k := observableKey{PID: 1}
	now := time.Now()
	s.AddMemcpy(k, 0, now)
	s.AddMemcpy(k, -10, now)
	if got := s.Peek(k); got.MemcpyBytes != 0 {
		t.Errorf("zero/neg memcpy should not accumulate, got %d", got.MemcpyBytes)
	}
}

func TestStepObservables_PruneStale(t *testing.T) {
	s := newStepObservables()
	t0 := time.Now()
	s.AddLaunch(observableKey{PID: 1}, 100*time.Microsecond, t0)
	s.AddLaunch(observableKey{PID: 2}, 100*time.Microsecond, t0.Add(2*time.Minute))
	s.AddLaunch(observableKey{PID: 3}, 100*time.Microsecond, t0.Add(7*time.Minute))

	pruned := s.PruneStale(t0.Add(8*time.Minute), 5*time.Minute)
	if pruned != 2 {
		t.Errorf("prune count = %d, want 2 (PID 1 and 2 stale)", pruned)
	}
	if s.Len() != 1 {
		t.Errorf("post-prune len = %d, want 1", s.Len())
	}
}

func TestStepObservables_ConcurrentAdds(t *testing.T) {
	s := newStepObservables()
	now := time.Now()
	const goroutines = 8
	const iters = 1000

	var wg sync.WaitGroup
	for g := 0; g < goroutines; g++ {
		wg.Add(1)
		go func(g int) {
			defer wg.Done()
			k := observableKey{PID: uint32(g)}
			for i := 0; i < iters; i++ {
				s.AddLaunch(k, 1*time.Microsecond, now)
				s.AddMemcpy(k, 1, now)
			}
		}(g)
	}
	wg.Wait()

	for g := 0; g < goroutines; g++ {
		k := observableKey{PID: uint32(g)}
		got := s.Peek(k)
		if got.LaunchCount != iters {
			t.Errorf("PID %d: LaunchCount = %d, want %d", g, got.LaunchCount, iters)
		}
		if got.MemcpyBytes != int64(iters) {
			t.Errorf("PID %d: MemcpyBytes = %d, want %d", g, got.MemcpyBytes, iters)
		}
	}
}

func TestStepObservables_LastUpdateRefreshed(t *testing.T) {
	s := newStepObservables()
	k := observableKey{PID: 1}
	t0 := time.Now()
	t1 := t0.Add(10 * time.Second)
	s.AddLaunch(k, 100*time.Microsecond, t0)
	s.AddLaunch(k, 100*time.Microsecond, t1)
	got := s.Peek(k)
	if !got.LastUpdate.Equal(t1) {
		t.Errorf("LastUpdate = %v, want %v", got.LastUpdate, t1)
	}
}
