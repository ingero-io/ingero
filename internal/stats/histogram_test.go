package stats

import (
	"math"
	"sync"
	"testing"
)

func TestHistogram_BasicObservations(t *testing.T) {
	h := NewHistogram([]float64{1.0, 10.0, 100.0})
	h.Observe(0.5)   // bucket 0 (< 1)
	h.Observe(1.0)   // bucket 0 (<= 1)
	h.Observe(5.0)   // bucket 1
	h.Observe(50.0)  // bucket 2
	h.Observe(500.0) // bucket 3 (overflow)

	s := h.Snapshot()
	if s.Count != 5 {
		t.Errorf("Count=%d want 5", s.Count)
	}
	want := []uint64{2, 1, 1, 1}
	for i, c := range want {
		if s.BucketCounts[i] != c {
			t.Errorf("BucketCounts[%d]=%d want %d (full=%v)", i, s.BucketCounts[i], c, s.BucketCounts)
		}
	}
	if s.Sum != 0.5+1.0+5.0+50.0+500.0 {
		t.Errorf("Sum=%v want 556.5", s.Sum)
	}
	if s.Min != 0.5 || s.Max != 500.0 {
		t.Errorf("Min/Max = %v/%v want 0.5/500", s.Min, s.Max)
	}
}

func TestHistogram_BoundaryValues(t *testing.T) {
	// Confirm exact-bound observations land in the lower bucket.
	// SearchFloat64s returns first i with bounds[i] >= v, so for
	// v == bounds[i], idx = i (the bucket whose upper bound is v).
	h := NewHistogram([]float64{1.0, 10.0})
	h.Observe(1.0)
	h.Observe(10.0)
	s := h.Snapshot()
	if s.BucketCounts[0] != 1 {
		t.Errorf("v=1.0 should land in bucket 0; got counts=%v", s.BucketCounts)
	}
	if s.BucketCounts[1] != 1 {
		t.Errorf("v=10.0 should land in bucket 1; got counts=%v", s.BucketCounts)
	}
}

func TestHistogram_EmptySnapshot(t *testing.T) {
	h := NewHistogram([]float64{1.0, 10.0})
	s := h.Snapshot()
	if s.Count != 0 || s.HasObservation {
		t.Errorf("empty snapshot should be Count=0 HasObservation=false; got %+v", s)
	}
	if len(s.BucketCounts) != 3 {
		t.Errorf("BucketCounts len=%d want 3", len(s.BucketCounts))
	}
}

func TestHistogram_Reset(t *testing.T) {
	h := NewHistogram([]float64{1.0, 10.0})
	h.Observe(5.0)
	h.Observe(20.0)
	h.Reset()
	s := h.Snapshot()
	if s.Count != 0 || s.Sum != 0 || s.HasObservation {
		t.Errorf("post-reset: %+v want zero", s)
	}
}

func TestHistogram_PanicsOnBadBounds(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic on out-of-order bounds")
		}
	}()
	_ = NewHistogram([]float64{10.0, 1.0})
}

func TestHistogram_PanicsOnDuplicateBounds(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Errorf("expected panic on duplicate bounds")
		}
	}()
	_ = NewHistogram([]float64{1.0, 1.0, 10.0})
}

func TestHistogram_SnapshotIsCopy(t *testing.T) {
	h := NewHistogram([]float64{1.0})
	h.Observe(5.0)
	s1 := h.Snapshot()
	s1.BucketCounts[0] = 999
	s1.ExplicitBounds[0] = 999
	s2 := h.Snapshot()
	if s2.BucketCounts[0] == 999 || s2.ExplicitBounds[0] == 999 {
		t.Errorf("Snapshot must return a copy; mutation leaked into internal state")
	}
}

func TestHistogram_ConcurrentObserve(t *testing.T) {
	h := NewHistogram([]float64{1.0, 10.0, 100.0})
	const G = 8
	const N = 1000
	var wg sync.WaitGroup
	for g := 0; g < G; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			for i := 0; i < N; i++ {
				h.Observe(float64(i % 200))
			}
		}()
	}
	wg.Wait()
	s := h.Snapshot()
	if s.Count != G*N {
		t.Errorf("Count=%d want %d (lost updates under concurrent observe)", s.Count, G*N)
	}
	var totalBuckets uint64
	for _, b := range s.BucketCounts {
		totalBuckets += b
	}
	if totalBuckets != s.Count {
		t.Errorf("sum of buckets %d != Count %d", totalBuckets, s.Count)
	}
}

func TestIsFinite(t *testing.T) {
	if !IsFinite(0) || !IsFinite(1.5) || !IsFinite(-100) {
		t.Errorf("real numbers should be finite")
	}
	if IsFinite(math.NaN()) || IsFinite(math.Inf(1)) || IsFinite(math.Inf(-1)) {
		t.Errorf("NaN/Inf must be reported non-finite")
	}
}

func TestDefaultMemcpyDurationBoundsMs_StrictlyIncreasing(t *testing.T) {
	for i := 1; i < len(DefaultMemcpyDurationBoundsMs); i++ {
		if DefaultMemcpyDurationBoundsMs[i] <= DefaultMemcpyDurationBoundsMs[i-1] {
			t.Errorf("bounds not strictly increasing at index %d: %v <= %v",
				i, DefaultMemcpyDurationBoundsMs[i], DefaultMemcpyDurationBoundsMs[i-1])
		}
	}
	// Sanity: covers sub-ms to seconds range.
	if DefaultMemcpyDurationBoundsMs[0] >= 1.0 {
		t.Errorf("first bound %v should be sub-millisecond", DefaultMemcpyDurationBoundsMs[0])
	}
	last := DefaultMemcpyDurationBoundsMs[len(DefaultMemcpyDurationBoundsMs)-1]
	if last < 500.0 {
		t.Errorf("last bound %v should reach seconds range", last)
	}
}
