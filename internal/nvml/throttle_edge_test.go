package nvml

import (
	"sync"
	"testing"
)

func TestThrottleEdgeDetector_RisingEdgeIncrements(t *testing.T) {
	d := NewThrottleEdgeDetector()
	d.Observe("GPU-A", ThrottleBuckets{}) // baseline: no throttle
	d.Observe("GPU-A", ThrottleBuckets{Power: true})
	d.Observe("GPU-A", ThrottleBuckets{Power: true}) // sustained, no new event
	d.Observe("GPU-A", ThrottleBuckets{Power: true, Thermal: true})

	got := d.Snapshot()
	if got.PowerEvents != 1 {
		t.Errorf("PowerEvents=%d want 1", got.PowerEvents)
	}
	if got.ThermalEvents != 1 {
		t.Errorf("ThermalEvents=%d want 1 (rising edge from prev=power-only)", got.ThermalEvents)
	}
}

func TestThrottleEdgeDetector_FallingEdgeAndReentrant(t *testing.T) {
	d := NewThrottleEdgeDetector()
	d.Observe("GPU-A", ThrottleBuckets{})
	d.Observe("GPU-A", ThrottleBuckets{Power: true})
	d.Observe("GPU-A", ThrottleBuckets{}) // falling edge: not counted
	d.Observe("GPU-A", ThrottleBuckets{Power: true})
	d.Observe("GPU-A", ThrottleBuckets{})

	got := d.Snapshot()
	if got.PowerEvents != 2 {
		t.Errorf("two power-active periods should produce 2 events; got %d", got.PowerEvents)
	}
}

func TestThrottleEdgeDetector_FirstObservationDoesNotIncrement(t *testing.T) {
	// A GPU's very first poll IS itself a rising edge (we have no prior
	// state to compare against). To avoid lying about how many events
	// happened during agent startup, the detector only counts edges on
	// observations 2+.
	d := NewThrottleEdgeDetector()
	d.Observe("GPU-A", ThrottleBuckets{Power: true})
	got := d.Snapshot()
	if got.PowerEvents != 0 {
		t.Errorf("first observation should not register an edge; got %d", got.PowerEvents)
	}
}

func TestThrottleEdgeDetector_PerGPUIndependence(t *testing.T) {
	d := NewThrottleEdgeDetector()
	d.Observe("GPU-A", ThrottleBuckets{})
	d.Observe("GPU-B", ThrottleBuckets{})
	d.Observe("GPU-A", ThrottleBuckets{Power: true})
	d.Observe("GPU-B", ThrottleBuckets{Thermal: true})
	got := d.Snapshot()
	if got.PowerEvents != 1 || got.ThermalEvents != 1 {
		t.Errorf("per-GPU edges should both fire; got %+v", got)
	}
}

func TestThrottleEdgeDetector_Reset(t *testing.T) {
	d := NewThrottleEdgeDetector()
	d.Observe("GPU-A", ThrottleBuckets{})
	d.Observe("GPU-A", ThrottleBuckets{Power: true})
	d.Reset()
	got := d.Snapshot()
	if got.PowerEvents != 0 {
		t.Errorf("Reset should zero counters; got %+v", got)
	}
}

func TestThrottleEdgeDetector_ConcurrentObserve(t *testing.T) {
	d := NewThrottleEdgeDetector()
	var wg sync.WaitGroup
	for g := 0; g < 4; g++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			uuid := "GPU-X"
			for i := 0; i < 100; i++ {
				if i%2 == 0 {
					d.Observe(uuid, ThrottleBuckets{Power: true})
				} else {
					d.Observe(uuid, ThrottleBuckets{})
				}
			}
		}(g)
	}
	wg.Wait()
	// We don't assert exact count under contention (interleavings
	// vary); just assert no race + nonzero result.
	got := d.Snapshot()
	if got.PowerEvents == 0 {
		t.Errorf("expected nonzero PowerEvents under concurrent observe; got 0")
	}
}
