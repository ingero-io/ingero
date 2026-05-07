package nvml

import "sync"

// ThrottleEdgeDetector turns the per-poll bitmask into per-bucket
// transition counts. Each rising edge (bucket goes from inactive
// to active between two polls) increments the corresponding
// counter exactly once.
//
// v0.15 item L: replaces the v0.12.10 per-window gauge with an
// event-count surface. Honest framing: this still rests on the
// underlying nvidia-smi poll, so a throttle burst entirely within
// a single poll interval is still missed (same floor as v0.12.10;
// documented in CHANGELOG). The detector is added on top of the
// poll, NOT a replacement for the gauge.
//
// Why edge detection and not real kernel-event hooking: the NVIDIA
// closed driver does not publicly name a throttle-state-change
// kprobe target. The cgo-NVML alternative (nvmlEventSetWait) would
// add a libnvidia-ml.so runtime dependency and a cgo build path
// that the agent has explicitly avoided since v0.10. Edge detection
// is the honest tradeoff: meaningfully better operator surface,
// no new dependencies, no false claims about sub-poll precision.
type ThrottleEdgeDetector struct {
	mu       sync.Mutex
	prev     map[string]ThrottleBuckets // keyed by GPU UUID
	counters ThrottleEventCounters
}

// ThrottleEventCounters carries per-bucket cumulative event counts.
type ThrottleEventCounters struct {
	PowerEvents   int64
	ThermalEvents int64
	SWEvents      int64
	HWEvents      int64
}

// NewThrottleEdgeDetector returns an empty detector.
func NewThrottleEdgeDetector() *ThrottleEdgeDetector {
	return &ThrottleEdgeDetector{prev: map[string]ThrottleBuckets{}}
}

// Observe records this poll's bucket state and increments any
// rising-edge counters. uuid identifies the GPU; b is the decoded
// bucket state (output of DecodeReasons).
func (d *ThrottleEdgeDetector) Observe(uuid string, b ThrottleBuckets) {
	d.mu.Lock()
	defer d.mu.Unlock()
	prev, seen := d.prev[uuid]
	if seen {
		if b.Power && !prev.Power {
			d.counters.PowerEvents++
		}
		if b.Thermal && !prev.Thermal {
			d.counters.ThermalEvents++
		}
		if b.SW && !prev.SW {
			d.counters.SWEvents++
		}
		if b.HW && !prev.HW {
			d.counters.HWEvents++
		}
	}
	d.prev[uuid] = b
}

// Snapshot returns the current cumulative counters.
func (d *ThrottleEdgeDetector) Snapshot() ThrottleEventCounters {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.counters
}

// Reset zeroes all counters. Test-only helper.
func (d *ThrottleEdgeDetector) Reset() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.prev = map[string]ThrottleBuckets{}
	d.counters = ThrottleEventCounters{}
}
