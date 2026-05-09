package stats

import (
	"math"
	"sort"
	"sync"
)

// Histogram is a fixed-bucket cumulative histogram with the OTLP
// explicit-bounds shape. Concurrent Observe calls are safe (mu).
//
// v0.15 item B/C: producer-side aggregation feeding the OTLP encoder.
// Each instance owns a per-attribute-set time series; Snapshot()
// returns a frozen view + an OTLP-compatible bucket layout.
//
// The bucket layout follows the OTLP convention: ExplicitBounds is
// strictly-increasing; BucketCounts has len(ExplicitBounds)+1 entries
// (the last bucket is the +inf overflow).
type Histogram struct {
	mu             sync.Mutex
	bounds         []float64 // strictly-increasing
	bucketCounts   []uint64  // len = len(bounds) + 1
	count          uint64
	sum            float64
	min, max       float64
	hasObservation bool
}

// NewHistogram returns a Histogram with the given strictly-increasing
// bucket boundaries. Caller is responsible for choosing bounds that
// match the metric's expected distribution. Empty bounds means a
// "summary-only" histogram (count + sum + min + max only); valid but
// unusual. Duplicate or out-of-order bounds panic.
func NewHistogram(bounds []float64) *Histogram {
	for i := 1; i < len(bounds); i++ {
		if bounds[i] <= bounds[i-1] {
			panic("stats.NewHistogram: bounds must be strictly increasing")
		}
	}
	cp := make([]float64, len(bounds))
	copy(cp, bounds)
	return &Histogram{
		bounds:       cp,
		bucketCounts: make([]uint64, len(cp)+1),
	}
}

// Observe records one value.
func (h *Histogram) Observe(v float64) {
	h.mu.Lock()
	defer h.mu.Unlock()
	idx := sort.SearchFloat64s(h.bounds, v)
	// SearchFloat64s returns smallest i such that bounds[i] >= v.
	// We want bucket idx where v <= bounds[idx]. Behavior:
	//   - v < bounds[0]: idx = 0 (first bucket)
	//   - bounds[i-1] < v <= bounds[i]: idx = i
	//   - v > bounds[last]: idx = len(bounds) (overflow bucket)
	// SearchFloat64s gives first i with bounds[i] >= v, so when
	// bounds[idx] == v we want this same idx; when bounds[idx] > v
	// we also want this idx. Both behaviors produce the same result.
	h.bucketCounts[idx]++
	h.count++
	h.sum += v
	if !h.hasObservation {
		h.min = v
		h.max = v
		h.hasObservation = true
	} else {
		if v < h.min {
			h.min = v
		}
		if v > h.max {
			h.max = v
		}
	}
}

// HistogramSnapshot is the frozen view used by exporters.
type HistogramSnapshot struct {
	Count          uint64
	Sum            float64
	Min, Max       float64
	HasObservation bool
	BucketCounts   []uint64  // len = len(ExplicitBounds) + 1
	ExplicitBounds []float64
}

// Snapshot returns a copy of the histogram's current state.
func (h *Histogram) Snapshot() HistogramSnapshot {
	h.mu.Lock()
	defer h.mu.Unlock()
	bounds := make([]float64, len(h.bounds))
	copy(bounds, h.bounds)
	counts := make([]uint64, len(h.bucketCounts))
	copy(counts, h.bucketCounts)
	return HistogramSnapshot{
		Count:          h.count,
		Sum:            h.sum,
		Min:            h.min,
		Max:            h.max,
		HasObservation: h.hasObservation,
		BucketCounts:   counts,
		ExplicitBounds: bounds,
	}
}

// Reset zeroes the histogram. Useful for delta-temporality emission.
func (h *Histogram) Reset() {
	h.mu.Lock()
	for i := range h.bucketCounts {
		h.bucketCounts[i] = 0
	}
	h.count = 0
	h.sum = 0
	h.min = 0
	h.max = 0
	h.hasObservation = false
	h.mu.Unlock()
}

// DefaultMemcpyDurationBoundsMs is the bucket layout for
// gpu.memcpy.duration_ms. Geometric progression covering host->dev
// and dev->dev memcpy operations from sub-millisecond to seconds.
// Calibrated against v0.12 hardware-validation traces (A10/H100/GH200)
// where p50 is typically 0.5-5 ms and p99 stretches to ~200 ms under
// fan-out load.
var DefaultMemcpyDurationBoundsMs = []float64{
	0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 25.0, 50.0, 100.0, 250.0, 500.0, 1000.0,
}

// IsFinite reports whether v is a real number (not NaN, not +/-Inf).
// Histogram callers should drop non-finite observations rather than
// poison the running sum + min/max.
func IsFinite(v float64) bool {
	return !math.IsNaN(v) && !math.IsInf(v, 0)
}

// DefaultInferStepDurationBoundsNs is the bucket layout for
// ingero.infer.step_duration_ns. Geometric progression covering
// inference engine iterations from sub-millisecond decodes to
// multi-second stalls. Calibrated against the v0.16.1 phase classifier
// regimes (vLLM/TGI/SGLang serving): decode p50 ~5 ms, prefill p95
// ~500 ms on 70B serving, and a long tail to 10 s for stuck steps.
//
// Why nanoseconds, not milliseconds: the metric name and the OTel
// histogram both carry the "ns" unit, so consumers don't need a
// conversion step. Bucket widths span 100us (sub-decode) to 10s
// (deep stall) so a single histogram covers every realistic regime.
var DefaultInferStepDurationBoundsNs = []float64{
	100_000, 250_000, 500_000,
	1_000_000, 2_500_000, 5_000_000,
	10_000_000, 25_000_000, 50_000_000,
	100_000_000, 250_000_000, 500_000_000,
	1_000_000_000, 2_500_000_000, 5_000_000_000,
	10_000_000_000,
}
