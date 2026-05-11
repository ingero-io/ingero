// Package infer establishes a per-workload step-duration baseline for
// inference workloads and classifies each step against it. A "step" is
// the wall-clock interval between consecutive cudaStreamSynchronize
// (or cudaDeviceSynchronize / driver ctxSynchronize) events on the
// same (pid, stream_handle); for continuous-batching servers (vLLM,
// SGLang, TGI) each step is one engine iteration, not a user-facing
// HTTP request.
//
// The package is consumed by the `ingero trace --inference` umbrella;
// the engine sits passively on the event stream and emits outlier
// events through OTLP histograms + the existing FOSS UDS socket so
// any consumer (including the EE orchestrator) can react.
package infer

import (
	"github.com/ingero-io/ingero/internal/health"
	"github.com/ingero-io/ingero/internal/stats"
)

// emaAlphaMean is the smoothing factor for the per-workload mean. We
// pick a faster alpha (0.05) than the node-level health Baseliner
// (0.001 floor / 0.1 fast) because workloads churn faster than the
// node-aggregate signals — a model deployment can change its step
// shape inside seconds, and a slow EMA would flag every deploy as a
// regression.
const emaAlphaMean = 0.05

// p95Quantile is fixed for v0.16. A future revision can parameterize
// this when more quantile choices are wanted.
const p95Quantile = 0.95

// WorkloadBaseliner tracks the running mean (EMA) and 95th-percentile
// (P² algorithm) of step durations for one (cgroup, pid, stream)
// workload. Single-threaded — the owning Engine guards concurrency.
//
// The P² estimator (Jain & Chlamtac, 1985) is single-pass, O(1)
// memory, and converges to the true p95 within a few hundred samples
// for stationary distributions. We pair it with an EMA mean so the
// outlier classifier can also report "ratio against typical step"
// rather than only "ratio against tail."
type WorkloadBaseliner struct {
	samples int
	mean    float64

	// P² state. Five markers; q[i] is the height (sample value) at
	// position n[i]. ns[i] is the *desired* marker position; dn[i] is
	// the per-sample increment for ns[i]. Only meaningful once the
	// initial-fill phase (5 samples) is complete.
	q     [5]float64
	n     [5]int     // actual position (1-indexed in the original P² paper)
	ns    [5]float64 // desired position
	dn    [5]float64 // desired-position increment per new sample
	init0 [5]float64 // first 5 raw samples, sorted at fill-complete
	filled int       // count of init samples observed (caps at 5)

	// hist is the cumulative step-duration distribution. v0.16.3
	// surface for OTLP/Prometheus emission so operators see the same
	// shape the EMA + P² baseliner does, without having to back-compute
	// it from the per-event JSON stream. Cumulative across the
	// baseliner's lifetime; never reset (matches the OTel cumulative
	// temporality used by every other histogram in this codebase).
	hist *stats.Histogram
}

// NewWorkloadBaseliner constructs an empty baseliner. Heights are
// initialized lazily on the first 5 samples.
func NewWorkloadBaseliner() *WorkloadBaseliner {
	b := &WorkloadBaseliner{
		hist: stats.NewHistogram(stats.DefaultInferStepDurationBoundsNs),
	}
	// Desired-position formula for an arbitrary quantile p:
	//   ns[0] = 1
	//   ns[1] = 1 + 2p
	//   ns[2] = 1 + 4p
	//   ns[3] = 3 + 2p
	//   ns[4] = 5
	// Per-sample increments dn[i] are the partial derivatives of
	// ns[i] w.r.t. the running sample count n: 0, p/2, p, (1+p)/2, 1.
	p := p95Quantile
	b.ns[0], b.ns[1], b.ns[2], b.ns[3], b.ns[4] = 1, 1+2*p, 1+4*p, 3+2*p, 5
	b.dn[0], b.dn[1], b.dn[2], b.dn[3], b.dn[4] = 0, p/2, p, (1+p)/2, 1
	return b
}

// Update folds one step duration (in nanoseconds) into the running
// mean and p95. Caller is responsible for filtering out non-finite
// or negative inputs; we assume sanitized input from the Engine.
func (b *WorkloadBaseliner) Update(stepNs float64) {
	stepNs = health.CleanFinite(stepNs)
	if stepNs <= 0 {
		return
	}
	b.samples++

	// EMA mean. CleanFinite already coerced any pathological value to
	// 0 and we filtered <=0 above, so the EMA stays well-formed.
	b.mean = health.EMAUpdate(b.mean, stepNs, emaAlphaMean)

	// Fold the step into the cumulative histogram. The Engine only
	// calls Update for healthy + warmup steps (post-warmup outliers
	// skip Update to keep the EMA/P² baseline clean), so the
	// histogram represents the workload's healthy distribution.
	// Outlier counts are visible separately via the per-bucket
	// counter (MetricInferOutlierTotal).
	if b.hist != nil {
		b.hist.Observe(stepNs)
	}

	// P² fill phase: collect the first 5 samples raw.
	if b.filled < 5 {
		b.init0[b.filled] = stepNs
		b.filled++
		if b.filled == 5 {
			b.completeInit()
		}
		return
	}

	b.p2Update(stepNs)
}

// completeInit sorts the first 5 samples into q[0..4] in ascending
// order and seeds the marker positions n[0..4] = 1..5.
func (b *WorkloadBaseliner) completeInit() {
	// Tiny insertion sort — 5 elements, no allocation.
	for i := 1; i < 5; i++ {
		v := b.init0[i]
		j := i
		for j > 0 && b.init0[j-1] > v {
			b.init0[j] = b.init0[j-1]
			j--
		}
		b.init0[j] = v
	}
	for i := 0; i < 5; i++ {
		b.q[i] = b.init0[i]
		b.n[i] = i + 1
	}
}

// p2Update applies one P² iteration on a sample x. The five-marker
// version of the algorithm: find the cell k that x lands in, bump
// the right-hand markers, then conditionally adjust q[1..3] using
// the parabolic-prediction step (or linear if parabolic would
// violate monotonicity).
func (b *WorkloadBaseliner) p2Update(x float64) {
	// Cell selection. x updates min/max heights at the boundary; for
	// the inner cells we find which (q[k], q[k+1]] interval contains
	// it. The original P² algorithm uses 1-indexed cells; here we use
	// 0-indexed k in [0,3].
	var k int
	switch {
	case x < b.q[0]:
		b.q[0] = x
		k = 0
	case x < b.q[1]:
		k = 0
	case x < b.q[2]:
		k = 1
	case x < b.q[3]:
		k = 2
	case x <= b.q[4]:
		k = 3
	default:
		b.q[4] = x
		k = 3
	}
	// Bump positions of markers to the right of cell k.
	for i := k + 1; i < 5; i++ {
		b.n[i]++
	}
	// Bump desired positions for all markers.
	for i := 0; i < 5; i++ {
		b.ns[i] += b.dn[i]
	}
	// Adjust the 3 inner markers if their actual position has drifted
	// from desired. The "if d >= 1 && n[i+1] - n[i] > 1" gates are the
	// classical P² conditions ensuring monotonic q[].
	for i := 1; i <= 3; i++ {
		d := b.ns[i] - float64(b.n[i])
		if (d >= 1 && b.n[i+1]-b.n[i] > 1) || (d <= -1 && b.n[i-1]-b.n[i] < -1) {
			ds := 1.0
			if d < 0 {
				ds = -1.0
			}
			qNew := b.parabolic(i, ds)
			if b.q[i-1] < qNew && qNew < b.q[i+1] {
				b.q[i] = qNew
			} else {
				b.q[i] = b.linear(i, ds)
			}
			b.n[i] += int(ds)
		}
	}
}

// parabolic is the P² parabolic-prediction step. d is +1 or -1.
func (b *WorkloadBaseliner) parabolic(i int, d float64) float64 {
	num1 := d / float64(b.n[i+1]-b.n[i-1])
	left := float64(b.n[i]-b.n[i-1]+int(d)) * (b.q[i+1] - b.q[i]) / float64(b.n[i+1]-b.n[i])
	right := float64(b.n[i+1]-b.n[i]-int(d)) * (b.q[i] - b.q[i-1]) / float64(b.n[i]-b.n[i-1])
	return b.q[i] + num1*(left+right)
}

// linear is the P² fallback when parabolic would violate
// monotonicity. d is +1 or -1.
func (b *WorkloadBaseliner) linear(i int, d float64) float64 {
	step := d * (b.q[i+int(d)] - b.q[i]) / float64(b.n[i+int(d)]-b.n[i])
	return b.q[i] + step
}

// Mean returns the current running mean. Defined as 0 before any
// successful Update.
func (b *WorkloadBaseliner) Mean() float64 { return b.mean }

// P95 returns the current 95th-percentile estimate. Defined as 0
// before the P² fill phase (5 samples) completes; once filled, it
// returns q[3] which is the marker tracking p=0.95.
//
// (Marker indices: q[0]=min, q[1]=p25, q[2]=p50, q[3]=p95, q[4]=max
// when the initial desired-position formula is fed p=0.95.)
func (b *WorkloadBaseliner) P95() float64 {
	if b.filled < 5 {
		return 0
	}
	return b.q[3]
}

// Samples returns the cumulative count of Update calls that produced
// a meaningful (>0, finite) input. Used by the Engine to gate
// classification on the warmup threshold.
func (b *WorkloadBaseliner) Samples() int { return b.samples }

// Warmed reports whether the baseliner has observed at least n
// samples AND completed the P² initial fill. When warmed, P95 is
// meaningful.
func (b *WorkloadBaseliner) Warmed(warmupSamples int) bool {
	return b.samples >= warmupSamples && b.filled == 5
}

// HistogramSnapshot returns a frozen copy of the cumulative
// step-duration histogram. Returns the zero value when the baseliner
// was constructed before histogram support landed (test-only legacy
// path); callers that emit OTLP can detect this via
// HasObservation==false.
func (b *WorkloadBaseliner) HistogramSnapshot() stats.HistogramSnapshot {
	if b.hist == nil {
		return stats.HistogramSnapshot{}
	}
	return b.hist.Snapshot()
}
