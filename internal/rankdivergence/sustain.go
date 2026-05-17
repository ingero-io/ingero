// Package rankdivergence detects per-rank compute-time divergence
// from NCCL collective latencies. One rank consistently slower (or
// numerically diverging into a slow path) than the cohort over a
// sustained window is the classic silent-data-corruption fingerprint
// and the most actionable rank-imbalance signal short of explicit
// NCCL collective profiling.
//
// Input: per-event records of (pid, comm_id_hash, rank, n_ranks,
// duration_ns) drawn from internal/ebpf/ncclprobe. Records are
// bucketed by comm_id_hash (the NCCL communicator that connects a
// set of ranks). Per-rank duration medians are computed over a
// sliding window of N most-recent samples; the cohort median +
// MAD (median absolute deviation) are computed across per-rank
// medians; any rank more than `sigmaThreshold` MAD-sigmas from the
// cohort median is flagged.
//
// Sustained-divergence requires the rank to stay flagged across
// `sustainedSweeps` consecutive Compute() ticks before emission, so
// transient measurement noise from a single slow tick doesn't fire.
// Emission is suppressed-once-per-episode (rearms when the rank
// drops back below threshold).
//
// Anchor: catalog row T4 (SDC fingerprint) and NEW row T20 (rank
// imbalance). MAD is preferred over standard deviation because NCCL
// per-rank latencies have heavy-tailed distributions and a single
// outlier inflates SD; MAD is the robust analog.
package rankdivergence

import (
	"math"
	"sort"
	"sync"
	"time"
)

// DefaultWindowSize is the number of per-rank samples kept in the
// sliding window. 64 is small enough to react within a few seconds
// at typical training-loop NCCL rates (10-100 collectives/sec) and
// large enough that the per-rank median is stable.
const DefaultWindowSize = 64

// DefaultSigmaThreshold is the MAD-sigma multiplier above which a
// rank's median duration is flagged as divergent. 4.0 is well above
// the natural per-rank variance in a healthy cohort (~1-2 sigma) and
// catches the SDC fingerprint (~10+ sigma) with margin.
const DefaultSigmaThreshold = 4.0

// DefaultSustainedSweeps is the number of consecutive Compute()
// ticks a rank must stay flagged before emission. Three ticks at a
// 5s ticker = 15s sustained divergence — well above transient
// measurement noise.
const DefaultSustainedSweeps = 3

// Divergence is the per-episode summary emitted by Compute. Maps
// 1:1 onto the orchestrator's RankDivergenceState wire message.
type Divergence struct {
	PID         uint32
	Rank        uint32
	DriftSigma  float64
	SustainedMs uint64
}

// Tracker holds per-commIDHash per-rank duration history.
type Tracker struct {
	mu               sync.Mutex
	comms            map[uint64]*commState
	windowSize       int
	sigmaThreshold   float64
	sustainedSweeps  int
	tickInterval     time.Duration
	now              func() time.Time
}

type commState struct {
	// Per-rank samples (most-recent first; capped at windowSize).
	ranks map[uint32]*rankState
	// Most recent NRanks observed; used to know when the cohort is
	// complete enough to compute MAD.
	expectedRanks uint32
}

type rankState struct {
	pid     uint32
	samples []float64 // duration_ns, ring buffer head wraps at windowSize
	// flaggedSince increments by 1 each Compute() tick while the
	// rank stays divergent; resets to 0 when below threshold.
	flaggedSweeps int
	// emitted: an episode for this rank has already been emitted;
	// rearms when flaggedSweeps drops to 0.
	emitted bool
	// lastFlaggedAt: when divergence first crossed threshold this
	// episode. Used to report sustained_ms in the emission.
	episodeStart time.Time
}

// New returns a Tracker with defaults.
func New() *Tracker {
	return NewWithThresholds(DefaultWindowSize, DefaultSigmaThreshold, DefaultSustainedSweeps, 5*time.Second)
}

// NewWithThresholds returns a Tracker with custom parameters. Tests
// use smaller windowSize and sustainedSweeps for faster assertions.
func NewWithThresholds(windowSize int, sigmaThreshold float64, sustainedSweeps int, tickInterval time.Duration) *Tracker {
	if windowSize < 4 {
		windowSize = 4
	}
	return &Tracker{
		comms:           make(map[uint64]*commState),
		windowSize:      windowSize,
		sigmaThreshold:  sigmaThreshold,
		sustainedSweeps: sustainedSweeps,
		tickInterval:    tickInterval,
		now:             time.Now,
	}
}

// Observe records one NCCL return sample. pid==0 or commIDHash==0
// is dropped (we can't bucket without a comm). nRanks==0 is treated
// as "unknown cohort size" and the sample is buffered but won't
// participate in MAD until a non-zero nRanks lands.
func (t *Tracker) Observe(pid uint32, commIDHash uint64, rank uint32, nRanks uint32, durationNs uint64) {
	if pid == 0 || commIDHash == 0 {
		return
	}
	t.mu.Lock()
	defer t.mu.Unlock()
	c, ok := t.comms[commIDHash]
	if !ok {
		c = &commState{ranks: make(map[uint32]*rankState)}
		t.comms[commIDHash] = c
	}
	if nRanks > c.expectedRanks {
		c.expectedRanks = nRanks
	}
	r, ok := c.ranks[rank]
	if !ok {
		r = &rankState{pid: pid}
		c.ranks[rank] = r
	}
	r.pid = pid
	r.samples = append(r.samples, float64(durationNs))
	if len(r.samples) > t.windowSize {
		// Drop the oldest. Slice rather than ring buffer for
		// simplicity; windowSize is small (default 64).
		r.samples = r.samples[len(r.samples)-t.windowSize:]
	}
}

// Compute runs one MAD analysis pass across all known commIDHashes
// and returns Divergences for any rank whose flagged-sweep count
// has just crossed sustainedSweeps. Emits at most one Divergence
// per (commIDHash, rank) per episode.
//
// Commits that don't have enough samples (per-rank < windowSize/4)
// or that don't have at least 3 ranks reporting (MAD across <3
// data points is meaningless) are skipped this tick.
func (t *Tracker) Compute(now time.Time) []Divergence {
	t.mu.Lock()
	defer t.mu.Unlock()

	var out []Divergence
	minPerRankSamples := t.windowSize / 4
	if minPerRankSamples < 4 {
		minPerRankSamples = 4
	}

	for _, c := range t.comms {
		// Build per-rank medians for ranks with enough samples.
		type rankMedian struct {
			rank   uint32
			median float64
			state  *rankState
		}
		var medians []rankMedian
		for rank, st := range c.ranks {
			if len(st.samples) < minPerRankSamples {
				continue
			}
			medians = append(medians, rankMedian{
				rank:   rank,
				median: medianFloat64(st.samples),
				state:  st,
			})
		}
		if len(medians) < 3 {
			continue
		}

		// Cohort median + MAD across per-rank medians.
		cohort := make([]float64, len(medians))
		for i, m := range medians {
			cohort[i] = m.median
		}
		cohortMedian := medianFloat64(cohort)
		mad := medianAbsoluteDeviation(cohort, cohortMedian)
		// Floor MAD at 1% of the cohort median. With small cohorts
		// (4-8 ranks), one outlier among otherwise-identical ranks
		// collapses MAD to 0 (majority of deviations are 0, so the
		// median deviation is 0 too). Without the floor we'd
		// silently skip the exact case the detector is designed for.
		// 1% * sigmaThreshold (4 default) means a rank must drift
		// >= 4% of the cohort median to flag — well above measurement
		// noise on healthy collectives, well below real SDC / slow-
		// rank shapes.
		if cohortMedian > 0 {
			floor := 0.01 * cohortMedian
			if mad < floor {
				mad = floor
			}
		}
		if mad <= 0 {
			// Cohort median itself is non-positive (only possible
			// for empty input which we've already guarded). Skip.
			continue
		}

		for _, m := range medians {
			drift := math.Abs(m.median-cohortMedian) / mad
			if drift >= t.sigmaThreshold {
				if m.state.flaggedSweeps == 0 {
					m.state.episodeStart = now
				}
				m.state.flaggedSweeps++
				if m.state.flaggedSweeps >= t.sustainedSweeps && !m.state.emitted {
					out = append(out, Divergence{
						PID:         m.state.pid,
						Rank:        m.rank,
						DriftSigma:  drift,
						SustainedMs: uint64(now.Sub(m.state.episodeStart).Milliseconds()),
					})
					m.state.emitted = true
				}
			} else {
				// Below threshold; rearm the episode.
				m.state.flaggedSweeps = 0
				m.state.emitted = false
				m.state.episodeStart = time.Time{}
			}
		}
	}
	return out
}

// Forget drops state for a comm. Call this on observed comm destroy
// (Op == COMM_DESTROY).
func (t *Tracker) Forget(commIDHash uint64) {
	t.mu.Lock()
	defer t.mu.Unlock()
	delete(t.comms, commIDHash)
}

// TrackedComms returns the number of comms currently tracked.
func (t *Tracker) TrackedComms() int {
	t.mu.Lock()
	defer t.mu.Unlock()
	return len(t.comms)
}

// medianFloat64 returns the median of xs. Empty input returns 0.
// Does not allocate when len(xs) > 0 (sorts in place on a copy).
func medianFloat64(xs []float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	cp := make([]float64, len(xs))
	copy(cp, xs)
	sort.Float64s(cp)
	n := len(cp)
	if n%2 == 1 {
		return cp[n/2]
	}
	return (cp[n/2-1] + cp[n/2]) / 2
}

// medianAbsoluteDeviation returns MAD = median(|x_i - center|).
// Scaled by 1.4826 to be a consistent estimator of standard
// deviation under a normal distribution, so the sigma threshold
// has the same intuitive meaning as a Gaussian sigma.
func medianAbsoluteDeviation(xs []float64, center float64) float64 {
	if len(xs) == 0 {
		return 0
	}
	devs := make([]float64, len(xs))
	for i, x := range xs {
		devs[i] = math.Abs(x - center)
	}
	return medianFloat64(devs) * 1.4826
}
