package infer

import "time"

// Phase labels one inference step by the kind of work it performed.
// Phase is an *observed* property — derived at step boundary from
// kernel-launch count, total kernel duration, memcpy bytes, and NCCL
// collective participation between consecutive cudaStreamSync events.
// It is NOT read from the engine; rule-based classification with no
// ML, ~90% accuracy on vLLM/TGI/SGLang per the eInfer / Beyla
// literature.
//
// Why phase exists: a single P² p95 baseline per `(cgroup, pid,
// stream)` produces false negatives on heterogeneous-task streams.
// A vLLM continuous-batching server interleaves prefill (~200ms,
// kernel-heavy) and decode (~5ms, sparse-launch) steps on the same
// hot-path stream; a 50ms decode step (10× normal) is invisible
// against the mixed p95 of ~180ms. Splitting baselines by phase
// makes each baseline unimodal and the apples-to-apples comparison
// possible.
type Phase string

const (
	PhasePrefill Phase = "prefill"
	PhaseDecode  Phase = "decode"
	PhaseMixed   Phase = "mixed"
	PhaseUnknown Phase = "unknown"
)

// PhaseConfig holds the rule-classifier thresholds. Defaults are
// LLM-tuned (7B-70B serving). Operators with non-LLM workloads
// (embedding, vision, MoE) tune individual thresholds via flags.
//
// Zero values resolve to defaults via Resolved().
//
// Why these rules are duration-invariant: a "slow decode" is still
// a decode (few launches, no memcpy, no NCCL); we want it to land
// in the decode bucket so a 10× slowdown fires as an outlier
// against the decode baseline. Using step duration as a primary
// signal would reclassify a slow decode as "unknown" or "mixed",
// defeating the apples-to-apples premise. Step duration is the
// VALUE we baseline; it cannot also be the SHAPE we classify on.
type PhaseConfig struct {
	// DecodeMaxLaunches + DecodeMaxMemcpy gate the decode rule.
	// A step is decode if launch_count < DecodeMaxLaunches AND
	// memcpy_bytes < DecodeMaxMemcpy AND NCCL == 0. Duration is
	// not used here.
	DecodeMaxLaunches int
	DecodeMaxMemcpy   int64

	// PrefillMinLaunches + PrefillMinAvgKernel gate the prefill
	// rule. A step is prefill if launch_count > PrefillMinLaunches
	// OR avg_kernel_duration > PrefillMinAvgKernel. NCCL>0 is also
	// prefill (separate top-priority rule).
	PrefillMinLaunches  int
	PrefillMinAvgKernel time.Duration

	// MixedMemcpyThreshold + MixedLaunchLow + MixedLaunchHigh gate
	// the mixed rule. Captures continuous-batching transition
	// steps that don't cleanly match prefill or decode.
	MixedMemcpyThreshold int64
	MixedLaunchLow       int
	MixedLaunchHigh      int

	// MemfragDecodeMin is the minimum memfrag IOCTL event count that
	// triggers the high-priority "memfrag pressure -> decode" rule.
	// v0.16.3: KV-cache eviction storms are decode-shape failures and
	// should be classified into the decode bucket so the slow step
	// fires against the decode baseline (not unknown / mixed). Zero
	// disables the rule. Default 1 (any memfrag activity counts).
	MemfragDecodeMin int
}

// DefaultPhaseConfig returns LLM-tuned thresholds derived from the
// vLLM and TGI literature (typical 7B-70B model serving). Embedding,
// vision, and MoE workloads should tune individual thresholds.
func DefaultPhaseConfig() PhaseConfig {
	return PhaseConfig{
		DecodeMaxLaunches:    50,
		DecodeMaxMemcpy:      1 * 1024 * 1024, // 1 MiB
		PrefillMinLaunches:   200,
		PrefillMinAvgKernel:  500 * time.Microsecond,
		MixedMemcpyThreshold: 10 * 1024 * 1024, // 10 MiB
		MixedLaunchLow:       50,
		MixedLaunchHigh:      200,
		MemfragDecodeMin:     1,
	}
}

// Resolved fills in zero-value fields with the LLM defaults. Mutating
// the returned copy after Resolved is safe; the original argument is
// unchanged.
func (c PhaseConfig) Resolved() PhaseConfig {
	d := DefaultPhaseConfig()
	if c.DecodeMaxLaunches <= 0 {
		c.DecodeMaxLaunches = d.DecodeMaxLaunches
	}
	if c.DecodeMaxMemcpy <= 0 {
		c.DecodeMaxMemcpy = d.DecodeMaxMemcpy
	}
	if c.PrefillMinLaunches <= 0 {
		c.PrefillMinLaunches = d.PrefillMinLaunches
	}
	if c.PrefillMinAvgKernel <= 0 {
		c.PrefillMinAvgKernel = d.PrefillMinAvgKernel
	}
	if c.MixedMemcpyThreshold <= 0 {
		c.MixedMemcpyThreshold = d.MixedMemcpyThreshold
	}
	if c.MixedLaunchLow <= 0 {
		c.MixedLaunchLow = d.MixedLaunchLow
	}
	if c.MixedLaunchHigh <= 0 {
		c.MixedLaunchHigh = d.MixedLaunchHigh
	}
	if c.MemfragDecodeMin <= 0 {
		c.MemfragDecodeMin = d.MemfragDecodeMin
	}
	return c
}

// ClassifyPhase returns the phase label for one step. Arguments are
// observed between consecutive cudaStreamSync events on the same
// (pid, stream):
//
//   - stepDuration: wall time from prev sync to this sync. Reserved
//     for future use; current rules are intentionally
//     duration-INVARIANT so a slow decode is still classified as
//     decode (and fires as an outlier against the decode baseline
//     instead of getting reclassified into a different bucket).
//   - launchCount:  cudaLaunchKernel events observed
//   - totalKernelNs: sum of kernel durations
//   - memcpyBytes:  total memcpy bytes (any direction)
//   - ncclCount:    NCCL collective events
//   - memfragCount: NVIDIA closed-driver IOCTL events (v0.15 W1
//     memfrag kprobe). Non-zero is a strong "decode-pressure" signal
//     (KV-cache eviction storm under VRAM pressure). v0.16.3.
//
// Rule order matters; first match wins. The bias is toward
// PhaseUnknown on ambiguous input — better a step bucket of its
// own than misclassification that pollutes a "real" baseline.
func ClassifyPhase(
	stepDuration time.Duration,
	launchCount int,
	totalKernelNs time.Duration,
	memcpyBytes int64,
	ncclCount int,
	memfragCount int,
	cfg PhaseConfig,
) Phase {
	_ = stepDuration // reserved; see doc comment above
	cfg = cfg.Resolved()

	// Rule 0: memfrag IOCTL pressure with low launch density is a
	// KV-cache eviction storm — decode-shape failure under VRAM
	// pressure. Higher priority than NCCL because operators want a
	// memfrag-storm step folded into the decode baseline (so the
	// slow decode fires as a decode-bucket outlier), not absorbed
	// into prefill where it gets dwarfed. The launch_count gate
	// keeps the rule from firing on prefill-shape steps that happen
	// to coincide with a memfrag event (memfrag during a 200-launch
	// prefill is a different scenario - probably allocation, not
	// eviction). Disabled when MemfragDecodeMin <= 0.
	if cfg.MemfragDecodeMin > 0 &&
		memfragCount >= cfg.MemfragDecodeMin &&
		launchCount < cfg.DecodeMaxLaunches {
		return PhaseDecode
	}

	// Rule 1: NCCL participation -> distributed prefill (allreduce,
	// allgather, etc.). In inference servers NCCL is overwhelmingly
	// tensor-parallel prefill; worth a top-priority rule because
	// NCCL participation is a strong, duration-invariant signal.
	if ncclCount > 0 {
		return PhasePrefill
	}

	// Rule 2: zero-launch step is an idle-poll (engine polling
	// stream readiness without queueing work). Not a real workload
	// step; bucket separately so it doesn't pollute decode/prefill
	// baselines.
	if launchCount == 0 && memcpyBytes == 0 {
		return PhaseUnknown
	}

	// Compute avg kernel duration once — used by rules 3 and 4.
	var avgKernel time.Duration
	if launchCount > 0 {
		avgKernel = totalKernelNs / time.Duration(launchCount)
	}

	// Rule 3: prefill via fat average kernel. Comes BEFORE decode
	// so a "30 launches × 1ms each" step (compute-heavy GEMM-style
	// prefill on a small batch) doesn't get misclassified as decode
	// just because its launch count is low. Average kernel duration
	// is the more discriminating signal here.
	if avgKernel > cfg.PrefillMinAvgKernel {
		return PhasePrefill
	}

	// Rule 4: prefill via many launches. Catches the typical LLM
	// prefill pattern (many small attention/MLP kernels per layer).
	if launchCount > cfg.PrefillMinLaunches {
		return PhasePrefill
	}

	// Rule 5: decode. Few launches AND small memcpy AND no NCCL.
	// Duration-invariant: a slow decode (10× p95) still has the
	// "few launches, small memcpy" shape, so it lands here.
	if launchCount < cfg.DecodeMaxLaunches && memcpyBytes < cfg.DecodeMaxMemcpy {
		return PhaseDecode
	}

	// Rule 6: mixed. Mid-range launch density OR bulk memcpy. Catches
	// continuous-batching transition steps that don't cleanly fit
	// prefill or decode (engine just picked up a new request, KV
	// cache shuffling, batch composition change).
	if launchCount >= cfg.MixedLaunchLow && launchCount <= cfg.MixedLaunchHigh {
		return PhaseMixed
	}
	if memcpyBytes >= cfg.MixedMemcpyThreshold {
		return PhaseMixed
	}

	// Rule 7: bias toward PhaseUnknown. Conservative — we don't
	// know what the workload is doing, so we don't fold it into
	// any of the named-phase baselines. The unknown bucket gets
	// its own baseline; outliers there fire normally but do NOT
	// trigger sampler degradation (callers gate on phase).
	return PhaseUnknown
}

// IsClassified returns true when the phase is one of the three
// "real" phases (prefill, decode, mixed). PhaseUnknown is excluded.
// Used by the engine to gate sampler degradation: we never bump the
// store sampler to 100% admission on an unknown-phase step because
// we lack the context to know whether the slowdown is meaningful.
func (p Phase) IsClassified() bool {
	switch p {
	case PhasePrefill, PhaseDecode, PhaseMixed:
		return true
	}
	return false
}
