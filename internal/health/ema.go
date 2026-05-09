package health

import "math"

// EMAUpdate folds one observation x into a running exponential moving
// average using the standard recurrence: prev*(1-alpha) + x*alpha.
//
// Exported so peer agent packages (notably internal/infer for the
// per-workload step-duration baseline) can reuse the same numerics
// without re-importing the math.
func EMAUpdate(prev, x, alpha float64) float64 {
	return alpha*x + (1-alpha)*prev
}

// BiasCorrectScalar applies s_hat_t = s_t / (1 - (1-alpha)^t) so that
// samples near t=0 are not pulled toward zero by the cold start. Returns
// 0 when t <= 0 or the correction denominator is non-positive (the
// only path that produces this is alpha=1, where bias correction is a
// no-op anyway).
func BiasCorrectScalar(raw, alpha float64, t int) float64 {
	if t <= 0 {
		return 0
	}
	correction := 1 - math.Pow(1-alpha, float64(t))
	if correction <= 0 {
		return 0
	}
	return raw / correction
}

// CleanFinite coerces NaN and Inf to 0. Used at the EMA boundary so a
// single bad observation cannot poison the running average.
func CleanFinite(x float64) float64 {
	if math.IsNaN(x) || math.IsInf(x, 0) {
		return 0
	}
	return x
}
