package cli

import (
	"errors"
	"fmt"
	"strings"
	"time"

	"github.com/ingero-io/ingero/internal/config"
	"github.com/spf13/cobra"
)

// resolvedInference is the composed --inference configuration after
// layering CLI flags over the on-disk YAML. Returned as a value (not
// out-params) so the resolution logic is unit-testable in isolation.
//
// Mirrors the resolveOTLPConfig shape in fleet_push.go — same
// flag-overrides-YAML semantics via cmd.Flags().Changed().
type resolvedInference struct {
	Enabled bool

	WarmupSamples         int
	OutlierThresholdRatio float64
	PauseOnSeverity       string
	SamplerDegradeOn      string

	DBRolloverSize string
	DBRolloverKeep int

	DaemonDuration time.Duration
	DaemonLogPath  string
}

// resolveInferenceConfig composes the --inference runtime values from
// a parsed YAML config and the trace command's CLI flags. CLI flags
// override YAML only when explicitly set (cmd.Flags().Changed); flags
// left at default inherit the YAML value, which itself defaults to the
// internal/infer package's defaults at construction time.
//
// cfgPath is included so the surfaced error mentions which file the
// operator should edit.
//
// Validation enforced here (single source of truth):
//   - --max-db AND --db-rollover-size on the same invocation are
//     mutually exclusive (in-place pruning vs. file rollover).
//   - SamplerDegradeOn, when non-empty, must be one of {"1.5x","2x","3x","off"}.
//   - WarmupSamples must be non-negative.
//   - OutlierThresholdRatio, when non-zero, must be > 1.0 (smaller would
//     classify the baseline itself as an outlier).
func resolveInferenceConfig(cfg *config.AgentConfig, cmd *cobra.Command, cfgPath string) (resolvedInference, error) {
	out := resolvedInference{
		Enabled:               cfg.Inference.Enabled,
		WarmupSamples:         cfg.Inference.Baseline.WarmupSamples,
		OutlierThresholdRatio: cfg.Inference.Outlier.ThresholdRatio,
		PauseOnSeverity:       cfg.Inference.Baseline.PauseOnSeverity,
		SamplerDegradeOn:      cfg.Inference.Outlier.SamplerDegradeOn,
		DBRolloverSize:        cfg.Inference.DBRollover.Size,
		DBRolloverKeep:        cfg.Inference.DBRollover.Keep,
		DaemonDuration:        cfg.Inference.Daemon.Duration,
		DaemonLogPath:         cfg.Inference.Daemon.LogPath,
	}

	if cmd.Flags().Changed("inference") {
		out.Enabled = traceInference
	}
	if cmd.Flags().Changed("inference-warmup") {
		out.WarmupSamples = traceInferenceWarmup
	}
	if cmd.Flags().Changed("inference-outlier-ratio") {
		out.OutlierThresholdRatio = traceInferenceOutlierRatio
	}
	if cmd.Flags().Changed("inference-pause-on-severity") {
		out.PauseOnSeverity = traceInferencePauseSeverity
	}
	if cmd.Flags().Changed("inference-sampler-degrade-on") {
		out.SamplerDegradeOn = traceInferenceSamplerDegradeOn
	}
	if cmd.Flags().Changed("db-rollover-size") {
		out.DBRolloverSize = traceDBRolloverSize
	}
	if cmd.Flags().Changed("db-rollover-keep") {
		out.DBRolloverKeep = traceDBRolloverKeep
	}

	// Validation. Run only when the umbrella is actually engaged so a
	// stray YAML key on a non-inference invocation doesn't fail
	// startup for unrelated workflows.
	if !out.Enabled {
		return out, nil
	}

	if out.WarmupSamples < 0 {
		return resolvedInference{}, fmt.Errorf("inference.baseline.warmup_samples must be >= 0 in %s (got %d)",
			cfgPath, out.WarmupSamples)
	}
	if out.OutlierThresholdRatio != 0 && out.OutlierThresholdRatio <= 1.0 {
		return resolvedInference{}, fmt.Errorf("inference.outlier.threshold_ratio must be > 1.0 in %s (got %g)",
			cfgPath, out.OutlierThresholdRatio)
	}
	switch strings.ToLower(strings.TrimSpace(out.SamplerDegradeOn)) {
	case "", "1.5x", "2x", "3x", "off":
		// valid
	default:
		return resolvedInference{}, fmt.Errorf("inference.outlier.sampler_degrade_on must be one of 1.5x|2x|3x|off in %s (got %q)",
			cfgPath, out.SamplerDegradeOn)
	}
	switch strings.ToUpper(strings.TrimSpace(out.PauseOnSeverity)) {
	case "", "HIGH", "MEDIUM", "LOW":
		// valid
	default:
		return resolvedInference{}, fmt.Errorf("inference.baseline.pause_on_severity must be one of HIGH|MEDIUM|LOW or empty in %s (got %q)",
			cfgPath, out.PauseOnSeverity)
	}

	mdb := strings.TrimSpace(traceMaxDB)
	maxDBSet := cmd.Flags().Changed("max-db") && mdb != "" && mdb != "0"
	rolloverSet := strings.TrimSpace(out.DBRolloverSize) != ""
	if maxDBSet && rolloverSet {
		return resolvedInference{}, errors.New("--max-db and --db-rollover-size are mutually exclusive: --max-db prunes rows in place; --db-rollover-size rotates the DB file")
	}
	if out.DBRolloverKeep < 0 {
		return resolvedInference{}, fmt.Errorf("inference.db_rollover.keep must be >= 0 in %s (got %d)",
			cfgPath, out.DBRolloverKeep)
	}

	return out, nil
}
