// Package config loads the agent's YAML configuration.
//
// v0.13 wires the otlp: and alerter.pagerduty: blocks of configs/ingero.yaml
// into the runtime: previously those blocks were documentation-only and
// operators had to pass --otlp-* / --pagerduty-routing-key flags. The flags
// remain available as overrides; the YAML is the source of truth when both
// are absent.
//
// Other YAML blocks (agent, probes, store, mcp, health, prometheus, fleet)
// are either consumed elsewhere (parseFleetNodesFromYAML in cli/node.go) or
// reserved for future expansion. KnownFields(false) ignores them silently.
package config

import (
	"bytes"
	"fmt"
	"os"
	"time"

	"gopkg.in/yaml.v3"
)

// DefaultConfigPath is the relative path used by the --config persistent
// flag when the operator does not override it. Matches the layout shipped
// in deploy/ and configs/ in the repo.
const DefaultConfigPath = "configs/ingero.yaml"

// AgentConfig is the on-disk shape of configs/ingero.yaml. Only the
// blocks consumed by Go subcommands are modeled; other blocks
// (health, prometheus, fleet) are passed through opaquely or reserved
// for future expansion. Adding a new block here requires extending one
// or both of the consumers in fleet_push.go / mcp_cmd.go.
type AgentConfig struct {
	OTLP      OTLPConfig      `yaml:"otlp"`
	Alerter   AlerterConfig   `yaml:"alerter"`
	Inference InferenceConfig `yaml:"inference"`
}

// InferenceConfig drives the v0.16 `--inference` umbrella. Mirrors
// the trace-command flag surface so operators can configure once in
// YAML and override per-invocation with CLI flags. All fields are
// optional — a zero InferenceConfig means "no umbrella defaults
// applied," and the agent behaves as if --inference were not set.
type InferenceConfig struct {
	// Enabled is the YAML equivalent of `--inference`. When true,
	// the trace command applies the umbrella defaults below.
	Enabled bool `yaml:"enabled"`

	Baseline   InferenceBaselineConfig   `yaml:"baseline"`
	Outlier    InferenceOutlierConfig    `yaml:"outlier"`
	DBRollover InferenceRolloverConfig   `yaml:"db_rollover"`
	Daemon     InferenceDaemonConfig     `yaml:"daemon"`
}

// InferenceBaselineConfig tunes the per-workload step-duration
// baseliner. Zero values resolve to the package defaults in
// internal/infer.
type InferenceBaselineConfig struct {
	// WarmupSamples is the number of healthy steps required before
	// outlier classification activates for a workload. Default 30.
	WarmupSamples int `yaml:"warmup_samples"`
	// PauseOnSeverity skips baseline updates while a causal chain at
	// this severity or higher is active for the PID. Default "HIGH".
	// Empty disables the gate.
	PauseOnSeverity string `yaml:"pause_on_severity"`
}

// InferenceOutlierConfig tunes the classifier and the
// outlier->sampler feedback loop.
type InferenceOutlierConfig struct {
	// ThresholdRatio is the multiplier applied to baseline p95 for
	// the LARGEST outlier bucket. Default 3.0.
	ThresholdRatio float64 `yaml:"threshold_ratio"`
	// SamplerDegradeOn is the smallest bucket that bumps the store
	// sampler to admit 100% of events. Allowed values: "1.5x" | "2x"
	// | "3x" | "off". Default "3x".
	SamplerDegradeOn string `yaml:"sampler_degrade_on"`
}

// InferenceRolloverConfig drives file-level DB rollover (rotates the
// SQLite trace DB when it crosses Size, retains Keep oldest rolled
// files). Mutually exclusive with --max-db (in-place row pruning).
type InferenceRolloverConfig struct {
	// Size is the size threshold; same human-friendly units as
	// --max-db ("1g", "500m"). Empty disables rollover.
	Size string `yaml:"size"`
	// Keep is the number of rolled files retained on disk. Default 6.
	Keep int `yaml:"keep"`
}

// InferenceDaemonConfig drives the daemon-mode defaults applied when
// the umbrella is engaged.
type InferenceDaemonConfig struct {
	// Duration is the trace duration. Empty / zero = run forever
	// (until ctx cancellation).
	Duration time.Duration `yaml:"duration"`
	// LogPath is the operator-supplied log path. When empty AND the
	// CLI also leaves --log unset, the umbrella picks a temp path so
	// stderr stays clean for systemd / k8s log collectors.
	LogPath string `yaml:"log_path"`
}

// OTLPConfig drives detection-event traces export. v0.13 ships HTTP only.
// The Protocol and ExportInterval keys are reserved in the on-disk YAML
// but unused at runtime; gRPC support and a tunable export interval defer.
type OTLPConfig struct {
	Enabled  bool   `yaml:"enabled"`
	Endpoint string `yaml:"endpoint"`
	Insecure bool   `yaml:"insecure"`
}

// AlerterConfig groups the agent-side alerter configuration. Today only
// PagerDuty is wired; Slack stays sidecar-only because the agent does not
// drive Slack webhooks directly.
type AlerterConfig struct {
	PagerDuty *PagerDutyYAML `yaml:"pagerduty,omitempty"`
}

// PagerDutyYAML is the minimal agent-facing PagerDuty config. We do NOT
// reuse alerter.PagerDutyConfig: that struct is the alerter sidecar's
// full config (with Severity and other fields) and pulling it in would
// drag the alerter package's surface into the agent's wiring.
//
// The mcp subcommand turns this into an *alerter.PagerDutyConfig at the
// last moment, when constructing the backend.
type PagerDutyYAML struct {
	RoutingKey string `yaml:"routing_key"`
}

// Load parses the config file at path. Returns a zero-value AgentConfig
// (no error) if path doesn't exist - callers can layer CLI flags on top
// of an empty baseline. A malformed file or any non-NotExist read error
// is surfaced; the caller decides whether to fall back or fail.
func Load(path string) (*AgentConfig, error) {
	data, err := os.ReadFile(path)
	if err != nil {
		if os.IsNotExist(err) {
			return &AgentConfig{}, nil
		}
		return nil, fmt.Errorf("read %s: %w", path, err)
	}
	var cfg AgentConfig
	dec := yaml.NewDecoder(bytes.NewReader(data))
	// Ignore unknown blocks (health, prometheus, fleet, agent, probes, ...).
	// We only model the blocks the agent's Go subcommands consume.
	dec.KnownFields(false)
	if err := dec.Decode(&cfg); err != nil {
		return nil, fmt.Errorf("parse %s: %w", path, err)
	}
	return &cfg, nil
}
