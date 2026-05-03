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
	OTLP    OTLPConfig    `yaml:"otlp"`
	Alerter AlerterConfig `yaml:"alerter"`
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
