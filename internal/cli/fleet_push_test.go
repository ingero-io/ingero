package cli

import (
	"strings"
	"testing"

	"github.com/spf13/cobra"

	"github.com/ingero-io/ingero/internal/config"
)

// newOTLPTestCmd builds a cobra.Command with just the three OTLP flags
// registered, mirroring fleet_push.go's init() registration. We don't
// reuse fleetPushCmd directly because its package-level state would
// bleed across tests (the package vars are mutated by Cobra parsing).
func newOTLPTestCmd() *cobra.Command {
	cmd := &cobra.Command{Use: "test"}
	cmd.Flags().BoolVar(&fleetPushOTLPEnabled, "otlp-enabled", false, "")
	cmd.Flags().StringVar(&fleetPushOTLPEndpoint, "otlp-endpoint", "", "")
	cmd.Flags().BoolVar(&fleetPushOTLPInsecure, "otlp-insecure", true, "")
	return cmd
}

func TestResolveOTLPConfig_YAMLOnly(t *testing.T) {
	t.Cleanup(func() { fleetPushOTLPEnabled = false; fleetPushOTLPEndpoint = ""; fleetPushOTLPInsecure = true })
	cmd := newOTLPTestCmd()
	if err := cmd.ParseFlags(nil); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	cfg := &config.AgentConfig{
		OTLP: config.OTLPConfig{Enabled: true, Endpoint: "yaml.example:4318", Insecure: false},
	}
	got, err := resolveOTLPConfig(cfg, cmd, "test.yaml")
	if err != nil {
		t.Fatalf("resolveOTLPConfig: %v", err)
	}
	if !got.Enabled || got.Endpoint != "yaml.example:4318" || got.Insecure {
		t.Errorf("YAML values not propagated: %+v", got)
	}
}

func TestResolveOTLPConfig_FlagOverridesYAML(t *testing.T) {
	t.Cleanup(func() { fleetPushOTLPEnabled = false; fleetPushOTLPEndpoint = ""; fleetPushOTLPInsecure = true })
	cmd := newOTLPTestCmd()
	if err := cmd.ParseFlags([]string{"--otlp-endpoint", "flag.example:4318", "--otlp-enabled=true"}); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	cfg := &config.AgentConfig{
		OTLP: config.OTLPConfig{Enabled: true, Endpoint: "yaml.example:4318", Insecure: false},
	}
	got, err := resolveOTLPConfig(cfg, cmd, "test.yaml")
	if err != nil {
		t.Fatalf("resolveOTLPConfig: %v", err)
	}
	if got.Endpoint != "flag.example:4318" {
		t.Errorf("flag did not override endpoint: got %q", got.Endpoint)
	}
	if !got.Enabled {
		t.Errorf("Enabled = false; want true")
	}
	// Insecure not changed on cmdline -> inherits YAML (false).
	if got.Insecure {
		t.Errorf("Insecure = true; want false (inherited from YAML)")
	}
}

func TestResolveOTLPConfig_FlagOnly(t *testing.T) {
	t.Cleanup(func() { fleetPushOTLPEnabled = false; fleetPushOTLPEndpoint = ""; fleetPushOTLPInsecure = true })
	cmd := newOTLPTestCmd()
	if err := cmd.ParseFlags([]string{"--otlp-enabled=true", "--otlp-endpoint", "host:4318", "--otlp-insecure=false"}); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	// Empty YAML.
	got, err := resolveOTLPConfig(&config.AgentConfig{}, cmd, "test.yaml")
	if err != nil {
		t.Fatalf("resolveOTLPConfig: %v", err)
	}
	if !got.Enabled || got.Endpoint != "host:4318" || got.Insecure {
		t.Errorf("flag-only path mis-resolved: %+v", got)
	}
}

func TestResolveOTLPConfig_EnabledNoEndpoint(t *testing.T) {
	t.Cleanup(func() { fleetPushOTLPEnabled = false; fleetPushOTLPEndpoint = ""; fleetPushOTLPInsecure = true })
	cmd := newOTLPTestCmd()
	if err := cmd.ParseFlags(nil); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	// Pathological YAML: enabled true, endpoint empty.
	cfg := &config.AgentConfig{
		OTLP: config.OTLPConfig{Enabled: true, Endpoint: "", Insecure: true},
	}
	_, err := resolveOTLPConfig(cfg, cmd, "configs/ingero.yaml")
	if err == nil {
		t.Fatalf("expected error on enabled-but-empty-endpoint; got nil")
	}
	if !strings.Contains(err.Error(), "configs/ingero.yaml") {
		t.Errorf("error missing config path: %v", err)
	}
	if !strings.Contains(err.Error(), "endpoint") {
		t.Errorf("error missing 'endpoint': %v", err)
	}
}

func TestResolveOTLPConfig_DisabledEmpty(t *testing.T) {
	t.Cleanup(func() { fleetPushOTLPEnabled = false; fleetPushOTLPEndpoint = ""; fleetPushOTLPInsecure = true })
	cmd := newOTLPTestCmd()
	if err := cmd.ParseFlags(nil); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	// Default zero AgentConfig: disabled, empty endpoint. This must NOT
	// error — it's the "no tracing" case which is the v0.13 default.
	got, err := resolveOTLPConfig(&config.AgentConfig{}, cmd, "test.yaml")
	if err != nil {
		t.Fatalf("disabled+empty must not error: %v", err)
	}
	if got.Enabled {
		t.Errorf("Enabled = true; want false")
	}
}

func TestResolveOTLPConfig_WhitespaceEndpointTreatedEmpty(t *testing.T) {
	t.Cleanup(func() { fleetPushOTLPEnabled = false; fleetPushOTLPEndpoint = ""; fleetPushOTLPInsecure = true })
	cmd := newOTLPTestCmd()
	if err := cmd.ParseFlags(nil); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	cfg := &config.AgentConfig{
		OTLP: config.OTLPConfig{Enabled: true, Endpoint: "   ", Insecure: true},
	}
	if _, err := resolveOTLPConfig(cfg, cmd, "x.yaml"); err == nil {
		t.Fatalf("whitespace-only endpoint must be rejected when enabled=true")
	}
}
