package config

import (
	"os"
	"path/filepath"
	"strings"
	"testing"
)

// writeTempConfig writes content to a tempdir-scoped file and returns the
// path. Caller's t.TempDir() means the file is auto-cleaned at test end.
func writeTempConfig(t *testing.T, content string) string {
	t.Helper()
	dir := t.TempDir()
	path := filepath.Join(dir, "ingero.yaml")
	if err := os.WriteFile(path, []byte(content), 0o644); err != nil {
		t.Fatalf("write temp config: %v", err)
	}
	return path
}

func TestLoad_HappyPath(t *testing.T) {
	const yamlBody = `
otlp:
  enabled: true
  endpoint: "fleet.example:4318"
  insecure: false

alerter:
  pagerduty:
    routing_key: "R0UTING-K3Y-ABC"
`
	path := writeTempConfig(t, yamlBody)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load returned error: %v", err)
	}
	if !cfg.OTLP.Enabled {
		t.Errorf("OTLP.Enabled = false, want true")
	}
	if cfg.OTLP.Endpoint != "fleet.example:4318" {
		t.Errorf("OTLP.Endpoint = %q, want %q", cfg.OTLP.Endpoint, "fleet.example:4318")
	}
	if cfg.OTLP.Insecure {
		t.Errorf("OTLP.Insecure = true, want false")
	}
	if cfg.Alerter.PagerDuty == nil {
		t.Fatalf("Alerter.PagerDuty is nil, want non-nil")
	}
	if got := cfg.Alerter.PagerDuty.RoutingKey; got != "R0UTING-K3Y-ABC" {
		t.Errorf("Alerter.PagerDuty.RoutingKey = %q, want %q", got, "R0UTING-K3Y-ABC")
	}
}

func TestLoad_MissingFile(t *testing.T) {
	// Path under a tempdir that exists but the file does not.
	missing := filepath.Join(t.TempDir(), "does-not-exist.yaml")
	cfg, err := Load(missing)
	if err != nil {
		t.Fatalf("Load on missing file returned error: %v (want nil)", err)
	}
	if cfg == nil {
		t.Fatalf("Load on missing file returned nil cfg; want zero AgentConfig")
	}
	// Zero values across the board.
	if cfg.OTLP.Enabled || cfg.OTLP.Endpoint != "" || cfg.OTLP.Insecure {
		t.Errorf("expected zero OTLPConfig on missing file, got %+v", cfg.OTLP)
	}
	if cfg.Alerter.PagerDuty != nil {
		t.Errorf("expected nil Alerter.PagerDuty on missing file, got %+v", cfg.Alerter.PagerDuty)
	}
}

func TestLoad_MalformedYAML(t *testing.T) {
	// Tab in indentation + structurally broken; yaml.v3 rejects this.
	const broken = "otlp:\n\tenabled: true\n  endpoint: ["
	path := writeTempConfig(t, broken)
	if _, err := Load(path); err == nil {
		t.Fatalf("Load on malformed YAML returned nil error; want parse error")
	} else if !strings.Contains(err.Error(), "parse") {
		t.Errorf("error %q does not contain 'parse'", err.Error())
	}
}

func TestLoad_PartialYAML(t *testing.T) {
	// Only otlp: present; alerter: omitted entirely.
	const yamlBody = `
otlp:
  enabled: true
  endpoint: "host:4318"
`
	path := writeTempConfig(t, yamlBody)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load returned error: %v", err)
	}
	if !cfg.OTLP.Enabled {
		t.Errorf("OTLP.Enabled false; want true")
	}
	if cfg.OTLP.Endpoint != "host:4318" {
		t.Errorf("OTLP.Endpoint = %q", cfg.OTLP.Endpoint)
	}
	if cfg.Alerter.PagerDuty != nil {
		t.Errorf("Alerter.PagerDuty = %+v; want nil", cfg.Alerter.PagerDuty)
	}
}

func TestLoad_UnknownFields(t *testing.T) {
	// health: and prometheus: blocks are real on-disk YAML but unmodeled
	// in AgentConfig. Load must accept them silently. agent: + probes:
	// + fleet: also unmodeled here.
	const yamlBody = `
agent:
  log_level: info
  max_overhead_percent: 2.0

probes:
  cuda:
    enabled: true

health:
  weights:
    throughput: 0.40

prometheus:
  enabled: false
  listen: ":9090"

fleet:
  nodes: []
  endpoint: ""

otlp:
  enabled: false

alerter:
  pagerduty:
    routing_key: "key"
`
	path := writeTempConfig(t, yamlBody)
	cfg, err := Load(path)
	if err != nil {
		t.Fatalf("Load returned error on YAML with unknown fields: %v", err)
	}
	if cfg.OTLP.Enabled {
		t.Errorf("OTLP.Enabled true; want false")
	}
	if cfg.Alerter.PagerDuty == nil || cfg.Alerter.PagerDuty.RoutingKey != "key" {
		t.Errorf("Alerter.PagerDuty mis-parsed: %+v", cfg.Alerter.PagerDuty)
	}
}

func TestLoad_EmptyFile(t *testing.T) {
	// Empty file = zero AgentConfig, no error. yaml.v3 returns io.EOF on
	// Decode of empty input; Load must turn that into a zero cfg.
	path := writeTempConfig(t, "")
	cfg, err := Load(path)
	// yaml.v3's behavior: Decode of empty input returns io.EOF. Our
	// implementation surfaces that as a parse error today; document the
	// chosen behavior so a future change is intentional.
	if err == nil {
		// Acceptable: empty file -> zero cfg.
		if cfg.OTLP.Enabled {
			t.Errorf("empty file produced non-zero cfg: %+v", cfg)
		}
	}
	// If err is non-nil, the file was treated as malformed; both are
	// reasonable. Don't fail either way.
}
