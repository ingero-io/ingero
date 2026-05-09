package cli

import (
	"testing"

	"github.com/spf13/cobra"

	"github.com/ingero-io/ingero/internal/config"
)

// newPagerDutyTestCmd builds a cobra.Command with just the
// --pagerduty-routing-key flag registered. Mirrors mcp_cmd.go's init().
func newPagerDutyTestCmd() *cobra.Command {
	cmd := &cobra.Command{Use: "test"}
	cmd.Flags().String("pagerduty-routing-key", "", "")
	return cmd
}

func TestResolvePagerDutyRoutingKey_YAMLOnly(t *testing.T) {
	cmd := newPagerDutyTestCmd()
	if err := cmd.ParseFlags(nil); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	cfg := &config.AgentConfig{
		Alerter: config.AlerterConfig{
			PagerDuty: &config.PagerDutyYAML{RoutingKey: "yaml-key-123"},
		},
	}
	got := resolvePagerDutyRoutingKey(cfg, cmd)
	if got != "yaml-key-123" {
		t.Errorf("got %q; want yaml-key-123", got)
	}
}

func TestResolvePagerDutyRoutingKey_FlagOverrides(t *testing.T) {
	cmd := newPagerDutyTestCmd()
	if err := cmd.ParseFlags([]string{"--pagerduty-routing-key", "flag-key-456"}); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	cfg := &config.AgentConfig{
		Alerter: config.AlerterConfig{
			PagerDuty: &config.PagerDutyYAML{RoutingKey: "yaml-key-123"},
		},
	}
	got := resolvePagerDutyRoutingKey(cfg, cmd)
	if got != "flag-key-456" {
		t.Errorf("got %q; want flag-key-456 (CLI must override YAML)", got)
	}
}

func TestResolvePagerDutyRoutingKey_FlagOnly(t *testing.T) {
	cmd := newPagerDutyTestCmd()
	if err := cmd.ParseFlags([]string{"--pagerduty-routing-key", "only-flag"}); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	got := resolvePagerDutyRoutingKey(&config.AgentConfig{}, cmd)
	if got != "only-flag" {
		t.Errorf("got %q; want only-flag", got)
	}
}

func TestResolvePagerDutyRoutingKey_NeitherSet(t *testing.T) {
	cmd := newPagerDutyTestCmd()
	if err := cmd.ParseFlags(nil); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	got := resolvePagerDutyRoutingKey(&config.AgentConfig{}, cmd)
	if got != "" {
		t.Errorf("got %q; want empty (PagerDuty not configured)", got)
	}
}

func TestResolvePagerDutyRoutingKey_NilAlerterBlock(t *testing.T) {
	cmd := newPagerDutyTestCmd()
	if err := cmd.ParseFlags(nil); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	// AlerterConfig present but PagerDuty nil — common when YAML omits
	// the block entirely. Must not panic, must return "".
	cfg := &config.AgentConfig{
		Alerter: config.AlerterConfig{PagerDuty: nil},
	}
	got := resolvePagerDutyRoutingKey(cfg, cmd)
	if got != "" {
		t.Errorf("got %q; want empty", got)
	}
}

func TestResolvePagerDutyRoutingKey_NilCfg(t *testing.T) {
	cmd := newPagerDutyTestCmd()
	if err := cmd.ParseFlags([]string{"--pagerduty-routing-key", "flag-only"}); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	// Nil cfg shouldn't panic; flag must still be honored.
	got := resolvePagerDutyRoutingKey(nil, cmd)
	if got != "flag-only" {
		t.Errorf("got %q; want flag-only", got)
	}
}

func TestResolvePagerDutyRoutingKey_FlagEmptyExplicit(t *testing.T) {
	// Operator explicitly passes --pagerduty-routing-key="" - the
	// "Changed" path triggers and overrides any YAML value with "".
	// Documents the chosen semantics.
	cmd := newPagerDutyTestCmd()
	if err := cmd.ParseFlags([]string{"--pagerduty-routing-key="}); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	cfg := &config.AgentConfig{
		Alerter: config.AlerterConfig{
			PagerDuty: &config.PagerDutyYAML{RoutingKey: "yaml-key"},
		},
	}
	got := resolvePagerDutyRoutingKey(cfg, cmd)
	if got != "" {
		t.Errorf("got %q; want '' (explicit empty flag must override YAML)", got)
	}
}

// v0.15 item A: pagerduty_trigger gates on listener identity.
func TestPagerDutyMCPEnabled_StdioModeAlways(t *testing.T) {
	if !pagerDutyMCPEnabled("", "") {
		t.Errorf("stdio mode (no http) should enable pagerduty_trigger by default; loopback by definition")
	}
	if !pagerDutyMCPEnabled("", "any-token") {
		t.Errorf("stdio mode with bearer should also enable")
	}
}

func TestPagerDutyMCPEnabled_HTTPRequiresBearer(t *testing.T) {
	if pagerDutyMCPEnabled(":8080", "") {
		t.Errorf("HTTP without bearer must NOT enable pagerduty_trigger (v0.14 R3 ★4 caveat)")
	}
	if pagerDutyMCPEnabled("0.0.0.0:8080", "") {
		t.Errorf("HTTP on all-interfaces without bearer must stay gated")
	}
}

func TestPagerDutyMCPEnabled_HTTPWithBearerEnables(t *testing.T) {
	if !pagerDutyMCPEnabled(":8080", "secret-token") {
		t.Errorf("HTTP + bearer should enable pagerduty_trigger; identity-bearing listener")
	}
	if !pagerDutyMCPEnabled("127.0.0.1:8080", "tok") {
		t.Errorf("HTTP loopback + bearer should enable")
	}
}
