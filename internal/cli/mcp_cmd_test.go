package cli

import (
	"strings"
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
	t.Setenv("INGERO_PAGERDUTY_KEY", "")
	cmd := newPagerDutyTestCmd()
	if err := cmd.ParseFlags(nil); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	cfg := &config.AgentConfig{
		Alerter: config.AlerterConfig{
			PagerDuty: &config.PagerDutyYAML{RoutingKey: "yaml-key-123"},
		},
	}
	got, err := resolvePagerDutyRoutingKey(cfg, cmd)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "yaml-key-123" {
		t.Errorf("got %q; want yaml-key-123", got)
	}
}

func TestResolvePagerDutyRoutingKey_EnvOverridesYAML(t *testing.T) {
	t.Setenv("INGERO_PAGERDUTY_KEY", "env-key-789")
	cmd := newPagerDutyTestCmd()
	if err := cmd.ParseFlags(nil); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	cfg := &config.AgentConfig{
		Alerter: config.AlerterConfig{
			PagerDuty: &config.PagerDutyYAML{RoutingKey: "yaml-key-123"},
		},
	}
	got, err := resolvePagerDutyRoutingKey(cfg, cmd)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "env-key-789" {
		t.Errorf("got %q; want env-key-789 (env must override YAML)", got)
	}
}

func TestResolvePagerDutyRoutingKey_FlagIsRefused(t *testing.T) {
	t.Setenv("INGERO_PAGERDUTY_KEY", "")
	cmd := newPagerDutyTestCmd()
	if err := cmd.ParseFlags([]string{"--pagerduty-routing-key", "flag-key-456"}); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	cfg := &config.AgentConfig{
		Alerter: config.AlerterConfig{
			PagerDuty: &config.PagerDutyYAML{RoutingKey: "yaml-key-123"},
		},
	}
	_, err := resolvePagerDutyRoutingKey(cfg, cmd)
	if err == nil {
		t.Fatal("expected refusal when --pagerduty-routing-key supplies a value")
	}
	if !strings.Contains(err.Error(), "INGERO_PAGERDUTY_KEY") {
		t.Errorf("error %q should name the env var", err)
	}
}

func TestResolvePagerDutyRoutingKey_FlagRefusedEvenWithoutYAML(t *testing.T) {
	t.Setenv("INGERO_PAGERDUTY_KEY", "")
	cmd := newPagerDutyTestCmd()
	if err := cmd.ParseFlags([]string{"--pagerduty-routing-key", "only-flag"}); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	_, err := resolvePagerDutyRoutingKey(&config.AgentConfig{}, cmd)
	if err == nil {
		t.Fatal("expected refusal for --pagerduty-routing-key regardless of YAML state")
	}
}

func TestResolvePagerDutyRoutingKey_FlagRefusedEvenWithEnvSet(t *testing.T) {
	// Refuse the leaky flag even when env is also configured. Silently
	// preferring env would let the flag survive in shell history without
	// the operator noticing.
	t.Setenv("INGERO_PAGERDUTY_KEY", "env-key-789")
	cmd := newPagerDutyTestCmd()
	if err := cmd.ParseFlags([]string{"--pagerduty-routing-key", "flag-key-456"}); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	_, err := resolvePagerDutyRoutingKey(&config.AgentConfig{}, cmd)
	if err == nil {
		t.Fatal("expected refusal even when env is set")
	}
}

func TestResolvePagerDutyRoutingKey_NeitherSet(t *testing.T) {
	t.Setenv("INGERO_PAGERDUTY_KEY", "")
	cmd := newPagerDutyTestCmd()
	if err := cmd.ParseFlags(nil); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	got, err := resolvePagerDutyRoutingKey(&config.AgentConfig{}, cmd)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "" {
		t.Errorf("got %q; want empty (PagerDuty not configured)", got)
	}
}

func TestResolvePagerDutyRoutingKey_NilAlerterBlock(t *testing.T) {
	t.Setenv("INGERO_PAGERDUTY_KEY", "")
	cmd := newPagerDutyTestCmd()
	if err := cmd.ParseFlags(nil); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	// AlerterConfig present but PagerDuty nil — common when YAML omits
	// the block entirely. Must not panic, must return "".
	cfg := &config.AgentConfig{
		Alerter: config.AlerterConfig{PagerDuty: nil},
	}
	got, err := resolvePagerDutyRoutingKey(cfg, cmd)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "" {
		t.Errorf("got %q; want empty", got)
	}
}

func TestResolvePagerDutyRoutingKey_NilCfgWithEnv(t *testing.T) {
	t.Setenv("INGERO_PAGERDUTY_KEY", "env-only")
	cmd := newPagerDutyTestCmd()
	if err := cmd.ParseFlags(nil); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	// Nil cfg shouldn't panic; env must still be honored.
	got, err := resolvePagerDutyRoutingKey(nil, cmd)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "env-only" {
		t.Errorf("got %q; want env-only", got)
	}
}

func TestResolvePagerDutyRoutingKey_FlagEmptyExplicitNoLongerOverrides(t *testing.T) {
	// Operator explicitly passes --pagerduty-routing-key="" - the empty
	// flag is treated as "no value supplied" (ResolveSecret only refuses
	// non-empty flag values), so YAML stands. This is a deliberate change
	// from the pre-deprecation behavior: the explicit-empty-overrides path
	// no longer exists because the flag itself is no longer a supported
	// input channel for the value.
	t.Setenv("INGERO_PAGERDUTY_KEY", "")
	cmd := newPagerDutyTestCmd()
	if err := cmd.ParseFlags([]string{"--pagerduty-routing-key="}); err != nil {
		t.Fatalf("parse flags: %v", err)
	}
	cfg := &config.AgentConfig{
		Alerter: config.AlerterConfig{
			PagerDuty: &config.PagerDutyYAML{RoutingKey: "yaml-key"},
		},
	}
	got, err := resolvePagerDutyRoutingKey(cfg, cmd)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != "yaml-key" {
		t.Errorf("got %q; want yaml-key (explicit empty flag no longer overrides YAML)", got)
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
		t.Errorf("HTTP without bearer must NOT enable pagerduty_trigger (v0.14 R3 caveat)")
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
