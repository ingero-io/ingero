package cli

import (
	"context"
	"fmt"
	"log"
	"log/slog"
	"os"
	"os/signal"
	"syscall"
	"time"

	"github.com/spf13/cobra"

	"github.com/ingero-io/ingero/internal/alerter"
	"github.com/ingero-io/ingero/internal/config"
	ingmcp "github.com/ingero-io/ingero/internal/mcp"
	"github.com/ingero-io/ingero/internal/store"
)

var mcpCmd = &cobra.Command{
	Use:   "mcp",
	Short: "Start MCP server for AI agent integration",
	Long: `Start an MCP (Model Context Protocol) server.

The MCP server exposes seven tools and one prompt to AI agents (e.g., Claude):

Tools:
  - get_check: Run system diagnostics
  - get_trace_stats: CUDA/host statistics (p50/p95/p99 or aggregate fallback)
  - get_causal_chains: Causal chains with severity and root cause
  - get_stacks: Resolved call stacks for CUDA/driver operations
  - run_demo: Run a synthetic demo scenario
  - get_test_report: GPU integration test report (JSON)
  - run_sql: Execute read-only SQL for ad-hoc analysis

Prompts:
  - /investigate: Guided investigation workflow (stats -> chains -> SQL)

By default, runs on stdio for Claude Code / MCP clients.
Use --http to start an HTTPS server (TLS 1.3) for curl and remote agents.
If no --tls-cert/--tls-key is provided, an ephemeral self-signed certificate
is generated automatically.

Requires 'ingero trace' to be running (or to have run recently)
so the SQLite database has events to query.

Examples:
  ingero mcp                     # start on stdio (for Claude Code)
  ingero mcp --http :8080        # HTTPS on port 8080 (self-signed cert)
  ingero mcp --http :8080 --tls-cert cert.pem --tls-key key.pem
  ingero mcp --db /path/to.db    # use custom database path

  # curl examples (with --http :8080):
  curl -sk https://localhost:8080/mcp \
    -H 'Content-Type: application/json' \
    -H 'Accept: application/json, text/event-stream' \
    -d '{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"get_check","arguments":{}}}' | jq`,

	RunE: mcpRunE,
}

var (
	mcpDBPath      string
	mcpHTTPAddr    string
	mcpTLSCert     string
	mcpTLSKey      string
	mcpLogPath     string
	mcpBearerToken string
)

func init() {
	mcpCmd.Flags().StringVar(&mcpDBPath, "db", "", "database path (default: ~/.ingero/ingero.db)")
	mcpCmd.Flags().StringVar(&mcpHTTPAddr, "http", "", "HTTPS listen address (e.g. :8080). If set, serves over HTTPS (TLS 1.3) instead of stdio")
	mcpCmd.Flags().StringVar(&mcpTLSCert, "tls-cert", "", "TLS certificate file (PEM). If omitted with --http, a self-signed cert is generated")
	mcpCmd.Flags().StringVar(&mcpTLSKey, "tls-key", "", "TLS private key file (PEM). Required if --tls-cert is set")
	mcpCmd.Flags().StringVar(&mcpLogPath, "log", "", "write log output to file (append, no rotation)")
	mcpCmd.Flags().StringVar(&mcpBearerToken, "mcp-bearer-token", "", "DEPRECATED — leaks via /proc/<pid>/cmdline. Set the token via the INGERO_MCP_BEARER_TOKEN environment variable. The flag is now refused at startup if non-empty.")
	mcpCmd.Flags().String("pagerduty-routing-key", "", "DEPRECATED — leaks via /proc/<pid>/cmdline. Set the routing key via the INGERO_PAGERDUTY_KEY environment variable or alerter.pagerduty.routing_key in --config. The flag is now refused at startup if non-empty.")
	rootCmd.AddCommand(mcpCmd)
}

// pagerDutyMCPEnabled answers whether the pagerduty_trigger MCP tool
// should be registered for this run. v0.15 item A replaces v0.14's
// blanket default-off with an identity-based gate:
//
//   - stdio mode (httpAddr == ""): loopback by definition; enabled.
//   - HTTP + bearer set: caller-identity enforced; enabled.
//   - HTTP + bearer empty: unauthenticated listener; stays gated.
//
// Standalone for unit testability without an MCP server.
func pagerDutyMCPEnabled(httpAddr, bearerToken string) bool {
	if httpAddr == "" {
		return true
	}
	return bearerToken != ""
}

// resolvePagerDutyRoutingKey composes the PagerDuty routing key from the
// parsed YAML and the INGERO_PAGERDUTY_KEY environment variable. The env var
// wins over YAML so operators can override the on-disk config without
// re-deploying the YAML. Empty string return means PagerDuty is not
// configured (the MCP tool stays registered but returns "not configured" at
// call time).
//
// The CLI flag is refused: --pagerduty-routing-key on the command line
// leaks the secret via /proc/<pid>/cmdline. Refusal is loud (a startup
// error) rather than silent because a routing key that survives in shell
// history defeats the entire point of constant-time compare downstream.
//
// Returned as a standalone function so the override-resolution logic is
// unit-testable without spinning up the MCP server.
func resolvePagerDutyRoutingKey(cfg *config.AgentConfig, cmd *cobra.Command) (string, error) {
	routingKey := ""
	if cfg != nil && cfg.Alerter.PagerDuty != nil {
		routingKey = cfg.Alerter.PagerDuty.RoutingKey
	}
	if cmd != nil && cmd.Flags().Changed("pagerduty-routing-key") {
		flagVal, _ := cmd.Flags().GetString("pagerduty-routing-key")
		if _, err := ResolveSecret("pagerduty-routing-key", "INGERO_PAGERDUTY_KEY", flagVal); err != nil {
			return "", err
		}
	}
	if envKey := os.Getenv("INGERO_PAGERDUTY_KEY"); envKey != "" {
		routingKey = envKey
	}
	return routingKey, nil
}

func mcpRunE(cmd *cobra.Command, args []string) error {
	// --log: redirect log output to a file (debug, no rotation).
	if mcpLogPath != "" {
		f, err := os.OpenFile(mcpLogPath, os.O_CREATE|os.O_WRONLY|os.O_APPEND, 0o644)
		if err != nil {
			return fmt.Errorf("opening log file %s: %w", mcpLogPath, err)
		}
		defer f.Close()
		log.SetOutput(f)
	}

	// Validate TLS flags: both or neither.
	if (mcpTLSCert == "") != (mcpTLSKey == "") {
		return fmt.Errorf("--tls-cert and --tls-key must be specified together")
	}

	// Bearer token resolution: env-only. Refuses --mcp-bearer-token to keep
	// the value out of /proc/<pid>/cmdline. Resolved into a local so the
	// package-level mcpBearerToken stays empty for the rest of the run.
	bearerToken, err := ResolveSecret("mcp-bearer-token", "INGERO_MCP_BEARER_TOKEN", mcpBearerToken)
	if err != nil {
		return err
	}

	dbPath := mcpDBPath
	if dbPath == "" {
		dbPath = store.DefaultDBPath()
	}

	// Open database if it exists. Tools that don't need the store (get_check,
	// run_demo, get_test_report) work without it; tools that do (get_trace_stats,
	// get_causal_chains, get_stacks, run_sql) return a helpful error.
	var s *store.Store
	if _, err := os.Stat(dbPath); err == nil {
		s, err = store.New(dbPath)
		if err != nil {
			return fmt.Errorf("opening database: %w", err)
		}
		defer s.Close()
	}
	debugf("mcp: db=%s http=%q store_available=%v", dbPath, mcpHTTPAddr, s != nil)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	srv := ingmcp.New(s)

	// Wire fleet.nodes from config for query_fleet tool.
	if fleetNodes := ReadFleetNodes(); len(fleetNodes) > 0 {
		srv.SetFleetNodes(fleetNodes)
		debugf("mcp: fleet nodes configured: %v", fleetNodes)
	}

	// Wire PagerDuty backend for pagerduty_trigger tool. v0.13: routing
	// key resolves from alerter.pagerduty.routing_key in the YAML, with
	// --pagerduty-routing-key as an explicit override. Routing key is
	// a secret; never log it. Read once into a local, hand to the
	// alerter, then drop the local. The alerter scrubs the key from
	// any error it returns.
	cfgPath, _ := cmd.Flags().GetString("config")
	agentCfg, err := config.Load(cfgPath)
	if err != nil {
		return fmt.Errorf("load config %s: %w", cfgPath, err)
	}
	pdKey, err := resolvePagerDutyRoutingKey(agentCfg, cmd)
	if err != nil {
		return err
	}
	if pdKey != "" {
		pd := alerter.NewPagerDuty(&alerter.PagerDutyConfig{RoutingKey: pdKey}, 5*time.Second, slog.Default())
		srv.SetPagerDuty(pd)
		pdKey = ""
		debugf("mcp: pagerduty backend configured")
	}

	pagerDutyEnabled := pagerDutyMCPEnabled(mcpHTTPAddr, bearerToken)
	srv.SetPagerDutyMCPEnabled(pagerDutyEnabled)
	debugf("mcp: pagerduty_trigger enabled=%v (http=%q bearer_set=%v)",
		pagerDutyEnabled, mcpHTTPAddr, bearerToken != "")

	if mcpHTTPAddr != "" {
		return srv.RunHTTP(ctx, mcpHTTPAddr, mcpTLSCert, mcpTLSKey, bearerToken)
	}
	return srv.Run(ctx)
}
