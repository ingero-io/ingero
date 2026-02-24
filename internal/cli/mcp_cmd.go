package cli

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"

	ingmcp "github.com/ingero-io/ingero/internal/mcp"
	"github.com/ingero-io/ingero/internal/store"
)

var mcpCmd = &cobra.Command{
	Use:   "mcp",
	Short: "Start MCP server for AI agent integration",
	Long: `Start an MCP (Model Context Protocol) server.

The MCP server exposes six tools to AI agents (e.g., Claude):
  - get_check: Run system diagnostics
  - get_trace_stats: Get recent CUDA/host statistics
  - query_events: Query stored events with filters
  - get_causal_chains: Analyze events and return causal chains
  - run_demo: Run a synthetic demo scenario
  - get_test_report: Return the GPU integration test report

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
	mcpDBPath   string
	mcpHTTPAddr string
	mcpTLSCert  string
	mcpTLSKey   string
)

func init() {
	mcpCmd.Flags().StringVar(&mcpDBPath, "db", "", "database path (default: ~/.ingero/ingero.db)")
	mcpCmd.Flags().StringVar(&mcpHTTPAddr, "http", "", "HTTPS listen address (e.g. :8080). If set, serves over HTTPS (TLS 1.3) instead of stdio")
	mcpCmd.Flags().StringVar(&mcpTLSCert, "tls-cert", "", "TLS certificate file (PEM). If omitted with --http, a self-signed cert is generated")
	mcpCmd.Flags().StringVar(&mcpTLSKey, "tls-key", "", "TLS private key file (PEM). Required if --tls-cert is set")
	rootCmd.AddCommand(mcpCmd)
}

func mcpRunE(cmd *cobra.Command, args []string) error {
	// Validate TLS flags: both or neither.
	if (mcpTLSCert == "") != (mcpTLSKey == "") {
		return fmt.Errorf("--tls-cert and --tls-key must be specified together")
	}

	dbPath := mcpDBPath
	if dbPath == "" {
		dbPath = store.DefaultDBPath()
	}

	// Open database if it exists. Tools that don't need the store (get_check,
	// run_demo) work without it; tools that do (get_trace_stats, query_events,
	// get_causal_chains) return a helpful error.
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

	if mcpHTTPAddr != "" {
		return srv.RunHTTP(ctx, mcpHTTPAddr, mcpTLSCert, mcpTLSKey)
	}
	return srv.Run(ctx)
}
