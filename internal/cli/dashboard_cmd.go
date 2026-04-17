package cli

import (
	"context"
	"fmt"
	"os"
	"os/signal"
	"syscall"

	"github.com/spf13/cobra"

	"github.com/ingero-io/ingero/internal/dashboard"
	"github.com/ingero-io/ingero/internal/store"
)

var dashboardCmd = &cobra.Command{
	Use:   "dashboard",
	Short: "Start GPU monitoring dashboard (HTTPS)",
	Long: `Start the Ingero GPU monitoring dashboard as an HTTPS server.

The dashboard provides a live browser-based view of GPU causal
observability data from the SQLite event store. Metrics that Ingero
does not collect (GPU utilization, SM stalls, NVLink, etc.) are
grayed out with tooltips naming the required external tool.

Requires 'ingero trace' to be running (or to have run recently)
so the SQLite database has events to query.

Examples:
  ingero dashboard                           # HTTPS on :8080 (self-signed TLS 1.3)
  ingero dashboard --addr :9090              # custom port
  ingero dashboard --db /path/to/ingero.db   # custom database
  ingero dashboard --tls-cert cert.pem --tls-key key.pem

  # Remote access via SSH tunnel:
  ssh -L 8080:localhost:8080 user@gpu-vm
  # Then open https://localhost:8080 in browser`,

	RunE: dashboardRunE,
}

var (
	dashAddr    string
	dashDBPath  string
	dashTLSCert string
	dashTLSKey  string
	dashNoTLS   bool
	dashToken   string
)

func init() {
	dashboardCmd.Flags().StringVar(&dashAddr, "addr", ":8080", "HTTPS listen address")
	dashboardCmd.Flags().StringVar(&dashDBPath, "db", "", "database path (default: ~/.ingero/ingero.db)")
	dashboardCmd.Flags().StringVar(&dashTLSCert, "tls-cert", "", "TLS certificate file (PEM). If omitted, a self-signed cert is generated")
	dashboardCmd.Flags().StringVar(&dashTLSKey, "tls-key", "", "TLS private key file (PEM). Required if --tls-cert is set")
	dashboardCmd.Flags().BoolVar(&dashNoTLS, "no-tls", false, "serve plain HTTP (for fleet queries on trusted networks)")
	dashboardCmd.Flags().StringVar(&dashToken, "token", "", "Bearer token required for /api/ endpoints (recommended when listening on non-loopback)")
	rootCmd.AddCommand(dashboardCmd)
}

func dashboardRunE(cmd *cobra.Command, args []string) error {
	// Validate TLS flags: both or neither.
	if (dashTLSCert == "") != (dashTLSKey == "") {
		return fmt.Errorf("--tls-cert and --tls-key must be specified together")
	}

	dbPath := dashDBPath
	if dbPath == "" {
		dbPath = store.DefaultDBPath()
	}

	// Open database if it exists.
	var s *store.Store
	if _, err := os.Stat(dbPath); err == nil {
		s, err = store.New(dbPath)
		if err != nil {
			return fmt.Errorf("opening database: %w", err)
		}
		defer s.Close()
	}
	debugf("dashboard: db=%s addr=%s store_available=%v", dbPath, dashAddr, s != nil)

	ctx, stop := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer stop()

	srv := dashboard.New(s, dashAddr, dashTLSCert, dashTLSKey)
	srv.SetNoTLS(dashNoTLS)
	if dashToken != "" {
		srv.SetToken(dashToken)
	}
	return srv.Start(ctx)
}
