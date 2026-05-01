// ingero-alerter is a sidecar that consumes straggler events from
// the Ingero agent's remediation UDS socket and dispatches them to
// Slack incoming webhooks and/or PagerDuty Events API v2.
//
// Run with:
//
//	ingero-alerter --config /etc/ingero-alerter/config.json
//
// The config schema is documented in internal/alerter/alerter.go;
// see deploy/systemd/ingero-alerter.service for a reference unit.
package main

import (
	"context"
	"flag"
	"fmt"
	"log/slog"
	"os"
	"os/signal"
	"syscall"

	"github.com/ingero-io/ingero/internal/alerter"
)

func main() {
	configPath := flag.String("config", "/etc/ingero-alerter/config.json", "path to alerter config JSON")
	verbose := flag.Bool("verbose", false, "enable debug logging")
	flag.Parse()

	level := slog.LevelInfo
	if *verbose {
		level = slog.LevelDebug
	}
	log := slog.New(slog.NewTextHandler(os.Stderr, &slog.HandlerOptions{Level: level}))

	cfg, err := alerter.LoadConfigFile(*configPath)
	if err != nil {
		fmt.Fprintln(os.Stderr, "ingero-alerter:", err)
		os.Exit(2)
	}
	if err := cfg.Validate(); err != nil {
		fmt.Fprintln(os.Stderr, "ingero-alerter:", err)
		os.Exit(2)
	}

	ctx, cancel := signal.NotifyContext(context.Background(), syscall.SIGINT, syscall.SIGTERM)
	defer cancel()

	if err := alerter.Run(ctx, cfg, log); err != nil {
		fmt.Fprintln(os.Stderr, "ingero-alerter:", err)
		os.Exit(1)
	}
}
