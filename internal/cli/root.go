// Package cli implements the ingero command-line interface using cobra.
package cli

import (
	"log"

	"github.com/ingero-io/ingero/internal/update"
	"github.com/ingero-io/ingero/internal/version"
	"github.com/spf13/cobra"
)

// updateCh receives the result of a background version check.
// Set in PersistentPreRun, read in PersistentPostRun.
var updateCh <-chan update.Result

// debugMode enables diagnostic output on stderr. Set via --debug flag.
var debugMode bool

// debugf prints a debug message if --debug is enabled.
// Zero cost when off: just a bool check (~1ns).
//
// Output goes to Go's log package — normally stderr, but redirected to a file
// when --log is set (via log.SetOutput). This is why we use log.Printf here
// instead of fmt.Fprintf(os.Stderr): --log must capture debug output.
//
// Use in initialization/setup paths (probe attachment, auto-detect, store open).
// For hot paths, use periodic counters — never call debugf per-event.
func debugf(format string, args ...any) {
	if debugMode {
		log.Printf("[DEBUG] "+format, args...)
	}
}

var rootCmd = &cobra.Command{
	Use:   "ingero",
	Short: "Kernel-level causal tracing for GPU production workloads",
	Long: `Kernel-level causal tracing for GPU production workloads with <2% overhead.

Attach to any running CUDA process — see latencies, allocations, anomalies in real time.
No code changes required.

  ingero demo               Auto-detect GPU, run all demo scenarios
  ingero demo cold-start    Run a specific scenario (cold-start, memcpy-bottleneck, periodic-spike)
  sudo ingero trace         Live-trace a running CUDA process
  sudo ingero check         Check system readiness`,

	// SilenceUsage prevents cobra from printing usage on every error.
	// We only want usage on --help, not on runtime failures.
	SilenceUsage: true,

	// SilenceErrors prevents cobra from printing errors itself.
	// We handle error display in main() for consistent formatting.
	// Without this, errors print twice: once by cobra, once by main().
	SilenceErrors: true,
}

// Execute runs the root command. Called from main().
func Execute() error {
	return rootCmd.Execute()
}

func init() {
	// --debug flag: available on all subcommands via PersistentFlags.
	rootCmd.PersistentFlags().BoolVar(&debugMode, "debug", false, "enable diagnostic output on stderr")

	// Set version string — cobra handles "ingero version" and "--version" for us.
	rootCmd.Version = version.String()

	// Disable the auto-generated "completion" command — not useful for a CLI tool
	// that users run directly (not in scripts that need shell completion).
	rootCmd.CompletionOptions.DisableDefaultCmd = true

	// Background update check: start before command runs, print after.
	// Skipped for: ingero (bare help), version (is version-display itself),
	// query (data pipeline), mcp (long-running server).
	//
	// NOTE: cobra "closest parent wins" — if ANY subcommand defines its own
	// PersistentPreRun[E], the root's hook is silently skipped for that command.
	// Currently no subcommand defines PersistentPreRun. Keep it that way.
	rootCmd.PersistentPreRun = func(cmd *cobra.Command, args []string) {
		switch cmd.Name() {
		case "ingero", "version", "query", "mcp":
			return
		}
		updateCh = update.CheckInBackground(version.Version())
	}

	rootCmd.PersistentPostRun = func(cmd *cobra.Command, args []string) {
		if updateCh != nil {
			update.PrintNotice(updateCh)
		}
	}

	// Subcommands are registered in their own files' init() functions.
	// See check.go and trace.go — each calls rootCmd.AddCommand().
}
