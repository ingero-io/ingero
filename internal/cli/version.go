package cli

import (
	"fmt"
	"time"

	"github.com/ingero-io/ingero/internal/update"
	"github.com/ingero-io/ingero/internal/version"
	"github.com/spf13/cobra"
)

// versionCmd prints build information and checks for updates.
// Unlike other commands (which use the async PrintNotice in PersistentPostRun),
// version does a synchronous check with a short timeout — because the command
// exits instantly and the async goroutine wouldn't have time to finish.
var versionCmd = &cobra.Command{
	Use:   "version",
	Short: "Print build information",
	RunE: func(cmd *cobra.Command, args []string) error {
		fmt.Printf("ingero %s\n", version.String())

		ch := update.CheckInBackground(version.Version())
		update.WaitNotice(ch, 3*time.Second)
		return nil
	},
}

func init() {
	rootCmd.AddCommand(versionCmd)
}
