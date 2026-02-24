package cli

import (
	"fmt"

	"github.com/ingero-io/ingero/internal/version"
	"github.com/spf13/cobra"
)

// versionCmd prints build information.
var versionCmd = &cobra.Command{
	Use:   "version",
	Short: "Print build information",
	RunE: func(cmd *cobra.Command, args []string) error {
		fmt.Printf("ingero %s\n", version.String())
		return nil
	},
}

func init() {
	rootCmd.AddCommand(versionCmd)
}
