package cli

import (
	"fmt"

	"github.com/ingero-io/ingero/internal/discover"
	"github.com/spf13/cobra"
)

var checkCmd = &cobra.Command{
	Use:   "check",
	Short: "Check system readiness for GPU tracing",
	Long: `Check kernel version, BTF support, NVIDIA driver, CUDA libraries,
and running GPU processes. Reports what Ingero can trace and what's missing.`,

	RunE: func(cmd *cobra.Command, args []string) error {
		debugf("check: running all system checks")
		fmt.Println("Ingero — System Readiness Check")
		fmt.Println()

		results := discover.RunAllChecks()

		allOK := true
		for _, r := range results {
			icon := "✓"
			if !r.OK {
				if r.Optional {
					icon = "~"
				} else {
					icon = "✗"
					allOK = false
				}
			}

			if r.Value != "" {
				fmt.Printf("  [%s] %s: %s\n", icon, r.Name, r.Value)
			} else {
				fmt.Printf("  [%s] %s\n", icon, r.Name)
			}
			if r.Detail != "" {
				fmt.Printf("      %s\n", r.Detail)
			}
		}

		debugf("check: %d checks completed, all_ok=%v", len(results), allOK)

		fmt.Println()
		if allOK {
			fmt.Println("All checks passed — ready to trace!")
		} else {
			fmt.Println("Some checks failed — see above for details.")
		}

		return nil
	},
}

func init() {
	rootCmd.AddCommand(checkCmd)
}
