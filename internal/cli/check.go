package cli

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/ingero-io/ingero/internal/discover"
	"github.com/spf13/cobra"
)

var checkJSON bool

var checkCmd = &cobra.Command{
	Use:   "check",
	Short: "Check system readiness for GPU tracing",
	Long: `Check kernel version, BTF support, NVIDIA driver, CUDA libraries,
and running GPU processes. Reports what Ingero can trace and what's missing.`,

	RunE: func(cmd *cobra.Command, args []string) error {
		debugf("check: running all system checks")

		results := discover.RunAllChecks()

		if checkJSON {
			return checkOutputJSON(results)
		}

		fmt.Println("Ingero — System Readiness Check")
		fmt.Println()

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
	checkCmd.Flags().BoolVar(&checkJSON, "json", false, "output as JSON")
	rootCmd.AddCommand(checkCmd)
}

// checkOutputJSON renders system check results as JSON.
func checkOutputJSON(results []discover.CheckResult) error {
	type jsonCheck struct {
		Name     string `json:"name"`
		Pass     bool   `json:"pass"`
		Optional bool   `json:"optional,omitempty"`
		Value    string `json:"value,omitempty"`
		Detail   string `json:"detail,omitempty"`
	}

	allOK := true
	items := make([]jsonCheck, len(results))
	for i, r := range results {
		items[i] = jsonCheck{
			Name:     r.Name,
			Pass:     r.OK,
			Optional: r.Optional,
			Value:    r.Value,
			Detail:   r.Detail,
		}
		if !r.OK && !r.Optional {
			allOK = false
		}
	}

	output := struct {
		AllPassed bool        `json:"all_passed"`
		Checks   []jsonCheck `json:"checks"`
	}{
		AllPassed: allOK,
		Checks:   items,
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(output)
}
