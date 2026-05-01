package cli

import (
	"encoding/json"
	"fmt"
	"os"

	"github.com/ingero-io/ingero/internal/discover"
	"github.com/ingero-io/ingero/internal/support"
	"github.com/ingero-io/ingero/internal/version"
	"github.com/spf13/cobra"
)

var (
	checkJSON          bool
	checkSupportBundle string
)

var checkCmd = &cobra.Command{
	Use:   "check",
	Short: "Check system readiness for GPU tracing",
	Long: `Check kernel version, BTF support, NVIDIA driver, CUDA libraries,
and running GPU processes. Reports what Ingero can trace and what's missing.

Add --support-bundle <path> to also write a tarball with kernel info,
BTF state, GPU + driver state, agent version, last 1000 lines of any
~/.ingero/{trace,sink}.log, and a redacted environment dump. Attach
the tarball to a support case so the developer triaging it has every
diagnostic input in one place.`,

	RunE: func(cmd *cobra.Command, args []string) error {
		debugf("check: running all system checks")

		results := discover.RunAllChecks()

		if checkSupportBundle != "" {
			path, n, err := support.Bundle(checkSupportBundle, version.Version(), version.Commit())
			if err != nil {
				fmt.Fprintf(os.Stderr, "ingero check: support bundle: %v\n", err)
			} else {
				fmt.Fprintf(os.Stderr, "ingero check: wrote support bundle to %s (%d entries)\n", path, n)
			}
		}

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
			if !r.OK && r.Recommendation != "" {
				fmt.Printf("      Recommendation: %s\n", r.Recommendation)
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
	checkCmd.Flags().StringVar(&checkSupportBundle, "support-bundle", "",
		"Write a diagnostic tarball at the given path (e.g. /tmp/ingero-bundle.tgz). Attach to support cases.")
	rootCmd.AddCommand(checkCmd)
}

// checkOutputJSON renders system check results as JSON.
func checkOutputJSON(results []discover.CheckResult) error {
	type jsonCheck struct {
		Name           string `json:"name"`
		Pass           bool   `json:"pass"`
		Optional       bool   `json:"optional,omitempty"`
		Value          string `json:"value,omitempty"`
		Detail         string `json:"detail,omitempty"`
		Recommendation string `json:"recommendation,omitempty"`
	}

	allOK := true
	items := make([]jsonCheck, len(results))
	for i, r := range results {
		items[i] = jsonCheck{
			Name:           r.Name,
			Pass:           r.OK,
			Optional:       r.Optional,
			Value:          r.Value,
			Detail:         r.Detail,
			Recommendation: r.Recommendation,
		}
		if !r.OK && !r.Optional {
			allOK = false
		}
	}

	output := struct {
		AllPassed bool        `json:"all_passed"`
		Checks    []jsonCheck `json:"checks"`
	}{
		AllPassed: allOK,
		Checks:    items,
	}

	enc := json.NewEncoder(os.Stdout)
	enc.SetIndent("", "  ")
	return enc.Encode(output)
}
