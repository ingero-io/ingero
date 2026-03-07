// Package main is the entry point for the ingero CLI.
package main

import (
	"fmt"
	"os"

	"github.com/ingero-io/ingero/internal/cli"
)

func main() {
	if err := cli.Execute(); err != nil {
		fmt.Fprintf(os.Stderr, "Error: %v\n", err)
		os.Exit(1)
	}
}
