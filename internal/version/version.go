// Package version provides build-time version information.
// Variables set via -ldflags at build time (see Makefile).
package version

import "fmt"

// These are set via -ldflags at build time. See Makefile.
var (
	version = "dev"
	commit  = "unknown"
	date    = "unknown"
)

// String returns a formatted version string for display.
func String() string {
	return fmt.Sprintf("%s (commit: %s, built: %s)", version, commit, date)
}

// Version returns just the version tag.
func Version() string {
	return version
}

// Commit returns the git commit hash.
func Commit() string {
	return commit
}
