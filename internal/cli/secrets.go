// Package cli — secrets.go provides cross-command helpers for safe-default
// handling of operator-supplied secrets and bind-address loopback checks.
//
// CLI arguments are visible to any process on the host via /proc/<pid>/cmdline,
// shell history, ps output, and systemd journal _CMDLINE. Secrets passed via
// --flag=<value> therefore leak before any constant-time comparison happens.
// AssertSecretSafe refuses values whose source is not in the safe set; the
// convenience wrapper ResolveSecret pairs that refusal with an env-var read.
//
// IsLoopback centralizes the "host:port or host is on a loopback interface"
// check that gates non-loopback warning/refusal paths across mcp, dashboard,
// trace, prometheus, and fleet commands.
package cli

import (
	"fmt"
	"net"
	"os"
	"strings"
)

// SecretSource enumerates how a secret arrived at the process. SourceFlag is
// considered leaky because CLI arguments are visible via /proc/<pid>/cmdline.
type SecretSource int

const (
	SourceFlag SecretSource = iota
	SourceEnv
	SourceFile
)

// String returns the lowercase name of the source, suitable for error text.
func (s SecretSource) String() string {
	switch s {
	case SourceFlag:
		return "flag"
	case SourceEnv:
		return "env"
	case SourceFile:
		return "file"
	}
	return "unknown"
}

// AssertSecretSafe returns nil iff source is in allowed. Callers pass
// []SecretSource{SourceEnv, SourceFile} to reject CLI flags. name is the
// secret's logical identifier (e.g. "mcp-bearer-token") so the operator can
// match the error back to their command line.
func AssertSecretSafe(name string, source SecretSource, allowed []SecretSource) error {
	for _, s := range allowed {
		if s == source {
			return nil
		}
	}
	names := make([]string, 0, len(allowed))
	for _, s := range allowed {
		names = append(names, s.String())
	}
	return fmt.Errorf(
		"secret %q supplied via %s; safe sources are [%s]. CLI flags leak the value via /proc/<pid>/cmdline, shell history, and process audit logs",
		name, source, strings.Join(names, ", "),
	)
}

// ResolveSecret returns the value of envName, refusing if flagValue is
// non-empty. When both env and flag are populated, the env-var path always
// wins; the flag path is treated as a hard configuration error rather than
// silently ignored, so the operator notices and stops embedding the secret
// in their shell command.
//
// Returns ("", nil) when neither source supplies a value — callers decide
// whether that disables the feature or is itself an error.
func ResolveSecret(flagName, envName, flagValue string) (string, error) {
	if flagValue != "" {
		if err := AssertSecretSafe(flagName, SourceFlag, []SecretSource{SourceEnv, SourceFile}); err != nil {
			return "", fmt.Errorf("%w; set %s in the environment instead of --%s", err, envName, flagName)
		}
	}
	return os.Getenv(envName), nil
}

// IsLoopback reports whether host (bare host or host:port) resolves to a
// loopback interface. Empty host (":9090" listen form) is treated as loopback
// for the warning path because net.Listen on "" binds to localhost on most
// platforms — but bind-on-all-interfaces (":port") is NOT empty host; the
// SplitHostPort path returns "" for the host component, which we treat as
// non-loopback because that is the bind-all interpretation.
//
// Recognized loopback forms: "localhost", "127.0.0.0/8" IPv4 addresses, "::1".
func IsLoopback(host string) bool {
	h, _, err := net.SplitHostPort(host)
	if err == nil {
		host = h
	}
	if host == "" {
		// ":port" form binds to all interfaces; not loopback-safe.
		return false
	}
	if host == "localhost" {
		return true
	}
	ip := net.ParseIP(host)
	return ip != nil && ip.IsLoopback()
}
