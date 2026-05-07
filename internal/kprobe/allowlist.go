// Package kprobe holds the experimental closed-driver kprobe surface
// (W1 memfrag, W2 throttle, kernel grid/block dims) shipped behind
// the --enable-experimental-kprobes flag.
//
// The kprobe targets attach to symbols inside the closed-source
// NVIDIA kernel module (nvidia_unlocked_ioctl). Both the symbol
// name and the IOCTL command numbers are stable across recent
// driver versions in our testing, but neither has a public
// stability guarantee from NVIDIA. Operators outside the tested
// driver + kernel matrix get a startup warning and the probes do
// not load; correctness is the responsibility of whoever expands
// the allowlist after re-validating against a new driver release.
package kprobe

import (
	"fmt"
	"os"
	"regexp"
	"strconv"
	"strings"
)

// Versions describes a tested driver + kernel pair.
type Versions struct {
	NVIDIADriver string // e.g. "535.86.10"
	LinuxKernel  string // e.g. "5.15.0-89-generic"
}

// AllowedPair is one entry in the allowlist. Driver and Kernel
// patterns are matched against the detected versions per
// MatchVersion semantics.
type AllowedPair struct {
	DriverPrefix string // e.g. "535." matches any 535.x driver
	KernelPrefix string // e.g. "5.15." matches any 5.15.x kernel
	Notes        string // free-form context for logs / docs
}

// DefaultAllowlist starts narrow: the Lambda baseline images at
// the v0.15 validation cycle. New entries are appended only after
// a real-hardware run confirms the kprobes attach + fire + decode
// correctly on that pair.
var DefaultAllowlist = []AllowedPair{
	{DriverPrefix: "535.", KernelPrefix: "5.15.", Notes: "Lambda A10 amd64, prior baseline"},
	{DriverPrefix: "535.", KernelPrefix: "6.5.", Notes: "Lambda GH200 arm64, prior baseline"},
	{DriverPrefix: "570.", KernelPrefix: "6.8.", Notes: "Lambda A10 amd64, current Ubuntu 22.04 image (validated 2026-05-07)"},
}

// MatchVersion reports whether want is a prefix of got, treating
// the trailing dot as a non-greedy boundary so "5.15." matches
// "5.15.0-89-generic" but NOT "5.150.0".
func MatchVersion(got, want string) bool {
	return strings.HasPrefix(got, want)
}

// IsAllowed reports whether (driver, kernel) is on the supplied
// allowlist. v is the detection output; allowlist may be the
// DefaultAllowlist or an operator-supplied override.
func IsAllowed(v Versions, allowlist []AllowedPair) bool {
	for _, a := range allowlist {
		if MatchVersion(v.NVIDIADriver, a.DriverPrefix) && MatchVersion(v.LinuxKernel, a.KernelPrefix) {
			return true
		}
	}
	return false
}

// DetectVersions reads the running driver + kernel versions from
// /proc. Empty fields when a file is missing or unreadable; the
// allowlist gate is not the right layer to hard-fail on detection
// errors. The caller logs and skips probe load.
func DetectVersions() Versions {
	v := Versions{
		NVIDIADriver: parseNVIDIADriverVersion(readFileSafe("/proc/driver/nvidia/version")),
		LinuxKernel:  strings.TrimSpace(readFileSafe("/proc/sys/kernel/osrelease")),
	}
	return v
}

// nvidiaVersionRE captures the version field on the NVIDIA driver
// banner line, e.g.:
//
//	NVRM version: NVIDIA UNIX x86_64 Kernel Module  535.86.10  Wed Jul 26 ...
//
// or the open-source kernel module shape:
//
//	NVRM version: NVIDIA UNIX Open Kernel Module for x86_64  535.104.05  Tue Aug 22 ...
var nvidiaVersionRE = regexp.MustCompile(`Module(?:\s+for\s+\S+)?\s+([0-9]+(?:\.[0-9]+){1,3})`)

func parseNVIDIADriverVersion(banner string) string {
	if banner == "" {
		return ""
	}
	for _, line := range strings.Split(banner, "\n") {
		if !strings.HasPrefix(line, "NVRM version:") {
			continue
		}
		m := nvidiaVersionRE.FindStringSubmatch(line)
		if len(m) > 1 {
			return m[1]
		}
	}
	return ""
}

func readFileSafe(path string) string {
	b, err := os.ReadFile(path)
	if err != nil {
		return ""
	}
	return string(b)
}

// DescribeStatus is a one-line summary suitable for the agent
// startup log. Returns "" when the experimental flag is off; the
// caller checks the flag before calling.
func DescribeStatus(v Versions, allowed bool, allowlist []AllowedPair) string {
	if !allowed {
		return fmt.Sprintf("experimental-kprobes: driver=%s kernel=%s not on allowlist (%d tested pairs); probes will NOT load",
			displayOrUnknown(v.NVIDIADriver), displayOrUnknown(v.LinuxKernel), len(allowlist))
	}
	return fmt.Sprintf("experimental-kprobes: driver=%s kernel=%s on allowlist; probes will load",
		v.NVIDIADriver, v.LinuxKernel)
}

func displayOrUnknown(s string) string {
	if s == "" {
		return "(unknown)"
	}
	return s
}

// ParseDriverMajor returns the leading "NNN" component of a driver
// version. Useful for crude release-line checks. Returns 0 on
// malformed input.
func ParseDriverMajor(driver string) int {
	if driver == "" {
		return 0
	}
	i := strings.IndexByte(driver, '.')
	if i < 1 {
		return 0
	}
	n, err := strconv.Atoi(driver[:i])
	if err != nil {
		return 0
	}
	return n
}
