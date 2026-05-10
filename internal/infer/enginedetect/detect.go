// Package enginedetect identifies a known inference engine from a
// running process's command line. The detection is engine-name only;
// confirming the detection (port + /metrics endpoint sniff) lives in
// the scrape package which calls Detect to know which parser to
// dispatch.
//
// Detection is intentionally cmdline-only — no port scanning, no
// process-tree walking. We trust the cmdline pattern as the strong
// signal because it's stable across the process lifetime and cheap
// to read (one /proc/PID/cmdline syscall, cached per-PID by the
// surrounding agent infrastructure).
//
// Coverage (v0.16.2): vLLM, TGI, SGLang, Triton. NIM passes through
// vLLM metrics so it's covered by the vLLM detector. TensorRT-LLM
// has no /metrics endpoint and is intentionally not detected.
package enginedetect

import (
	"fmt"
	"os"
	"strconv"
	"strings"
)

// Engine names a known inference engine. UnknownEngine is returned
// for processes the detector does not recognize; callers treat that
// as "skip scraping for this PID."
type Engine string

const (
	UnknownEngine Engine = ""
	VLLM          Engine = "vllm"
	TGI           Engine = "tgi"
	SGLang        Engine = "sglang"
	Triton        Engine = "triton"
)

// IsKnown reports whether e is one of the supported engines.
func (e Engine) IsKnown() bool {
	switch e {
	case VLLM, TGI, SGLang, Triton:
		return true
	}
	return false
}

// DefaultPort returns the engine's typical /metrics-serving port.
// The cmdline often overrides this via --port; callers should
// prefer the cmdline-extracted port when present.
func (e Engine) DefaultPort() uint16 {
	switch e {
	case VLLM:
		return 8000
	case TGI:
		return 8080
	case SGLang:
		return 30000
	case Triton:
		return 8002
	}
	return 0
}

// MetricsPath returns the HTTP path the engine exposes Prometheus
// metrics on. All four engines use /metrics today; NIM uses
// /v1/metrics but the vLLM-named output passes through.
func (e Engine) MetricsPath() string {
	return "/metrics"
}

// Detection is the result of cmdline analysis: the recognized engine
// (or UnknownEngine), the listening port (DefaultPort if not present
// on cmdline), and the model name when extractable. Model name is
// best-effort; absent for unknown engines.
type Detection struct {
	Engine Engine
	Port   uint16
	Model  string
}

// readCmdlineArgs reads /proc/[pid]/cmdline, splits on the
// NUL separator, and trims trailing empties. Returns a slice
// of zero-or-more args. Empty slice on read failure or empty
// cmdline (e.g. zombie process).
func readCmdlineArgs(procPath string, pid uint32) []string {
	if procPath == "" {
		procPath = "/proc"
	}
	path := fmt.Sprintf("%s/%d/cmdline", procPath, pid)
	data, err := os.ReadFile(path)
	if err != nil {
		return nil
	}
	// Trim trailing NULs (the kernel pads), then split.
	s := strings.TrimRight(string(data), "\x00")
	if s == "" {
		return nil
	}
	args := strings.Split(s, "\x00")
	out := make([]string, 0, len(args))
	for _, a := range args {
		if a != "" {
			out = append(out, a)
		}
	}
	return out
}

// ListEnginePIDs scans /proc and returns the PIDs of every running
// process whose cmdline matches a known inference engine. Used by
// the v0.16.4 continuous re-detection loop in the scrape package
// when the agent runs system-wide (no --pid filter); the loop calls
// this on each tick to discover engines that started after the agent.
//
// procPath defaults to "/proc" when empty (test injection point,
// matches Detect's contract). The function is best-effort: any
// /proc/<pid> that becomes inaccessible mid-walk is silently skipped.
// Returns nil on a missing /proc or an empty result; allocates the
// slice only when at least one match is found.
func ListEnginePIDs(procPath string) []uint32 {
	if procPath == "" {
		procPath = "/proc"
	}
	entries, err := os.ReadDir(procPath)
	if err != nil {
		return nil
	}
	var out []uint32
	for _, e := range entries {
		if !e.IsDir() {
			continue
		}
		name := e.Name()
		// PIDs are decimal; reject non-numeric entries cheaply.
		pid64, err := strconv.ParseUint(name, 10, 32)
		if err != nil {
			continue
		}
		pid := uint32(pid64)
		if _, ok := detectAt(procPath, pid); ok {
			out = append(out, pid)
		}
	}
	return out
}

// Detect classifies a process by cmdline. Reads /proc/[pid]/cmdline
// and matches against the known engine patterns. procPath defaults
// to /proc when empty (test injection point).
//
// Match rules (first-wins):
//
//   - "vllm.entrypoints" or "vllm" + "serve"           → VLLM
//   - "text-generation-launcher"                       → TGI
//   - "sglang.launch_server" or "sglang" + module args → SGLang
//   - "tritonserver"                                   → Triton
//
// Port extraction: scans subsequent args for "--port <N>" or
// "--port=<N>" or (vLLM/SGLang) "--http-port <N>". Falls back to
// the engine's DefaultPort.
//
// Model extraction: scans for "--model <NAME>" / "--model=<NAME>"
// (vLLM, SGLang, Triton "--model-repository" is a path, not a
// model). TGI uses "--model-id". Best-effort; empty when absent.
func Detect(pid uint32) (Detection, bool) {
	return detectAt("/proc", pid)
}

// detectAt is the test seam allowing /proc replacement.
func detectAt(procPath string, pid uint32) (Detection, bool) {
	args := readCmdlineArgs(procPath, pid)
	if len(args) == 0 {
		return Detection{}, false
	}

	det := Detection{}
	joined := strings.Join(args, " ")

	switch {
	case strings.Contains(joined, "vllm.entrypoints") ||
		(strings.Contains(joined, "vllm") && containsArg(args, "serve")):
		det.Engine = VLLM
	case strings.Contains(joined, "text-generation-launcher"):
		det.Engine = TGI
	case strings.Contains(joined, "sglang.launch_server"):
		det.Engine = SGLang
	case strings.Contains(joined, "tritonserver"):
		det.Engine = Triton
	default:
		return Detection{}, false
	}

	det.Port = extractPort(args, det.Engine)
	det.Model = extractModel(args, det.Engine)
	return det, true
}

// containsArg reports whether s appears as a standalone arg (not as
// a substring of another arg). Used to tighten the vLLM "vllm" +
// "serve" match so a user-named "vllm-helper" subcommand doesn't
// false-match.
func containsArg(args []string, s string) bool {
	for _, a := range args {
		if a == s {
			return true
		}
	}
	return false
}

// extractPort scans args for the first --port flag (or engine-
// specific equivalent). Returns engine.DefaultPort() when none
// found or parse fails.
func extractPort(args []string, engine Engine) uint16 {
	candidates := []string{"--port", "--http-port"}
	for i, a := range args {
		// "--port=8000"
		for _, k := range candidates {
			if strings.HasPrefix(a, k+"=") {
				if p, ok := parsePortDigits(a[len(k)+1:]); ok {
					return p
				}
			}
		}
		// "--port 8000"
		for _, k := range candidates {
			if a == k && i+1 < len(args) {
				if p, ok := parsePortDigits(args[i+1]); ok {
					return p
				}
			}
		}
	}
	return engine.DefaultPort()
}

func parsePortDigits(s string) (uint16, bool) {
	n, err := strconv.ParseUint(strings.TrimSpace(s), 10, 16)
	if err != nil || n == 0 {
		return 0, false
	}
	return uint16(n), true
}

// extractModel pulls the model identifier from cmdline when present.
// Each engine uses a slightly different flag; we cover the common
// cases. Returns "" when nothing matches.
func extractModel(args []string, engine Engine) string {
	keys := []string{"--model"}
	switch engine {
	case TGI:
		keys = []string{"--model-id"}
	case SGLang:
		// SGLang's launch flag is `--model-path`; older builds
		// accept `--model` as an alias. Canonical form first so a
		// process passing both wins via the right key.
		keys = []string{"--model-path", "--model"}
	case Triton:
		// Triton uses --model-repository (a directory), which is
		// not a model identifier. Skip.
		return ""
	}
	for i, a := range args {
		for _, k := range keys {
			if strings.HasPrefix(a, k+"=") {
				return a[len(k)+1:]
			}
			if a == k && i+1 < len(args) {
				return args[i+1]
			}
		}
	}
	return ""
}
