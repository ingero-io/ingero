package symtab

import (
	"fmt"
	"os"
	"regexp"
	"strings"
)

// PythonInfo describes a CPython interpreter found in a process.
type PythonInfo struct {
	Version    string // e.g., "3.10", "3.11", "3.12"
	Major      int    // 3
	Minor      int    // 10, 11, 12
	LibPath    string // path to libpython3.X.so or python3.X binary
	RuntimeAddr uint64 // address of _PyRuntime symbol (0 if not found)
}

// pythonRe matches libpython or python binary paths in /proc/maps.
// Examples:
//
//	/usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0
//	/usr/bin/python3.12
//	/home/user/miniconda3/lib/libpython3.11.so.1.0
var pythonRe = regexp.MustCompile(`(?:lib)?python(3)\.(\d+)`)

// DetectPython checks if a process has CPython loaded. Returns nil if
// no Python interpreter is found.
//
// Two signals are consulted in order:
//
//  1. /proc/<pid>/maps — executable regions whose path matches
//     libpython3.X.so or python3.X. This is the primary signal and
//     gives us the exact library/binary the process has mapped.
//
//  2. /proc/<pid>/exe — the kernel sets this symlink at exec() time,
//     before ld-linux runs. When the maps scan happens very early
//     during process startup (the first cuda event can fire before
//     the binary's executable PT_LOAD is mapped — observed on
//     /opt/pytorch/bin/python3.12 workloads launched immediately
//     after trace start), the maps are incomplete but the exe
//     symlink is already correct. Without this fallback, the
//     per-PID dedup in the caller marks the PID as "not Python"
//     forever and the walker never fires.
func DetectPython(pid uint32) *PythonInfo {
	regions, _ := ParseProcMaps(pid)
	if info := detectPythonFromRegions(regions); info != nil {
		return info
	}
	return detectPythonFromProcExe(pid)
}

// detectPythonFromProcExe resolves /proc/<pid>/exe and runs the python
// name regex on the symlink target. Loader-early fallback: see
// DetectPython. Extracted from DetectPython so the pure name-matching
// logic in detectPythonFromExeTarget is unit-testable.
func detectPythonFromProcExe(pid uint32) *PythonInfo {
	target, err := os.Readlink(fmt.Sprintf("/proc/%d/exe", pid))
	if err != nil {
		return nil
	}
	return detectPythonFromExeTarget(target)
}

// detectPythonFromExeTarget matches a resolved /proc/<pid>/exe path
// against the python name regex and returns a minimal PythonInfo.
// LibPath is the exe target itself; findPyRuntimeAddr opens this path
// to resolve _PyRuntime, so it must point at an ELF with symbol info
// (the python binary, which is statically linked on uv distributions
// and Ubuntu's vanilla packages both).
func detectPythonFromExeTarget(exeTarget string) *PythonInfo {
	m := pythonRe.FindStringSubmatch(exeTarget)
	if m == nil {
		return nil
	}
	major := 3 // m[1] is always "3"
	minor := 0
	fmt.Sscanf(m[2], "%d", &minor)
	return &PythonInfo{
		Version: fmt.Sprintf("%d.%d", major, minor),
		Major:   major,
		Minor:   minor,
		LibPath: exeTarget,
	}
}

// detectPythonFromRegions scans map regions for a Python interpreter.
// Prefers libpython (shared library) over the python binary, since the
// shared library has the full symbol table including _PyRuntime.
// If no libpython is found, returns the binary path as fallback (covers
// statically-linked Python like Ubuntu's /usr/bin/python3.10).
func detectPythonFromRegions(regions []MapRegion) *PythonInfo {
	var fallback *PythonInfo

	for _, r := range regions {
		if r.Path == "" {
			continue
		}

		matches := pythonRe.FindStringSubmatch(r.Path)
		if matches == nil {
			continue
		}

		major := 3 // matches[1] is always "3"
		minor := 0
		fmt.Sscanf(matches[2], "%d", &minor)

		isLib := strings.Contains(r.Path, "libpython")

		info := &PythonInfo{
			Version: fmt.Sprintf("%d.%d", major, minor),
			Major:   major,
			Minor:   minor,
			LibPath: r.Path,
		}

		if isLib {
			return info // libpython found — best match, return immediately.
		}

		// Binary match — keep as fallback, continue scanning for libpython.
		if fallback == nil {
			fallback = info
		}
	}

	return fallback
}

// IsSupportedVersion returns true if we have hardcoded offsets for this version.
func (p *PythonInfo) IsSupportedVersion() bool {
	if p == nil {
		return false
	}
	switch p.Minor {
	case 10, 11, 12:
		return true
	default:
		return false
	}
}
