package symtab

import (
	"fmt"
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

// DetectPython checks if a process has CPython loaded.
// Returns nil if no Python interpreter is found.
func DetectPython(pid uint32) *PythonInfo {
	regions, err := ParseProcMaps(pid)
	if err != nil {
		return nil
	}

	return detectPythonFromRegions(regions)
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
