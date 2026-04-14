package symtab

import "fmt"

// Known-offsets database keyed by libpython build-id.
//
// Purpose:
//   Distro packagers (Ubuntu, Debian, RHEL/AL2023, etc.) frequently patch
//   CPython in ways that shift internal struct layouts — most commonly by
//   retaining the 3.11-style _PyCFrame indirection in 3.12 for ABI stability,
//   or by adding/reordering fields. When ingero cannot obtain DWARF debug
//   info (the libpython3.X-dbgsym / -dbg packages are not installed), it
//   would otherwise fall back to upstream hardcoded offsets, which are wrong
//   for these patched builds and cause silent frame-walker corruption.
//
//   This database ships precomputed offsets keyed by the libpython ELF
//   build-id (.note.gnu.build-id), which uniquely identifies a specific
//   binary build. A build-id match guarantees the offsets are correct for
//   that exact build — it's effectively a fingerprint of the compiled layout.
//
// Fallback chain (see GetPyOffsetsBest and readDebugOffsets):
//   1. _Py_DebugOffsets (CPython 3.12+, read from target memory)
//   2. Known-offsets DB (this file, keyed by build-id)       <-- here
//   3. DWARF extraction (requires libpython-dbgsym)
//   4. Hardcoded upstream offsets (GetPyOffsets)
//
// How to add a new entry:
//   1. On a target system (e.g. Ubuntu 24.04 amd64) install libpython3.X-dbgsym.
//   2. Get the build-id of the release libpython:
//        readelf -n /usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0 | grep 'Build ID'
//   3. Run ingero's DWARF extractor against that libpython (or use
//      BuildPyOffsetsFromDWARF directly against the -dbgsym .debug file) and
//      copy the resulting PyOffsets values into a new map entry below.
//   4. Add a comment documenting CPython version, distribution/arch, and how
//      the offsets were obtained.
//
// Why entries are version-specific:
//   Even patch releases (3.12.3 -> 3.12.4) may change offsets, and the same
//   upstream version built by different distros will have different build-ids.
//   Do NOT reuse an entry across builds — always key by the exact build-id.
//
// The map ships empty: real entries must be populated via a separate process
// that runs the offset extractor on actual target systems (Ubuntu 22.04/24.04,
// Debian 12, AL2023, etc.). The infrastructure here is functional and ready
// to be consumed by LookupByBuildID once entries are added.
var knownPyOffsets = map[string]*PyOffsets{
	// Placeholder entries documenting the structure — real entries need
	// to be populated by building ingero on target distros and running
	// the offset extractor. The infrastructure ships empty but functional.
	//
	// Example entry (commented out until we have real build-ids):
	// "a1b2c3d4e5f6...": &PyOffsets{
	//     Version:                 "3.12.3-ubuntu-24.04",
	//     RuntimeInterpretersHead: 40,
	//     InterpTstateHead:        80,
	//     TstateNext:              16,
	//     TstateThreadID:          40,
	//     TstateNativeThreadID:    48,
	//     TstateFrame:             56,
	//     FrameBack:               8,
	//     FrameCode:               32,
	//     CodeFilename:            112,
	//     CodeName:                120,
	//     UnicodeLength:           16,
	//     UnicodeData:             48,
	//     UnicodeState:            32,
	//     NewStyleFrames:          true,
	// },
}

// LookupByBuildID reads the build-id from the libpython binary and looks up
// known offsets for that exact build. Returns nil if not found, with no error
// for the common case of an unknown build-id. Returns an error only if the
// build-id read itself failed (malformed ELF, I/O error).
func LookupByBuildID(libPath string) (*PyOffsets, error) {
	buildID, err := ReadBuildID(libPath)
	if err != nil {
		return nil, fmt.Errorf("reading build-id from %s: %w", libPath, err)
	}
	if buildID == "" {
		return nil, nil
	}
	if offsets, ok := knownPyOffsets[buildID]; ok {
		return offsets, nil
	}
	return nil, nil
}
