package symtab

// CPython struct offsets for extracting Python frames from process memory.
//
// These offsets are version-specific because CPython's internal structs change
// between versions. Sources:
//   - parca-dev/runtime-data (pre-computed tables)
//   - CPython source code (Include/cpython/*.h)
//   - Verified with offsetof() test programs
//
// Two layout families:
//   CPython 3.10 and earlier:
//     PyInterpreterState → tstate_head → PyThreadState → frame → PyFrameObject
//     PyFrameObject: f_back, f_code → PyCodeObject
//
//   CPython 3.11+:
//     PyInterpreterState → tstate_head → PyThreadState → cframe → current_frame → _PyInterpreterFrame
//     _PyInterpreterFrame: previous, f_executable → PyCodeObject

// PyOffsets holds the version-specific struct offsets.
type PyOffsets struct {
	Version string

	// PyInterpreterState offsets (from _PyRuntime)
	RuntimeInterpretersHead uint64 // _PyRuntime.interpreters.head

	// PyInterpreterState offsets
	InterpTstateHead uint64 // PyInterpreterState.tstate_head (3.10) or .threads.head (3.12)

	// PyThreadState offsets
	TstateNext           uint64 // PyThreadState.next
	TstateThreadID       uint64 // PyThreadState.thread_id (pthread_self(), NOT kernel TID)
	TstateNativeThreadID uint64 // PyThreadState.native_thread_id (gettid(), kernel TID — 3.8+)
	TstateFrame          uint64 // PyThreadState.frame (3.10) or .cframe (3.11) or .current_frame (3.12)

	// For 3.11: cframe → current_frame
	CframeCurrentFrame uint64 // _PyCFrame.current_frame (3.11 only, 0 for others)

	// PyFrameObject / _PyInterpreterFrame offsets
	FrameBack     uint64 // f_back (3.10) or previous (3.11+)
	FrameCode     uint64 // f_code (3.10) or f_executable (3.11+)

	// PyCodeObject offsets
	CodeFilename    uint64 // co_filename
	CodeName        uint64 // co_name (function name)
	CodeFirstLineNo uint64 // co_firstlineno

	// PyUnicodeObject offsets (for reading strings)
	UnicodeLength uint64 // PyASCIIObject.length
	UnicodeData   uint64 // PyASCIIObject.data (compact ASCII) or hash+data
	UnicodeState  uint64 // PyASCIIObject.state (for ascii/compact detection)

	// Use New-style frames? (3.11+)
	NewStyleFrames bool
}

// GetPyOffsetsBest tries DWARF debug info first, then falls back to hardcoded offsets.
//
// When libpython (or its separate .debug file) has DWARF info, we extract the real
// struct offsets at runtime. This handles distro-patched builds (e.g. Ubuntu 22.04)
// where struct layouts differ from upstream CPython source.
//
// If no debug info is found, falls back to GetPyOffsets() which returns the
// hardcoded upstream offsets. Returns nil if neither path produces offsets.
func GetPyOffsetsBest(libPath string, minor int) *PyOffsets {
	// Try DWARF first.
	symDebugf("searching for DWARF debug info for %s (Python 3.%d)", libPath, minor)
	debugPath, err := FindDebugFile(libPath)
	if err != nil {
		symDebugf("DWARF debug file search failed for %s: %v", libPath, err)
	}

	if debugPath != "" {
		offsets, err := BuildPyOffsetsFromDWARF(debugPath, minor)
		if err != nil {
			symDebugf("DWARF offset extraction failed for %s: %v", debugPath, err)
		} else {
			symDebugf("using DWARF offsets from %s for Python 3.%d", debugPath, minor)
			logOffsetComparison(offsets, minor)
			return offsets
		}
	}

	// Fall back to hardcoded offsets.
	offsets := GetPyOffsets(minor)
	if offsets != nil {
		symDebugf("using hardcoded offsets for Python 3.%d (no DWARF debug info)", minor)
	}
	return offsets
}

// logOffsetComparison logs the difference between DWARF and hardcoded offsets
// when --debug is active. Helps diagnose distro-patched builds.
func logOffsetComparison(dwarf *PyOffsets, minor int) {
	hardcoded := GetPyOffsets(minor)
	if hardcoded == nil || debugLogFn == nil {
		return
	}

	type field struct {
		name          string
		dwarfVal, hcVal uint64
	}

	fields := []field{
		{"RuntimeInterpretersHead", dwarf.RuntimeInterpretersHead, hardcoded.RuntimeInterpretersHead},
		{"InterpTstateHead", dwarf.InterpTstateHead, hardcoded.InterpTstateHead},
		{"TstateNext", dwarf.TstateNext, hardcoded.TstateNext},
		{"TstateThreadID", dwarf.TstateThreadID, hardcoded.TstateThreadID},
		{"TstateNativeThreadID", dwarf.TstateNativeThreadID, hardcoded.TstateNativeThreadID},
		{"TstateFrame", dwarf.TstateFrame, hardcoded.TstateFrame},
		{"CframeCurrentFrame", dwarf.CframeCurrentFrame, hardcoded.CframeCurrentFrame},
		{"FrameBack", dwarf.FrameBack, hardcoded.FrameBack},
		{"FrameCode", dwarf.FrameCode, hardcoded.FrameCode},
		{"CodeFilename", dwarf.CodeFilename, hardcoded.CodeFilename},
		{"CodeName", dwarf.CodeName, hardcoded.CodeName},
		{"CodeFirstLineNo", dwarf.CodeFirstLineNo, hardcoded.CodeFirstLineNo},
		{"UnicodeLength", dwarf.UnicodeLength, hardcoded.UnicodeLength},
		{"UnicodeData", dwarf.UnicodeData, hardcoded.UnicodeData},
		{"UnicodeState", dwarf.UnicodeState, hardcoded.UnicodeState},
	}

	diffs := 0
	for _, f := range fields {
		if f.dwarfVal != f.hcVal {
			symDebugf("  DWARF vs hardcoded MISMATCH: %s = %d (DWARF) vs %d (hardcoded)", f.name, f.dwarfVal, f.hcVal)
			diffs++
		}
	}
	if diffs == 0 {
		symDebugf("  DWARF offsets match hardcoded for Python 3.%d (all %d fields)", minor, len(fields))
	} else {
		symDebugf("  %d/%d offset fields differ — distro-patched build detected", diffs, len(fields))
	}

	// Log all offsets for diagnostic reference.
	symDebugf("  DWARF offset table: RuntimeInterpretersHead=%d InterpTstateHead=%d TstateNext=%d TstateThreadID=%d TstateFrame=%d FrameBack=%d FrameCode=%d CodeFilename=%d CodeName=%d UnicodeData=%d",
		dwarf.RuntimeInterpretersHead, dwarf.InterpTstateHead, dwarf.TstateNext,
		dwarf.TstateThreadID, dwarf.TstateFrame, dwarf.FrameBack, dwarf.FrameCode,
		dwarf.CodeFilename, dwarf.CodeName, dwarf.UnicodeData)
}

// GetPyOffsets returns the offset table for a given CPython version.
// Returns nil if the version is not supported.
func GetPyOffsets(minor int) *PyOffsets {
	switch minor {
	case 10:
		return pyOffsets310()
	case 11:
		return pyOffsets311()
	case 12:
		return pyOffsets312()
	default:
		return nil
	}
}

// CPython 3.10 (Ubuntu 22.04 default)
//
// Frame layout: PyThreadState.frame → PyFrameObject (linked list via f_back)
// PyCodeObject: co_filename, co_name, co_firstlineno
//
// Offsets from CPython 3.10.12 source + parca-dev/runtime-data
func pyOffsets310() *PyOffsets {
	return &PyOffsets{
		Version:                "3.10",
		RuntimeInterpretersHead: 40, // _PyRuntime.interpreters.head

		InterpTstateHead: 8, // PyInterpreterState.tstate_head

		TstateNext:           8,   // PyThreadState.next
		TstateThreadID:       176, // PyThreadState.thread_id (pthread_self)
		TstateNativeThreadID: 184, // PyThreadState.native_thread_id (gettid, 3.8+)
		TstateFrame:          24,  // PyThreadState.frame (PyFrameObject*)

		CframeCurrentFrame: 0, // Not used in 3.10

		FrameBack: 24, // PyFrameObject.f_back
		FrameCode: 16, // PyFrameObject.f_code

		CodeFilename:    104, // PyCodeObject.co_filename
		CodeName:        112, // PyCodeObject.co_name
		CodeFirstLineNo: 48,  // PyCodeObject.co_firstlineno

		UnicodeLength: 16, // PyASCIIObject.length
		UnicodeData:   48, // Compact ASCII data offset
		UnicodeState:  32, // PyASCIIObject.state (after length:8 + hash:8)

		NewStyleFrames: false,
	}
}

// CPython 3.11 (transitional — changed frame layout)
//
// Frame layout: PyThreadState.cframe → _PyCFrame.current_frame → _PyInterpreterFrame
// _PyInterpreterFrame has .previous and .f_executable (was .f_code)
//
// Offsets from CPython 3.11.8 source + parca-dev/runtime-data
func pyOffsets311() *PyOffsets {
	return &PyOffsets{
		Version:                "3.11",
		RuntimeInterpretersHead: 40,

		InterpTstateHead: 8,

		TstateNext:           8,
		TstateThreadID:       176, // pthread_self
		TstateNativeThreadID: 184, // gettid (3.8+)
		TstateFrame:          40,  // PyThreadState.cframe (_PyCFrame*)

		CframeCurrentFrame: 0, // _PyCFrame.current_frame (offset 0)

		FrameBack: 24, // _PyInterpreterFrame.previous
		FrameCode: 0,  // _PyInterpreterFrame.f_code (offset 0 in 3.11)

		CodeFilename:    104,
		CodeName:        112,
		CodeFirstLineNo: 48,

		UnicodeLength: 16,
		UnicodeData:   48,
		UnicodeState:  20,

		NewStyleFrames: true,
	}
}

// CPython 3.12 (conda default, common in ML)
//
// Frame layout: PyThreadState.current_frame → _PyInterpreterFrame (direct, no cframe indirection)
// _PyInterpreterFrame.f_executable → PyCodeObject
//
// Offsets from CPython 3.12.2 source + parca-dev/runtime-data
func pyOffsets312() *PyOffsets {
	return &PyOffsets{
		Version:                "3.12",
		RuntimeInterpretersHead: 40,

		InterpTstateHead: 16, // PyInterpreterState.threads.head

		TstateNext:           8,
		TstateThreadID:       184, // pthread_self (shifted in 3.12)
		TstateNativeThreadID: 192, // gettid (3.8+)
		TstateFrame:          56,  // PyThreadState.current_frame (_PyInterpreterFrame* direct)

		CframeCurrentFrame: 0, // Not used in 3.12 (direct pointer)

		FrameBack: 24, // _PyInterpreterFrame.previous
		FrameCode: 0,  // _PyInterpreterFrame.f_executable

		CodeFilename:    112, // Shifted in 3.12
		CodeName:        120,
		CodeFirstLineNo: 48,

		UnicodeLength: 16,
		UnicodeData:   48,
		UnicodeState:  20,

		NewStyleFrames: true,
	}
}
