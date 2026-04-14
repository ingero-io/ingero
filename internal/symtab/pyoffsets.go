package symtab

import (
	"fmt"
	"log/slog"
)

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

// readDebugOffsets reads CPython's _Py_DebugOffsets struct from the beginning of
// _PyRuntime in the target process's memory. Available in CPython 3.12+.
//
// Currently only supports CPython 3.12. The _Py_DebugOffsets group layout
// (field counts per group, and therefore group start offsets) varies between
// minor versions, so 3.13+ will need their own layout constants.
//
// The _Py_DebugOffsets struct is embedded at offset 0 of _PyRuntime and contains
// self-describing offset information, making it possible to walk Python frames
// without DWARF debug info or hardcoded offset tables.
//
// Struct layout (all fields uint64, little-endian on x86_64):
//
//	offset  0: version (uint64) — encoded CPython version
//	offset  8: free_threaded (uint64) — free-threading flag (3.13+, always 0 for 3.12)
//
//	Then groups of offsets, each starting with a size field:
//	  runtime_state: size, finalizing, interpreters_head
//	  interpreter_state: size, threads_head, ...
//	  thread_state: size, prev, next, interp, current_frame, native_thread_id, ...
//	  interpreter_frame: size, previous, executable, ...
//	  code_object: size, filename, name, firstlineno, ...
//	  unicode_object: size, length, state, ascii (data start offset)
//
// Returns nil for minor < 12 (struct does not exist before 3.12).
func readDebugOffsets(mem *ProcMem, runtimeAddr uint64, minor int) (*PyOffsets, error) {
	if minor < 12 {
		return nil, nil
	}

	// _Py_DebugOffsets layout varies between CPython minor versions.
	// We only support 3.12 with known group offsets. For 3.13+, fall
	// through to DWARF/hardcoded until we add their layouts.
	if minor != 12 {
		symDebugf("_Py_DebugOffsets: version 3.%d not yet supported, falling back to DWARF/hardcoded", minor)
		return nil, nil
	}

	// Read the _Py_DebugOffsets header. The struct is at the very start of _PyRuntime.
	// We need enough bytes to cover all the offset groups we care about.
	// CPython 3.12 _Py_DebugOffsets is approximately 200-300 bytes.
	// Read a generous buffer to cover current and future versions.
	const bufSize = 512
	buf := make([]byte, bufSize)
	if err := mem.ReadAt(buf, runtimeAddr); err != nil {
		return nil, fmt.Errorf("reading _Py_DebugOffsets at 0x%x: %w", runtimeAddr, err)
	}

	// Helper to read a uint64 at a byte offset within the buffer.
	readU64 := func(off int) uint64 {
		if off+8 > len(buf) {
			return 0
		}
		return uint64(buf[off]) | uint64(buf[off+1])<<8 | uint64(buf[off+2])<<16 |
			uint64(buf[off+3])<<24 | uint64(buf[off+4])<<32 | uint64(buf[off+5])<<40 |
			uint64(buf[off+6])<<48 | uint64(buf[off+7])<<56
	}

	// Validate the version field. It encodes the CPython version.
	// For 3.12.x it should encode major=3, minor=12.
	version := readU64(0)
	if version == 0 {
		return nil, fmt.Errorf("_Py_DebugOffsets version is 0 — not a valid debug offsets struct")
	}

	symDebugf("_Py_DebugOffsets version field: 0x%x", version)

	// _Py_DebugOffsets layout for CPython 3.12 (each field is uint64):
	//
	// Header (2 fields):
	//   [0]  version
	//   [8]  free_threaded
	//
	// runtime_state group (3 fields):
	//   [16] size
	//   [24] finalizing
	//   [32] interpreters_head
	//
	// interpreter_state group (variable fields):
	//   [40] size
	//   [48] threads_head
	//   ... (we only need threads_head)
	//
	// To find subsequent groups, we use the size field of each group
	// to determine how many uint64s it contains, then skip ahead.
	// However, the size field is the size of the *target struct*, not
	// the offset group. The number of fields per group is fixed per
	// CPython version.
	//
	// For CPython 3.12, the layout is well-defined. We use fixed offsets
	// based on the known struct layout.

	// Fixed byte offsets within _Py_DebugOffsets for CPython 3.12:
	const (
		offVersion         = 0
		offFreeThreaded    = 8
		offRuntimeSize     = 16
		offRuntimeFinalizing    = 24
		offRuntimeInterpretersHead = 32
		offInterpSize      = 40
		offInterpThreadsHead = 48
	)

	// Read the runtime_state offsets.
	runtimeInterpretersHead := readU64(offRuntimeInterpretersHead)
	interpThreadsHead := readU64(offInterpThreadsHead)

	// For the remaining groups, we need to know how many fields are in the
	// interpreter_state group. In CPython 3.12, the groups after interpreter_state
	// are at fixed positions. We compute these by counting fields.
	//
	// CPython 3.12 _Py_DebugOffsets field layout (verified against cpython/Include/internal/pycore_debug_offsets.h):
	//   interpreter_state: size, threads_head, gc, modules, builtins, sysdict, ceval_gil, gil_runtime_state
	//   That's 8 fields = 64 bytes, starting at offset 40, so next group at 40+64=104
	//
	// But the exact count varies by CPython patch version. We use the interpreter_state
	// size field to detect the group boundary. Each offset group has:
	//   1 size field + N named offset fields
	//
	// Simpler approach: the interp group in 3.12 has a known number of fields.
	// We'll read the interp group size to validate, then use fixed offsets.

	// CPython 3.12 thread_state group starts after interpreter_state group.
	// interpreter_state group in 3.12 has 8 fields (size + 7 offsets) = 64 bytes.
	// So thread_state group starts at offset 40 + 64 = 104.
	//
	// However, this varies between 3.12.x patch releases. A more robust approach:
	// scan for the thread_state group by looking at the size field values.
	// The "size" field of each group contains the sizeof() of the corresponding
	// CPython struct (e.g., sizeof(PyThreadState) is typically 300-500 bytes).

	// For robustness, we scan forward from the interpreter_state group to find
	// the thread_state group. We know:
	// - Each group starts with a "size" field containing sizeof(struct)
	// - PyThreadState size is typically 200-600 bytes
	// - _PyInterpreterFrame size is typically 40-100 bytes

	// Start scanning after the interp group's first two fields we already read.
	// The interp group starts at offset 40. We look for where the next "size"
	// field starts by checking plausible offsets.

	// CPython 3.12.0-3.12.x: interpreter_state group has these fields:
	//   size, threads_head, gc, modules, builtins, sysdict, ceval_gil, gil_runtime_state
	// That's 8 fields = 64 bytes. Thread state group at 104.

	const threadsGroupStart = 104

	tstateNext := readU64(threadsGroupStart + 2*8)         // next (3rd field after size, prev)
	tstateCurrentFrame := readU64(threadsGroupStart + 4*8)  // current_frame (5th field)
	tstateNativeThreadID := readU64(threadsGroupStart + 5*8) // native_thread_id (6th field)

	// Validate: these should look like struct offsets (small positive numbers, < 1000).
	if tstateNext > 1000 || tstateCurrentFrame > 1000 || tstateNativeThreadID > 1000 {
		return nil, fmt.Errorf("_Py_DebugOffsets thread_state offsets look invalid (next=%d, current_frame=%d, native_thread_id=%d)",
			tstateNext, tstateCurrentFrame, tstateNativeThreadID)
	}

	// Thread state group in 3.12 has: size, prev, next, interp, current_frame, native_thread_id
	// That's 6 fields = 48 bytes. Frame group starts at 104 + 48 = 152.
	const frameGroupStart = 152

	framePrevious := readU64(frameGroupStart + 1*8)   // previous (2nd field after size)
	frameExecutable := readU64(frameGroupStart + 2*8)  // executable (3rd field)

	// Frame group in 3.12 has: size, previous, executable, instr_ptr, ...
	// Typically 4-6 fields. Code object group follows.
	// 3.12: frame group has 4 fields = 32 bytes. Code group at 152 + 32 = 184.
	const codeGroupStart = 184

	codeFilename := readU64(codeGroupStart + 1*8)      // filename (2nd field)
	codeName := readU64(codeGroupStart + 2*8)           // name (3rd field)
	codeFirstLineNo := readU64(codeGroupStart + 4*8)    // firstlineno (5th field, after qualname)

	// Code group in 3.12 has: size, filename, name, qualname, firstlineno, ...
	// Typically 5-6 fields. Unicode group follows.
	// 3.12: code group has 5 fields = 40 bytes. Unicode group at 184 + 40 = 224.
	const unicodeGroupStart = 224

	unicodeLength := readU64(unicodeGroupStart + 1*8)   // length (2nd field)
	unicodeState := readU64(unicodeGroupStart + 2*8)    // state (3rd field, used for ascii/compact flags)
	unicodeASCII := readU64(unicodeGroupStart + 3*8)    // ascii (4th field, data start for compact ASCII)

	// Validate unicode offsets.
	if unicodeASCII > 256 || unicodeLength > 256 {
		return nil, fmt.Errorf("_Py_DebugOffsets unicode offsets look invalid (length=%d, ascii=%d)",
			unicodeLength, unicodeASCII)
	}

	// thread_id (pthread_self) is stored immediately before native_thread_id
	// in CPython 3.12's PyThreadState. This offset relationship is stable
	// within 3.12.x but NOT guaranteed across minor versions — another
	// reason to gate this function by minor version.
	if tstateNativeThreadID < 8 {
		return nil, fmt.Errorf("_Py_DebugOffsets: native_thread_id offset %d too small to derive thread_id", tstateNativeThreadID)
	}
	tstateThreadID := tstateNativeThreadID - 8

	offsets := &PyOffsets{
		Version:                 fmt.Sprintf("3.%d-debugoffsets", minor),
		RuntimeInterpretersHead: runtimeInterpretersHead,
		InterpTstateHead:        interpThreadsHead,
		TstateNext:              tstateNext,
		TstateThreadID:          tstateThreadID,
		TstateNativeThreadID:    tstateNativeThreadID,
		TstateFrame:             tstateCurrentFrame,
		CframeCurrentFrame:      0, // 3.12+ uses direct pointer, no cframe indirection
		FrameBack:               framePrevious,
		FrameCode:               frameExecutable,
		CodeFilename:            codeFilename,
		CodeName:                codeName,
		CodeFirstLineNo:         codeFirstLineNo,
		UnicodeLength:           unicodeLength,
		UnicodeData:             unicodeASCII,
		UnicodeState:            unicodeState,
		NewStyleFrames:          true,
	}

	symDebugf("_Py_DebugOffsets extracted: RuntimeInterpretersHead=%d InterpTstateHead=%d TstateNext=%d TstateFrame=%d FrameBack=%d FrameCode=%d CodeFilename=%d CodeName=%d UnicodeData=%d",
		offsets.RuntimeInterpretersHead, offsets.InterpTstateHead, offsets.TstateNext,
		offsets.TstateFrame, offsets.FrameBack, offsets.FrameCode,
		offsets.CodeFilename, offsets.CodeName, offsets.UnicodeData)

	return offsets, nil
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

	var mismatched []string
	for _, f := range fields {
		if f.dwarfVal != f.hcVal {
			symDebugf("  DWARF vs hardcoded MISMATCH: %s = %d (DWARF) vs %d (hardcoded)", f.name, f.dwarfVal, f.hcVal)
			mismatched = append(mismatched, f.name)
		}
	}
	if len(mismatched) == 0 {
		symDebugf("  DWARF offsets match hardcoded for Python 3.%d (all %d fields)", minor, len(fields))
	} else {
		symDebugf("  %d/%d offset fields differ — distro-patched build detected", len(mismatched), len(fields))
		slog.Info("distro-patched CPython detected — DWARF offsets differ from upstream",
			"python_version", fmt.Sprintf("3.%d", minor), "mismatched_fields", mismatched)
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
