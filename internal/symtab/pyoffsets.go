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
	InterpNext       uint64 // PyInterpreterState.next — first field across 3.9..3.14, so 0

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

// CPython's _Py_DebugOffsets struct is embedded as the first field of
// _PyRuntimeState starting in CPython 3.13 (earlier versions did NOT have
// this struct — the field at _PyRuntime+0 was just `int _initialized`).
// It contains self-describing byte offsets for the runtime, interpreter,
// thread, frame, code, and unicode structs, letting an out-of-process
// observer walk Python frames without DWARF debug info or hardcoded
// offset tables.
//
// The struct is decidedly NOT backwards compatible: members are added,
// reordered, and resized between minor versions. CPython documents this
// directly in pycore_runtime.h: "only guaranteed to be stable between
// patch versions for a given minor version". So we hardcode one layout
// table per supported minor.
//
// For each minor we care about the byte offset within _Py_DebugOffsets
// of the specific fields we actually use. The VALUES at those byte
// offsets are themselves offsets (into PyThreadState, _PyInterpreterFrame,
// PyCodeObject, PyASCIIObject, etc.) emitted by the CPython build system.
//
// Layouts verified against Include/internal/pycore_debug_offsets.h (3.14)
// and Include/internal/pycore_runtime.h (3.13 — the 3.13 struct lives in
// the runtime header, not a dedicated file).
type debugOffsetsLayout struct {
	runtimeInterpretersHead uint64 // -> _PyRuntimeState.interpreters.head
	interpNext              uint64 // -> PyInterpreterState.next
	interpThreadsHead       uint64 // -> PyInterpreterState.threads.head
	tstateCurrentFrame      uint64 // -> PyThreadState.current_frame
	tstateThreadID          uint64 // -> PyThreadState.thread_id (pthread_self)
	tstateNativeThreadID    uint64 // -> PyThreadState.native_thread_id (gettid)
	framePrevious           uint64 // -> _PyInterpreterFrame.previous
	frameExecutable         uint64 // -> _PyInterpreterFrame.f_executable
	codeFilename            uint64 // -> PyCodeObject.co_filename
	codeName                uint64 // -> PyCodeObject.co_name
	codeFirstLineNo         uint64 // -> PyCodeObject.co_firstlineno
	unicodeState            uint64 // -> PyASCIIObject.state
	unicodeLength           uint64 // -> PyASCIIObject.length
	unicodeASCII            uint64 // -> PyASCIIObject ASCII inline-data start
}

// debugOffsetsLayouts maps a CPython minor version to the field offsets
// inside _Py_DebugOffsets for that version. Offsets computed from the
// field order in the upstream struct definition: 24-byte header
// (cookie[8] + version + free_threaded) followed by per-group sub-structs
// of uint64 fields. Each field is 8 bytes.
//
// 3.13 (13 fields in interpreter_state, 6 in interpreter_frame, 10 in
// code_object, 3 pyobject+4 type+3 tuple+3 list+3 dict+2 float+3 long+3
// bytes groups before unicode_object):
//
// 3.14 (16 fields in interpreter_state after threads_main + code/tlbc
// generations; 8 fields in interpreter_frame after stackpointer +
// tlbc_index; 11 fields in code_object after co_tlbc; plus a new 4-field
// set_object group between list_object and dict_object):
var debugOffsetsLayouts = map[int]debugOffsetsLayout{
	13: {
		runtimeInterpretersHead: 40,
		interpNext:              64,
		interpThreadsHead:       72,
		tstateCurrentFrame:      184,
		tstateThreadID:          192,
		tstateNativeThreadID:    200,
		framePrevious:           232,
		frameExecutable:         240,
		codeFilename:            280,
		codeName:                288,
		codeFirstLineNo:         312,
		unicodeState:            544,
		unicodeLength:           552,
		unicodeASCII:            560,
	},
	14: {
		runtimeInterpretersHead: 40,
		interpNext:              64,
		interpThreadsHead:       72,
		tstateCurrentFrame:      208,
		tstateThreadID:          216,
		tstateNativeThreadID:    224,
		framePrevious:           256,
		frameExecutable:         264,
		codeFilename:            320,
		codeName:                328,
		codeFirstLineNo:         352,
		unicodeState:            624,
		unicodeLength:           632,
		unicodeASCII:            640,
	},
}

// debugOffsetsCookie is the 8-byte magic value CPython 3.13+ writes at
// _Py_DebugOffsets.cookie (see _Py_Debug_Cookie in pycore_debug_offsets.h).
// Validating it before trusting anything else rejects both pre-3.13
// builds (no cookie at all, just `int _initialized`) and accidental
// bad-pointer reads.
const debugOffsetsCookie = "xdebugpy"

// ReadDebugOffsetsFromPID opens /proc/<pid>/mem and extracts the
// _Py_DebugOffsets struct from a running CPython 3.13+ process. The
// struct is embedded at offset 0 of _PyRuntime, so runtimeAddr is the
// address the caller already located via the _PyRuntime ELF symbol.
//
// Returns (nil, nil) when:
//   - minor is < 13 (the struct did not exist before 3.13)
//   - runtimeAddr is 0 (caller couldn't resolve _PyRuntime)
//   - cookie validation fails (stale or partially-initialized memory)
//   - no layout is registered for this minor
//
// Returns an error only for actual memory-access failures the caller
// may want to log. All callers treat a nil result as "fall through to
// DWARF / hardcoded offsets".
//
// When it DOES return a populated PyOffsets, that table is authoritative
// for the exact CPython build the workload is running: immune to
// distro-patch layout drift, release-vs-debug ABI differences, and DWARF
// lookups from sibling libraries.
func ReadDebugOffsetsFromPID(pid uint32, runtimeAddr uint64, minor int) (*PyOffsets, error) {
	if minor < 13 || runtimeAddr == 0 {
		return nil, nil
	}
	mem, err := OpenProcMem(pid)
	if err != nil {
		return nil, fmt.Errorf("opening procmem for pid %d: %w", pid, err)
	}
	defer mem.Close()
	return readDebugOffsets(mem, runtimeAddr, minor)
}

// readDebugOffsets reads the _Py_DebugOffsets struct and parses it via
// parseDebugOffsets. Split out so parseDebugOffsets can be unit-tested
// against synthetic byte buffers without going through /proc.
func readDebugOffsets(mem *ProcMem, runtimeAddr uint64, minor int) (*PyOffsets, error) {
	if minor < 13 {
		return nil, nil
	}
	// 3.14's _Py_DebugOffsets is ~648 bytes; 3.13's is ~584 bytes. 1024
	// leaves comfortable headroom for any future minor that grows the
	// struct without requiring a re-size on every patch bump.
	const bufSize = 1024
	buf := make([]byte, bufSize)
	if err := mem.ReadAt(buf, runtimeAddr); err != nil {
		return nil, fmt.Errorf("reading _Py_DebugOffsets at 0x%x: %w", runtimeAddr, err)
	}
	return parseDebugOffsets(buf, minor)
}

// parseDebugOffsets extracts field offsets from a raw _Py_DebugOffsets
// byte buffer. Pure function — no I/O, no globals beyond the per-version
// layout tables — so the parser can be exercised in unit tests against
// fixture buffers built with Go.
func parseDebugOffsets(buf []byte, minor int) (*PyOffsets, error) {
	layout, ok := debugOffsetsLayouts[minor]
	if !ok {
		symDebugf("_Py_DebugOffsets: no layout registered for CPython 3.%d, falling back to DWARF/hardcoded", minor)
		return nil, nil
	}

	// Cookie: exactly 8 bytes, no NUL terminator. If the cookie is wrong
	// we're almost certainly reading the wrong address or a pre-3.13
	// build; either way, walking the rest of the struct would produce
	// garbage offsets.
	if len(buf) < 24 {
		return nil, fmt.Errorf("_Py_DebugOffsets buffer too short: %d bytes", len(buf))
	}
	if string(buf[0:8]) != debugOffsetsCookie {
		return nil, fmt.Errorf("_Py_DebugOffsets cookie mismatch: got %q, want %q", string(buf[0:8]), debugOffsetsCookie)
	}

	readU64 := func(off uint64) uint64 {
		if off+8 > uint64(len(buf)) {
			return 0
		}
		return uint64(buf[off]) | uint64(buf[off+1])<<8 | uint64(buf[off+2])<<16 |
			uint64(buf[off+3])<<24 | uint64(buf[off+4])<<32 | uint64(buf[off+5])<<40 |
			uint64(buf[off+6])<<48 | uint64(buf[off+7])<<56
	}

	// Version field is PY_VERSION_HEX, a packed integer whose high bits
	// encode major.minor. Validate the minor to reject "cookie matched
	// by chance on stale memory" and "we're reading the wrong struct".
	// PY_VERSION_HEX format: (major << 24) | (minor << 16) | (micro << 8) | (release << 4) | serial.
	version := readU64(8)
	gotMajor := int((version >> 24) & 0xff)
	gotMinor := int((version >> 16) & 0xff)
	if gotMajor != 3 || gotMinor != minor {
		return nil, fmt.Errorf("_Py_DebugOffsets version mismatch: buffer encodes 3.%d, caller asked for 3.%d (version=0x%x)",
			gotMinor, minor, version)
	}

	// Extract the offsets we actually use.
	tstateNativeThreadID := readU64(layout.tstateNativeThreadID)
	offsets := &PyOffsets{
		Version:                 fmt.Sprintf("3.%d-debugoffsets", minor),
		RuntimeInterpretersHead: readU64(layout.runtimeInterpretersHead),
		InterpTstateHead:        readU64(layout.interpThreadsHead),
		InterpNext:              readU64(layout.interpNext),
		// PyThreadState.next is not exported via _Py_DebugOffsets but has
		// been at byte 8 since 3.9 (stable across all supported versions).
		// Hardcoding it here keeps the layout table focused on fields that
		// genuinely drift between versions.
		TstateNext:           8,
		TstateThreadID:       readU64(layout.tstateThreadID),
		TstateNativeThreadID: tstateNativeThreadID,
		TstateFrame:          readU64(layout.tstateCurrentFrame),
		CframeCurrentFrame:   0, // 3.13+ dropped cframe; current_frame is direct.
		FrameBack:            readU64(layout.framePrevious),
		FrameCode:            readU64(layout.frameExecutable),
		CodeFilename:         readU64(layout.codeFilename),
		CodeName:             readU64(layout.codeName),
		CodeFirstLineNo:      readU64(layout.codeFirstLineNo),
		UnicodeLength:        readU64(layout.unicodeLength),
		UnicodeState:         readU64(layout.unicodeState),
		UnicodeData:          readU64(layout.unicodeASCII),
		NewStyleFrames:       true,
	}

	// Basic sanity check: every offset here is a position within a
	// CPython struct, so kilobyte-scale values almost certainly mean we
	// mis-parsed the buffer. PyInterpreterState has grown past 7000
	// bytes in 3.13, so use a forgiving upper bound.
	if offsets.InterpTstateHead > 32768 || offsets.TstateFrame > 4096 ||
		offsets.FrameCode > 4096 || offsets.CodeFilename > 4096 ||
		offsets.UnicodeData > 256 {
		return nil, fmt.Errorf("_Py_DebugOffsets sanity check failed for 3.%d: InterpTstateHead=%d TstateFrame=%d FrameCode=%d CodeFilename=%d UnicodeData=%d",
			minor, offsets.InterpTstateHead, offsets.TstateFrame, offsets.FrameCode, offsets.CodeFilename, offsets.UnicodeData)
	}

	symDebugf("_Py_DebugOffsets 3.%d parsed: RuntimeInterpretersHead=%d InterpTstateHead=%d TstateFrame=%d FrameBack=%d FrameCode=%d CodeFilename=%d CodeName=%d UnicodeData=%d",
		minor,
		offsets.RuntimeInterpretersHead, offsets.InterpTstateHead,
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
	// Try known-offsets DB first — fastest path, no DWARF parsing needed.
	if dbOffsets, err := LookupByBuildID(libPath); err != nil {
		symDebugf("known-offsets DB lookup failed: %v", err)
	} else if dbOffsets != nil {
		slog.Info("using known offsets from DB", "python_version", fmt.Sprintf("3.%d", minor), "source", dbOffsets.Version)
		return dbOffsets
	}

	// Try DWARF next.
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
	case 9:
		return pyOffsets39()
	case 10:
		return pyOffsets310()
	case 11:
		return pyOffsets311()
	case 12:
		return pyOffsets312()
	case 13:
		return pyOffsets313()
	case 14:
		return pyOffsets314()
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

		TstateNext:     8,   // PyThreadState.next
		TstateThreadID: 176, // PyThreadState.thread_id (pthread_self)
		// native_thread_id (gettid) was added to PyThreadState upstream in
		// 3.11, not 3.8 as an earlier comment in this file claimed. Ubuntu
		// backports the field to 3.10 via a distro patch (where it sits at
		// byte 184 but is never populated for the main thread, which is why
		// find_thread_state needs the single-tstate fallback). On vanilla
		// 3.9/3.10 builds the field does not exist at all; the walker reads
		// whatever happens to live at byte 184 and relies entirely on the
		// single-tstate fallback to still return a correct tstate.
		TstateNativeThreadID: 184,
		TstateFrame:          24, // PyThreadState.frame (PyFrameObject*)

		CframeCurrentFrame: 0, // Not used in 3.10

		// PyFrameObject on x86_64: PyObject_VAR_HEAD (24) + f_back (8) + f_code (8) + ...
		// The 16 value previously hardcoded here matched an older (3.7/3.8) layout.
		FrameBack: 24, // PyFrameObject.f_back
		FrameCode: 32, // PyFrameObject.f_code (verified on Ubuntu 22.04 CPython 3.10.12)

		CodeFilename:    104, // PyCodeObject.co_filename
		CodeName:        112, // PyCodeObject.co_name
		// Verified on Ubuntu 22.04 CPython 3.10.12: co_firstlineno sits at
		// byte 40 (u32), just after co_argcount/co_posonlyargcount/... which
		// pack into the preceding 8 bytes. The 48 value previously here
		// matched a different build's layout and produced garbage line
		// numbers on vanilla 3.10.
		CodeFirstLineNo: 40,

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
		TstateNativeThreadID: 184, // gettid (added upstream in 3.11)
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

// CPython 3.9 uses the legacy PyFrameObject layout (no _PyInterpreterFrame,
// no cframe indirection). Routed to walker_310 by the dispatcher.
//
// Offsets derived from cpython v3.9.25 upstream headers:
//
//   _PyRuntime (pycore_runtime.h):
//     4 ints (preinitializing..initialized) -> 16
//     _Py_atomic_address _finalizing         -> +8  = 24
//     pyinterpreters.mutex                   -> +8  = 32
//     pyinterpreters.head                    -> RuntimeInterpretersHead = 32
//
//   PyInterpreterState (pycore_interp.h):
//     next (8) -> 0
//     tstate_head -> InterpTstateHead = 8
//
//   PyThreadState (cpython/pystate.h):
//     prev/next/interp (3 * 8)         -> 24
//     frame                            -> TstateFrame = 24
//     recursion_depth/overflowed/...   -> +8  = 32
//     stackcheck/tracing/use_tracing   -> +16 = 48
//     pad+c_profilefunc/c_tracefunc/c_profileobj/c_traceobj (4*8)
//     curexc_type/value/traceback (3*8) -> +32 = 104
//     exc_state (_PyErr_StackItem, 32) -> +32 = 144
//     exc_info/dict (2*8)              -> 160
//     gilstate_counter+pad+async_exc   -> 176
//     thread_id                        -> TstateThreadID = 176
//
//   native_thread_id does NOT exist in vanilla 3.9 (upstream added it in
//   3.11). The walker still sets TstateNativeThreadID = 184 (inherited
//   from 3.10) because find_thread_state reads that offset unconditionally;
//   on 3.9 it ends up reading `trash_delete_nesting` (int) and fails the
//   kernel-tid match, at which point the single-tstate fallback returns
//   the first (and only) tstate for the common single-threaded case.
//
//   PyFrameObject (cpython/frameobject.h):
//     PyObject_VAR_HEAD (24) + f_back (8) + f_code (8) = FrameCode offset 32
//     FrameBack = 24
//
//   PyCodeObject (cpython/code.h) layout is identical to 3.10, so
//   CodeFilename=104, CodeName=112, CodeFirstLineNo=40 carry over unchanged.
//
// Open question: the plan's pre-validation matrix reported "3.9: 0 frames,
// walker enters, emits nothing". That observation was made before the
// precursor single-tstate fallback landed; with that fallback now in place,
// the walker should emit frames for single-threaded 3.9 workloads. Live
// /proc/<pid>/mem verification on a uv-distributed cpython-3.9.x process
// is still pending to rule out any build-specific quirk not covered by the
// upstream header derivation above.
func pyOffsets39() *PyOffsets {
	o := pyOffsets310()
	o.Version = "3.9"
	o.RuntimeInterpretersHead = 32
	return o
}

// CPython 3.13 (stable since 2024-10)
//
// 3.13 dropped the _PyCFrame indirection and made PyThreadState.current_frame
// a direct pointer to _PyInterpreterFrame, so the frame-walk path matches 3.12.
// _PyInterpreterFrame layout is unchanged from 3.12 on x86_64 (verified against
// CPython v3.13.0..v3.13.13 Include/internal/pycore_frame.h):
//
//   offset 0:  PyObject *f_executable   (code object)
//   offset 8:  _PyInterpreterFrame *previous
//   offset 16: PyObject *f_funcobj
//   offset 24: PyObject *f_globals
//   offset 32: PyObject *f_builtins
//   offset 40: PyObject *f_locals
//   ...
//
// The real 3.12→3.13 deltas are outside the frame struct:
//   - _PyRuntime grew a _Py_DebugOffsets prefix, pushing interpreters.head
//     from byte 40 to byte 632.
//   - PyInterpreterState grew (PEP 684 per-interpreter state), pushing
//     threads.head from byte 16 to byte 7344.
//   - PyThreadState.current_frame moved from byte 56 to byte 232.
//   - PyCodeObject.co_firstlineno moved from byte 48 to byte 68.
//
// _Py_DebugOffsets is the authoritative source for 3.13+ but only comes into
// play when the reader at `ReadDebugOffsetsFromPID` has the 3.13 layout
// constants wired up; these fallback values unblock the walker in the
// meantime.
func pyOffsets313() *PyOffsets {
	return &PyOffsets{
		Version: "3.13",
		// 3.13 embeds _Py_DebugOffsets at the start of _PyRuntime. That
		// prefix pushes the `interpreters.head` field far past 3.12's
		// byte 40 (empirically byte 632 on cpython 3.13.13). Once the
		// _Py_DebugOffsets reader is extended to 3.13, this fallback
		// becomes a backstop; until then it's the only way the walker
		// can reach the interpreter list on 3.13.
		RuntimeInterpretersHead: 632,

		// PyInterpreterState grew dramatically in 3.13 (PEP 684 per-
		// interpreter state + debug offsets + more). threads.head sits
		// at byte 7344, far beyond the 16 that was valid in 3.12.
		InterpTstateHead: 7344,

		TstateNext:           8,
		TstateThreadID:       152,
		TstateNativeThreadID: 160,
		// current_frame in 3.13 sits at byte 72 of PyThreadState, not 232.
		// Layout per Include/cpython/pystate.h: prev/next/interp (24) +
		// eval_breaker (8) + _status+_whence+state+py_recursion_* +
		// c_recursion_*+tracing+what_event (10 ints = 40) + 4-byte pad = 72.
		// Verified on /proc/<pid>/mem of cpython-3.13.13 (uv distribution)
		// by walking `previous` from every pointer in tstate[:2KB] and
		// confirming only the one at +72 chains to sys._getframe().f_frame.
		// The earlier 232 pointed at datastack_chunk (a _PyStackChunk
		// header), which explains the "1 garbage frame then 0" symptom.
		TstateFrame:          72,

		CframeCurrentFrame: 0,

		// _PyInterpreterFrame field order on 3.13 matches 3.12:
		// f_executable at 0, previous at 8.
		FrameBack: 8,
		FrameCode: 0,

		CodeFilename:    112,
		CodeName:        120,
		CodeFirstLineNo: 68,

		// PyASCIIObject layout shifted in 3.13 (PEP 623 cleanup, removed
		// wstr/wstr_length). State and inline-data offsets are not the
		// same as 3.12's (20 and 48).
		UnicodeLength: 16,
		UnicodeState:  32,
		UnicodeData:   40,

		NewStyleFrames: true,
	}
}

// CPython 3.14 uses direct current_frame like 3.13. _Py_DebugOffsets grew
// again, shifting interpreters.head from 3.13's 632 to 808 on cpython
// 3.14.4 (empirical). _PyInterpreterFrame field order at frame-walk offsets
// (0 and 8) is unchanged, so FrameCode and FrameBack from 3.13 carry over.
//
// NOTE: 3.14 re-typed f_executable from `PyObject *` to the `_PyStackRef`
// union (Include/internal/pycore_interpframe_structs.h). The low bits of
// the 8-byte word carry a tag (Py_TAG_REFCNT = 1 in default GIL build).
// The eBPF walker masks `code_ptr &= ~0x7ULL` after reading f_executable
// to strip the tag before treating the value as a PyCodeObject pointer;
// see walk_python_frames_312 in bpf/python_walker.bpf.h.
func pyOffsets314() *PyOffsets {
	o := pyOffsets313()
	o.Version = "3.14"
	o.RuntimeInterpretersHead = 808
	return o
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
