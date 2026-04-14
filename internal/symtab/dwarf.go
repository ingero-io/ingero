package symtab

import (
	"debug/dwarf"
	"debug/elf"
	"encoding/binary"
	"encoding/hex"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
)

// debugLogFn is called for DWARF diagnostic messages when set.
// Wire from CLI via SetDebugLog(). Nil = silent (default).
var debugLogFn func(format string, args ...any)

// SetDebugLog sets the debug logging function for the symtab package.
// Pass cli.debugf (or equivalent) when --debug is enabled.
// Pass nil to disable debug logging.
func SetDebugLog(fn func(format string, args ...any)) {
	debugLogFn = fn
}

// symDebugf logs a debug message if debug logging is enabled.
func symDebugf(format string, args ...any) {
	if debugLogFn != nil {
		debugLogFn(format, args...)
	}
}

// FindDebugFile locates separate DWARF debug info for a stripped ELF binary.
//
// Strategy (in priority order):
//  1. Build ID: read .note.gnu.build-id → /usr/lib/debug/.build-id/XX/XXXX.debug
//  2. Debuglink: read .gnu_debuglink → search dir/, dir/.debug/, /usr/lib/debug/dir/
//  3. Inline: check if libPath itself has .debug_info (unstripped, e.g. conda builds)
//  4. Debug build: try libpythonX.Yd.so (Ubuntu's -dbg package includes full DWARF).
//     Since CPython 3.8, Py_DEBUG no longer implies Py_TRACE_REFS, so the debug
//     build's struct layouts match the release build (no _PyObject_HEAD_EXTRA).
//     This is the most reliable fallback on Ubuntu when -dbgsym packages aren't
//     available from ddebs.ubuntu.com.
//
// Returns "" with nil error if no debug info is found (caller should fall back).
func FindDebugFile(libPath string) (string, error) {
	f, err := elf.Open(libPath)
	if err != nil {
		return "", fmt.Errorf("opening ELF %s: %w", libPath, err)
	}
	defer f.Close()

	// Strategy 1: Build ID lookup.
	if path, err := findByBuildID(f); err == nil && path != "" {
		symDebugf("found debug file via build-id: %s", path)
		return path, nil
	}

	// Strategy 2: .gnu_debuglink lookup.
	if path, err := findByDebuglink(f, libPath); err == nil && path != "" {
		symDebugf("found debug file via debuglink: %s", path)
		return path, nil
	}

	// Strategy 3: Check if the binary itself has DWARF sections.
	if hasInlineDWARF(f) {
		symDebugf("found inline DWARF in %s", libPath)
		return libPath, nil
	}

	// Strategy 4: Try the debug build library (libpythonX.Yd.so).
	// Ubuntu's libpython3.X-dbg package installs a debug build with full DWARF.
	// Since CPython 3.8, Py_DEBUG no longer implies Py_TRACE_REFS, so struct
	// layouts match the release build (verified on Ubuntu 22.04 with offsetof).
	if path := findDebugBuildLib(libPath); path != "" {
		return path, nil
	}

	symDebugf("no DWARF debug info found for %s (install libpython3.X-dbgsym or libpython3.X-dbg)", libPath)
	return "", nil
}

// ReadBuildID reads the .note.gnu.build-id section of an ELF file and
// returns the build-id as a lowercase hex string (e.g., "a1b2c3d4...").
// Returns "" if the section is missing or malformed.
func ReadBuildID(libPath string) (string, error) {
	f, err := elf.Open(libPath)
	if err != nil {
		return "", fmt.Errorf("opening ELF %s: %w", libPath, err)
	}
	defer f.Close()

	return readBuildIDFromELF(f)
}

// readBuildIDFromELF parses .note.gnu.build-id from an already-opened ELF file
// and returns the build-id as a lowercase hex string. Returns "" if the section
// is missing or malformed.
//
// The .note.gnu.build-id section contains an ELF note with:
//   - 4 bytes: name size (should be 4 for "GNU\0")
//   - 4 bytes: desc size (typically 20 for SHA-1)
//   - 4 bytes: type (NT_GNU_BUILD_ID = 3)
//   - name (padded to 4-byte alignment)
//   - desc (the build ID bytes)
func readBuildIDFromELF(f *elf.File) (string, error) {
	sect := f.Section(".note.gnu.build-id")
	if sect == nil {
		return "", nil
	}

	data, err := sect.Data()
	if err != nil {
		return "", err
	}

	// Parse ELF note. Minimum: 12 bytes header + 4 bytes name + 1 byte desc.
	if len(data) < 16 {
		return "", nil
	}

	var bo binary.ByteOrder
	switch f.ByteOrder {
	case binary.LittleEndian:
		bo = binary.LittleEndian
	default:
		bo = binary.BigEndian
	}

	nameSize := bo.Uint32(data[0:4])
	descSize := bo.Uint32(data[4:8])
	noteType := bo.Uint32(data[8:12])

	// NT_GNU_BUILD_ID = 3
	if noteType != 3 {
		return "", nil
	}

	// Sanity check: sizes must fit within the section data.
	dataLen := uint32(len(data))
	if nameSize > dataLen || descSize > dataLen {
		return "", nil
	}

	// Name starts at offset 12, padded to 4-byte alignment.
	nameEnd := 12 + align4(nameSize)
	descEnd := nameEnd + descSize

	// Check for overflow and bounds.
	if nameEnd < 12 || descEnd < nameEnd || dataLen < descEnd {
		return "", nil
	}

	buildID := data[nameEnd:descEnd]
	if len(buildID) < 2 {
		return "", nil
	}

	return hex.EncodeToString(buildID), nil
}

// findByBuildID reads .note.gnu.build-id and looks for
// /usr/lib/debug/.build-id/XX/XXXXXXXX.debug.
func findByBuildID(f *elf.File) (string, error) {
	idHex, err := readBuildIDFromELF(f)
	if err != nil {
		return "", err
	}
	if idHex == "" || len(idHex) < 4 {
		return "", nil
	}

	// Build path: /usr/lib/debug/.build-id/XX/YYYY.debug
	debugPath := filepath.Join("/usr/lib/debug/.build-id", idHex[:2], idHex[2:]+".debug")

	if _, err := os.Stat(debugPath); err == nil {
		return debugPath, nil
	}

	return "", nil
}

// findByDebuglink reads .gnu_debuglink and searches standard paths.
//
// The section contains: null-terminated filename + padding + 4-byte CRC32.
// We search for the debug file in:
//   - dir(libPath)/filename
//   - dir(libPath)/.debug/filename
//   - /usr/lib/debug/dir(libPath)/filename
func findByDebuglink(f *elf.File, libPath string) (string, error) {
	sect := f.Section(".gnu_debuglink")
	if sect == nil {
		return "", nil
	}

	data, err := sect.Data()
	if err != nil {
		return "", err
	}

	// Extract null-terminated filename.
	var nameEnd int
	for nameEnd = 0; nameEnd < len(data); nameEnd++ {
		if data[nameEnd] == 0 {
			break
		}
	}
	if nameEnd == 0 || nameEnd >= len(data) {
		return "", nil
	}

	debugName := string(data[:nameEnd])
	dir := filepath.Dir(libPath)

	// Search order matches GDB's algorithm.
	candidates := []string{
		filepath.Join(dir, debugName),
		filepath.Join(dir, ".debug", debugName),
		filepath.Join("/usr/lib/debug", dir, debugName),
	}

	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c, nil
		}
	}

	return "", nil
}

// findDebugBuildLib checks for a debug build of libpython (libpythonX.Yd.so).
//
// On Ubuntu, `apt install libpython3.10-dbg` installs libpython3.10d.so with
// full DWARF debug info. Since CPython 3.8, Py_DEBUG no longer implies
// Py_TRACE_REFS, so struct layouts match the release build.
//
// Handles two input patterns:
//
//  1. Shared library: "/usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0"
//     → tries libpython3.10d.so.1.0, libpython3.10d.so in same dir
//
//  2. Static binary: "/usr/bin/python3.10" (Ubuntu default, statically linked)
//     → tries libpython3.10d.so in standard lib dirs
func findDebugBuildLib(libPath string) string {
	base := filepath.Base(libPath)

	// Case 1: libpythonX.Y.so[.suffix]
	const libPrefix = "libpython"
	if len(base) >= len(libPrefix)+4 && base[:len(libPrefix)] == libPrefix {
		return findDebugBuildFromLib(libPath, base)
	}

	// Case 2: python3.X or python3.10 binary (statically linked)
	if ver := extractPythonVersionFromBinary(base); ver != "" {
		return findDebugBuildFromBinary(ver)
	}

	return ""
}

func findDebugBuildFromLib(libPath, base string) string {
	dir := filepath.Dir(libPath)
	const prefix = "libpython"

	rest := base[len(prefix):] // "3.10.so.1.0"
	dotSo := ".so"
	soIdx := -1
	for i := 0; i+len(dotSo) <= len(rest); i++ {
		if rest[i:i+len(dotSo)] == dotSo {
			soIdx = i
			break
		}
	}
	if soIdx < 0 {
		return ""
	}

	ver := rest[:soIdx]    // "3.10"
	suffix := rest[soIdx:] // ".so.1.0"

	candidates := []string{
		filepath.Join(dir, prefix+ver+"d"+suffix),
	}
	if suffix != ".so" {
		candidates = append(candidates, filepath.Join(dir, prefix+ver+"d.so"))
	}

	return findFirstDWARFLib(candidates)
}

func findDebugBuildFromBinary(ver string) string {
	const prefix = "libpython"
	debugLib := prefix + ver + "d.so"

	// Multiarch directory: x86_64-linux-gnu on amd64, aarch64-linux-gnu on arm64.
	archDir := "x86_64-linux-gnu"
	if runtime.GOARCH == "arm64" {
		archDir = "aarch64-linux-gnu"
	}

	dirs := []string{
		"/usr/lib/" + archDir,
		"/lib/" + archDir,
		"/usr/lib",
		"/lib",
	}

	var candidates []string
	for _, dir := range dirs {
		candidates = append(candidates,
			filepath.Join(dir, debugLib+".1.0"),
			filepath.Join(dir, debugLib),
		)
	}

	return findFirstDWARFLib(candidates)
}

func extractPythonVersionFromBinary(base string) string {
	const prefix = "python"
	if len(base) < len(prefix)+3 {
		return ""
	}
	if base[:len(prefix)] != prefix {
		return ""
	}
	ver := base[len(prefix):]
	if len(ver) < 3 || ver[0] < '0' || ver[0] > '9' {
		return ""
	}
	hasDot := false
	for _, c := range ver {
		if c == '.' {
			hasDot = true
		}
	}
	if !hasDot {
		return ""
	}
	return ver
}

func findFirstDWARFLib(candidates []string) string {
	for _, c := range candidates {
		if _, err := os.Stat(c); err != nil {
			continue
		}
		df, err := elf.Open(c)
		if err != nil {
			continue
		}
		ok := hasInlineDWARF(df)
		df.Close()
		if ok {
			symDebugf("found debug build library with DWARF: %s", c)
			return c
		}
	}
	return ""
}

// hasInlineDWARF checks if the ELF file itself contains DWARF debug info.
// True for unstripped builds (conda, pyenv, debug builds, Go binaries).
// Checks both uncompressed (.debug_info) and compressed (.zdebug_info) sections.
func hasInlineDWARF(f *elf.File) bool {
	if f.Section(".debug_info") != nil {
		return true
	}
	// Go 1.16+ uses zlib-compressed DWARF sections by default.
	if f.Section(".zdebug_info") != nil {
		return true
	}
	// Some toolchains use SHF_COMPRESSED on .debug_info (SHT_PROGBITS with flag).
	// Go's elf.File.DWARF() handles these transparently, so try it as a last resort.
	_, err := f.DWARF()
	return err == nil
}

// align4 rounds n up to the next 4-byte boundary.
func align4(n uint32) uint32 {
	return (n + 3) &^ 3
}

// ExtractStructOffsets opens an ELF file, parses DWARF, and returns a map of
// field_name → byte_offset for the named struct.
//
// For example, ExtractStructOffsets("/usr/lib/debug/...", "PyThreadState")
// returns {"next": 8, "thread_id": 176, "frame": 24, ...}.
//
// Returns an error if the ELF has no DWARF info or the struct isn't found.
// Skips forward-declaration (incomplete) structs.
func ExtractStructOffsets(elfPath, structName string) (map[string]int64, error) {
	st, err := extractStructType(elfPath, structName)
	if err != nil {
		return nil, err
	}

	offsets := make(map[string]int64, len(st.Field))
	for _, f := range st.Field {
		offsets[f.Name] = f.ByteOffset
	}
	return offsets, nil
}

// extractStructType returns the full *dwarf.StructType for the named struct.
// Used when we need both offsets and the struct's total Size.
//
// Handles both direct struct types and typedefs (e.g., CPython's
// PyThreadState = typedef for struct _ts).
func extractStructType(elfPath, structName string) (*dwarf.StructType, error) {
	f, err := elf.Open(elfPath)
	if err != nil {
		return nil, fmt.Errorf("opening ELF %s: %w", elfPath, err)
	}
	defer f.Close()

	dw, err := f.DWARF()
	if err != nil {
		return nil, fmt.Errorf("reading DWARF from %s: %w", elfPath, err)
	}

	st, err := findStructInDWARF(dw, structName)
	if err != nil {
		return nil, fmt.Errorf("%w of %s", err, elfPath)
	}

	return st, nil
}

// extractNestedFieldOffset extracts offset of a field inside a nested struct.
//
// For example, _PyRuntime.interpreters.head:
//  1. Find pyruntimestate struct → offset of "interpreters" field
//  2. Follow type of "interpreters" → find its "head" field offset
//  3. Return sum of both offsets
//
// The parentStruct is the top-level struct name, fields is the dot-separated path.
func extractNestedFieldOffset(elfPath, parentStruct string, fields []string) (int64, error) {
	if len(fields) < 2 {
		return 0, fmt.Errorf("need at least 2 fields for nested offset, got %d", len(fields))
	}

	f, err := elf.Open(elfPath)
	if err != nil {
		return 0, fmt.Errorf("opening ELF %s: %w", elfPath, err)
	}
	defer f.Close()

	dw, err := f.DWARF()
	if err != nil {
		return 0, fmt.Errorf("reading DWARF from %s: %w", elfPath, err)
	}

	// Find the top-level struct.
	st, err := findStructInDWARF(dw, parentStruct)
	if err != nil {
		return 0, err
	}

	var totalOffset int64

	// Walk through the field chain.
	for i, fieldName := range fields {
		field := findField(st, fieldName)
		if field == nil {
			return 0, fmt.Errorf("field %q not found in struct %s", fieldName, st.StructName)
		}

		totalOffset += field.ByteOffset

		// If there are more fields to traverse, resolve the field's type to a struct.
		if i < len(fields)-1 {
			inner, ok := resolveToStruct(field.Type)
			if !ok {
				return 0, fmt.Errorf("field %q in %s is not a struct type", fieldName, st.StructName)
			}
			st = inner
		}
	}

	return totalOffset, nil
}

// findStructInDWARF searches DWARF data for a named struct type.
//
// Searches both DW_TAG_structure_type entries and DW_TAG_typedef entries.
// CPython uses typedefs extensively: PyThreadState is typedef for struct _ts,
// PyInterpreterState for struct _is, etc. When the name matches a typedef,
// we follow the type chain to find the underlying struct.
func findStructInDWARF(dw *dwarf.Data, name string) (*dwarf.StructType, error) {
	reader := dw.Reader()
	for {
		entry, err := reader.Next()
		if err != nil {
			return nil, fmt.Errorf("iterating DWARF: %w", err)
		}
		if entry == nil {
			break
		}

		// Match struct tags directly, or typedefs that resolve to structs.
		if entry.Tag != dwarf.TagStructType && entry.Tag != dwarf.TagTypedef {
			continue
		}

		sname, _ := entry.Val(dwarf.AttrName).(string)
		if sname != name {
			continue
		}

		typ, err := dw.Type(entry.Offset)
		if err != nil {
			continue
		}

		// For direct struct types:
		if st, ok := typ.(*dwarf.StructType); ok && !st.Incomplete {
			return st, nil
		}

		// For typedefs: follow the chain (typedef → qualifier → struct).
		if st, ok := resolveToStruct(typ); ok {
			return st, nil
		}
	}

	return nil, fmt.Errorf("struct %q not found in DWARF", name)
}

// findField finds a field by name in a struct type.
func findField(st *dwarf.StructType, name string) *dwarf.StructField {
	for i := range st.Field {
		if st.Field[i].Name == name {
			return st.Field[i]
		}
	}
	return nil
}

// resolveToStruct follows typedefs, pointers, and qualifiers to find a struct.
// Handles chains like: typedef → const → struct, or pointer → struct.
func resolveToStruct(t dwarf.Type) (*dwarf.StructType, bool) {
	for {
		switch v := t.(type) {
		case *dwarf.StructType:
			if !v.Incomplete {
				return v, true
			}
			return nil, false
		case *dwarf.TypedefType:
			t = v.Type
		case *dwarf.QualType:
			t = v.Type
		case *dwarf.PtrType:
			t = v.Type
		default:
			return nil, false
		}
	}
}

// BuildPyOffsetsFromDWARF extracts CPython struct offsets from DWARF debug info.
//
// This is the core function that replaces hardcoded offset tables when debug info
// is available. It extracts offsets for all the structs the frame walker needs:
//   - pyruntimestate (_PyRuntime) with nested interpreters.head
//   - PyInterpreterState
//   - PyThreadState (version-specific field names)
//   - _PyCFrame (3.11 only)
//   - PyFrameObject (3.10) / _PyInterpreterFrame (3.11+)
//   - PyCodeObject
//   - PyASCIIObject (including struct size for data offset)
//
// minor is the CPython minor version (10, 11, 12, 13, ...).
func BuildPyOffsetsFromDWARF(debugPath string, minor int) (*PyOffsets, error) {
	offsets := &PyOffsets{
		Version: fmt.Sprintf("3.%d-dwarf", minor),
	}

	// --- _PyRuntime.interpreters.head ---
	// This is always a nested path: pyruntimestate → interpreters → head
	runtimeHead, err := extractNestedFieldOffset(debugPath, "pyruntimestate", []string{"interpreters", "head"})
	if err != nil {
		return nil, fmt.Errorf("_PyRuntime.interpreters.head: %w", err)
	}
	offsets.RuntimeInterpretersHead = uint64(runtimeHead)

	// --- PyInterpreterState ---
	interpOffsets, err := ExtractStructOffsets(debugPath, "PyInterpreterState")
	if err != nil {
		return nil, fmt.Errorf("PyInterpreterState: %w", err)
	}

	// 3.12+ renamed tstate_head → threads.head (nested struct).
	if off, ok := interpOffsets["tstate_head"]; ok {
		offsets.InterpTstateHead = uint64(off)
	} else {
		// 3.12+: threads is a nested struct containing head.
		nestedOff, err := extractNestedFieldOffset(debugPath, "PyInterpreterState", []string{"threads", "head"})
		if err != nil {
			return nil, fmt.Errorf("PyInterpreterState.threads.head: %w", err)
		}
		offsets.InterpTstateHead = uint64(nestedOff)
	}

	// --- PyThreadState ---
	tstateOffsets, err := ExtractStructOffsets(debugPath, "PyThreadState")
	if err != nil {
		return nil, fmt.Errorf("PyThreadState: %w", err)
	}

	if off, ok := tstateOffsets["next"]; ok {
		offsets.TstateNext = uint64(off)
	}
	if off, ok := tstateOffsets["thread_id"]; ok {
		offsets.TstateThreadID = uint64(off)
	}
	if off, ok := tstateOffsets["native_thread_id"]; ok {
		offsets.TstateNativeThreadID = uint64(off)
	}

	// Frame pointer field is version-dependent:
	//   3.10: frame (PyFrameObject*)
	//   3.11: cframe (_PyCFrame*)
	//   3.12+: current_frame (_PyInterpreterFrame*)
	switch {
	case minor >= 12:
		if off, ok := tstateOffsets["current_frame"]; ok {
			// Upstream CPython 3.12+: current_frame directly in PyThreadState.
			offsets.TstateFrame = uint64(off)
			offsets.NewStyleFrames = true
		} else if off, ok := tstateOffsets["cframe"]; ok {
			// Distro-patched builds (e.g. Ubuntu 24.04's 3.12.3) retain the
			// 3.11-style _PyCFrame indirection for ABI stability.
			offsets.TstateFrame = uint64(off)
			offsets.NewStyleFrames = true

			cframeOffsets, err := ExtractStructOffsets(debugPath, "_PyCFrame")
			if err != nil {
				return nil, fmt.Errorf("_PyCFrame (3.%d cframe fallback): %w", minor, err)
			}
			if cfOff, ok := cframeOffsets["current_frame"]; ok {
				offsets.CframeCurrentFrame = uint64(cfOff)
			}
		} else {
			return nil, fmt.Errorf("PyThreadState.current_frame (or cframe) not found for 3.%d", minor)
		}

	case minor == 11:
		if off, ok := tstateOffsets["cframe"]; ok {
			offsets.TstateFrame = uint64(off)
			offsets.NewStyleFrames = true

			// _PyCFrame.current_frame offset
			cframeOffsets, err := ExtractStructOffsets(debugPath, "_PyCFrame")
			if err != nil {
				return nil, fmt.Errorf("_PyCFrame: %w", err)
			}
			if cfOff, ok := cframeOffsets["current_frame"]; ok {
				offsets.CframeCurrentFrame = uint64(cfOff)
			}
		} else {
			return nil, fmt.Errorf("PyThreadState.cframe not found for 3.11")
		}

	default: // 3.10 and earlier
		if off, ok := tstateOffsets["frame"]; ok {
			offsets.TstateFrame = uint64(off)
			offsets.NewStyleFrames = false
		} else {
			return nil, fmt.Errorf("PyThreadState.frame not found for 3.%d", minor)
		}
	}

	// --- Frame struct offsets ---
	if minor >= 11 {
		// _PyInterpreterFrame (3.11+)
		frameOffsets, err := ExtractStructOffsets(debugPath, "_PyInterpreterFrame")
		if err != nil {
			return nil, fmt.Errorf("_PyInterpreterFrame: %w", err)
		}

		if off, ok := frameOffsets["previous"]; ok {
			offsets.FrameBack = uint64(off)
		}

		// 3.11 uses f_code, 3.12+ uses f_executable
		if minor >= 12 {
			if off, ok := frameOffsets["f_executable"]; ok {
				offsets.FrameCode = uint64(off)
			} else if off, ok := frameOffsets["f_code"]; ok {
				offsets.FrameCode = uint64(off)
			}
		} else {
			if off, ok := frameOffsets["f_code"]; ok {
				offsets.FrameCode = uint64(off)
			}
		}
	} else {
		// PyFrameObject (3.10)
		frameOffsets, err := ExtractStructOffsets(debugPath, "PyFrameObject")
		if err != nil {
			return nil, fmt.Errorf("PyFrameObject: %w", err)
		}

		if off, ok := frameOffsets["f_back"]; ok {
			offsets.FrameBack = uint64(off)
		}
		if off, ok := frameOffsets["f_code"]; ok {
			offsets.FrameCode = uint64(off)
		}
	}

	// --- PyCodeObject ---
	codeOffsets, err := ExtractStructOffsets(debugPath, "PyCodeObject")
	if err != nil {
		return nil, fmt.Errorf("PyCodeObject: %w", err)
	}

	if off, ok := codeOffsets["co_filename"]; ok {
		offsets.CodeFilename = uint64(off)
	}
	if off, ok := codeOffsets["co_name"]; ok {
		offsets.CodeName = uint64(off)
	}
	if off, ok := codeOffsets["co_firstlineno"]; ok {
		offsets.CodeFirstLineNo = uint64(off)
	}

	// --- PyASCIIObject ---
	// UnicodeData = sizeof(PyASCIIObject) because compact ASCII data immediately follows.
	st, err := extractStructType(debugPath, "PyASCIIObject")
	if err != nil {
		return nil, fmt.Errorf("PyASCIIObject: %w", err)
	}

	asciiOffsets := make(map[string]int64, len(st.Field))
	for _, f := range st.Field {
		asciiOffsets[f.Name] = f.ByteOffset
	}

	if off, ok := asciiOffsets["length"]; ok {
		offsets.UnicodeLength = uint64(off)
	}
	if off, ok := asciiOffsets["state"]; ok {
		offsets.UnicodeState = uint64(off)
	}

	// Compact ASCII data starts right after the struct.
	offsets.UnicodeData = uint64(st.Size())

	// --- Validation ---
	// On 64-bit, pointer-chased fields are always at offset > 0.
	// FrameCode and CframeCurrentFrame can legitimately be 0.
	if err := validateDWARFOffsets(offsets); err != nil {
		return nil, err
	}

	return offsets, nil
}

// validateDWARFOffsets checks that critical offsets were actually found in DWARF.
// Fields that must be > 0 on any 64-bit build:
//   - TstateNext: always after the prev pointer, never at offset 0
//   - TstateThreadID: deep in the struct, never at offset 0
//   - TstateFrame: pointer to frame, never at offset 0
//   - CodeFilename, CodeName: deep in PyCodeObject, never at offset 0
//   - UnicodeData: sizeof(PyASCIIObject), always > 0
//
// Fields that CAN be 0: FrameCode (3.11/3.12), CframeCurrentFrame (3.11).
func validateDWARFOffsets(o *PyOffsets) error {
	type check struct {
		name string
		val  uint64
	}
	checks := []check{
		{"TstateNext", o.TstateNext},
		{"TstateThreadID", o.TstateThreadID},
		{"TstateFrame", o.TstateFrame},
		{"CodeFilename", o.CodeFilename},
		{"CodeName", o.CodeName},
		{"UnicodeData", o.UnicodeData},
	}

	for _, c := range checks {
		if c.val == 0 {
			return fmt.Errorf("DWARF extraction produced zero offset for %s — likely missing field in debug info", c.name)
		}
	}

	return nil
}
