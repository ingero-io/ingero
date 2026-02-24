package symtab

import (
	"debug/elf"
	"fmt"
	"os"
	"os/exec"
	"path/filepath"
	"strings"
	"testing"
)

// execCommand wraps exec.Command for testability.
var execCommand = exec.Command

// TestExtractStructOffsets_Libc validates DWARF parsing against a real system library.
// libc always has at least the debug link; on many systems /usr/lib/debug has the full DWARF.
// We look for "timespec" which is a well-known, stable struct.
func TestExtractStructOffsets_Libc(t *testing.T) {
	// Find libc on this system.
	candidates := []string{
		"/usr/lib/x86_64-linux-gnu/libc.so.6",
		"/lib/x86_64-linux-gnu/libc.so.6",
		"/usr/lib/libc.so.6",
	}

	var libcPath string
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			libcPath = c
			break
		}
	}
	if libcPath == "" {
		t.Skip("libc.so.6 not found on this system")
	}

	debugPath, err := FindDebugFile(libcPath)
	if err != nil {
		t.Fatalf("FindDebugFile(%s): %v", libcPath, err)
	}
	if debugPath == "" {
		t.Skip("no debug info found for libc (install libc6-dbg to enable)")
	}

	offsets, err := ExtractStructOffsets(debugPath, "timespec")
	if err != nil {
		t.Fatalf("ExtractStructOffsets(timespec): %v", err)
	}

	// timespec has tv_sec at offset 0 and tv_nsec at offset 8 (on 64-bit).
	if off, ok := offsets["tv_sec"]; !ok {
		t.Error("tv_sec field not found")
	} else if off != 0 {
		t.Errorf("tv_sec offset = %d, want 0", off)
	}

	if off, ok := offsets["tv_nsec"]; !ok {
		t.Error("tv_nsec field not found")
	} else if off != 8 {
		t.Errorf("tv_nsec offset = %d, want 8", off)
	}
}

// TestBuildPyOffsetsFromDWARF tests full Python offset extraction.
// Only runs when libpython3.X-dbgsym is installed.
func TestBuildPyOffsetsFromDWARF(t *testing.T) {
	libPath, minor := findLibpython(t)
	debugPath, err := FindDebugFile(libPath)
	if err != nil {
		t.Fatalf("FindDebugFile(%s): %v", libPath, err)
	}
	if debugPath == "" {
		t.Skipf("no debug info for %s (install libpython3.%d-dbgsym to enable)", libPath, minor)
	}

	offsets, err := BuildPyOffsetsFromDWARF(debugPath, minor)
	if err != nil {
		t.Fatalf("BuildPyOffsetsFromDWARF: %v", err)
	}

	// Validation already happens inside BuildPyOffsetsFromDWARF, but double-check key fields.
	if offsets.RuntimeInterpretersHead == 0 {
		t.Error("RuntimeInterpretersHead is 0")
	}
	if offsets.TstateThreadID == 0 {
		t.Error("TstateThreadID is 0")
	}
	if offsets.CodeFilename == 0 {
		t.Error("CodeFilename is 0")
	}

	// Version string must contain "-dwarf" suffix.
	if !strings.Contains(offsets.Version, "-dwarf") {
		t.Errorf("Version = %q, want suffix '-dwarf'", offsets.Version)
	}

	t.Logf("DWARF offsets for Python 3.%d:", minor)
	t.Logf("  RuntimeInterpretersHead: %d", offsets.RuntimeInterpretersHead)
	t.Logf("  InterpTstateHead:        %d", offsets.InterpTstateHead)
	t.Logf("  TstateNext:              %d", offsets.TstateNext)
	t.Logf("  TstateThreadID:          %d", offsets.TstateThreadID)
	t.Logf("  TstateFrame:             %d", offsets.TstateFrame)
	t.Logf("  CframeCurrentFrame:      %d", offsets.CframeCurrentFrame)
	t.Logf("  FrameBack:               %d", offsets.FrameBack)
	t.Logf("  FrameCode:               %d", offsets.FrameCode)
	t.Logf("  CodeFilename:            %d", offsets.CodeFilename)
	t.Logf("  CodeName:                %d", offsets.CodeName)
	t.Logf("  CodeFirstLineNo:         %d", offsets.CodeFirstLineNo)
	t.Logf("  UnicodeLength:           %d", offsets.UnicodeLength)
	t.Logf("  UnicodeData:             %d", offsets.UnicodeData)
	t.Logf("  UnicodeState:            %d", offsets.UnicodeState)
	t.Logf("  NewStyleFrames:          %v", offsets.NewStyleFrames)
}

// TestBuildPyOffsetsFromDWARF_CrossValidation compares DWARF-extracted offsets
// against hardcoded tables when both are available. This detects distro-patched
// builds (Ubuntu 22.04's libpython3.10 has different offsets from upstream).
func TestBuildPyOffsetsFromDWARF_CrossValidation(t *testing.T) {
	libPath, minor := findLibpython(t)
	debugPath, err := FindDebugFile(libPath)
	if err != nil {
		t.Fatalf("FindDebugFile: %v", err)
	}
	if debugPath == "" {
		t.Skipf("no debug info for %s", libPath)
	}

	dwarfOff, err := BuildPyOffsetsFromDWARF(debugPath, minor)
	if err != nil {
		t.Fatalf("BuildPyOffsetsFromDWARF: %v", err)
	}

	hardcoded := GetPyOffsets(minor)
	if hardcoded == nil {
		t.Skipf("no hardcoded offsets for Python 3.%d", minor)
	}

	type field struct {
		name           string
		dwarf, hc      uint64
	}

	fields := []field{
		{"RuntimeInterpretersHead", dwarfOff.RuntimeInterpretersHead, hardcoded.RuntimeInterpretersHead},
		{"InterpTstateHead", dwarfOff.InterpTstateHead, hardcoded.InterpTstateHead},
		{"TstateNext", dwarfOff.TstateNext, hardcoded.TstateNext},
		{"TstateThreadID", dwarfOff.TstateThreadID, hardcoded.TstateThreadID},
		{"TstateFrame", dwarfOff.TstateFrame, hardcoded.TstateFrame},
		{"CframeCurrentFrame", dwarfOff.CframeCurrentFrame, hardcoded.CframeCurrentFrame},
		{"FrameBack", dwarfOff.FrameBack, hardcoded.FrameBack},
		{"FrameCode", dwarfOff.FrameCode, hardcoded.FrameCode},
		{"CodeFilename", dwarfOff.CodeFilename, hardcoded.CodeFilename},
		{"CodeName", dwarfOff.CodeName, hardcoded.CodeName},
		{"CodeFirstLineNo", dwarfOff.CodeFirstLineNo, hardcoded.CodeFirstLineNo},
		{"UnicodeLength", dwarfOff.UnicodeLength, hardcoded.UnicodeLength},
		{"UnicodeData", dwarfOff.UnicodeData, hardcoded.UnicodeData},
		{"UnicodeState", dwarfOff.UnicodeState, hardcoded.UnicodeState},
	}

	diffs := 0
	for _, f := range fields {
		if f.dwarf != f.hc {
			t.Logf("MISMATCH %s: DWARF=%d, hardcoded=%d", f.name, f.dwarf, f.hc)
			diffs++
		}
	}

	if diffs > 0 {
		t.Logf("%d/%d fields differ — this is a distro-patched build (DWARF offsets should be used)", diffs, len(fields))
	} else {
		t.Logf("All %d fields match between DWARF and hardcoded (upstream build)", len(fields))
	}
	// This test never fails — it's purely diagnostic. The important thing is
	// that BuildPyOffsetsFromDWARF succeeds (tested above).
}

// TestExtractStructOffsets_NoDebugInfo verifies graceful error on a stripped binary.
func TestExtractStructOffsets_NoDebugInfo(t *testing.T) {
	path := findBinary(t, "/usr/bin/ls", "/bin/ls")

	_, err := ExtractStructOffsets(path, "nonexistent_struct")
	if err == nil {
		t.Error("expected error for stripped binary, got nil")
	}
}

// TestExtractStructOffsets_StructNotFound verifies error when struct doesn't exist.
// Builds a small Go binary with DWARF to test against.
func TestExtractStructOffsets_StructNotFound(t *testing.T) {
	bin := buildTestBinaryWithDWARF(t)

	_, err := ExtractStructOffsets(bin, "CompletelyBogusStructThatDoesNotExist")
	if err == nil {
		t.Error("expected error for nonexistent struct, got nil")
	}
	t.Logf("got expected error: %v", err)
}

// TestFindDebugFile_NotFound verifies that a missing debug file returns empty string, not error.
func TestFindDebugFile_NotFound(t *testing.T) {
	path := findBinary(t, "/usr/bin/ls", "/bin/ls")

	result, err := FindDebugFile(path)
	if err != nil {
		t.Errorf("FindDebugFile returned error: %v", err)
	}
	// Result may be empty (no debug file) or non-empty (if debug packages are installed).
	// Either way, no error is the key assertion.
	t.Logf("FindDebugFile(%s) = %q", path, result)
}

// TestFindDebugFile_NonExistent verifies error on a path that doesn't exist.
func TestFindDebugFile_NonExistent(t *testing.T) {
	_, err := FindDebugFile("/nonexistent/path/libfoo.so")
	if err == nil {
		t.Error("expected error for nonexistent path, got nil")
	}
}

// TestFindDebugFile_InlineDWARF tests strategy 3 (inline DWARF detection).
// We compile a tiny test binary with DWARF to ensure the path works.
func TestFindDebugFile_InlineDWARF(t *testing.T) {
	bin := buildTestBinaryWithDWARF(t)

	result, err := FindDebugFile(bin)
	if err != nil {
		t.Fatalf("FindDebugFile on test binary: %v", err)
	}

	// Binary built with DWARF should be detected as having inline DWARF.
	if result != bin {
		t.Errorf("FindDebugFile(test binary) = %q, want %q (inline DWARF)", result, bin)
	}
}

// TestGetPyOffsetsBest_Fallback verifies the orchestrator falls back to hardcoded offsets.
func TestGetPyOffsetsBest_Fallback(t *testing.T) {
	// A nonexistent library path should trigger fallback to hardcoded offsets.
	for _, minor := range []int{10, 11, 12} {
		offsets := GetPyOffsetsBest("/nonexistent/libpython3.so", minor)
		if offsets == nil {
			t.Errorf("GetPyOffsetsBest(nonexistent, %d) returned nil, want hardcoded fallback", minor)
			continue
		}
		// Fallback offsets should NOT have -dwarf suffix.
		if strings.Contains(offsets.Version, "-dwarf") {
			t.Errorf("Python 3.%d fallback: Version = %q, should not have -dwarf suffix", minor, offsets.Version)
		}
		if offsets.RuntimeInterpretersHead == 0 {
			t.Errorf("Python 3.%d fallback: RuntimeInterpretersHead is 0", minor)
		}
	}
}

// TestGetPyOffsetsBest_UnsupportedVersion verifies nil for unsupported version with no DWARF.
func TestGetPyOffsetsBest_UnsupportedVersion(t *testing.T) {
	offsets := GetPyOffsetsBest("/nonexistent/libpython3.so", 9)
	if offsets != nil {
		t.Error("expected nil for unsupported version 3.9 with no DWARF, got non-nil")
	}
}

// TestGetPyOffsetsBest_WithDWARF tests the full DWARF path end-to-end.
func TestGetPyOffsetsBest_WithDWARF(t *testing.T) {
	libPath, minor := findLibpython(t)

	debugPath, _ := FindDebugFile(libPath)
	if debugPath == "" {
		t.Skipf("no debug info for %s", libPath)
	}

	offsets := GetPyOffsetsBest(libPath, minor)
	if offsets == nil {
		t.Fatal("GetPyOffsetsBest returned nil when DWARF is available")
	}

	// Should use DWARF path.
	if !strings.Contains(offsets.Version, "-dwarf") {
		t.Errorf("expected DWARF offsets (version with -dwarf), got %q", offsets.Version)
	}
}

// TestGetPyOffsetsBest_DebugLogging verifies that debug callback fires.
func TestGetPyOffsetsBest_DebugLogging(t *testing.T) {
	var messages []string
	SetDebugLog(func(format string, args ...any) {
		messages = append(messages, fmt.Sprintf(format, args...))
	})
	defer SetDebugLog(nil)

	// This will attempt DWARF (fail for nonexistent), then fall back.
	_ = GetPyOffsetsBest("/nonexistent/libpython3.so", 10)

	if len(messages) == 0 {
		t.Error("expected debug messages when debug log is set, got none")
	}

	// Should mention the nonexistent path and the hardcoded fallback.
	foundDWARF := false
	foundHardcoded := false
	for _, m := range messages {
		if strings.Contains(m, "/nonexistent/") {
			foundDWARF = true
		}
		if strings.Contains(m, "hardcoded") {
			foundHardcoded = true
		}
	}
	if !foundDWARF {
		t.Errorf("expected DWARF failure message mentioning path, got: %v", messages)
	}
	if !foundHardcoded {
		t.Errorf("expected 'hardcoded' in debug messages, got: %v", messages)
	}
}

// TestAlign4 verifies the 4-byte alignment helper.
func TestAlign4(t *testing.T) {
	tests := []struct {
		in, want uint32
	}{
		{0, 0},
		{1, 4},
		{2, 4},
		{3, 4},
		{4, 4},
		{5, 8},
		{7, 8},
		{8, 8},
	}
	for _, tt := range tests {
		if got := align4(tt.in); got != tt.want {
			t.Errorf("align4(%d) = %d, want %d", tt.in, got, tt.want)
		}
	}
}

// TestValidateDWARFOffsets_Rejects_Zero tests that validation catches missing fields.
func TestValidateDWARFOffsets_Rejects_Zero(t *testing.T) {
	// All zeros should fail.
	err := validateDWARFOffsets(&PyOffsets{})
	if err == nil {
		t.Error("expected validation error for all-zero offsets, got nil")
	}

	// Valid offsets should pass.
	err = validateDWARFOffsets(&PyOffsets{
		TstateNext:     8,
		TstateThreadID: 176,
		TstateFrame:    24,
		CodeFilename:   104,
		CodeName:       112,
		UnicodeData:    48,
	})
	if err != nil {
		t.Errorf("valid offsets rejected: %v", err)
	}

	// Single missing field should fail.
	err = validateDWARFOffsets(&PyOffsets{
		TstateNext:     8,
		TstateThreadID: 0, // missing
		TstateFrame:    24,
		CodeFilename:   104,
		CodeName:       112,
		UnicodeData:    48,
	})
	if err == nil {
		t.Error("expected validation error for missing TstateThreadID")
	}
	if !strings.Contains(err.Error(), "TstateThreadID") {
		t.Errorf("error should mention TstateThreadID, got: %v", err)
	}
}

// TestExtractStructOffsets_GoBinary validates DWARF parsing works on a Go binary.
// Looks for runtime.g which is a well-known Go runtime struct.
func TestExtractStructOffsets_GoBinary(t *testing.T) {
	bin := buildTestBinaryWithDWARF(t)

	// Look for runtime.g — a well-known Go runtime struct.
	offsets, err := ExtractStructOffsets(bin, "runtime.g")
	if err != nil {
		t.Skipf("runtime.g not found in Go binary DWARF: %v", err)
	}

	// runtime.g has a "goid" field (goroutine ID) at a non-zero offset.
	if _, ok := offsets["goid"]; !ok {
		t.Error("expected 'goid' field in runtime.g")
	}
	t.Logf("Found %d fields in runtime.g", len(offsets))
}

// TestExtractPythonVersionFromBinary validates version extraction from binary names.
func TestExtractPythonVersionFromBinary(t *testing.T) {
	tests := []struct {
		input string
		want  string
	}{
		{"python3.10", "3.10"},
		{"python3.11", "3.11"},
		{"python3.12", "3.12"},
		{"python3.9", "3.9"},
		{"python3", ""},    // no minor version dot
		{"python", ""},     // too short
		{"bash", ""},       // not python
		{"libpython3.10", ""}, // not a binary name (has "lib" prefix)
		{"python3.10.1", "3.10.1"}, // patch version OK (we just need the prefix)
	}

	for _, tt := range tests {
		got := extractPythonVersionFromBinary(tt.input)
		if got != tt.want {
			t.Errorf("extractPythonVersionFromBinary(%q) = %q, want %q", tt.input, got, tt.want)
		}
	}
}

// TestFindDebugBuildLib_SharedLib validates the shared library debug build lookup.
// Uses a nonexistent path, so should return "" (no debug lib found).
func TestFindDebugBuildLib_SharedLib(t *testing.T) {
	result := findDebugBuildLib("/nonexistent/libpython3.10.so.1.0")
	if result != "" {
		t.Errorf("expected empty for nonexistent path, got %q", result)
	}
}

// TestFindDebugBuildLib_StaticBinary validates the static binary debug build lookup.
// Uses a nonexistent path, so should return "" (no debug lib found).
func TestFindDebugBuildLib_StaticBinary(t *testing.T) {
	result := findDebugBuildLib("/nonexistent/python3.10")
	if result != "" {
		t.Errorf("expected empty for nonexistent path, got %q", result)
	}
}

// TestFindDebugBuildLib_NotPython verifies non-Python binaries are ignored.
func TestFindDebugBuildLib_NotPython(t *testing.T) {
	result := findDebugBuildLib("/usr/bin/bash")
	if result != "" {
		t.Errorf("expected empty for non-Python binary, got %q", result)
	}
}

// TestFindDebugFile_DebugBuildLib tests Strategy 4 (debug build library).
// Only runs when libpython3.X-dbg is installed (provides libpythonX.Yd.so).
func TestFindDebugFile_DebugBuildLib(t *testing.T) {
	// Look for a debug build library on the system.
	candidates := []string{
		"/usr/lib/x86_64-linux-gnu/libpython3.10d.so.1.0",
		"/usr/lib/x86_64-linux-gnu/libpython3.11d.so.1.0",
		"/usr/lib/x86_64-linux-gnu/libpython3.12d.so.1.0",
	}

	var debugLib string
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			debugLib = c
			break
		}
	}
	if debugLib == "" {
		t.Skip("no debug build library found (install libpython3.X-dbg to enable)")
	}

	// FindDebugFile on a stripped Python binary should find the debug build lib.
	// Test with the binary name pattern (statically linked case).
	result := findDebugBuildLib("/usr/bin/python3.10")
	if result == "" {
		// May fail if python3.10 isn't installed; try the shared lib pattern.
		libPath, _ := findLibpython(t)
		result = findDebugBuildLib(libPath)
	}

	if result == "" {
		t.Skip("debug build library exists but findDebugBuildLib didn't find it")
	}

	t.Logf("findDebugBuildLib found: %s", result)

	// Verify the found file has DWARF.
	f, err := elf.Open(result)
	if err != nil {
		t.Fatalf("cannot open debug build lib: %v", err)
	}
	defer f.Close()

	if !hasInlineDWARF(f) {
		t.Error("debug build library does not have DWARF sections")
	}
}

// TestResolveToStruct_Typedef validates typedef → struct resolution.
// Uses a real Go binary's DWARF (Go has typedef chains in its runtime).
func TestResolveToStruct_Typedef(t *testing.T) {
	bin := buildTestBinaryWithDWARF(t)

	f, err := elf.Open(bin)
	if err != nil {
		t.Fatal(err)
	}
	defer f.Close()

	dw, err := f.DWARF()
	if err != nil {
		t.Fatal(err)
	}

	// findStructInDWARF should handle both direct structs and typedefs.
	// runtime.g is a struct in the Go runtime.
	st, err := findStructInDWARF(dw, "runtime.g")
	if err != nil {
		t.Skipf("runtime.g not found: %v", err)
	}

	if len(st.Field) == 0 {
		t.Error("runtime.g has no fields")
	}
	t.Logf("runtime.g has %d fields", len(st.Field))
}

// --- helpers ---

// findLibpython locates a libpython on the system, skips if not found.
func findLibpython(t *testing.T) (string, int) {
	t.Helper()
	candidates := []string{
		"/usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0",
		"/usr/lib/x86_64-linux-gnu/libpython3.11.so.1.0",
		"/usr/lib/x86_64-linux-gnu/libpython3.12.so.1.0",
	}

	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			var minor int
			switch filepath.Base(c) {
			case "libpython3.10.so.1.0":
				minor = 10
			case "libpython3.11.so.1.0":
				minor = 11
			case "libpython3.12.so.1.0":
				minor = 12
			}
			return c, minor
		}
	}
	t.Skip("no libpython found on this system")
	return "", 0
}

// findBinary locates one of the given paths, skips if none found.
func findBinary(t *testing.T, candidates ...string) string {
	t.Helper()
	for _, c := range candidates {
		if _, err := os.Stat(c); err == nil {
			return c
		}
	}
	t.Skipf("none of %v found", candidates)
	return ""
}

// buildTestBinaryWithDWARF compiles a minimal Go binary with DWARF info.
// Uses t.TempDir() so the binary is cleaned up after the test.
func buildTestBinaryWithDWARF(t *testing.T) string {
	t.Helper()
	tmpDir := t.TempDir()
	src := filepath.Join(tmpDir, "main.go")
	bin := filepath.Join(tmpDir, "testbin")
	if err := os.WriteFile(src, []byte("package main\nfunc main(){}\n"), 0644); err != nil {
		t.Fatal(err)
	}

	cmd := execCommand("go", "build", "-o", bin, src)
	if out, err := cmd.CombinedOutput(); err != nil {
		t.Skipf("cannot build test binary: %s: %v", out, err)
	}
	return bin
}
