package symtab

import (
	"encoding/binary"
	"strings"
	"testing"
)

// buildDebugOffsetsBuf produces a byte buffer that matches the
// _Py_DebugOffsets layout for minor version minor, with the specified
// field values written at the positions debugOffsetsLayouts[minor]
// describes. Header is populated with the correct cookie and a synthetic
// version number encoding 3.<minor>.0. Fields not set in values default
// to zero in the buffer, which forces parseDebugOffsets's sanity checks
// to decide on a per-field basis whether zero is acceptable.
func buildDebugOffsetsBuf(t *testing.T, minor int, values map[uint64]uint64) []byte {
	t.Helper()
	layout, ok := debugOffsetsLayouts[minor]
	if !ok {
		t.Fatalf("no layout registered for minor %d", minor)
	}

	// 1024 matches the production read size in readDebugOffsets.
	buf := make([]byte, 1024)
	copy(buf[0:8], debugOffsetsCookie)

	// PY_VERSION_HEX format: (major << 24) | (minor << 16) | (micro << 8) | ...
	version := uint64(3)<<24 | uint64(minor)<<16
	binary.LittleEndian.PutUint64(buf[8:16], version)

	// Write every known layout field; zeros where not set in values.
	for off, v := range map[uint64]uint64{
		layout.runtimeInterpretersHead: values[layout.runtimeInterpretersHead],
		layout.interpThreadsHead:       values[layout.interpThreadsHead],
		layout.tstateCurrentFrame:      values[layout.tstateCurrentFrame],
		layout.tstateThreadID:          values[layout.tstateThreadID],
		layout.tstateNativeThreadID:    values[layout.tstateNativeThreadID],
		layout.framePrevious:           values[layout.framePrevious],
		layout.frameExecutable:         values[layout.frameExecutable],
		layout.codeFilename:            values[layout.codeFilename],
		layout.codeName:                values[layout.codeName],
		layout.codeFirstLineNo:         values[layout.codeFirstLineNo],
		layout.unicodeState:            values[layout.unicodeState],
		layout.unicodeLength:           values[layout.unicodeLength],
		layout.unicodeASCII:            values[layout.unicodeASCII],
	} {
		binary.LittleEndian.PutUint64(buf[off:off+8], v)
	}
	return buf
}

func TestParseDebugOffsets_Valid313(t *testing.T) {
	layout := debugOffsetsLayouts[13]
	// Plausible 3.13 values from the live probe we ran on cpython-3.13.13:
	// tstate.current_frame=72, thread_id=152, native_thread_id=160,
	// PyInterpreterState.threads.head=7344, _PyRuntime.interpreters.head=632.
	values := map[uint64]uint64{
		layout.runtimeInterpretersHead: 632,
		layout.interpThreadsHead:       7344,
		layout.tstateCurrentFrame:      72,
		layout.tstateThreadID:          152,
		layout.tstateNativeThreadID:    160,
		layout.framePrevious:           8,
		layout.frameExecutable:         0,
		layout.codeFilename:            112,
		layout.codeName:                120,
		layout.codeFirstLineNo:         68,
		layout.unicodeState:            32,
		layout.unicodeLength:           16,
		layout.unicodeASCII:            40,
	}
	buf := buildDebugOffsetsBuf(t, 13, values)

	got, err := parseDebugOffsets(buf, 13)
	if err != nil {
		t.Fatalf("parseDebugOffsets: %v", err)
	}
	if got == nil {
		t.Fatal("expected non-nil PyOffsets")
	}
	if got.Version != "3.13-debugoffsets" {
		t.Errorf("Version = %q, want %q", got.Version, "3.13-debugoffsets")
	}
	checks := []struct {
		name string
		got  uint64
		want uint64
	}{
		{"RuntimeInterpretersHead", got.RuntimeInterpretersHead, 632},
		{"InterpTstateHead", got.InterpTstateHead, 7344},
		{"TstateFrame", got.TstateFrame, 72},
		{"TstateThreadID", got.TstateThreadID, 152},
		{"TstateNativeThreadID", got.TstateNativeThreadID, 160},
		{"FrameBack", got.FrameBack, 8},
		{"FrameCode", got.FrameCode, 0},
		{"CodeFilename", got.CodeFilename, 112},
		{"CodeName", got.CodeName, 120},
		{"CodeFirstLineNo", got.CodeFirstLineNo, 68},
		{"UnicodeState", got.UnicodeState, 32},
		{"UnicodeLength", got.UnicodeLength, 16},
		{"UnicodeData", got.UnicodeData, 40},
		{"TstateNext", got.TstateNext, 8}, // hardcoded in parser
	}
	for _, c := range checks {
		if c.got != c.want {
			t.Errorf("%s = %d, want %d", c.name, c.got, c.want)
		}
	}
	if !got.NewStyleFrames {
		t.Error("NewStyleFrames should be true for 3.13")
	}
	if got.CframeCurrentFrame != 0 {
		t.Errorf("CframeCurrentFrame = %d, want 0 (3.13+ dropped cframe)", got.CframeCurrentFrame)
	}
}

func TestParseDebugOffsets_Valid314(t *testing.T) {
	layout := debugOffsetsLayouts[14]
	// 3.14 field *positions* inside _Py_DebugOffsets differ from 3.13
	// (threads_main + code_object_generation + tlbc_generation in
	// interpreter_state; stackpointer + tlbc_index in interpreter_frame;
	// co_tlbc in code_object; new set_object group before dict_object).
	// The *values* we emit here are the same shape as 3.13's.
	values := map[uint64]uint64{
		layout.runtimeInterpretersHead: 808,
		layout.interpThreadsHead:       7344,
		layout.tstateCurrentFrame:      72,
		layout.tstateThreadID:          152,
		layout.tstateNativeThreadID:    160,
		layout.framePrevious:           8,
		layout.frameExecutable:         0,
		layout.codeFilename:            112,
		layout.codeName:                120,
		layout.codeFirstLineNo:         68,
		layout.unicodeState:            32,
		layout.unicodeLength:           16,
		layout.unicodeASCII:            40,
	}
	buf := buildDebugOffsetsBuf(t, 14, values)

	got, err := parseDebugOffsets(buf, 14)
	if err != nil {
		t.Fatalf("parseDebugOffsets: %v", err)
	}
	if got == nil {
		t.Fatal("expected non-nil PyOffsets")
	}
	if got.RuntimeInterpretersHead != 808 {
		t.Errorf("RuntimeInterpretersHead = %d, want 808", got.RuntimeInterpretersHead)
	}
	if got.TstateFrame != 72 {
		t.Errorf("TstateFrame = %d, want 72", got.TstateFrame)
	}
	if got.TstateThreadID != 152 {
		t.Errorf("TstateThreadID = %d, want 152", got.TstateThreadID)
	}
	if got.FrameCode != 0 {
		t.Errorf("FrameCode = %d, want 0", got.FrameCode)
	}
}

func TestParseDebugOffsets_CookieMismatch(t *testing.T) {
	buf := make([]byte, 1024)
	copy(buf[0:8], "NOTVALID")
	// Write a plausible version so that would-have-been-valid reads
	// don't accidentally succeed past the cookie check.
	binary.LittleEndian.PutUint64(buf[8:16], uint64(3)<<24|uint64(13)<<16)

	got, err := parseDebugOffsets(buf, 13)
	if err == nil {
		t.Fatalf("expected error for bad cookie, got %+v", got)
	}
	if !strings.Contains(err.Error(), "cookie mismatch") {
		t.Errorf("error should mention cookie mismatch, got %v", err)
	}
	if got != nil {
		t.Errorf("expected nil offsets on cookie mismatch, got %+v", got)
	}
}

func TestParseDebugOffsets_VersionMismatch(t *testing.T) {
	// Build a buffer claiming to be 3.12 and ask parseDebugOffsets to
	// treat it as 3.13. The cookie check would never succeed on a real
	// 3.12 process (struct doesn't exist), but this simulates reading
	// the wrong address from a genuine 3.14 process and accidentally
	// landing on a stale 3.13-shaped struct.
	buf := make([]byte, 1024)
	copy(buf[0:8], debugOffsetsCookie)
	binary.LittleEndian.PutUint64(buf[8:16], uint64(3)<<24|uint64(12)<<16)

	_, err := parseDebugOffsets(buf, 13)
	if err == nil {
		t.Fatal("expected error for version mismatch")
	}
	if !strings.Contains(err.Error(), "version mismatch") {
		t.Errorf("error should mention version mismatch, got %v", err)
	}
}

func TestParseDebugOffsets_UnsupportedMinor(t *testing.T) {
	// Minor versions we don't have a layout for should produce
	// (nil, nil) — "fall through to DWARF / hardcoded" signal, not an
	// error. 15 is plausibly a future version.
	buf := make([]byte, 1024)
	copy(buf[0:8], debugOffsetsCookie)

	got, err := parseDebugOffsets(buf, 15)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if got != nil {
		t.Errorf("expected nil for unsupported minor, got %+v", got)
	}
}

func TestParseDebugOffsets_SanityCheckRejectsGarbage(t *testing.T) {
	// A buffer whose cookie and version match but whose offset fields
	// look like random memory (very large values) should be rejected so
	// the walker doesn't push garbage into the BPF map.
	layout := debugOffsetsLayouts[13]
	values := map[uint64]uint64{
		layout.interpThreadsHead: 99999, // way larger than the 32 KiB tolerance
		layout.tstateCurrentFrame:      72,
		layout.frameExecutable:         0,
		layout.codeFilename:            112,
		layout.unicodeASCII:            40,
	}
	buf := buildDebugOffsetsBuf(t, 13, values)

	_, err := parseDebugOffsets(buf, 13)
	if err == nil {
		t.Fatal("expected sanity check to reject garbage")
	}
	if !strings.Contains(err.Error(), "sanity check") {
		t.Errorf("error should mention sanity check, got %v", err)
	}
}

func TestReadDebugOffsetsFromPID_Pre313ReturnsNil(t *testing.T) {
	// Pre-3.13 is not the caller's problem: the struct doesn't exist in
	// 3.12, so we return (nil, nil) before ever touching /proc.
	got, err := ReadDebugOffsetsFromPID(1, 0x12345, 12)
	if err != nil {
		t.Errorf("expected no error for minor 12, got %v", err)
	}
	if got != nil {
		t.Errorf("expected nil PyOffsets for minor 12, got %+v", got)
	}
}

func TestReadDebugOffsetsFromPID_ZeroRuntimeAddr(t *testing.T) {
	// If the caller couldn't resolve _PyRuntime we shouldn't try to
	// read from address 0.
	got, err := ReadDebugOffsetsFromPID(1, 0, 13)
	if err != nil {
		t.Errorf("expected no error for zero runtimeAddr, got %v", err)
	}
	if got != nil {
		t.Errorf("expected nil PyOffsets for zero runtimeAddr, got %+v", got)
	}
}
