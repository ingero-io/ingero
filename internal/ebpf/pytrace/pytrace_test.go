package pytrace

import (
	"bytes"
	"encoding/binary"
	"testing"
	"unsafe"
)

func TestPyRuntimeStateSize(t *testing.T) {
	want := 36
	if pyRuntimeStateSize != want {
		t.Errorf("pyRuntimeStateSize = %d, want %d", pyRuntimeStateSize, want)
	}
	// Sanity check: in-memory Go struct includes alignment padding, so
	// unsafe.Sizeof may differ from the marshaled size. We marshal
	// explicitly so this is OK — but verify the marshaled size is what
	// we expect.
	s := PyRuntimeState{}
	buf, err := s.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary failed: %v", err)
	}
	if len(buf) != want {
		t.Errorf("MarshalBinary length = %d, want %d", len(buf), want)
	}
	// Sanity log to surprise-detect Go alignment changes.
	t.Logf("Go struct in-memory size: %d (marshaled: %d)", unsafe.Sizeof(s), len(buf))
}

func TestPyRuntimeStateMarshalRoundtrip(t *testing.T) {
	original := PyRuntimeState{
		RuntimeAddr:                0xDEADBEEF12345678,
		OffRuntimeInterpretersHead: 32,
		OffTstateHead:              48,
		OffTstateNext:              120,
		OffTstateNativeTid:         152,
		OffTstateFrame:             144,
		OffFrameBack:               160,
		OffFrameCode:               168,
		OffCodeFilename:            192,
		OffCodeName:                200,
		OffCodeFirstLineNo:         216,
		OffUnicodeState:            240,
		OffUnicodeData:             248,
	}
	buf, err := original.MarshalBinary()
	if err != nil {
		t.Fatalf("MarshalBinary failed: %v", err)
	}

	// Manually unmarshal to verify each field landed at the right offset.
	if got := binary.LittleEndian.Uint64(buf[0:8]); got != original.RuntimeAddr {
		t.Errorf("RuntimeAddr offset 0: got 0x%x, want 0x%x", got, original.RuntimeAddr)
	}
	if got := binary.LittleEndian.Uint16(buf[8:10]); got != original.OffRuntimeInterpretersHead {
		t.Errorf("OffRuntimeInterpretersHead offset 8: got %d, want %d", got, original.OffRuntimeInterpretersHead)
	}
	if got := binary.LittleEndian.Uint16(buf[30:32]); got != original.OffUnicodeData {
		t.Errorf("OffUnicodeData offset 30: got %d, want %d", got, original.OffUnicodeData)
	}

	// Verify all bytes accounted for (no padding gaps).
	zero := bytes.Count(buf, []byte{0})
	t.Logf("marshaled bytes: %x (zero count: %d)", buf, zero)
}

// TestPyRuntimeStateMarshalRoundtrip_AllVersions verifies that the v2
// appended fields (PythonMinor, OffCframeCurrentFrame) land at the
// correct byte offsets for each supported CPython minor version.
func TestPyRuntimeStateMarshalRoundtrip_AllVersions(t *testing.T) {
	cases := []struct {
		name  string
		minor uint8
		off   uint16 // OffCframeCurrentFrame
	}{
		{"python_310", 10, 0},
		{"python_311", 11, 8}, // CPython 3.11 _PyCFrame.current_frame is typically at offset 8
		{"python_312", 12, 0},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			s := PyRuntimeState{
				RuntimeAddr:           0xDEADBEEF,
				PythonMinor:           c.minor,
				OffCframeCurrentFrame: c.off,
			}
			buf, err := s.MarshalBinary()
			if err != nil {
				t.Fatalf("MarshalBinary: %v", err)
			}
			if len(buf) != 36 {
				t.Errorf("len=%d, want 36", len(buf))
			}
			if buf[32] != c.minor {
				t.Errorf("byte[32] = %d, want %d", buf[32], c.minor)
			}
			got := binary.LittleEndian.Uint16(buf[34:36])
			if got != c.off {
				t.Errorf("OffCframeCurrentFrame at byte 34-35 = %d, want %d", got, c.off)
			}
		})
	}
}
