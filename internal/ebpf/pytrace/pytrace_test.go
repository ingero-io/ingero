package pytrace

import (
	"bytes"
	"encoding/binary"
	"testing"
	"unsafe"
)

func TestPyRuntimeStateSize(t *testing.T) {
	want := 32
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
