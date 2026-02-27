package events

import (
	"encoding/binary"
	"testing"
)

func TestParseStackIPs_Valid(t *testing.T) {
	// Build a buffer: 56-byte base + stack section with 3 IPs.
	base := 56
	buf := make([]byte, base+8+3*8) // header(8) + 3 IPs

	binary.LittleEndian.PutUint16(buf[base:], 3)   // depth = 3
	binary.LittleEndian.PutUint64(buf[base+8:], 0xAAAA)
	binary.LittleEndian.PutUint64(buf[base+16:], 0xBBBB)
	binary.LittleEndian.PutUint64(buf[base+24:], 0xCCCC)

	frames := ParseStackIPs(buf, base)
	if len(frames) != 3 {
		t.Fatalf("got %d frames, want 3", len(frames))
	}
	if frames[0].IP != 0xAAAA || frames[1].IP != 0xBBBB || frames[2].IP != 0xCCCC {
		t.Errorf("unexpected IPs: %+v", frames)
	}
}

func TestParseStackIPs_DepthZero(t *testing.T) {
	base := 56
	buf := make([]byte, base+8+8)
	binary.LittleEndian.PutUint16(buf[base:], 0) // depth = 0

	if frames := ParseStackIPs(buf, base); frames != nil {
		t.Errorf("depth=0 should return nil, got %d frames", len(frames))
	}
}

func TestParseStackIPs_DepthTooLarge(t *testing.T) {
	base := 56
	buf := make([]byte, base+8+8)
	binary.LittleEndian.PutUint16(buf[base:], 65) // depth > MAX_STACK_DEPTH (64)

	if frames := ParseStackIPs(buf, base); frames != nil {
		t.Errorf("depth=65 should return nil, got %d frames", len(frames))
	}
}

func TestParseStackIPs_BufferTooShort(t *testing.T) {
	// Buffer shorter than baseOffset + 8 (stack header).
	buf := make([]byte, 60) // base=56, need 64 minimum

	if frames := ParseStackIPs(buf, 56); frames != nil {
		t.Errorf("short buffer should return nil, got %d frames", len(frames))
	}
}

func TestParseStackIPs_Truncated(t *testing.T) {
	// Depth claims 4 IPs but only 2 fit in the buffer.
	base := 56
	buf := make([]byte, base+8+2*8) // header + 2 IPs only
	binary.LittleEndian.PutUint16(buf[base:], 4)   // claims 4
	binary.LittleEndian.PutUint64(buf[base+8:], 0x1111)
	binary.LittleEndian.PutUint64(buf[base+16:], 0x2222)

	frames := ParseStackIPs(buf, base)
	if len(frames) != 2 {
		t.Fatalf("truncated buffer should yield 2 frames, got %d", len(frames))
	}
}

func TestParseStackIPs_ZeroIPTerminates(t *testing.T) {
	// Depth claims 3 but second IP is zero — stops early.
	base := 56
	buf := make([]byte, base+8+3*8)
	binary.LittleEndian.PutUint16(buf[base:], 3)
	binary.LittleEndian.PutUint64(buf[base+8:], 0xDEAD)
	binary.LittleEndian.PutUint64(buf[base+16:], 0) // zero = end
	binary.LittleEndian.PutUint64(buf[base+24:], 0xBEEF)

	frames := ParseStackIPs(buf, base)
	if len(frames) != 1 {
		t.Fatalf("zero IP should terminate, got %d frames", len(frames))
	}
	if frames[0].IP != 0xDEAD {
		t.Errorf("got IP %#x, want 0xDEAD", frames[0].IP)
	}
}
