package events

import "encoding/binary"

// ParseStackIPs extracts stack frames from the stack section of a cuda_event_stack.
// The stack section starts at baseOffset (byte 56 for CUDA/driver events):
//
//	offset baseOffset+0: stack_depth (uint16 LE) — number of valid IPs
//	offset baseOffset+2: _stack_pad[3] (6 bytes)
//	offset baseOffset+8: stack_ips[MAX_STACK_DEPTH] (up to 64 * 8 = 512 bytes)
//
// Shared by CUDA runtime and driver tracers (identical struct layout).
func ParseStackIPs(raw []byte, baseOffset int) []StackFrame {
	if len(raw) < baseOffset+8 {
		return nil
	}

	depth := binary.LittleEndian.Uint16(raw[baseOffset : baseOffset+2])
	if depth == 0 || depth > 64 {
		return nil
	}

	ipsOffset := baseOffset + 8 // skip stack_depth (2) + pad (6)
	frames := make([]StackFrame, 0, depth)
	for i := uint16(0); i < depth; i++ {
		off := ipsOffset + int(i)*8
		if off+8 > len(raw) {
			break
		}
		ip := binary.LittleEndian.Uint64(raw[off : off+8])
		if ip == 0 {
			break // zero IP marks end of valid frames
		}
		frames = append(frames, StackFrame{IP: ip})
	}
	return frames
}
