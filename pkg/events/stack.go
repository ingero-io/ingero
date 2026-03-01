package events

import (
	"encoding/binary"
	"hash/fnv"
)

// ParseStackIPs extracts stack frames from the stack section of a cuda_event_stack.
// The stack section starts at baseOffset (byte 64 for CUDA/driver events):
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

// HashStackIPs computes an FNV-64a hash of a stack trace's raw instruction
// pointers. Two stacks with the same IPs in the same order produce the same
// hash. Used as the primary key in the stack_traces interning table.
//
// Hashes raw uint64 bytes directly — no JSON serialization or hex formatting.
//
// WARNING: ASLR makes this unsuitable for cross-process deduplication.
// Each process maps shared libraries at different virtual addresses, so
// logically identical call stacks (same symbols, same code paths) produce
// different hashes. Use HashStackSymbols for sampling/dedup across PIDs.
func HashStackIPs(stack []StackFrame) uint64 {
	h := fnv.New64a()
	var buf [8]byte
	for _, f := range stack {
		binary.LittleEndian.PutUint64(buf[:], f.IP)
		h.Write(buf[:])
	}
	return h.Sum64()
}

// HashStackSymbols computes an FNV-64a hash of a stack trace's resolved
// symbol names, ignoring raw instruction pointers. This is ASLR-independent:
// the same logical call path across different processes (with different library
// base addresses) produces the same hash.
//
// Used for stack sampling deduplication in trace.go, where the goal is to limit
// stored events per *logical* call path, not per *physical* IP sequence.
//
// Hashes: SymbolName, File (full path from /proc/[pid]/maps), PyFile, PyFunc, PyLine.
// Raw IPs are excluded (they vary with ASLR). Each field is null-separated
// to prevent "ab"+"c" colliding with "a"+"bc" (same technique Git uses
// for tree entry hashing).
//
// Note: File is the full path, not basename. This is correct for same-host
// multi-process (ASLR) dedup. For v0.7 container support, consider using
// filepath.Base(File) to handle container vs host path differences.
func HashStackSymbols(stack []StackFrame) uint64 {
	h := fnv.New64a()
	var buf [4]byte
	sep := []byte{0}
	for _, f := range stack {
		h.Write([]byte(f.SymbolName))
		h.Write(sep)
		h.Write([]byte(f.File))
		h.Write(sep)
		h.Write([]byte(f.PyFile))
		h.Write(sep)
		h.Write([]byte(f.PyFunc))
		h.Write(sep)
		binary.LittleEndian.PutUint32(buf[:], uint32(f.PyLine))
		h.Write(buf[:])
	}
	return h.Sum64()
}
