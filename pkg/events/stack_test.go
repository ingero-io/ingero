package events

import (
	"encoding/binary"
	"testing"
)

func TestParseStackIPs_Valid(t *testing.T) {
	// Build a buffer: 64-byte base (v0.7) + stack section with 3 IPs.
	base := 64
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
	base := 64
	buf := make([]byte, base+8+8)
	binary.LittleEndian.PutUint16(buf[base:], 0) // depth = 0

	if frames := ParseStackIPs(buf, base); frames != nil {
		t.Errorf("depth=0 should return nil, got %d frames", len(frames))
	}
}

func TestParseStackIPs_DepthTooLarge(t *testing.T) {
	base := 64
	buf := make([]byte, base+8+8)
	binary.LittleEndian.PutUint16(buf[base:], 65) // depth > MAX_STACK_DEPTH (64)

	if frames := ParseStackIPs(buf, base); frames != nil {
		t.Errorf("depth=65 should return nil, got %d frames", len(frames))
	}
}

func TestParseStackIPs_BufferTooShort(t *testing.T) {
	// Buffer shorter than baseOffset + 8 (stack header).
	buf := make([]byte, 68) // base=64, need 72 minimum

	if frames := ParseStackIPs(buf, 64); frames != nil {
		t.Errorf("short buffer should return nil, got %d frames", len(frames))
	}
}

func TestParseStackIPs_Truncated(t *testing.T) {
	// Depth claims 4 IPs but only 2 fit in the buffer.
	base := 64
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
	base := 64
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

func TestParseStackIPs_MaxDepth(t *testing.T) {
	// Boundary test: depth == 64 (MAX_STACK_DEPTH) should be accepted.
	base := 64
	buf := make([]byte, base+8+64*8) // header + 64 IPs
	binary.LittleEndian.PutUint16(buf[base:], 64)
	for i := 0; i < 64; i++ {
		binary.LittleEndian.PutUint64(buf[base+8+i*8:], uint64(0x1000+i))
	}

	frames := ParseStackIPs(buf, base)
	if len(frames) != 64 {
		t.Fatalf("depth=64 should yield 64 frames, got %d", len(frames))
	}
	if frames[0].IP != 0x1000 || frames[63].IP != 0x103F {
		t.Errorf("boundary IPs wrong: first=%#x last=%#x", frames[0].IP, frames[63].IP)
	}
}

// TestHashStackSymbols_ASLRIndependent verifies that HashStackSymbols produces
// the same hash for logically identical stacks across PIDs with different ASLR
// bases, while HashStackIPs produces different hashes.
//
// Teaching note: ASLR (Address Space Layout Randomization) maps shared libraries
// at random virtual addresses per process. Two processes calling cudaMalloc from
// the same source line get different instruction pointers (IPs) but identical
// symbol names. HashStackIPs (raw IPs) → different hashes. HashStackSymbols
// (resolved names) → same hash. This is why stack sampling must use symbol-based
// hashing to correctly deduplicate across processes.
func TestHashStackSymbols_ASLRIndependent(t *testing.T) {
	// Same logical stack: cudaMalloc → _PyEval_EvalFrameDefault → forward
	// Process A: libraries at 0x7f1000...
	stackA := []StackFrame{
		{IP: 0x7f1234, SymbolName: "cudaMalloc", File: "libcudart.so.12"},
		{IP: 0x7f5678, SymbolName: "_PyEval_EvalFrameDefault", File: "libpython3.12.so"},
		{IP: 0x7f9abc, PyFile: "train.py", PyFunc: "forward", PyLine: 142},
	}
	// Process B: same libraries at 0x7e9000... (different ASLR base)
	stackB := []StackFrame{
		{IP: 0x7e9234, SymbolName: "cudaMalloc", File: "libcudart.so.12"},
		{IP: 0x7ed678, SymbolName: "_PyEval_EvalFrameDefault", File: "libpython3.12.so"},
		{IP: 0x7f1abc, PyFile: "train.py", PyFunc: "forward", PyLine: 142},
	}

	// HashStackIPs: different (raw IPs differ due to ASLR)
	ipHashA := HashStackIPs(stackA)
	ipHashB := HashStackIPs(stackB)
	if ipHashA == ipHashB {
		t.Error("HashStackIPs should differ for different ASLR bases")
	}

	// HashStackSymbols: same (symbols are identical)
	symHashA := HashStackSymbols(stackA)
	symHashB := HashStackSymbols(stackB)
	if symHashA != symHashB {
		t.Errorf("HashStackSymbols should match across ASLR bases: %d != %d", symHashA, symHashB)
	}
}

func TestHashStackSymbols_DifferentSymbols(t *testing.T) {
	stackA := []StackFrame{
		{IP: 0x1234, SymbolName: "cudaMalloc", File: "libcudart.so.12"},
	}
	stackB := []StackFrame{
		{IP: 0x1234, SymbolName: "cudaFree", File: "libcudart.so.12"},
	}

	if HashStackSymbols(stackA) == HashStackSymbols(stackB) {
		t.Error("HashStackSymbols should differ for different symbol names")
	}
}

func TestHashStackSymbols_EmptyStack(t *testing.T) {
	h := HashStackSymbols(nil)
	if h == 0 {
		t.Error("HashStackSymbols(nil) should return the FNV offset basis, not 0")
	}
}

func TestHashStackSymbols_UnresolvedFrames(t *testing.T) {
	// Stacks with no resolved symbols hash based on empty strings + depth.
	// This is correct: we can't distinguish unresolved frames at the same depth.
	stack1 := []StackFrame{{IP: 0x1111}, {IP: 0x2222}}
	stack2 := []StackFrame{{IP: 0x3333}, {IP: 0x4444}}

	// Same depth, all unresolved → same symbol hash (correct for sampling)
	if HashStackSymbols(stack1) != HashStackSymbols(stack2) {
		t.Error("unresolved stacks of same depth should have same symbol hash")
	}

	// Different depth → different hash
	stack3 := []StackFrame{{IP: 0x5555}}
	if HashStackSymbols(stack1) == HashStackSymbols(stack3) {
		t.Error("stacks of different depth should have different symbol hashes")
	}
}

func TestHashStackSymbols_MixedResolvedUnresolved(t *testing.T) {
	// Real-world pattern: some frames resolved (CUDA, Python), some not (stripped libs).
	// Two stacks with same resolved frames but different unresolved IPs → same hash.
	stackA := []StackFrame{
		{IP: 0x7f1234, SymbolName: "cudaMalloc", File: "libcudart.so.12"},
		{IP: 0x7f5000, File: "libcudnn_ops.so.9"},         // stripped: file but no symbol
		{IP: 0x7f9000},                                      // fully unresolved
		{IP: 0x400100, PyFile: "train.py", PyFunc: "forward", PyLine: 42},
	}
	stackB := []StackFrame{
		{IP: 0x7e1234, SymbolName: "cudaMalloc", File: "libcudart.so.12"},
		{IP: 0x7e5000, File: "libcudnn_ops.so.9"},         // same file, different IP
		{IP: 0x7e9000},                                      // different IP, still unresolved
		{IP: 0x500100, PyFile: "train.py", PyFunc: "forward", PyLine: 42},
	}

	if HashStackSymbols(stackA) != HashStackSymbols(stackB) {
		t.Error("mixed resolved/unresolved stacks with same symbols should have same hash")
	}
}

func TestHashStackSymbols_SymbolOffset(t *testing.T) {
	// The resolver produces names like "cudaMalloc+0x1a". Different offsets within
	// the same function are different call sites → should produce different hashes.
	stackA := []StackFrame{
		{IP: 0x1234, SymbolName: "cudaMalloc+0x1a", File: "libcudart.so.12"},
	}
	stackB := []StackFrame{
		{IP: 0x1250, SymbolName: "cudaMalloc+0x2f", File: "libcudart.so.12"},
	}

	if HashStackSymbols(stackA) == HashStackSymbols(stackB) {
		t.Error("different symbol offsets should produce different hashes")
	}
}

func TestHashStackSymbols_SameSymbolDifferentFiles(t *testing.T) {
	// Same symbol name in different libraries → different logical stacks.
	stackA := []StackFrame{
		{IP: 0x1234, SymbolName: "malloc", File: "libc.so.6"},
	}
	stackB := []StackFrame{
		{IP: 0x1234, SymbolName: "malloc", File: "libcudart.so.12"},
	}

	if HashStackSymbols(stackA) == HashStackSymbols(stackB) {
		t.Error("same symbol in different files should produce different hashes")
	}
}

func TestHashStackSymbols_FileOnlyFrame(t *testing.T) {
	// Stripped library: File populated but SymbolName empty.
	// Must differ from a completely unresolved frame.
	withFile := []StackFrame{
		{IP: 0x1234, File: "libcudnn_ops.so.9"},
	}
	noFile := []StackFrame{
		{IP: 0x1234},
	}

	if HashStackSymbols(withFile) == HashStackSymbols(noFile) {
		t.Error("file-only frame should differ from fully unresolved frame")
	}
}

func TestHashStackSymbols_PythonOnly(t *testing.T) {
	// Pure Python stack (CPython frames only, no native symbols).
	stackA := []StackFrame{
		{PyFile: "train.py", PyFunc: "forward", PyLine: 42},
		{PyFile: "model.py", PyFunc: "__call__", PyLine: 100},
	}
	stackB := []StackFrame{
		{PyFile: "train.py", PyFunc: "forward", PyLine: 42},
		{PyFile: "model.py", PyFunc: "__call__", PyLine: 101}, // different line
	}

	if HashStackSymbols(stackA) == HashStackSymbols(stackB) {
		t.Error("Python stacks with different line numbers should differ")
	}
}
