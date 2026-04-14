package symtab

import (
	"testing"

	"github.com/ingero-io/ingero/pkg/events"
)

func TestResolver_MergePythonFrames(t *testing.T) {
	r := NewResolver()

	// Create a stack with native frames, one of which is in libpython.
	evt := &events.Event{
		PID: 1234,
		TID: 1234,
		Stack: []events.StackFrame{
			{IP: 0x7f0001, SymbolName: "cudaMalloc", File: "/usr/lib/libcudart.so.12"},
			{IP: 0x7f0002, SymbolName: "torch::autograd::execute", File: "/usr/lib/libtorch_cuda.so"},
			{IP: 0x7f0003, SymbolName: "_PyEval_EvalFrameDefault", File: "/usr/lib/libpython3.10.so.1.0"},
			{IP: 0x7f0004, SymbolName: "_start", File: "/usr/bin/python3.10"},
		},
	}

	pyFrames := []PyFrame{
		{Filename: "train.py", Function: "forward", Line: 47},
		{Filename: "model.py", Function: "__call__", Line: 123},
	}

	r.mergePythonFrames(evt, pyFrames)

	// Expect Python frames inserted before the libpython frame.
	// Stack should be:
	//   0: cudaMalloc (native)
	//   1: torch::autograd::execute (native)
	//   2: train.py:47 in forward() (Python)
	//   3: model.py:123 in __call__() (Python)
	//   4: _PyEval_EvalFrameDefault (native, libpython)
	//   5: _start (native)
	if len(evt.Stack) != 6 {
		t.Fatalf("expected 6 frames, got %d", len(evt.Stack))
	}

	// Check Python frame positions.
	if evt.Stack[2].PyFile != "train.py" {
		t.Errorf("frame[2].PyFile = %q, want %q", evt.Stack[2].PyFile, "train.py")
	}
	if evt.Stack[2].PyFunc != "forward" {
		t.Errorf("frame[2].PyFunc = %q, want %q", evt.Stack[2].PyFunc, "forward")
	}
	if evt.Stack[2].PyLine != 47 {
		t.Errorf("frame[2].PyLine = %d, want %d", evt.Stack[2].PyLine, 47)
	}
	if evt.Stack[3].PyFile != "model.py" {
		t.Errorf("frame[3].PyFile = %q, want %q", evt.Stack[3].PyFile, "model.py")
	}

	// Check native frames preserved.
	if evt.Stack[0].SymbolName != "cudaMalloc" {
		t.Errorf("frame[0].SymbolName = %q, want %q", evt.Stack[0].SymbolName, "cudaMalloc")
	}
	if evt.Stack[4].SymbolName != "_PyEval_EvalFrameDefault" {
		t.Errorf("frame[4].SymbolName = %q, want %q", evt.Stack[4].SymbolName, "_PyEval_EvalFrameDefault")
	}
}

func TestResolver_MergePythonFrames_NoPython(t *testing.T) {
	r := NewResolver()

	// Stack with no libpython frame — Python frames should be prepended.
	evt := &events.Event{
		Stack: []events.StackFrame{
			{IP: 0x7f0001, SymbolName: "cudaMalloc", File: "/usr/lib/libcudart.so.12"},
		},
	}

	pyFrames := []PyFrame{
		{Filename: "train.py", Function: "main", Line: 10},
	}

	r.mergePythonFrames(evt, pyFrames)

	if len(evt.Stack) != 2 {
		t.Fatalf("expected 2 frames, got %d", len(evt.Stack))
	}
	if evt.Stack[0].PyFile != "train.py" {
		t.Errorf("frame[0].PyFile = %q, want %q", evt.Stack[0].PyFile, "train.py")
	}
	if evt.Stack[1].SymbolName != "cudaMalloc" {
		t.Errorf("frame[1].SymbolName = %q, want %q", evt.Stack[1].SymbolName, "cudaMalloc")
	}
}

// TestResolveStack_UsesBPFPythonFrames verifies that when Event.PythonFrames
// is already populated (by the kernel-side eBPF walker), the resolver
// merges those frames into Stack via mergeBPFPythonFrames rather than
// invoking the userspace PyFrameWalker (which would need /proc/pid/mem).
//
// This is the observable contract of the Step 4 change: BPF frames take
// precedence, and the resolver is a no-op on the userspace walker path
// when BPF frames are present.
func TestResolveStack_UsesBPFPythonFrames(t *testing.T) {
	r := NewResolver()

	// Construct an event with a trivial native stack and BPF-captured
	// Python frames. The native frame has no libpython marker, so the
	// merge path prepends Python frames (see mergePythonFrames).
	evt := &events.Event{
		PID: 1234,
		TID: 1234,
		Stack: []events.StackFrame{
			// IP set but no resolvable region — resolveFrame is a no-op
			// for this PID (no /proc/1234/maps on a CI host). That's
			// fine: we only care about the merge behavior here.
			{IP: 0xABCD},
		},
		PythonFrames: []events.PyFrame{
			{Filename: "test.py", Function: "main", Line: 42},
		},
	}

	r.ResolveStack(evt)

	// Expect the BPF frame to have been merged into the stack with
	// PyFile/PyFunc/PyLine populated — exactly the fields mergePythonFrames
	// would have produced from a symtab.PyFrame.
	foundPyFrame := false
	for _, sf := range evt.Stack {
		if sf.PyFile == "test.py" && sf.PyFunc == "main" && sf.PyLine == 42 {
			foundPyFrame = true
			break
		}
	}
	if !foundPyFrame {
		t.Errorf("expected BPF Python frame to be merged into stack; got %+v", evt.Stack)
	}
}

// TestResolveStack_BPFFramesShortCircuitUserspace is a smoke test that
// mergeBPFPythonFrames converts events.PyFrame to symtab.PyFrame and
// calls into mergePythonFrames. We don't exercise the userspace walker
// fallback here — that path requires a live /proc/pid/mem which isn't
// available in unit tests.
func TestResolveStack_BPFFramesShortCircuitUserspace(t *testing.T) {
	r := NewResolver()
	evt := &events.Event{
		PID: 9999,
		TID: 9999,
		Stack: []events.StackFrame{
			{IP: 0x1, SymbolName: "_PyEval_EvalFrameDefault", File: "/usr/lib/libpython3.12.so.1.0"},
		},
		PythonFrames: []events.PyFrame{
			{Filename: "train.py", Function: "step", Line: 17},
			{Filename: "model.py", Function: "forward", Line: 88},
		},
	}

	r.ResolveStack(evt)

	// Both Python frames should precede the libpython frame, in order.
	if len(evt.Stack) != 3 {
		t.Fatalf("expected 3 frames (2 py + 1 native libpython), got %d: %+v", len(evt.Stack), evt.Stack)
	}
	if evt.Stack[0].PyFile != "train.py" || evt.Stack[0].PyLine != 17 {
		t.Errorf("frame[0] = %+v, want PyFile=train.py PyLine=17", evt.Stack[0])
	}
	if evt.Stack[1].PyFile != "model.py" || evt.Stack[1].PyLine != 88 {
		t.Errorf("frame[1] = %+v, want PyFile=model.py PyLine=88", evt.Stack[1])
	}
	if evt.Stack[2].SymbolName != "_PyEval_EvalFrameDefault" {
		t.Errorf("frame[2] = %+v, want _PyEval_EvalFrameDefault preserved", evt.Stack[2])
	}
}

func TestIsLibPythonFrame(t *testing.T) {
	tests := []struct {
		path string
		want bool
	}{
		{"/usr/lib/x86_64-linux-gnu/libpython3.10.so.1.0", true},
		{"/usr/bin/python3.10", true},
		{"/usr/lib/libcudart.so.12", false},
		{"/usr/lib/libc.so.6", false},
		{"", false},
	}

	for _, tt := range tests {
		if got := isLibPythonFrame(tt.path); got != tt.want {
			t.Errorf("isLibPythonFrame(%q) = %v, want %v", tt.path, got, tt.want)
		}
	}
}
