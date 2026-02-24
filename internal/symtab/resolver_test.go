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
