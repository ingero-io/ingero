package symtab

import (
	"fmt"
	"sync"
)

// PyFrame represents a single Python frame in the call stack.
type PyFrame struct {
	Filename string // e.g., "train.py"
	Function string // e.g., "forward"
	Line     int    // e.g., 47
}

// String returns a human-readable representation: "train.py:47 in forward()"
func (f PyFrame) String() string {
	if f.Line > 0 {
		return fmt.Sprintf("%s:%d in %s()", f.Filename, f.Line, f.Function)
	}
	return fmt.Sprintf("%s in %s()", f.Filename, f.Function)
}

// maxPyFrameDepth limits how deep we walk the Python frame chain.
const maxPyFrameDepth = 64

// PyFrameWalker extracts Python stack frames from a target process.
// It caches the _PyRuntime address and thread state lookups.
//
// Thread-safe: can be called from multiple goroutines.
type PyFrameWalker struct {
	mu sync.Mutex

	// Per-PID cached state.
	cache map[uint32]*pyProcessState
}

type pyProcessState struct {
	info       *PythonInfo
	offsets    *PyOffsets
	runtimeAddr uint64 // address of _PyRuntime in process memory
	mem        *ProcMem
}

// NewPyFrameWalker creates a new Python frame walker.
func NewPyFrameWalker() *PyFrameWalker {
	return &PyFrameWalker{
		cache: make(map[uint32]*pyProcessState),
	}
}

// WalkPythonFrames extracts the Python call stack for a given thread.
//
// Algorithm:
//  1. Find _PyRuntime symbol address in libpython (ELF symtab + /proc maps)
//  2. Read PyInterpreterState head from _PyRuntime
//  3. Walk PyThreadState linked list, match by OS thread ID (TID)
//  4. Follow frame chain: f_back (3.10) or .previous (3.11+)
//  5. For each frame: read PyCodeObject → extract filename + function name + line
//
// Returns nil if Python is not detected or the thread has no Python frames.
func (w *PyFrameWalker) WalkPythonFrames(pid, tid uint32) ([]PyFrame, error) {
	state, err := w.getProcessState(pid)
	if err != nil || state == nil {
		return nil, err
	}

	// Find the PyThreadState for this OS thread.
	tstate, err := w.findThreadState(state, tid)
	if err != nil || tstate == 0 {
		return nil, err
	}

	// Get the current frame pointer.
	framePtr, err := w.getCurrentFrame(state, tstate)
	if err != nil || framePtr == 0 {
		return nil, err
	}

	// Walk the frame chain.
	return w.walkFrames(state, framePtr)
}

// getProcessState returns cached Python process info, initializing if needed.
func (w *PyFrameWalker) getProcessState(pid uint32) (*pyProcessState, error) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if s, ok := w.cache[pid]; ok {
		return s, nil
	}

	// Detect Python in the process.
	info := DetectPython(pid)
	if info == nil {
		// Not a Python process — cache the negative result.
		w.cache[pid] = nil
		return nil, nil
	}

	if !info.IsSupportedVersion() {
		w.cache[pid] = nil
		return nil, nil
	}

	offsets := GetPyOffsetsBest(info.LibPath, info.Minor)
	if offsets == nil {
		w.cache[pid] = nil
		return nil, nil
	}

	// Find _PyRuntime address.
	runtimeAddr, err := findPyRuntimeAddr(pid, info)
	if err != nil || runtimeAddr == 0 {
		w.cache[pid] = nil
		return nil, err
	}

	// Open /proc/[pid]/mem.
	mem, err := OpenProcMem(pid)
	if err != nil {
		w.cache[pid] = nil
		return nil, err
	}

	state := &pyProcessState{
		info:        info,
		offsets:     offsets,
		runtimeAddr: runtimeAddr,
		mem:         mem,
	}
	w.cache[pid] = state
	return state, nil
}

// findPyRuntimeAddr locates the _PyRuntime symbol in the process's memory.
func findPyRuntimeAddr(pid uint32, info *PythonInfo) (uint64, error) {
	// Parse /proc/[pid]/maps to find libpython's load address.
	regions, err := ParseProcMaps(pid)
	if err != nil {
		return 0, err
	}

	// Find the region containing the Python library.
	var pyRegion *MapRegion
	for i := range regions {
		if regions[i].Path == info.LibPath && regions[i].IsExecutable() {
			pyRegion = &regions[i]
			break
		}
	}
	if pyRegion == nil {
		return 0, fmt.Errorf("libpython region not found for PID %d", pid)
	}

	// Parse ELF symbols from libpython.
	elfs, err := ParseELFSymbols(info.LibPath)
	if err != nil {
		return 0, err
	}

	// Find _PyRuntime symbol.
	for _, sym := range elfs.Symbols {
		if sym.Name == "_PyRuntime" {
			// Calculate the runtime address in process memory.
			// For PIE/shared libs: addr = region.Start - region.Offset + sym.Value - baseVA
			addr := pyRegion.Start - pyRegion.Offset + sym.Value
			if elfs.PIE {
				addr -= elfs.BaseVA
			}
			return addr, nil
		}
	}

	// _PyRuntime might be in .data, not .text. Search all regions.
	allRegions, _ := parseMapsFile(fmt.Sprintf("/proc/%d/maps", pid))
	for i := range allRegions {
		if allRegions[i].Path == info.LibPath {
			for _, sym := range elfs.Symbols {
				if sym.Name == "_PyRuntime" {
					addr := allRegions[i].Start - allRegions[i].Offset + sym.Value
					if elfs.PIE {
						addr -= elfs.BaseVA
					}
					return addr, nil
				}
			}
		}
	}

	return 0, fmt.Errorf("_PyRuntime symbol not found in %s", info.LibPath)
}

// findThreadState walks the PyThreadState linked list to find the one matching tid.
func (w *PyFrameWalker) findThreadState(state *pyProcessState, tid uint32) (uint64, error) {
	offsets := state.offsets
	mem := state.mem

	// Read interpreters.head from _PyRuntime.
	interpPtr, err := mem.ReadPtr(state.runtimeAddr + offsets.RuntimeInterpretersHead)
	if err != nil {
		return 0, fmt.Errorf("reading interpreters.head: %w", err)
	}
	if interpPtr == 0 {
		return 0, nil
	}

	// Read tstate_head from PyInterpreterState.
	tstatePtr, err := mem.ReadPtr(interpPtr + offsets.InterpTstateHead)
	if err != nil {
		return 0, fmt.Errorf("reading tstate_head: %w", err)
	}

	// Walk the linked list.
	targetTID := uint64(tid)
	for i := 0; tstatePtr != 0 && i < 256; i++ {
		// Read this thread state's OS thread ID.
		threadID, err := mem.ReadUint64(tstatePtr + offsets.TstateThreadID)
		if err != nil {
			break
		}

		if threadID == targetTID {
			return tstatePtr, nil
		}

		// Next thread state.
		tstatePtr, err = mem.ReadPtr(tstatePtr + offsets.TstateNext)
		if err != nil {
			break
		}
	}

	return 0, nil // Thread not found (may not be running Python code)
}

// getCurrentFrame extracts the current frame pointer from a PyThreadState.
func (w *PyFrameWalker) getCurrentFrame(state *pyProcessState, tstate uint64) (uint64, error) {
	offsets := state.offsets
	mem := state.mem

	framePtr, err := mem.ReadPtr(tstate + offsets.TstateFrame)
	if err != nil {
		return 0, err
	}

	// For 3.11: tstate.cframe → _PyCFrame.current_frame
	if offsets.NewStyleFrames && offsets.CframeCurrentFrame > 0 && framePtr != 0 {
		framePtr, err = mem.ReadPtr(framePtr + offsets.CframeCurrentFrame)
		if err != nil {
			return 0, err
		}
	}

	return framePtr, nil
}

// walkFrames walks the Python frame chain and extracts frame information.
func (w *PyFrameWalker) walkFrames(state *pyProcessState, framePtr uint64) ([]PyFrame, error) {
	offsets := state.offsets
	mem := state.mem

	var frames []PyFrame

	for i := 0; framePtr != 0 && i < maxPyFrameDepth; i++ {
		// Read the code object pointer.
		codePtr, err := mem.ReadPtr(framePtr + offsets.FrameCode)
		if err != nil {
			break
		}

		if codePtr != 0 {
			frame, err := w.readCodeObject(state, codePtr)
			if err == nil {
				frames = append(frames, frame)
			}
		}

		// Move to parent frame.
		framePtr, err = mem.ReadPtr(framePtr + offsets.FrameBack)
		if err != nil {
			break
		}
	}

	return frames, nil
}

// readCodeObject extracts filename, function name, and line number from a PyCodeObject.
func (w *PyFrameWalker) readCodeObject(state *pyProcessState, codePtr uint64) (PyFrame, error) {
	offsets := state.offsets
	mem := state.mem

	// Read co_filename (PyUnicodeObject*).
	filenamePtr, err := mem.ReadPtr(codePtr + offsets.CodeFilename)
	if err != nil {
		return PyFrame{}, err
	}

	// Read co_name (PyUnicodeObject*).
	namePtr, err := mem.ReadPtr(codePtr + offsets.CodeName)
	if err != nil {
		return PyFrame{}, err
	}

	// Read co_firstlineno (int).
	firstLineNo, err := mem.ReadInt32(codePtr + offsets.CodeFirstLineNo)
	if err != nil {
		return PyFrame{}, err
	}

	// Read string values.
	filename, _ := mem.ReadPyUnicodeString(filenamePtr, offsets, 256)
	funcName, _ := mem.ReadPyUnicodeString(namePtr, offsets, 128)

	return PyFrame{
		Filename: filename,
		Function: funcName,
		Line:     int(firstLineNo),
	}, nil
}

// InvalidatePID removes cached state for a process (e.g., after exit).
func (w *PyFrameWalker) InvalidatePID(pid uint32) {
	w.mu.Lock()
	defer w.mu.Unlock()

	if state, ok := w.cache[pid]; ok && state != nil && state.mem != nil {
		state.mem.Close()
	}
	delete(w.cache, pid)
}

// Close releases all resources.
func (w *PyFrameWalker) Close() {
	w.mu.Lock()
	defer w.mu.Unlock()

	for _, state := range w.cache {
		if state != nil && state.mem != nil {
			state.mem.Close()
		}
	}
	w.cache = nil
}
