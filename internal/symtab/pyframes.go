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
// Uses RWMutex so concurrent cache hits (reads) don't serialize.
type PyFrameWalker struct {
	mu sync.RWMutex

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
	if err != nil {
		symDebugf("findThreadState failed for PID %d TID %d: %v", pid, tid, err)
		return nil, err
	}
	if tstate == 0 {
		return nil, nil // Thread not running Python code right now
	}

	// Get the current frame pointer.
	framePtr, err := w.getCurrentFrame(state, tstate)
	if err != nil {
		symDebugf("getCurrentFrame failed for PID %d TID %d: %v", pid, tid, err)
		return nil, err
	}
	if framePtr == 0 {
		return nil, nil
	}

	// Walk the frame chain.
	frames, err := w.walkFrames(state, framePtr)
	if err != nil {
		symDebugf("walkFrames failed for PID %d: %v", pid, err)
	}
	return frames, err
}

// getProcessState returns cached Python process info, initializing if needed.
// Uses RLock for cache hits (fast path), upgrading to Lock only on cache miss.
func (w *PyFrameWalker) getProcessState(pid uint32) (*pyProcessState, error) {
	// Fast path: RLock for cache hit.
	w.mu.RLock()
	if s, ok := w.cache[pid]; ok {
		w.mu.RUnlock()
		return s, nil
	}
	w.mu.RUnlock()

	// Slow path: Lock for cache miss + initialization.
	w.mu.Lock()
	defer w.mu.Unlock()

	// Double-check: another goroutine may have initialized while we upgraded.
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

	symDebugf("detected Python %s in PID %d (lib: %s)", info.Version, pid, info.LibPath)

	if !info.IsSupportedVersion() {
		symDebugf("Python %s not supported for frame walking (need 3.10-3.12)", info.Version)
		w.cache[pid] = nil
		return nil, nil
	}

	offsets := GetPyOffsetsBest(info.LibPath, info.Minor)
	if offsets == nil {
		symDebugf("no offsets available for Python %s", info.Version)
		w.cache[pid] = nil
		return nil, nil
	}

	// Find _PyRuntime address.
	runtimeAddr, err := findPyRuntimeAddr(pid, info)
	if err != nil || runtimeAddr == 0 {
		symDebugf("_PyRuntime resolution failed for PID %d: %v", pid, err)
		w.cache[pid] = nil
		return nil, err
	}
	symDebugf("_PyRuntime at 0x%x in PID %d", runtimeAddr, pid)

	// Open /proc/[pid]/mem.
	mem, err := OpenProcMem(pid)
	if err != nil {
		symDebugf("cannot open /proc/%d/mem: %v", pid, err)
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
//
// _PyRuntime is a global data object (STT_OBJECT in ELF), not a function.
// ParseELFSymbols() only keeps STT_FUNC symbols (for stack resolution), so
// it can never find _PyRuntime. We use FindSymbolByName() which searches
// all symbol types.
func findPyRuntimeAddr(pid uint32, info *PythonInfo) (uint64, error) {
	// Look up _PyRuntime by name — searches STT_FUNC, STT_OBJECT, STT_NOTYPE.
	symValue, pie, baseVA, found := FindSymbolByName(info.LibPath, "_PyRuntime")
	if !found {
		return 0, fmt.Errorf("_PyRuntime symbol not found in %s", info.LibPath)
	}

	// Parse /proc/[pid]/maps to find the library's load address.
	// _PyRuntime lives in .data (read-write), so we search all regions
	// for this path, not just executable ones.
	regions, err := parseMapsFile(fmt.Sprintf("/proc/%d/maps", pid))
	if err != nil {
		return 0, err
	}

	// Find the first region for this library path.
	for i := range regions {
		if regions[i].Path == info.LibPath {
			// Calculate the runtime address in process memory.
			// For PIE/shared libs: addr = region.Start - region.Offset + sym.Value - baseVA
			addr := regions[i].Start - regions[i].Offset + symValue
			if pie {
				addr -= baseVA
			}
			return addr, nil
		}
	}

	return 0, fmt.Errorf("library region not found for %s in PID %d", info.LibPath, pid)
}

// findThreadState walks the PyThreadState linked list to find the one matching tid.
func (w *PyFrameWalker) findThreadState(state *pyProcessState, tid uint32) (uint64, error) {
	offsets := state.offsets
	mem := state.mem

	// Read interpreters.head from _PyRuntime.
	interpPtr, err := mem.ReadPtr(state.runtimeAddr + offsets.RuntimeInterpretersHead)
	if err != nil {
		return 0, fmt.Errorf("reading interpreters.head at 0x%x: %w", state.runtimeAddr+offsets.RuntimeInterpretersHead, err)
	}
	if interpPtr == 0 {
		symDebugf("interpreters.head is NULL — Python not initialized?")
		return 0, nil
	}

	// Read tstate_head from PyInterpreterState.
	tstatePtr, err := mem.ReadPtr(interpPtr + offsets.InterpTstateHead)
	if err != nil {
		return 0, fmt.Errorf("reading tstate_head at 0x%x: %w", interpPtr+offsets.InterpTstateHead, err)
	}

	// Walk the linked list.
	// Match by native_thread_id (gettid/kernel TID, 3.8+), which matches the TID
	// from eBPF's bpf_get_current_pid_tgid(). Falls back to thread_id (pthread_self)
	// if native_thread_id is 0 (shouldn't happen on Linux 3.8+).
	targetTID := uint64(tid)
	walked := 0
	firstTstate := tstatePtr // Save for single-thread fallback.
	for i := 0; tstatePtr != 0 && i < 256; i++ {
		walked++

		// Try native_thread_id first (gettid — matches eBPF TID).
		if offsets.TstateNativeThreadID > 0 {
			nativeTID, err := mem.ReadUint64(tstatePtr + offsets.TstateNativeThreadID)
			if err == nil && nativeTID == targetTID {
				return tstatePtr, nil
			}
		}

		// Fallback: try thread_id (pthread_self — won't match eBPF TID on Linux,
		// but kept for completeness on platforms where native_thread_id isn't set).
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

	// Single-thread fallback: if there's exactly one PyThreadState and no TID
	// matched (common when hardcoded offsets don't match the distro's build),
	// use it anyway — for single-threaded Python processes, it's unambiguous.
	if walked == 1 && firstTstate != 0 {
		symDebugf("TID %d: using single PyThreadState (offset mismatch, single-thread fallback)", tid)
		return firstTstate, nil
	}

	symDebugf("TID %d not found in %d PyThreadState entries", tid, walked)
	return 0, nil // Thread not found (may not be running Python code)
}

// getCurrentFrame extracts the current frame pointer from a PyThreadState.
func (w *PyFrameWalker) getCurrentFrame(state *pyProcessState, tstate uint64) (uint64, error) {
	offsets := state.offsets
	mem := state.mem

	framePtr, err := mem.ReadPtr(tstate + offsets.TstateFrame)
	if err != nil {
		symDebugf("getCurrentFrame: error reading frame ptr at tstate+%d: %v", offsets.TstateFrame, err)
		return 0, err
	}

	if framePtr == 0 {
		symDebugf("getCurrentFrame: frame ptr is NULL (thread not in Python code)")
		return 0, nil
	}

	symDebugf("getCurrentFrame: raw framePtr=0x%x (from tstate+%d)", framePtr, offsets.TstateFrame)

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
			symDebugf("walkFrames[%d]: error reading f_code at frame 0x%x + %d: %v",
				i, framePtr, offsets.FrameCode, err)
			break
		}

		if codePtr != 0 {
			frame, err := w.readCodeObject(state, codePtr)
			if err == nil {
				frames = append(frames, frame)
			} else if i == 0 {
				symDebugf("walkFrames[%d]: readCodeObject(0x%x) failed: %v", i, codePtr, err)
			}
		} else if i == 0 {
			symDebugf("walkFrames[0]: f_code is NULL at frame 0x%x + %d", framePtr, offsets.FrameCode)
		}

		// Move to parent frame.
		framePtr, err = mem.ReadPtr(framePtr + offsets.FrameBack)
		if err != nil {
			break
		}
	}

	if len(frames) > 0 {
		symDebugf("walkFrames: extracted %d Python frames", len(frames))
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
	filename, fnErr := mem.ReadPyUnicodeString(filenamePtr, offsets, 256)
	funcName, nameErr := mem.ReadPyUnicodeString(namePtr, offsets, 128)

	if fnErr != nil {
		symDebugf("readCodeObject: ReadPyUnicodeString(filename=0x%x) failed: %v", filenamePtr, fnErr)
	}
	if nameErr != nil {
		symDebugf("readCodeObject: ReadPyUnicodeString(funcName=0x%x) failed: %v", namePtr, nameErr)
	}
	if filename != "" || funcName != "" {
		symDebugf("readCodeObject: filename=%q func=%q line=%d", filename, funcName, firstLineNo)
	}

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
