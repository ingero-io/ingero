package symtab

import (
	"fmt"
	"strings"
	"sync"

	"github.com/ingero-io/ingero/pkg/events"
)

// Resolver resolves raw instruction pointers to symbol names.
// It caches /proc/[pid]/maps and ELF symbol tables for performance.
// Also integrates CPython frame extraction when Python is detected.
//
// Thread-safe: can be called from multiple goroutines.
type Resolver struct {
	mu sync.RWMutex

	// Per-PID map region cache. Refreshed if a lookup misses
	// (process may have dlopen'd new libraries).
	pidMaps map[uint32][]MapRegion

	// Per-file ELF symbol cache. Keyed by file path.
	// ELF symbols don't change, so these are cached permanently.
	elfCache map[string]*ELFSymbols

	// Per-file ELF parse errors (avoid retrying failed parses).
	elfErrors map[string]bool

	// Python frame walker (initialized on first Python process encounter).
	pyWalker *PyFrameWalker
}

// NewResolver creates a new symbol resolver with empty caches.
func NewResolver() *Resolver {
	return &Resolver{
		pidMaps:   make(map[uint32][]MapRegion),
		elfCache:  make(map[string]*ELFSymbols),
		elfErrors: make(map[string]bool),
		pyWalker:  NewPyFrameWalker(),
	}
}

// ResolveStack resolves all frames in an event's stack trace.
// Modifies the event's Stack in place, filling in SymbolName and File fields.
// Also attempts CPython frame extraction: if a native frame is inside
// libpython's eval loop, the corresponding Python source frames are
// injected with PyFile/PyFunc/PyLine fields.
func (r *Resolver) ResolveStack(evt *events.Event) {
	if len(evt.Stack) == 0 {
		return
	}

	// Get or refresh the /proc/[pid]/maps for this process.
	// Process may have exited — in that case, regions will be nil and we
	// skip native resolution, but BPF Python frames (captured at event
	// time in kernel) are still valid and can be merged.
	regions := r.getRegions(evt.PID)

	// Resolve native symbols (skipped if no regions — e.g. process exited).
	if regions != nil {
		for i := range evt.Stack {
			r.resolveFrame(&evt.Stack[i], regions)
		}
	}

	// Attempt Python frame extraction.
	//
	// Precedence: when the BPF walker (--py-walker=ebpf) has already
	// captured frames for this event, use those directly — no /proc/pid/mem
	// read, no ptrace_scope dependency. Fall back to the userspace walker
	// only when the kernel-side walker produced nothing.
	if len(evt.PythonFrames) > 0 {
		r.mergeBPFPythonFrames(evt)
	} else if regions != nil && r.pyWalker != nil {
		pyFrames, err := r.pyWalker.WalkPythonFrames(evt.PID, evt.TID)
		if err != nil {
			symDebugf("Python frame walk failed for PID %d TID %d: %v", evt.PID, evt.TID, err)
		}
		if len(pyFrames) > 0 {
			// Merge Python frames into the stack. Insert them before the
			// first native frame that's inside libpython (the eval loop).
			r.mergePythonFrames(evt, pyFrames)
		}
	}
}

// mergeBPFPythonFrames adapts BPF-captured frames (events.PyFrame, with
// kernel-truncated strings) to the userspace PyFrame representation and
// then delegates to the existing merge logic. This keeps the merge
// insertion strategy (before the first libpython frame) identical for
// both capture paths.
//
// The two PyFrame types are structurally identical today — the conversion
// is defensive: keeping the boundary means events.PyFrame may add fields
// (column offsets, qualname) without breaking the merge invariants.
func (r *Resolver) mergeBPFPythonFrames(evt *events.Event) {
	if len(evt.PythonFrames) == 0 {
		return
	}
	pyFrames := make([]PyFrame, 0, len(evt.PythonFrames))
	for _, f := range evt.PythonFrames {
		pyFrames = append(pyFrames, PyFrame(f))
	}
	r.mergePythonFrames(evt, pyFrames)
}

// resolveFrame resolves a single stack frame.
func (r *Resolver) resolveFrame(frame *events.StackFrame, regions []MapRegion) {
	ip := frame.IP
	if ip == 0 {
		return
	}

	// Find which memory region contains this IP.
	region := FindRegion(regions, ip)
	if region == nil {
		return
	}

	// Set the file path (shared library or binary).
	frame.File = region.Path

	// Compute file offset for symbol lookup.
	// For memory-mapped files: file_offset = ip - region.Start + region.Offset
	fileOffset := ip - region.Start + region.Offset

	// Get ELF symbols for this file.
	elfs := r.getELFSymbols(region.Path)
	if elfs == nil {
		return
	}

	// Look up the symbol.
	name, offset := elfs.Lookup(fileOffset)
	if name != "" {
		if offset > 0 {
			frame.SymbolName = fmt.Sprintf("%s+0x%x", name, offset)
		} else {
			frame.SymbolName = name
		}
	}
}

// getRegions returns cached /proc/[pid]/maps regions, parsing if needed.
func (r *Resolver) getRegions(pid uint32) []MapRegion {
	r.mu.RLock()
	regions, ok := r.pidMaps[pid]
	r.mu.RUnlock()

	if ok {
		return regions
	}

	// Parse fresh.
	regions, err := ParseProcMaps(pid)
	if err != nil {
		return nil
	}

	r.mu.Lock()
	r.pidMaps[pid] = regions
	r.mu.Unlock()

	return regions
}

// getELFSymbols returns cached ELF symbols, parsing if needed.
func (r *Resolver) getELFSymbols(path string) *ELFSymbols {
	r.mu.RLock()
	elfs, ok := r.elfCache[path]
	isError := r.elfErrors[path]
	r.mu.RUnlock()

	if ok {
		return elfs
	}
	if isError {
		return nil
	}

	// Parse fresh.
	elfs, err := ParseELFSymbols(path)
	if err != nil {
		r.mu.Lock()
		r.elfErrors[path] = true
		r.mu.Unlock()
		return nil
	}

	r.mu.Lock()
	r.elfCache[path] = elfs
	r.mu.Unlock()

	return elfs
}

// mergePythonFrames inserts Python frames into the native stack.
// Python frames are placed before the first native frame inside libpython
// (typically _PyEval_EvalFrameDefault or similar eval loop functions).
func (r *Resolver) mergePythonFrames(evt *events.Event, pyFrames []PyFrame) {
	// Convert Python frames to StackFrame format.
	pyStackFrames := make([]events.StackFrame, len(pyFrames))
	for i, pf := range pyFrames {
		pyStackFrames[i] = events.StackFrame{
			PyFile: pf.Filename,
			PyFunc: pf.Function,
			PyLine: pf.Line,
		}
	}

	// Find insertion point: first frame inside libpython.
	insertIdx := -1
	for i, f := range evt.Stack {
		if isLibPythonFrame(f.File) {
			insertIdx = i
			break
		}
	}

	if insertIdx < 0 {
		// No libpython frame found — prepend Python frames.
		evt.Stack = append(pyStackFrames, evt.Stack...)
	} else {
		// Insert Python frames before the libpython frame.
		newStack := make([]events.StackFrame, 0, len(evt.Stack)+len(pyStackFrames))
		newStack = append(newStack, evt.Stack[:insertIdx]...)
		newStack = append(newStack, pyStackFrames...)
		newStack = append(newStack, evt.Stack[insertIdx:]...)
		evt.Stack = newStack
	}
}

// isLibPythonFrame returns true if the file path is a libpython shared library.
func isLibPythonFrame(path string) bool {
	if path == "" {
		return false
	}
	return strings.Contains(path, "libpython") || strings.Contains(path, "python3.")
}

// InvalidatePID removes cached maps for a PID (e.g., after process exit).
func (r *Resolver) InvalidatePID(pid uint32) {
	r.mu.Lock()
	delete(r.pidMaps, pid)
	r.mu.Unlock()
	if r.pyWalker != nil {
		r.pyWalker.InvalidatePID(pid)
	}
}

// Stats returns cache statistics for debugging.
func (r *Resolver) Stats() (pidCount, elfCount, elfErrors int) {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.pidMaps), len(r.elfCache), len(r.elfErrors)
}
