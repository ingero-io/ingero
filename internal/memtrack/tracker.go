package memtrack

import (
	"sync"

	"github.com/ingero-io/ingero/pkg/events"
)

// MemoryState represents the current VRAM allocation state for a single PID.
// JSON field names are the cross-language contract with the Rust orchestrator.
type MemoryState struct {
	PID            uint32  `json:"pid"`
	AllocatedBytes uint64  `json:"allocated_bytes"`
	TotalVRAM      uint64  `json:"total_vram"`
	UtilizationPct float64 `json:"utilization_pct"`
	LastAllocSize  uint64  `json:"last_alloc_size"`
	TimestampNs    int64   `json:"timestamp_ns"`
}

// Sink is a callback invoked after each event with the resulting MemoryState.
// nil means non-remediate mode (no emission).
type Sink func(MemoryState)

// pidState tracks the running allocation balance for a single PID.
type pidState struct {
	allocatedBytes uint64
}

// Tracker maintains per-PID VRAM allocation balances from cudaMalloc/cudaFree events.
// Thread-safe: protected by sync.RWMutex for concurrent access.
// Integration: called inline from the trace command's event loop.
type Tracker struct {
	mu        sync.RWMutex
	pids      map[uint32]*pidState
	totalVRAM uint64
	sink      Sink
}

// NewTracker creates a memory balance tracker.
// totalVRAM: total GPU VRAM in bytes (queried from nvidia-smi at startup, passed in).
// sink: callback for MemoryState emission. Pass nil for non-remediate mode.
func NewTracker(totalVRAM uint64, sink Sink) *Tracker {
	return &Tracker{
		pids:      make(map[uint32]*pidState),
		totalVRAM: totalVRAM,
		sink:      sink,
	}
}

// ProcessEvent updates the VRAM balance for the event's PID if the event
// is a memory allocation or free from either the CUDA Runtime API or Driver API.
//
// Runtime API (SourceCUDA):
//   - cudaMalloc: Args[0] = allocation size in bytes
//   - cudaFree: Args[0] = device pointer (not size). No balance decrement.
//
// Driver API (SourceDriver):
//   - cuMemAlloc_v2: Args[0] = allocation size in bytes
//   - cuMemAllocManaged: Args[0] = allocation size in bytes
//
// PyTorch's caching allocator calls cuMemAlloc_v2 (Driver API) for pool
// allocations — without Driver API support, those allocations are invisible.
//
// The tracker does NOT decrement on free — balance is a monotonically
// increasing upper bound (conservative for OOM detection).
func (t *Tracker) ProcessEvent(evt events.Event) {
	var lastAllocSize uint64
	var isAlloc bool

	switch evt.Source {
	case events.SourceCUDA:
		op := events.CUDAOp(evt.Op)
		switch op {
		case events.CUDAMalloc:
			lastAllocSize = evt.Args[0]
			isAlloc = true
		case events.CUDAFree:
			// cudaFree's Args[0] is the device pointer, not the size.
			// No-op on balance (conservative upper bound). Sink still fires.
		default:
			return
		}
	case events.SourceDriver:
		op := events.DriverOp(evt.Op)
		switch op {
		case events.DriverMemAlloc, events.DriverMemAllocManaged:
			lastAllocSize = evt.Args[0]
			isAlloc = true
		default:
			return
		}
	default:
		return
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	state, ok := t.pids[evt.PID]
	if !ok {
		state = &pidState{}
		t.pids[evt.PID] = state
	}

	if isAlloc {
		state.allocatedBytes += lastAllocSize
	}

	t.emit(evt.PID, lastAllocSize, evt.Timestamp.UnixNano())
}

// RecordMalloc adds size bytes to the PID's balance. Emits to sink.
// Exposed for testing. In production, use ProcessEvent with real events.
func (t *Tracker) RecordMalloc(pid uint32, size uint64, timestampNs int64) {
	t.mu.Lock()
	defer t.mu.Unlock()

	state, ok := t.pids[pid]
	if !ok {
		state = &pidState{}
		t.pids[pid] = state
	}

	state.allocatedBytes += size
	t.emit(pid, size, timestampNs)
}

// RecordFree subtracts size bytes from the PID's balance, clamping at 0.
// Used by the test suite to verify underflow protection.
// In production event flow, cudaFree goes through ProcessEvent which
// does not decrement (see ProcessEvent docs for rationale).
func (t *Tracker) RecordFree(pid uint32, size uint64, timestampNs int64) {
	t.mu.Lock()
	defer t.mu.Unlock()

	state, ok := t.pids[pid]
	if !ok {
		state = &pidState{}
		t.pids[pid] = state
	}

	if size > state.allocatedBytes {
		state.allocatedBytes = 0
	} else {
		state.allocatedBytes -= size
	}

	t.emit(pid, 0, timestampNs)
}

// emit sends a MemoryState to the sink. Caller must hold t.mu.
func (t *Tracker) emit(pid uint32, lastAllocSize uint64, timestampNs int64) {
	if t.sink == nil {
		return
	}
	state := t.pids[pid]
	var utilPct float64
	if t.totalVRAM > 0 {
		utilPct = float64(state.allocatedBytes) / float64(t.totalVRAM) * 100
	}
	t.sink(MemoryState{
		PID:            pid,
		AllocatedBytes: state.allocatedBytes,
		TotalVRAM:      t.totalVRAM,
		UtilizationPct: utilPct,
		LastAllocSize:  lastAllocSize,
		TimestampNs:    timestampNs,
	})
}
