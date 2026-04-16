package memtrack

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"sync"

	"github.com/ingero-io/ingero/pkg/events"
)

// MemoryState represents the current VRAM allocation state for a single PID on a single GPU.
// JSON field names are the cross-language contract with external consumers (see docs/remediation-protocol.md).
//
// v0.10: Comm carries the kernel-captured process name from bpf_get_current_comm()
// for human-readable orchestrator logs and PID-reuse detection. May be empty when
// the BPF probe could not capture comm (softirq context, edge cases) or when the
// emitting helper does not have a current event (test fixtures).
type MemoryState struct {
	PID            uint32  `json:"pid"`
	Comm           string  `json:"comm,omitempty"`
	GPUID          uint32  `json:"gpu_id"`
	AllocatedBytes uint64  `json:"allocated_bytes"`
	TotalVRAM      uint64  `json:"total_vram"`
	UtilizationPct float64 `json:"utilization_pct"`
	LastAllocSize  uint64  `json:"last_alloc_size"`
	TimestampNs    int64   `json:"timestamp_ns"`
}

// Sink is a callback invoked after each event with the resulting MemoryState.
// nil means non-remediate mode (no emission).
type Sink func(MemoryState)

// pidGpuKey is the map key for per-PID per-GPU allocation tracking.
type pidGpuKey struct {
	pid   uint32
	gpuID uint32
}

// PIDGPUState tracks the running allocation balance for a single PID on a single GPU.
// v0.10: caches the most recently observed non-empty comm for the PID so MemoryState
// emissions carry process identity even from event paths that don't supply comm
// (RecordMalloc/RecordFree test helpers, etc.).
type PIDGPUState struct {
	allocatedBytes uint64
	comm           string
}

// Tracker maintains per-PID per-GPU VRAM allocation balances from cudaMalloc/cudaFree events.
// Thread-safe: protected by sync.Mutex for concurrent access.
// Integration: called inline from the trace command's event loop.
type Tracker struct {
	mu      sync.Mutex
	pids    map[pidGpuKey]*PIDGPUState
	gpuVRAM map[uint32]uint64 // per-GPU total VRAM in bytes
	sink    Sink
}

// NewTracker creates a memory balance tracker.
// gpuVRAM: map of GPU ID to total VRAM in bytes (queried via DetectGPUVRAM at startup).
// sink: callback for MemoryState emission. Pass nil for non-remediate mode.
func NewTracker(gpuVRAM map[uint32]uint64, sink Sink) *Tracker {
	return &Tracker{
		pids:    make(map[pidGpuKey]*PIDGPUState),
		gpuVRAM: gpuVRAM,
		sink:    sink,
	}
}

// ProcessEvent updates the VRAM balance for the event's PID+GPU if the event
// is a memory allocation or free from either the CUDA Runtime API or Driver API.
//
// Runtime API (SourceCUDA):
//   - cudaMalloc: Args[0] = allocation size in bytes, Args[1] = devPtr param address
//   - cudaFree: Args[0] = device pointer, Args[1] = freed size in bytes (0 if unknown)
//
// Driver API (SourceDriver):
//   - cuMemAlloc_v2: Args[0] = allocation size in bytes
//   - cuMemAllocManaged: Args[0] = allocation size in bytes
//
// Each event's GPUID field (from BPF cuda_event.gpu_id) determines which GPU's balance is updated.
func (t *Tracker) ProcessEvent(evt events.Event) {
	var lastAllocSize uint64
	var isAlloc bool
	var isFree bool
	var freedBytes uint64

	switch evt.Source {
	case events.SourceCUDA:
		op := events.CUDAOp(evt.Op)
		switch op {
		case events.CUDAMalloc:
			lastAllocSize = evt.Args[0]
			isAlloc = true
		case events.CUDAFree:
			freedBytes = evt.Args[1] // freed size from eBPF alloc_sizes map (0 if unknown)
			if freedBytes > 0 {
				isFree = true
			}
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

	key := pidGpuKey{pid: evt.PID, gpuID: evt.GPUID}

	t.mu.Lock()
	state, ok := t.pids[key]
	if !ok {
		state = &PIDGPUState{}
		t.pids[key] = state
	}

	// Cache the most recent non-empty comm for this PID+GPU. Empty comm
	// (BPF edge case) does not overwrite a previously valid value.
	if evt.Comm != "" {
		state.comm = evt.Comm
	}

	if isAlloc {
		state.allocatedBytes += lastAllocSize
	}
	if isFree {
		if freedBytes > state.allocatedBytes {
			state.allocatedBytes = 0 // underflow clamp
		} else {
			state.allocatedBytes -= freedBytes
		}
	}

	ms := t.buildState(key, lastAllocSize, evt.Timestamp.UnixNano())
	t.mu.Unlock()

	if t.sink != nil {
		t.sink(ms)
	}
}

// RecordMalloc adds size bytes to the PID+GPU balance. Emits to sink.
// Exposed for testing. In production, use ProcessEvent with real events.
func (t *Tracker) RecordMalloc(pid uint32, gpuID uint32, size uint64, timestampNs int64) {
	key := pidGpuKey{pid: pid, gpuID: gpuID}

	t.mu.Lock()
	state, ok := t.pids[key]
	if !ok {
		state = &PIDGPUState{}
		t.pids[key] = state
	}

	state.allocatedBytes += size
	ms := t.buildState(key, size, timestampNs)
	t.mu.Unlock()

	if t.sink != nil {
		t.sink(ms)
	}
}

// RecordFree subtracts size bytes from the PID+GPU balance, clamping at 0.
func (t *Tracker) RecordFree(pid uint32, gpuID uint32, size uint64, timestampNs int64) {
	key := pidGpuKey{pid: pid, gpuID: gpuID}

	t.mu.Lock()
	state, ok := t.pids[key]
	if !ok {
		state = &PIDGPUState{}
		t.pids[key] = state
	}

	if size > state.allocatedBytes {
		state.allocatedBytes = 0
	} else {
		state.allocatedBytes -= size
	}

	ms := t.buildState(key, 0, timestampNs)
	t.mu.Unlock()

	if t.sink != nil {
		t.sink(ms)
	}
}

// buildState constructs a MemoryState snapshot. Caller must hold t.mu.
func (t *Tracker) buildState(key pidGpuKey, lastAllocSize uint64, timestampNs int64) MemoryState {
	state := t.pids[key]
	totalVRAM := t.gpuVRAM[key.gpuID]
	var utilPct float64
	if totalVRAM > 0 {
		utilPct = float64(state.allocatedBytes) / float64(totalVRAM) * 100
	}
	return MemoryState{
		PID:            key.pid,
		Comm:           state.comm, // v0.10: cached from most recent event with non-empty comm
		GPUID:          key.gpuID,
		AllocatedBytes: state.allocatedBytes,
		TotalVRAM:      totalVRAM,
		UtilizationPct: utilPct,
		LastAllocSize:  lastAllocSize,
		TimestampNs:    timestampNs,
	}
}

// DetectGPUVRAM queries nvidia-smi for per-GPU total VRAM and returns a map of
// GPU index to VRAM in bytes. Each line of nvidia-smi output corresponds to one GPU.
func DetectGPUVRAM() (map[uint32]uint64, error) {
	out, err := exec.Command("nvidia-smi", "--query-gpu=memory.total", "--format=csv,noheader,nounits").Output()
	if err != nil {
		return nil, fmt.Errorf("DetectGPUVRAM: %w", err)
	}
	return parseGPUVRAMOutput(string(out))
}

// parseGPUVRAMOutput parses nvidia-smi memory.total output into a per-GPU VRAM map.
func parseGPUVRAMOutput(output string) (map[uint32]uint64, error) {
	result := make(map[uint32]uint64)
	lines := strings.Split(strings.TrimSpace(output), "\n")
	for i, line := range lines {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		mib, err := strconv.ParseUint(line, 10, 64)
		if err != nil {
			return nil, fmt.Errorf("parseGPUVRAMOutput: GPU %d: parsing %q: %w", i, line, err)
		}
		result[uint32(i)] = mib * 1024 * 1024
	}
	if len(result) == 0 {
		return nil, fmt.Errorf("parseGPUVRAMOutput: no GPUs detected")
	}
	return result, nil
}
