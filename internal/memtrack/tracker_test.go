package memtrack

import (
	"encoding/json"
	"testing"
	"time"

	"github.com/ingero-io/ingero/pkg/events"
)

const fixedTS = 1711180800000000000

// singleGPUVRAM returns a per-GPU VRAM map for single-GPU (GPU 0) tests.
func singleGPUVRAM(totalVRAM uint64) map[uint32]uint64 {
	return map[uint32]uint64{0: totalVRAM}
}

// makeCUDAEvent creates a CUDA Runtime API event (GPU 0 by default).
func makeCUDAEvent(pid uint32, op events.CUDAOp, arg0 uint64) events.Event {
	return events.Event{
		Timestamp: time.Unix(0, fixedTS),
		PID:       pid,
		TID:       pid,
		Source:    events.SourceCUDA,
		Op:        uint8(op),
		Args:      [2]uint64{arg0, 0},
	}
}

// makeCUDAFreeEvent creates a cudaFree event with the freed size in Args[1].
func makeCUDAFreeEvent(pid uint32, devPtr uint64, freedBytes uint64) events.Event {
	return events.Event{
		Timestamp: time.Unix(0, fixedTS),
		PID:       pid,
		TID:       pid,
		Source:    events.SourceCUDA,
		Op:        uint8(events.CUDAFree),
		Args:      [2]uint64{devPtr, freedBytes},
	}
}

// makeDriverEvent creates a CUDA Driver API event (GPU 0 by default).
func makeDriverEvent(pid uint32, op events.DriverOp, size uint64) events.Event {
	return events.Event{
		Timestamp: time.Unix(0, fixedTS),
		PID:       pid,
		TID:       pid,
		Source:    events.SourceDriver,
		Op:        uint8(op),
		Args:      [2]uint64{size, 0},
	}
}

// makeCUDAEventGPU creates a CUDA event targeting a specific GPU.
func makeCUDAEventGPU(pid uint32, gpuID uint32, op events.CUDAOp, arg0 uint64) events.Event {
	e := makeCUDAEvent(pid, op, arg0)
	e.GPUID = gpuID
	return e
}

// makeCUDAFreeEventGPU creates a cudaFree event targeting a specific GPU.
func makeCUDAFreeEventGPU(pid uint32, gpuID uint32, devPtr uint64, freedBytes uint64) events.Event {
	e := makeCUDAFreeEvent(pid, devPtr, freedBytes)
	e.GPUID = gpuID
	return e
}

// makeDriverEventGPU creates a Driver event targeting a specific GPU.
func makeDriverEventGPU(pid uint32, gpuID uint32, op events.DriverOp, size uint64) events.Event {
	e := makeDriverEvent(pid, op, size)
	e.GPUID = gpuID
	return e
}

func TestTracker(t *testing.T) {
	tests := []struct {
		name string
		run  func(t *testing.T)
	}{
		{
			name: "malloc_increases_balance",
			run: func(t *testing.T) {
				var last MemoryState
				sink := func(ms MemoryState) { last = ms }
				tr := NewTracker(singleGPUVRAM(16*1024*1024*1024), sink)

				sizes := []uint64{256 * 1024 * 1024, 512 * 1024 * 1024, 128 * 1024 * 1024}
				var expected uint64
				for _, sz := range sizes {
					expected += sz
					tr.ProcessEvent(makeCUDAEvent(1000, events.CUDAMalloc, sz))
				}

				if last.AllocatedBytes != expected {
					t.Errorf("allocated_bytes = %d, want %d", last.AllocatedBytes, expected)
				}
			},
		},
		{
			name: "free_decreases_balance",
			run: func(t *testing.T) {
				var last MemoryState
				sink := func(ms MemoryState) { last = ms }
				tr := NewTracker(singleGPUVRAM(16*1024*1024*1024), sink)

				tr.RecordMalloc(1000, 0, 1000, fixedTS)
				tr.RecordFree(1000, 0, 400, fixedTS)

				if last.AllocatedBytes != 600 {
					t.Errorf("allocated_bytes = %d, want 600", last.AllocatedBytes)
				}
			},
		},
		{
			name: "utilization_percentage",
			run: func(t *testing.T) {
				var last MemoryState
				sink := func(ms MemoryState) { last = ms }
				tr := NewTracker(singleGPUVRAM(1000), sink)

				tr.RecordMalloc(1000, 0, 625, fixedTS)

				if last.UtilizationPct != 62.5 {
					t.Errorf("utilization_pct = %f, want 62.5", last.UtilizationPct)
				}
			},
		},
		{
			name: "free_without_malloc_clamps_zero",
			run: func(t *testing.T) {
				var last MemoryState
				sink := func(ms MemoryState) { last = ms }
				tr := NewTracker(singleGPUVRAM(16*1024*1024*1024), sink)

				tr.RecordFree(9999, 0, 500*1024*1024, fixedTS)

				if last.AllocatedBytes != 0 {
					t.Errorf("allocated_bytes = %d, want 0 (clamped)", last.AllocatedBytes)
				}
			},
		},
		{
			name: "independent_pid_tracking",
			run: func(t *testing.T) {
				states := make(map[uint32]MemoryState)
				sink := func(ms MemoryState) { states[ms.PID] = ms }
				tr := NewTracker(singleGPUVRAM(16*1024*1024*1024), sink)

				tr.ProcessEvent(makeCUDAEvent(100, events.CUDAMalloc, 300))
				tr.ProcessEvent(makeCUDAEvent(200, events.CUDAMalloc, 700))
				tr.ProcessEvent(makeCUDAEvent(100, events.CUDAMalloc, 200))

				if states[100].AllocatedBytes != 500 {
					t.Errorf("PID 100 allocated_bytes = %d, want 500", states[100].AllocatedBytes)
				}
				if states[200].AllocatedBytes != 700 {
					t.Errorf("PID 200 allocated_bytes = %d, want 700", states[200].AllocatedBytes)
				}
			},
		},
		{
			name: "sink_receives_correct_state",
			run: func(t *testing.T) {
				var received []MemoryState
				sink := func(ms MemoryState) { received = append(received, ms) }
				totalVRAM := uint64(17179869184) // 16 GB
				tr := NewTracker(singleGPUVRAM(totalVRAM), sink)

				allocSize := uint64(268435456) // 256 MB
				evt := makeCUDAEvent(12345, events.CUDAMalloc, allocSize)
				tr.ProcessEvent(evt)

				if len(received) != 1 {
					t.Fatalf("expected 1 emission, got %d", len(received))
				}
				ms := received[0]
				if ms.PID != 12345 {
					t.Errorf("pid = %d, want 12345", ms.PID)
				}
				if ms.GPUID != 0 {
					t.Errorf("gpu_id = %d, want 0", ms.GPUID)
				}
				if ms.AllocatedBytes != allocSize {
					t.Errorf("allocated_bytes = %d, want %d", ms.AllocatedBytes, allocSize)
				}
				if ms.TotalVRAM != totalVRAM {
					t.Errorf("total_vram = %d, want %d", ms.TotalVRAM, totalVRAM)
				}
				expectedUtil := float64(allocSize) / float64(totalVRAM) * 100
				if ms.UtilizationPct != expectedUtil {
					t.Errorf("utilization_pct = %f, want %f", ms.UtilizationPct, expectedUtil)
				}
				if ms.LastAllocSize != allocSize {
					t.Errorf("last_alloc_size = %d, want %d", ms.LastAllocSize, allocSize)
				}
				if ms.TimestampNs != evt.Timestamp.UnixNano() {
					t.Errorf("timestamp_ns = %d, want %d", ms.TimestampNs, evt.Timestamp.UnixNano())
				}
			},
		},
		{
			name: "driver_cuMemAlloc_increases_balance",
			run: func(t *testing.T) {
				var last MemoryState
				sink := func(ms MemoryState) { last = ms }
				tr := NewTracker(singleGPUVRAM(16*1024*1024*1024), sink)

				tr.ProcessEvent(makeDriverEvent(1000, events.DriverMemAlloc, 2*1024*1024*1024))

				if last.AllocatedBytes != 2*1024*1024*1024 {
					t.Errorf("allocated_bytes = %d, want %d", last.AllocatedBytes, uint64(2*1024*1024*1024))
				}
				if last.PID != 1000 {
					t.Errorf("pid = %d, want 1000", last.PID)
				}
			},
		},
		{
			name: "driver_cuMemAllocManaged_increases_balance",
			run: func(t *testing.T) {
				var last MemoryState
				sink := func(ms MemoryState) { last = ms }
				tr := NewTracker(singleGPUVRAM(16*1024*1024*1024), sink)

				tr.ProcessEvent(makeDriverEvent(1000, events.DriverMemAllocManaged, 512*1024*1024))

				if last.AllocatedBytes != 512*1024*1024 {
					t.Errorf("allocated_bytes = %d, want %d", last.AllocatedBytes, uint64(512*1024*1024))
				}
			},
		},
		{
			name: "driver_non_alloc_ops_ignored",
			run: func(t *testing.T) {
				callCount := 0
				sink := func(ms MemoryState) { callCount++ }
				tr := NewTracker(singleGPUVRAM(16*1024*1024*1024), sink)

				tr.ProcessEvent(makeDriverEvent(1000, events.DriverLaunchKernel, 0))
				tr.ProcessEvent(makeDriverEvent(1000, events.DriverMemcpy, 4096))
				tr.ProcessEvent(makeDriverEvent(1000, events.DriverCtxSync, 0))

				if callCount != 0 {
					t.Errorf("sink called %d times, want 0 (non-alloc driver ops should be ignored)", callCount)
				}
			},
		},
		{
			name: "mixed_runtime_and_driver_allocs_accumulate",
			run: func(t *testing.T) {
				var last MemoryState
				sink := func(ms MemoryState) { last = ms }
				tr := NewTracker(singleGPUVRAM(16*1024*1024*1024), sink)

				// cudaMalloc (Runtime API)
				tr.ProcessEvent(makeCUDAEvent(1000, events.CUDAMalloc, 1*1024*1024*1024))
				// cuMemAlloc_v2 (Driver API)
				tr.ProcessEvent(makeDriverEvent(1000, events.DriverMemAlloc, 2*1024*1024*1024))

				expected := uint64(3 * 1024 * 1024 * 1024)
				if last.AllocatedBytes != expected {
					t.Errorf("allocated_bytes = %d, want %d", last.AllocatedBytes, expected)
				}
			},
		},
		{
			name: "no_sink_no_panic",
			run: func(t *testing.T) {
				tr := NewTracker(singleGPUVRAM(16*1024*1024*1024), nil)

				// Must not panic
				tr.ProcessEvent(makeCUDAEvent(1000, events.CUDAMalloc, 256*1024*1024))
				tr.ProcessEvent(makeCUDAFreeEvent(1000, 0x7f0000, 0))
				tr.RecordMalloc(1000, 0, 100, fixedTS)
				tr.RecordFree(1000, 0, 50, fixedTS)
			},
		},
		// --- Net balance tests ---
		{
			name: "net_balance/alloc_then_free_returns_to_zero",
			run: func(t *testing.T) {
				var last MemoryState
				sink := func(ms MemoryState) { last = ms }
				tr := NewTracker(singleGPUVRAM(16*1024*1024*1024), sink)

				allocSize := uint64(1024 * 1024) // 1MB
				tr.ProcessEvent(makeCUDAEvent(1000, events.CUDAMalloc, allocSize))
				tr.ProcessEvent(makeCUDAFreeEvent(1000, 0x7f0000, allocSize))

				if last.AllocatedBytes != 0 {
					t.Errorf("allocated_bytes = %d, want 0", last.AllocatedBytes)
				}
			},
		},
		{
			name: "net_balance/multiple_alloc_partial_free",
			run: func(t *testing.T) {
				var last MemoryState
				sink := func(ms MemoryState) { last = ms }
				tr := NewTracker(singleGPUVRAM(16*1024*1024*1024), sink)

				tr.ProcessEvent(makeCUDAEvent(1000, events.CUDAMalloc, 1*1024*1024))  // 1MB
				tr.ProcessEvent(makeCUDAEvent(1000, events.CUDAMalloc, 2*1024*1024))  // 2MB
				tr.ProcessEvent(makeCUDAFreeEvent(1000, 0x7f0000, 1*1024*1024))       // free 1MB

				expected := uint64(2 * 1024 * 1024)
				if last.AllocatedBytes != expected {
					t.Errorf("allocated_bytes = %d, want %d", last.AllocatedBytes, expected)
				}
			},
		},
		{
			name: "net_balance/free_unknown_pointer_no_change",
			run: func(t *testing.T) {
				var last MemoryState
				sink := func(ms MemoryState) { last = ms }
				tr := NewTracker(singleGPUVRAM(16*1024*1024*1024), sink)

				allocSize := uint64(1024 * 1024) // 1MB
				tr.ProcessEvent(makeCUDAEvent(1000, events.CUDAMalloc, allocSize))
				// Free with Args[1]=0 (unknown pointer)
				tr.ProcessEvent(makeCUDAFreeEvent(1000, 0xdeadbeef, 0))

				if last.AllocatedBytes != allocSize {
					t.Errorf("allocated_bytes = %d, want %d (unchanged)", last.AllocatedBytes, allocSize)
				}
			},
		},
		{
			name: "net_balance/free_exceeds_balance_clamps_zero",
			run: func(t *testing.T) {
				var last MemoryState
				sink := func(ms MemoryState) { last = ms }
				tr := NewTracker(singleGPUVRAM(16*1024*1024*1024), sink)

				tr.ProcessEvent(makeCUDAEvent(1000, events.CUDAMalloc, 1*1024*1024)) // 1MB
				tr.ProcessEvent(makeCUDAFreeEvent(1000, 0x7f0000, 2*1024*1024))      // free 2MB

				if last.AllocatedBytes != 0 {
					t.Errorf("allocated_bytes = %d, want 0 (clamped)", last.AllocatedBytes)
				}
			},
		},
		{
			name: "net_balance/interleaved_pids_independent",
			run: func(t *testing.T) {
				states := make(map[uint32]MemoryState)
				sink := func(ms MemoryState) { states[ms.PID] = ms }
				tr := NewTracker(singleGPUVRAM(16*1024*1024*1024), sink)

				tr.ProcessEvent(makeCUDAEvent(100, events.CUDAMalloc, 1*1024*1024))  // PID A: +1MB
				tr.ProcessEvent(makeCUDAEvent(200, events.CUDAMalloc, 2*1024*1024))  // PID B: +2MB
				tr.ProcessEvent(makeCUDAFreeEvent(100, 0x7f0000, 1*1024*1024))       // PID A: -1MB

				if states[100].AllocatedBytes != 0 {
					t.Errorf("PID 100 allocated_bytes = %d, want 0", states[100].AllocatedBytes)
				}
				if states[200].AllocatedBytes != 2*1024*1024 {
					t.Errorf("PID 200 allocated_bytes = %d, want %d", states[200].AllocatedBytes, uint64(2*1024*1024))
				}
			},
		},
		{
			name: "net_balance/1000_alloc_free_cycles_returns_to_baseline",
			run: func(t *testing.T) {
				var last MemoryState
				sink := func(ms MemoryState) { last = ms }
				totalVRAM := uint64(16 * 1024 * 1024 * 1024) // 16GB
				tr := NewTracker(singleGPUVRAM(totalVRAM), sink)

				allocSize := uint64(1024 * 1024) // 1MB each
				for i := 0; i < 1000; i++ {
					tr.ProcessEvent(makeCUDAEvent(1000, events.CUDAMalloc, allocSize))
					tr.ProcessEvent(makeCUDAFreeEvent(1000, uint64(0x7f0000+i*0x100000), allocSize))
				}

				if last.AllocatedBytes != 0 {
					t.Errorf("allocated_bytes = %d, want 0 after 1000 alloc/free cycles", last.AllocatedBytes)
				}
				if last.UtilizationPct > 5.0 {
					t.Errorf("utilization_pct = %f, want within 5%% of 0", last.UtilizationPct)
				}
			},
		},
		{
			name: "net_balance/utilization_pct_decreases_on_free",
			run: func(t *testing.T) {
				var last MemoryState
				sink := func(ms MemoryState) { last = ms }
				totalVRAM := uint64(10 * 1024 * 1024 * 1024) // 10GB
				tr := NewTracker(singleGPUVRAM(totalVRAM), sink)

				// Allocate to ~80%
				allocSize := uint64(8 * 1024 * 1024 * 1024) // 8GB
				tr.ProcessEvent(makeCUDAEvent(1000, events.CUDAMalloc, allocSize))
				if last.UtilizationPct != 80.0 {
					t.Errorf("after alloc: utilization_pct = %f, want 80.0", last.UtilizationPct)
				}

				// Free half
				tr.ProcessEvent(makeCUDAFreeEvent(1000, 0x7f0000, allocSize/2))
				expectedUtil := 40.0
				if last.UtilizationPct != expectedUtil {
					t.Errorf("after free: utilization_pct = %f, want %f", last.UtilizationPct, expectedUtil)
				}
			},
		},
		// --- Backward compatibility: single-GPU emits gpu_id: 0 ---
		{
			name: "backward_compat/single_gpu_emits_gpu_id_zero",
			run: func(t *testing.T) {
				var last MemoryState
				sink := func(ms MemoryState) { last = ms }
				tr := NewTracker(singleGPUVRAM(16*1024*1024*1024), sink)

				tr.ProcessEvent(makeCUDAEvent(1000, events.CUDAMalloc, 256*1024*1024))

				if last.GPUID != 0 {
					t.Errorf("gpu_id = %d, want 0", last.GPUID)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, tt.run)
	}
}

func TestMultiGPU(t *testing.T) {
	gpuVRAM := map[uint32]uint64{
		0: 16 * 1024 * 1024 * 1024, // GPU 0: 16 GB
		1: 16 * 1024 * 1024 * 1024, // GPU 1: 16 GB
		2: 16 * 1024 * 1024 * 1024, // GPU 2: 16 GB
		3: 16 * 1024 * 1024 * 1024, // GPU 3: 16 GB
	}

	tests := []struct {
		name string
		run  func(t *testing.T)
	}{
		{
			name: "gpu_isolation/alloc_on_gpu0_does_not_affect_gpu1",
			run: func(t *testing.T) {
				var received []MemoryState
				sink := func(ms MemoryState) { received = append(received, ms) }
				tr := NewTracker(gpuVRAM, sink)

				tr.ProcessEvent(makeCUDAEventGPU(1000, 0, events.CUDAMalloc, 4*1024*1024*1024))

				if len(received) != 1 {
					t.Fatalf("expected 1 emission, got %d", len(received))
				}
				if received[0].GPUID != 0 {
					t.Errorf("gpu_id = %d, want 0", received[0].GPUID)
				}
				if received[0].AllocatedBytes != 4*1024*1024*1024 {
					t.Errorf("allocated_bytes = %d, want %d", received[0].AllocatedBytes, uint64(4*1024*1024*1024))
				}

				// Allocate on GPU 1 — should have independent state
				tr.ProcessEvent(makeCUDAEventGPU(1000, 1, events.CUDAMalloc, 2*1024*1024*1024))

				if len(received) != 2 {
					t.Fatalf("expected 2 emissions, got %d", len(received))
				}
				if received[1].GPUID != 1 {
					t.Errorf("gpu_id = %d, want 1", received[1].GPUID)
				}
				if received[1].AllocatedBytes != 2*1024*1024*1024 {
					t.Errorf("GPU 1 allocated_bytes = %d, want %d", received[1].AllocatedBytes, uint64(2*1024*1024*1024))
				}
			},
		},
		{
			name: "gpu_isolation/same_pid_different_gpus_independent_balances",
			run: func(t *testing.T) {
				type pgKey struct {
					pid   uint32
					gpuID uint32
				}
				states := make(map[pgKey]MemoryState)
				sink := func(ms MemoryState) { states[pgKey{ms.PID, ms.GPUID}] = ms }
				tr := NewTracker(gpuVRAM, sink)

				// PID 1000 allocates on GPU 0, 1, 2, 3
				tr.ProcessEvent(makeCUDAEventGPU(1000, 0, events.CUDAMalloc, 1*1024*1024*1024))
				tr.ProcessEvent(makeCUDAEventGPU(1000, 1, events.CUDAMalloc, 2*1024*1024*1024))
				tr.ProcessEvent(makeCUDAEventGPU(1000, 2, events.CUDAMalloc, 3*1024*1024*1024))
				tr.ProcessEvent(makeCUDAEventGPU(1000, 3, events.CUDAMalloc, 4*1024*1024*1024))

				expected := map[uint32]uint64{
					0: 1 * 1024 * 1024 * 1024,
					1: 2 * 1024 * 1024 * 1024,
					2: 3 * 1024 * 1024 * 1024,
					3: 4 * 1024 * 1024 * 1024,
				}
				for gpuID, wantBytes := range expected {
					got := states[pgKey{1000, gpuID}].AllocatedBytes
					if got != wantBytes {
						t.Errorf("GPU %d allocated_bytes = %d, want %d", gpuID, got, wantBytes)
					}
				}
			},
		},
		{
			name: "gpu_isolation/free_on_gpu0_does_not_affect_gpu1",
			run: func(t *testing.T) {
				type pgKey struct {
					pid   uint32
					gpuID uint32
				}
				states := make(map[pgKey]MemoryState)
				sink := func(ms MemoryState) { states[pgKey{ms.PID, ms.GPUID}] = ms }
				tr := NewTracker(gpuVRAM, sink)

				tr.ProcessEvent(makeCUDAEventGPU(1000, 0, events.CUDAMalloc, 4*1024*1024*1024))
				tr.ProcessEvent(makeCUDAEventGPU(1000, 1, events.CUDAMalloc, 4*1024*1024*1024))

				// Free on GPU 0
				tr.ProcessEvent(makeCUDAFreeEventGPU(1000, 0, 0x7f0000, 4*1024*1024*1024))

				if states[pgKey{1000, 0}].AllocatedBytes != 0 {
					t.Errorf("GPU 0 allocated_bytes = %d, want 0", states[pgKey{1000, 0}].AllocatedBytes)
				}
				if states[pgKey{1000, 1}].AllocatedBytes != 4*1024*1024*1024 {
					t.Errorf("GPU 1 allocated_bytes = %d, want %d (unchanged)", states[pgKey{1000, 1}].AllocatedBytes, uint64(4*1024*1024*1024))
				}
			},
		},
		{
			name: "gpu_isolation/per_gpu_utilization_independent",
			run: func(t *testing.T) {
				type pgKey struct {
					pid   uint32
					gpuID uint32
				}
				states := make(map[pgKey]MemoryState)
				sink := func(ms MemoryState) { states[pgKey{ms.PID, ms.GPUID}] = ms }
				tr := NewTracker(gpuVRAM, sink)

				// GPU 0: 50% utilization, GPU 1: 25% utilization
				tr.ProcessEvent(makeCUDAEventGPU(1000, 0, events.CUDAMalloc, 8*1024*1024*1024))
				tr.ProcessEvent(makeCUDAEventGPU(1000, 1, events.CUDAMalloc, 4*1024*1024*1024))

				if states[pgKey{1000, 0}].UtilizationPct != 50.0 {
					t.Errorf("GPU 0 utilization_pct = %f, want 50.0", states[pgKey{1000, 0}].UtilizationPct)
				}
				if states[pgKey{1000, 1}].UtilizationPct != 25.0 {
					t.Errorf("GPU 1 utilization_pct = %f, want 25.0", states[pgKey{1000, 1}].UtilizationPct)
				}
			},
		},
		{
			name: "gpu_isolation/different_pids_different_gpus",
			run: func(t *testing.T) {
				type pgKey struct {
					pid   uint32
					gpuID uint32
				}
				states := make(map[pgKey]MemoryState)
				sink := func(ms MemoryState) { states[pgKey{ms.PID, ms.GPUID}] = ms }
				tr := NewTracker(gpuVRAM, sink)

				// DDP pattern: PID 1001 → GPU 0, PID 1002 → GPU 1
				tr.ProcessEvent(makeCUDAEventGPU(1001, 0, events.CUDAMalloc, 6*1024*1024*1024))
				tr.ProcessEvent(makeCUDAEventGPU(1002, 1, events.CUDAMalloc, 8*1024*1024*1024))

				if states[pgKey{1001, 0}].AllocatedBytes != 6*1024*1024*1024 {
					t.Errorf("PID 1001 GPU 0 = %d, want %d", states[pgKey{1001, 0}].AllocatedBytes, uint64(6*1024*1024*1024))
				}
				if states[pgKey{1002, 1}].AllocatedBytes != 8*1024*1024*1024 {
					t.Errorf("PID 1002 GPU 1 = %d, want %d", states[pgKey{1002, 1}].AllocatedBytes, uint64(8*1024*1024*1024))
				}
			},
		},
		{
			name: "gpu_isolation/underflow_clamp_per_gpu",
			run: func(t *testing.T) {
				type pgKey struct {
					pid   uint32
					gpuID uint32
				}
				states := make(map[pgKey]MemoryState)
				sink := func(ms MemoryState) { states[pgKey{ms.PID, ms.GPUID}] = ms }
				tr := NewTracker(gpuVRAM, sink)

				tr.ProcessEvent(makeCUDAEventGPU(1000, 0, events.CUDAMalloc, 1*1024*1024))
				tr.ProcessEvent(makeCUDAEventGPU(1000, 1, events.CUDAMalloc, 2*1024*1024))

				// Free more than allocated on GPU 0 — should clamp to 0, GPU 1 unaffected
				tr.ProcessEvent(makeCUDAFreeEventGPU(1000, 0, 0x7f0000, 10*1024*1024))

				if states[pgKey{1000, 0}].AllocatedBytes != 0 {
					t.Errorf("GPU 0 allocated_bytes = %d, want 0 (clamped)", states[pgKey{1000, 0}].AllocatedBytes)
				}
				if states[pgKey{1000, 1}].AllocatedBytes != 2*1024*1024 {
					t.Errorf("GPU 1 allocated_bytes = %d, want %d (unchanged)", states[pgKey{1000, 1}].AllocatedBytes, uint64(2*1024*1024))
				}
			},
		},
		{
			name: "gpu_isolation/driver_api_multi_gpu",
			run: func(t *testing.T) {
				type pgKey struct {
					pid   uint32
					gpuID uint32
				}
				states := make(map[pgKey]MemoryState)
				sink := func(ms MemoryState) { states[pgKey{ms.PID, ms.GPUID}] = ms }
				tr := NewTracker(gpuVRAM, sink)

				tr.ProcessEvent(makeDriverEventGPU(1000, 0, events.DriverMemAlloc, 1*1024*1024*1024))
				tr.ProcessEvent(makeDriverEventGPU(1000, 2, events.DriverMemAllocManaged, 3*1024*1024*1024))

				if states[pgKey{1000, 0}].AllocatedBytes != 1*1024*1024*1024 {
					t.Errorf("GPU 0 = %d, want %d", states[pgKey{1000, 0}].AllocatedBytes, uint64(1*1024*1024*1024))
				}
				if states[pgKey{1000, 2}].AllocatedBytes != 3*1024*1024*1024 {
					t.Errorf("GPU 2 = %d, want %d", states[pgKey{1000, 2}].AllocatedBytes, uint64(3*1024*1024*1024))
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, tt.run)
	}
}

func TestParseGPUVRAMOutput(t *testing.T) {
	tests := []struct {
		name    string
		input   string
		want    map[uint32]uint64
		wantErr bool
	}{
		{
			name:  "single_gpu",
			input: "15360\n",
			want:  map[uint32]uint64{0: 15360 * 1024 * 1024},
		},
		{
			name:  "four_gpus",
			input: "15360\n15360\n15360\n15360\n",
			want: map[uint32]uint64{
				0: 15360 * 1024 * 1024,
				1: 15360 * 1024 * 1024,
				2: 15360 * 1024 * 1024,
				3: 15360 * 1024 * 1024,
			},
		},
		{
			name:  "mixed_gpu_sizes",
			input: "81920\n15360\n",
			want: map[uint32]uint64{
				0: 81920 * 1024 * 1024,
				1: 15360 * 1024 * 1024,
			},
		},
		{
			name:  "whitespace_trimmed",
			input: "  15360  \n  15360  \n",
			want: map[uint32]uint64{
				0: 15360 * 1024 * 1024,
				1: 15360 * 1024 * 1024,
			},
		},
		{
			name:    "empty_output",
			input:   "",
			wantErr: true,
		},
		{
			name:    "invalid_number",
			input:   "not_a_number\n",
			wantErr: true,
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got, err := parseGPUVRAMOutput(tt.input)
			if tt.wantErr {
				if err == nil {
					t.Errorf("expected error, got nil")
				}
				return
			}
			if err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
			if len(got) != len(tt.want) {
				t.Fatalf("gpu count = %d, want %d", len(got), len(tt.want))
			}
			for gpuID, wantVRAM := range tt.want {
				if got[gpuID] != wantVRAM {
					t.Errorf("GPU %d vram = %d, want %d", gpuID, got[gpuID], wantVRAM)
				}
			}
		})
	}
}

func TestMemoryStateJSON(t *testing.T) {
	ms := MemoryState{
		PID:            12345,
		GPUID:          2,
		AllocatedBytes: 10737418240,
		TotalVRAM:      16106127360,
		UtilizationPct: 66.7,
		LastAllocSize:  268435456,
		TimestampNs:    1711180800000000000,
	}

	data, err := json.Marshal(ms)
	if err != nil {
		t.Fatalf("json.Marshal: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("json.Unmarshal: %v", err)
	}

	// Verify gpu_id field exists and has correct value
	gpuID, ok := parsed["gpu_id"]
	if !ok {
		t.Fatal("gpu_id field missing from JSON")
	}
	if gpuID.(float64) != 2.0 {
		t.Errorf("gpu_id = %v, want 2", gpuID)
	}

	// Verify pid field
	pid, ok := parsed["pid"]
	if !ok {
		t.Fatal("pid field missing from JSON")
	}
	if pid.(float64) != 12345.0 {
		t.Errorf("pid = %v, want 12345", pid)
	}

	// Roundtrip: unmarshal back to MemoryState
	var roundtrip MemoryState
	if err := json.Unmarshal(data, &roundtrip); err != nil {
		t.Fatalf("roundtrip unmarshal: %v", err)
	}
	if roundtrip.GPUID != 2 {
		t.Errorf("roundtrip gpu_id = %d, want 2", roundtrip.GPUID)
	}
	if roundtrip.PID != 12345 {
		t.Errorf("roundtrip pid = %d, want 12345", roundtrip.PID)
	}
}
