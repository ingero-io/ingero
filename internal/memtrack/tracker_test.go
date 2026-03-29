package memtrack

import (
	"testing"
	"time"

	"github.com/ingero-io/ingero/pkg/events"
)

// makeCUDAEvent creates a CUDA Runtime API event with the given PID, op, and arg0/arg1.
// For cudaMalloc: arg0 = allocation size, arg1 = devPtr param address.
// For cudaFree: arg0 = device pointer, arg1 = freed size in bytes (0 if unknown).
func makeCUDAEvent(pid uint32, op events.CUDAOp, arg0 uint64) events.Event {
	return events.Event{
		Timestamp: time.Unix(0, 1711180800000000000), // fixed for determinism
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
		Timestamp: time.Unix(0, 1711180800000000000),
		PID:       pid,
		TID:       pid,
		Source:    events.SourceCUDA,
		Op:        uint8(events.CUDAFree),
		Args:      [2]uint64{devPtr, freedBytes},
	}
}

// makeDriverEvent creates a CUDA Driver API event with the given PID, op, and size in Args[0].
func makeDriverEvent(pid uint32, op events.DriverOp, size uint64) events.Event {
	return events.Event{
		Timestamp: time.Unix(0, 1711180800000000000),
		PID:       pid,
		TID:       pid,
		Source:    events.SourceDriver,
		Op:        uint8(op),
		Args:      [2]uint64{size, 0},
	}
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
				tr := NewTracker(16*1024*1024*1024, sink)

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
				tr := NewTracker(16*1024*1024*1024, sink)

				tr.RecordMalloc(1000, 1000, 1711180800000000000)
				tr.RecordFree(1000, 400, 1711180800000000000)

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
				tr := NewTracker(1000, sink)

				tr.RecordMalloc(1000, 625, 1711180800000000000)

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
				tr := NewTracker(16*1024*1024*1024, sink)

				tr.RecordFree(9999, 500*1024*1024, 1711180800000000000)

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
				tr := NewTracker(16*1024*1024*1024, sink)

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
				tr := NewTracker(totalVRAM, sink)

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
				tr := NewTracker(16*1024*1024*1024, sink)

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
				tr := NewTracker(16*1024*1024*1024, sink)

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
				tr := NewTracker(16*1024*1024*1024, sink)

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
				tr := NewTracker(16*1024*1024*1024, sink)

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
				tr := NewTracker(16*1024*1024*1024, nil)

				// Must not panic
				tr.ProcessEvent(makeCUDAEvent(1000, events.CUDAMalloc, 256*1024*1024))
				tr.ProcessEvent(makeCUDAFreeEvent(1000, 0x7f0000, 0))
				tr.RecordMalloc(1000, 100, 1711180800000000000)
				tr.RecordFree(1000, 50, 1711180800000000000)
			},
		},
		// --- Net balance tests (alloc/free tracking) ---
		{
			name: "net_balance/alloc_then_free_returns_to_zero",
			run: func(t *testing.T) {
				var last MemoryState
				sink := func(ms MemoryState) { last = ms }
				tr := NewTracker(16*1024*1024*1024, sink)

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
				tr := NewTracker(16*1024*1024*1024, sink)

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
				tr := NewTracker(16*1024*1024*1024, sink)

				allocSize := uint64(1024 * 1024) // 1MB
				tr.ProcessEvent(makeCUDAEvent(1000, events.CUDAMalloc, allocSize))
				// Free with Args[1]=0 (unknown pointer, not in alloc_sizes map)
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
				tr := NewTracker(16*1024*1024*1024, sink)

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
				tr := NewTracker(16*1024*1024*1024, sink)

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
				tr := NewTracker(totalVRAM, sink)

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
				tr := NewTracker(totalVRAM, sink)

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
	}

	for _, tt := range tests {
		t.Run(tt.name, tt.run)
	}
}
