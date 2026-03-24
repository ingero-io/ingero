package memtrack

import (
	"testing"
	"time"

	"github.com/ingero-io/ingero/pkg/events"
)

// makeCUDAEvent creates a CUDA event with the given PID, op, and size in Args[0].
func makeCUDAEvent(pid uint32, op events.CUDAOp, size uint64) events.Event {
	return events.Event{
		Timestamp: time.Unix(0, 1711180800000000000), // fixed for determinism
		PID:       pid,
		TID:       pid,
		Source:    events.SourceCUDA,
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
			name: "no_sink_no_panic",
			run: func(t *testing.T) {
				tr := NewTracker(16*1024*1024*1024, nil)

				// Must not panic
				tr.ProcessEvent(makeCUDAEvent(1000, events.CUDAMalloc, 256*1024*1024))
				tr.ProcessEvent(makeCUDAEvent(1000, events.CUDAFree, 0))
				tr.RecordMalloc(1000, 100, 1711180800000000000)
				tr.RecordFree(1000, 50, 1711180800000000000)
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, tt.run)
	}
}
