package cuda

import (
	"encoding/binary"
	"testing"
	"time"

	"github.com/ingero-io/ingero/pkg/events"
)

// buildCUDAEventBytes constructs a raw byte buffer matching the C struct cuda_event layout:
//
//	struct ingero_event_hdr {   // 20 bytes
//	    __u64 timestamp_ns;     // offset 0
//	    __u32 pid;              // offset 8
//	    __u32 tid;              // offset 12
//	    __u8  source;           // offset 16
//	    __u8  op;               // offset 17
//	    __u16 _pad;             // offset 18
//	};                          // + 4 bytes alignment padding → offset 24
//	struct cuda_event {
//	    struct ingero_event_hdr hdr;  // 0-23
//	    __u64 duration_ns;           // 24
//	    __u64 arg0;                  // 32
//	    __u64 arg1;                  // 40
//	    __s32 return_code;           // 48
//	    __u32 gpu_id;                // 52
//	};                               // total: 56 bytes
func buildCUDAEventBytes(tsNs uint64, pid, tid uint32, source, op uint8,
	durationNs, arg0, arg1 uint64, retCode int32, gpuID uint32) []byte {
	buf := make([]byte, 56)
	binary.LittleEndian.PutUint64(buf[0:8], tsNs)
	binary.LittleEndian.PutUint32(buf[8:12], pid)
	binary.LittleEndian.PutUint32(buf[12:16], tid)
	buf[16] = source
	buf[17] = op
	// buf[18:24] = pad (zeros)
	binary.LittleEndian.PutUint64(buf[24:32], durationNs)
	binary.LittleEndian.PutUint64(buf[32:40], arg0)
	binary.LittleEndian.PutUint64(buf[40:48], arg1)
	binary.LittleEndian.PutUint32(buf[48:52], uint32(retCode))
	binary.LittleEndian.PutUint32(buf[52:56], gpuID)
	return buf
}

func TestParseEventCUDAMalloc(t *testing.T) {
	tsNs := uint64(1000000000)
	raw := buildCUDAEventBytes(tsNs, 1234, 1235,
		uint8(events.SourceCUDA), uint8(events.CUDAMalloc),
		5000, // 5µs
		4096, // 4KB allocation
		0,    // no arg1
		0,    // success
		0,    // gpu 0
	)

	evt, err := parseEvent(raw)
	if err != nil {
		t.Fatalf("parseEvent() error: %v", err)
	}

	if evt.Source != events.SourceCUDA {
		t.Errorf("Source = %v, want SourceCUDA", evt.Source)
	}
	if evt.Op != uint8(events.CUDAMalloc) {
		t.Errorf("Op = %d, want %d (CUDAMalloc)", evt.Op, events.CUDAMalloc)
	}
	if evt.PID != 1234 {
		t.Errorf("PID = %d, want 1234", evt.PID)
	}
	if evt.TID != 1235 {
		t.Errorf("TID = %d, want 1235", evt.TID)
	}
	if evt.Duration != 5*time.Microsecond {
		t.Errorf("Duration = %v, want 5µs", evt.Duration)
	}
	if evt.Args[0] != 4096 {
		t.Errorf("Args[0] = %d, want 4096", evt.Args[0])
	}
	if evt.RetCode != 0 {
		t.Errorf("RetCode = %d, want 0", evt.RetCode)
	}
	if evt.Stack != nil {
		t.Errorf("Stack should be nil for base event")
	}
}

func TestParseEventCUDAFree(t *testing.T) {
	raw := buildCUDAEventBytes(1000000000, 1234, 1235,
		uint8(events.SourceCUDA), uint8(events.CUDAFree),
		2000,           // 2µs
		0x7f0a00001000, // devPtr being freed
		0,              // no arg1
		0,              // success
		0,              // gpu 0
	)

	evt, err := parseEvent(raw)
	if err != nil {
		t.Fatalf("parseEvent() error: %v", err)
	}

	if evt.Op != uint8(events.CUDAFree) {
		t.Errorf("Op = %d, want %d (CUDAFree)", evt.Op, events.CUDAFree)
	}
	if evt.Args[0] != 0x7f0a00001000 {
		t.Errorf("Args[0] = %#x, want 0x7f0a00001000 (devPtr)", evt.Args[0])
	}
	if evt.Duration != 2*time.Microsecond {
		t.Errorf("Duration = %v, want 2µs", evt.Duration)
	}
}

func TestParseEventTooShort(t *testing.T) {
	_, err := parseEvent([]byte{1, 2, 3})
	if err == nil {
		t.Fatal("parseEvent() should fail on short buffer")
	}
}

func TestParseEventWithStack(t *testing.T) {
	// Build a 576-byte stack event: 56 base + 2 depth + 6 pad + 512 IPs.
	buf := buildCUDAEventBytes(2000000000, 5678, 5679,
		uint8(events.SourceCUDA), uint8(events.CUDALaunchKernel),
		10000, // 10µs
		0xDEADBEEF, 0,
		0, 0,
	)
	// Extend to 576 bytes.
	stackSection := make([]byte, 576-56)
	// stack_depth = 3 at offset 0 of stack section.
	binary.LittleEndian.PutUint16(stackSection[0:2], 3)
	// IPs start at offset 8 of stack section.
	binary.LittleEndian.PutUint64(stackSection[8:16], 0x7f0001000)
	binary.LittleEndian.PutUint64(stackSection[16:24], 0x7f0002000)
	binary.LittleEndian.PutUint64(stackSection[24:32], 0x7f0003000)
	buf = append(buf, stackSection...)

	evt, err := parseEvent(buf)
	if err != nil {
		t.Fatalf("parseEvent() error: %v", err)
	}

	if len(evt.Stack) != 3 {
		t.Fatalf("Stack length = %d, want 3", len(evt.Stack))
	}
	if evt.Stack[0].IP != 0x7f0001000 {
		t.Errorf("Stack[0].IP = 0x%x, want 0x7f0001000", evt.Stack[0].IP)
	}
	if evt.Stack[2].IP != 0x7f0003000 {
		t.Errorf("Stack[2].IP = 0x%x, want 0x7f0003000", evt.Stack[2].IP)
	}
}

func TestParseEventTruncatedStack(t *testing.T) {
	// Base event + partial stack section (too short for any IPs).
	buf := buildCUDAEventBytes(3000000000, 100, 101,
		uint8(events.SourceCUDA), uint8(events.CUDAMemcpy),
		1000, 1024, 1, // 1KB H2D copy
		0, 0,
	)
	// Only 4 extra bytes — not enough for stack section.
	buf = append(buf, 0, 0, 0, 0)

	evt, err := parseEvent(buf)
	if err != nil {
		t.Fatalf("parseEvent() error: %v", err)
	}

	// Should parse successfully but with no stack.
	if evt.Stack != nil {
		t.Errorf("Stack should be nil for truncated stack section")
	}
	if evt.Op != uint8(events.CUDAMemcpy) {
		t.Errorf("Op = %d, want %d", evt.Op, events.CUDAMemcpy)
	}
}
