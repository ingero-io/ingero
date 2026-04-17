package cuda

import (
	"encoding/binary"
	"testing"
	"time"
	"unsafe"

	"github.com/ingero-io/ingero/pkg/events"
)

// Compile-time size assertion: ensures the bpf2go-generated struct matches
// the expected 80 bytes (v0.10: 48-byte header with cgroup_id+comm + 32 bytes payload).
// Fails compilation immediately if the struct size changes, preventing silent
// misparsing of ring buffer events.
var _ [80 - unsafe.Sizeof(cudaTraceCudaEvent{})]byte

// Compile-time size assertion: ingero_config is the value type of config_map.
// If the C struct is appended (e.g., new sampling knob) without regenerating
// the BPF objects via `make generate`, the kernel-side map's value_size stays
// at the old size while this Go struct grows, producing the runtime error:
//
//	marshal value: []uint8 doesn't marshal to N bytes
//
// on the first ConfigMap.Put(). This assertion catches the drift at compile time.
var _ [12 - unsafe.Sizeof(cudaTraceIngeroConfig{})]byte

// buildCUDAEventBytes constructs a raw byte buffer matching the C struct cuda_event layout:
//
//	struct ingero_event_hdr {   // 48 bytes (v0.10)
//	    __u64 timestamp_ns;     // offset 0
//	    __u32 pid;              // offset 8
//	    __u32 tid;              // offset 12
//	    __u8  source;           // offset 16
//	    __u8  op;               // offset 17
//	    __u16 _pad;             // offset 18
//	    __u32 _pad2;            // offset 20 (explicit alignment padding)
//	    __u64 cgroup_id;        // offset 24
//	    char  comm[16];         // offset 32 (v0.10 PID hardening)
//	};
//	struct cuda_event {
//	    struct ingero_event_hdr hdr;  // 0-47
//	    __u64 duration_ns;           // 48
//	    __u64 arg0;                  // 56
//	    __u64 arg1;                  // 64
//	    __s32 return_code;           // 72
//	    __u32 gpu_id;                // 76
//	};                               // total: 80 bytes
//
// comm is a Go string copied into the 16-byte field, NUL-padded; longer
// strings are truncated to TASK_COMM_LEN-1 with a trailing NUL terminator
// (matching the Linux kernel's invariant for comm).
func buildCUDAEventBytes(tsNs uint64, pid, tid uint32, source, op uint8,
	durationNs, arg0, arg1 uint64, retCode int32, gpuID uint32, cgroupID uint64, comm string) []byte {
	buf := make([]byte, 80)
	binary.LittleEndian.PutUint64(buf[0:8], tsNs)
	binary.LittleEndian.PutUint32(buf[8:12], pid)
	binary.LittleEndian.PutUint32(buf[12:16], tid)
	buf[16] = source
	buf[17] = op
	// buf[18:20] = _pad (zeros)
	// buf[20:24] = _pad2 (zeros)
	binary.LittleEndian.PutUint64(buf[24:32], cgroupID)
	// buf[32:48] = comm[16] — NUL-padded; truncated to 15 chars + NUL if too long.
	commBytes := []byte(comm)
	if len(commBytes) > 15 {
		commBytes = commBytes[:15]
	}
	copy(buf[32:48], commBytes)
	binary.LittleEndian.PutUint64(buf[48:56], durationNs)
	binary.LittleEndian.PutUint64(buf[56:64], arg0)
	binary.LittleEndian.PutUint64(buf[64:72], arg1)
	binary.LittleEndian.PutUint32(buf[72:76], uint32(retCode))
	binary.LittleEndian.PutUint32(buf[76:80], gpuID)
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
		42,   // cgroup_id
		"python3", // comm
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
	if evt.CGroupID != 42 {
		t.Errorf("CGroupID = %d, want 42", evt.CGroupID)
	}
	if evt.Comm != "python3" {
		t.Errorf("Comm = %q, want %q", evt.Comm, "python3")
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
		0,              // bare-metal (no cgroup)
		"",             // comm — exercises empty-comm tolerance (BPF edge case)
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
	if evt.CGroupID != 0 {
		t.Errorf("CGroupID = %d, want 0", evt.CGroupID)
	}
	if evt.Comm != "" {
		t.Errorf("Comm = %q, want empty string", evt.Comm)
	}
}

func TestParseEventTooShort(t *testing.T) {
	_, err := parseEvent([]byte{1, 2, 3})
	if err == nil {
		t.Fatal("parseEvent() should fail on short buffer")
	}
}

func TestParseEventWithStack(t *testing.T) {
	// Build a 600-byte stack event: 80 base (v0.10 with comm) + 2 depth + 6 pad + 512 IPs.
	buf := buildCUDAEventBytes(2000000000, 5678, 5679,
		uint8(events.SourceCUDA), uint8(events.CUDALaunchKernel),
		10000, // 10µs
		0xDEADBEEF, 0,
		0, 0,
		99,         // cgroup_id
		"workload", // comm
	)
	// Extend to 600 bytes (80 base + 520 stack section).
	stackSection := make([]byte, 600-80)
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
	if evt.CGroupID != 99 {
		t.Errorf("CGroupID = %d, want 99", evt.CGroupID)
	}
}

func TestParseEventTruncatedStack(t *testing.T) {
	// Base event + partial stack section (too short for any IPs).
	buf := buildCUDAEventBytes(3000000000, 100, 101,
		uint8(events.SourceCUDA), uint8(events.CUDAMemcpy),
		1000, 1024, 1, // 1KB H2D copy
		0, 0,
		0,  // cgroup_id
		"", // comm
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
