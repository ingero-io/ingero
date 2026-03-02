package driver

import (
	"encoding/binary"
	"testing"
	"time"
	"unsafe"

	"github.com/ingero-io/ingero/pkg/events"
)

// Compile-time size assertion: ensures bpf2go-generated struct matches 64 bytes.
var _ [64 - unsafe.Sizeof(driverTraceCudaEvent{})]byte

// buildDriverEventBytes constructs a raw byte buffer matching the C struct cuda_event layout
// (reused for driver events with source=EVENT_SRC_DRIVER). Same 64-byte layout as CUDA events.
func buildDriverEventBytes(tsNs uint64, pid, tid uint32, source, op uint8,
	durationNs, arg0, arg1 uint64, retCode int32, gpuID uint32, cgroupID uint64) []byte {
	buf := make([]byte, 64)
	binary.LittleEndian.PutUint64(buf[0:8], tsNs)
	binary.LittleEndian.PutUint32(buf[8:12], pid)
	binary.LittleEndian.PutUint32(buf[12:16], tid)
	buf[16] = source
	buf[17] = op
	// buf[18:24] = padding (zeros)
	binary.LittleEndian.PutUint64(buf[24:32], cgroupID)
	binary.LittleEndian.PutUint64(buf[32:40], durationNs)
	binary.LittleEndian.PutUint64(buf[40:48], arg0)
	binary.LittleEndian.PutUint64(buf[48:56], arg1)
	binary.LittleEndian.PutUint32(buf[56:60], uint32(retCode))
	binary.LittleEndian.PutUint32(buf[60:64], gpuID)
	return buf
}

func TestParseEventDriverLaunchKernel(t *testing.T) {
	tsNs := uint64(1000000000)
	raw := buildDriverEventBytes(tsNs, 1234, 1235,
		uint8(events.SourceDriver), uint8(events.DriverLaunchKernel),
		8000,       // 8µs
		0xCAFEBABE, // function handle
		0,
		0, 0,
		55, // cgroup_id
	)

	evt, err := parseEvent(raw)
	if err != nil {
		t.Fatalf("parseEvent() error: %v", err)
	}

	if evt.Source != events.SourceDriver {
		t.Errorf("Source = %v, want SourceDriver", evt.Source)
	}
	if evt.Op != uint8(events.DriverLaunchKernel) {
		t.Errorf("Op = %d, want %d (DriverLaunchKernel)", evt.Op, events.DriverLaunchKernel)
	}
	if evt.PID != 1234 {
		t.Errorf("PID = %d, want 1234", evt.PID)
	}
	if evt.Duration != 8*time.Microsecond {
		t.Errorf("Duration = %v, want 8µs", evt.Duration)
	}
	if evt.Args[0] != 0xCAFEBABE {
		t.Errorf("Args[0] = 0x%x, want 0xCAFEBABE", evt.Args[0])
	}
	if evt.CGroupID != 55 {
		t.Errorf("CGroupID = %d, want 55", evt.CGroupID)
	}
}

func TestParseEventDriverTooShort(t *testing.T) {
	_, err := parseEvent([]byte{1, 2, 3})
	if err == nil {
		t.Fatal("parseEvent() should fail on short buffer")
	}
}

func TestParseEventDriverWithStack(t *testing.T) {
	buf := buildDriverEventBytes(2000000000, 5678, 5679,
		uint8(events.SourceDriver), uint8(events.DriverMemcpy),
		15000, 65536, 0,
		0, 0,
		0, // cgroup_id
	)
	stackSection := make([]byte, 584-64)
	binary.LittleEndian.PutUint16(stackSection[0:2], 2)
	binary.LittleEndian.PutUint64(stackSection[8:16], 0x7f1001000)
	binary.LittleEndian.PutUint64(stackSection[16:24], 0x7f1002000)
	buf = append(buf, stackSection...)

	evt, err := parseEvent(buf)
	if err != nil {
		t.Fatalf("parseEvent() error: %v", err)
	}

	if len(evt.Stack) != 2 {
		t.Fatalf("Stack length = %d, want 2", len(evt.Stack))
	}
	if evt.Stack[0].IP != 0x7f1001000 {
		t.Errorf("Stack[0].IP = 0x%x, want 0x7f1001000", evt.Stack[0].IP)
	}
}

func TestParseEventDriverMemAllocManaged(t *testing.T) {
	tsNs := uint64(5000000000) // 5s
	allocSize := uint64(1 << 20) // 1 MiB unified memory
	raw := buildDriverEventBytes(tsNs, 4000, 4001,
		uint8(events.SourceDriver), uint8(events.DriverMemAllocManaged),
		120000,    // 120µs duration
		allocSize, // arg0 = allocation size in bytes
		0,
		0, 1, // gpuID = 1
		99, // cgroup_id
	)

	evt, err := parseEvent(raw)
	if err != nil {
		t.Fatalf("parseEvent() error: %v", err)
	}

	if evt.Source != events.SourceDriver {
		t.Errorf("Source = %v, want SourceDriver", evt.Source)
	}
	if evt.Op != uint8(events.DriverMemAllocManaged) {
		t.Errorf("Op = %d, want %d (DriverMemAllocManaged)", evt.Op, events.DriverMemAllocManaged)
	}
	if evt.PID != 4000 {
		t.Errorf("PID = %d, want 4000", evt.PID)
	}
	if evt.Duration != 120*time.Microsecond {
		t.Errorf("Duration = %v, want 120µs", evt.Duration)
	}
	if evt.Args[0] != allocSize {
		t.Errorf("Args[0] (alloc size) = %d, want %d", evt.Args[0], allocSize)
	}
	if evt.GPUID != 1 {
		t.Errorf("GPUID = %d, want 1", evt.GPUID)
	}
	if evt.CGroupID != 99 {
		t.Errorf("CGroupID = %d, want 99", evt.CGroupID)
	}
}

func TestParseEventDriverTruncatedStack(t *testing.T) {
	buf := buildDriverEventBytes(3000000000, 100, 101,
		uint8(events.SourceDriver), uint8(events.DriverMemcpyAsync),
		500, 2048, 0,
		0, 0,
		0, // cgroup_id
	)
	buf = append(buf, 0, 0, 0, 0)

	evt, err := parseEvent(buf)
	if err != nil {
		t.Fatalf("parseEvent() error: %v", err)
	}

	if evt.Stack != nil {
		t.Errorf("Stack should be nil for truncated stack section")
	}
	if evt.Op != uint8(events.DriverMemcpyAsync) {
		t.Errorf("Op = %d, want %d", evt.Op, events.DriverMemcpyAsync)
	}
}
