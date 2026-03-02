package net

import (
	"encoding/binary"
	"testing"
	"time"
	"unsafe"

	"github.com/ingero-io/ingero/pkg/events"
)

// Compile-time size assertion: ensures bpf2go-generated struct matches 56 bytes.
//
//	struct ingero_net_event {
//	    struct ingero_event_hdr hdr;  // offset 0-31  (32 bytes)
//	    __u64 duration_ns;           // offset 32-39
//	    __u32 fd;                    // offset 40-43
//	    __u32 bytes;                 // offset 44-47
//	    __u8  direction;             // offset 48
//	    __u8  _pad_net[7];          // offset 49-55
//	};                               // total: 56 bytes
var _ [56 - unsafe.Sizeof(netTraceIngeroNetEvent{})]byte

// buildNetEventBytes constructs a raw byte buffer matching the C struct layout.
func buildNetEventBytes(tsNs uint64, pid, tid uint32, op uint8,
	durationNs uint64, fd, bytesTransferred uint32, direction uint8, cgroupID uint64) []byte {
	buf := make([]byte, 56)
	binary.LittleEndian.PutUint64(buf[0:8], tsNs)
	binary.LittleEndian.PutUint32(buf[8:12], pid)
	binary.LittleEndian.PutUint32(buf[12:16], tid)
	buf[16] = uint8(events.SourceNet) // source
	buf[17] = op
	// buf[18:20] = _pad (zeros)
	// buf[20:24] = _pad2 (zeros)
	binary.LittleEndian.PutUint64(buf[24:32], cgroupID)
	binary.LittleEndian.PutUint64(buf[32:40], durationNs)
	binary.LittleEndian.PutUint32(buf[40:44], fd)
	binary.LittleEndian.PutUint32(buf[44:48], bytesTransferred)
	buf[48] = direction
	return buf
}

func TestParseEventNetSend(t *testing.T) {
	tsNs := uint64(1000000000)
	raw := buildNetEventBytes(tsNs, 1234, 1235, uint8(events.NetSend),
		500000, // 500us duration
		5,      // fd
		4096,   // bytes sent
		uint8(events.NetSend),
		88, // cgroup_id
	)

	tr := &Tracer{}
	evt, ok := tr.parseEvent(raw)
	if !ok {
		t.Fatal("parseEvent() returned false")
	}

	if evt.Source != events.SourceNet {
		t.Errorf("Source = %v, want SourceNet", evt.Source)
	}
	if evt.Op != uint8(events.NetSend) {
		t.Errorf("Op = %d, want %d (NetSend)", evt.Op, events.NetSend)
	}
	if evt.PID != 1234 {
		t.Errorf("PID = %d, want 1234", evt.PID)
	}
	if evt.TID != 1235 {
		t.Errorf("TID = %d, want 1235", evt.TID)
	}
	if evt.Duration != 500*time.Microsecond {
		t.Errorf("Duration = %v, want 500us", evt.Duration)
	}
	if evt.Args[0] != 5 { // fd
		t.Errorf("Args[0] (fd) = %d, want 5", evt.Args[0])
	}
	if evt.Args[1] != 4096 { // bytes
		t.Errorf("Args[1] (bytes) = %d, want 4096", evt.Args[1])
	}
	if evt.CGroupID != 88 {
		t.Errorf("CGroupID = %d, want 88", evt.CGroupID)
	}
}

func TestParseEventNetRecv(t *testing.T) {
	raw := buildNetEventBytes(2000000000, 5678, 5679, uint8(events.NetRecv),
		1000000, // 1ms
		10,      // fd
		8192,    // bytes received
		uint8(events.NetRecv),
		0,
	)

	tr := &Tracer{}
	evt, ok := tr.parseEvent(raw)
	if !ok {
		t.Fatal("parseEvent() returned false")
	}

	if evt.Op != uint8(events.NetRecv) {
		t.Errorf("Op = %d, want %d (NetRecv)", evt.Op, events.NetRecv)
	}
	if evt.Duration != time.Millisecond {
		t.Errorf("Duration = %v, want 1ms", evt.Duration)
	}
	if evt.Args[1] != 8192 {
		t.Errorf("Args[1] (bytes) = %d, want 8192", evt.Args[1])
	}
}

func TestParseEventNetZeroBytes(t *testing.T) {
	// Edge case: sendto/recvfrom with 0 bytes (e.g., probing, keepalive).
	raw := buildNetEventBytes(5000000000, 300, 301, uint8(events.NetSend),
		100000, // 100µs
		7,      // fd
		0,      // zero bytes
		uint8(events.NetSend),
		0,
	)

	tr := &Tracer{}
	evt, ok := tr.parseEvent(raw)
	if !ok {
		t.Fatal("parseEvent() returned false")
	}

	if evt.Op != uint8(events.NetSend) {
		t.Errorf("Op = %d, want %d", evt.Op, events.NetSend)
	}
	if evt.Args[1] != 0 {
		t.Errorf("Args[1] (bytes) = %d, want 0", evt.Args[1])
	}
	if evt.Duration != 100*time.Microsecond {
		t.Errorf("Duration = %v, want 100µs", evt.Duration)
	}
}

func TestParseEventNetLargeTransfer(t *testing.T) {
	// Large transfer: 1 GiB in a single recv call (e.g., NCCL collective).
	bigBytes := uint32(1 << 30) // 1 GiB
	raw := buildNetEventBytes(6000000000, 400, 401, uint8(events.NetRecv),
		50_000_000, // 50ms duration
		42,         // fd
		bigBytes,
		uint8(events.NetRecv),
		123,
	)

	tr := &Tracer{}
	evt, ok := tr.parseEvent(raw)
	if !ok {
		t.Fatal("parseEvent() returned false")
	}

	if evt.Op != uint8(events.NetRecv) {
		t.Errorf("Op = %d, want %d (NetRecv)", evt.Op, events.NetRecv)
	}
	if evt.Args[1] != uint64(bigBytes) {
		t.Errorf("Args[1] (bytes) = %d, want %d", evt.Args[1], bigBytes)
	}
	if evt.Duration != 50*time.Millisecond {
		t.Errorf("Duration = %v, want 50ms", evt.Duration)
	}
	if evt.CGroupID != 123 {
		t.Errorf("CGroupID = %d, want 123", evt.CGroupID)
	}
}

func TestParseEventTooShort(t *testing.T) {
	tr := &Tracer{}
	_, ok := tr.parseEvent([]byte{1, 2, 3})
	if ok {
		t.Fatal("parseEvent() should return false on short buffer")
	}
}
