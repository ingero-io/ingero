package host

import (
	"encoding/binary"
	"testing"
	"time"
	"unsafe"

	"github.com/ingero-io/ingero/pkg/events"
)

// Compile-time size assertion: ensures bpf2go-generated struct matches 48 bytes.
var _ [48 - unsafe.Sizeof(hostTraceHostEvent{})]byte

// buildHostEventBytes constructs a raw byte buffer matching the C struct host_event layout:
//
//	struct ingero_event_hdr {   // 32 bytes (v0.7)
//	    __u64 timestamp_ns;     // offset 0
//	    __u32 pid;              // offset 8
//	    __u32 tid;              // offset 12
//	    __u8  source;           // offset 16
//	    __u8  op;               // offset 17
//	    __u16 _pad;             // offset 18
//	    __u32 _pad2;            // offset 20
//	    __u64 cgroup_id;        // offset 24
//	};
//	struct host_event {
//	    struct ingero_event_hdr hdr;  // offset 0-31
//	    __u64 duration_ns;           // offset 32
//	    __u32 cpu;                   // offset 40
//	    __u32 target_pid;            // offset 44
//	};                               // total: 48 bytes
func buildHostEventBytes(tsNs uint64, pid, tid uint32, source, op uint8,
	durationNs uint64, cpu, targetPID uint32, cgroupID uint64) []byte {
	buf := make([]byte, 48)
	binary.LittleEndian.PutUint64(buf[0:8], tsNs)
	binary.LittleEndian.PutUint32(buf[8:12], pid)
	binary.LittleEndian.PutUint32(buf[12:16], tid)
	buf[16] = source
	buf[17] = op
	// buf[18:20] = _pad (zeros)
	// buf[20:24] = _pad2 (zeros)
	binary.LittleEndian.PutUint64(buf[24:32], cgroupID)
	binary.LittleEndian.PutUint64(buf[32:40], durationNs)
	binary.LittleEndian.PutUint32(buf[40:44], cpu)
	binary.LittleEndian.PutUint32(buf[44:48], targetPID)
	return buf
}

func TestParseEventSchedSwitch(t *testing.T) {
	tsNs := uint64(1000000000) // 1 second
	raw := buildHostEventBytes(tsNs, 1234, 1235, uint8(events.SourceHost), uint8(events.HostSchedSwitch),
		5000000, // 5ms off-CPU duration
		2,       // cpu
		1234,    // target_pid
		77,      // cgroup_id
	)

	evt, err := parseEvent(raw)
	if err != nil {
		t.Fatalf("parseEvent() error: %v", err)
	}

	if evt.Source != events.SourceHost {
		t.Errorf("Source = %v, want SourceHost", evt.Source)
	}
	if evt.Op != uint8(events.HostSchedSwitch) {
		t.Errorf("Op = %d, want %d (HostSchedSwitch)", evt.Op, events.HostSchedSwitch)
	}
	if evt.PID != 1234 {
		t.Errorf("PID = %d, want 1234", evt.PID)
	}
	if evt.TID != 1235 {
		t.Errorf("TID = %d, want 1235", evt.TID)
	}
	if evt.Duration != 5*time.Millisecond {
		t.Errorf("Duration = %v, want 5ms", evt.Duration)
	}
	if evt.Args[1] != 1234 {
		t.Errorf("Args[1] (target_pid) = %d, want 1234", evt.Args[1])
	}
	if evt.CGroupID != 77 {
		t.Errorf("CGroupID = %d, want 77", evt.CGroupID)
	}
	if evt.Timestamp != events.KtimeToWallClock(tsNs) {
		t.Errorf("Timestamp = %v, want %v", evt.Timestamp, events.KtimeToWallClock(tsNs))
	}
}

func TestParseEventPageAlloc(t *testing.T) {
	allocBytes := uint64(65536) // 16 pages
	raw := buildHostEventBytes(2000000000, 5678, 5679, uint8(events.SourceHost), uint8(events.HostPageAlloc),
		allocBytes, // duration_ns carries alloc_bytes for page_alloc
		0,          // cpu
		0,          // target_pid
		0,          // cgroup_id
	)

	evt, err := parseEvent(raw)
	if err != nil {
		t.Fatalf("parseEvent() error: %v", err)
	}

	if evt.Op != uint8(events.HostPageAlloc) {
		t.Errorf("Op = %d, want %d (HostPageAlloc)", evt.Op, events.HostPageAlloc)
	}
	// For page_alloc, Duration should be 0 and Args[0] should carry alloc_bytes.
	if evt.Duration != 0 {
		t.Errorf("Duration = %v, want 0 (page_alloc stores bytes in Args[0])", evt.Duration)
	}
	if evt.Args[0] != allocBytes {
		t.Errorf("Args[0] (alloc_bytes) = %d, want %d", evt.Args[0], allocBytes)
	}
}

func TestParseEventOOMKill(t *testing.T) {
	raw := buildHostEventBytes(3000000000, 100, 100, uint8(events.SourceHost), uint8(events.HostOOMKill),
		0,    // no duration
		3,    // cpu
		9999, // victim PID
		0,    // cgroup_id
	)

	evt, err := parseEvent(raw)
	if err != nil {
		t.Fatalf("parseEvent() error: %v", err)
	}

	if evt.Op != uint8(events.HostOOMKill) {
		t.Errorf("Op = %d, want %d (HostOOMKill)", evt.Op, events.HostOOMKill)
	}
	if evt.Args[1] != 9999 {
		t.Errorf("Args[1] (victim_pid) = %d, want 9999", evt.Args[1])
	}
}

func TestParseEventSchedWakeup(t *testing.T) {
	raw := buildHostEventBytes(4000000000, 200, 201, uint8(events.SourceHost), uint8(events.HostSchedWakeup),
		0,    // no duration for wakeup
		1,    // cpu
		5678, // wakee PID
		0,    // cgroup_id
	)

	evt, err := parseEvent(raw)
	if err != nil {
		t.Fatalf("parseEvent() error: %v", err)
	}

	if evt.Op != uint8(events.HostSchedWakeup) {
		t.Errorf("Op = %d, want %d (HostSchedWakeup)", evt.Op, events.HostSchedWakeup)
	}
	if evt.Duration != 0 {
		t.Errorf("Duration = %v, want 0", evt.Duration)
	}
	if evt.Args[1] != 5678 {
		t.Errorf("Args[1] (wakee_pid) = %d, want 5678", evt.Args[1])
	}
}

func TestParseEventPodRestart(t *testing.T) {
	// Pod lifecycle events are synthetic host events injected by the K8s PodCache.
	// They use the same host_event struct but with pod-specific op codes.
	raw := buildHostEventBytes(7000000000, 0, 0, uint8(events.SourceHost), uint8(events.HostPodRestart),
		0,    // no duration (point-in-time event)
		0,    // cpu
		8888, // target_pid = the pod's main container PID
		500,  // cgroup_id
	)

	evt, err := parseEvent(raw)
	if err != nil {
		t.Fatalf("parseEvent() error: %v", err)
	}

	if evt.Op != uint8(events.HostPodRestart) {
		t.Errorf("Op = %d, want %d (HostPodRestart)", evt.Op, events.HostPodRestart)
	}
	if evt.Args[1] != 8888 {
		t.Errorf("Args[1] (target_pid) = %d, want 8888", evt.Args[1])
	}
	if evt.CGroupID != 500 {
		t.Errorf("CGroupID = %d, want 500", evt.CGroupID)
	}
}

func TestParseEventPodEviction(t *testing.T) {
	raw := buildHostEventBytes(8000000000, 0, 0, uint8(events.SourceHost), uint8(events.HostPodEviction),
		0, 0,
		7777, // evicted pod's PID
		600,
	)

	evt, err := parseEvent(raw)
	if err != nil {
		t.Fatalf("parseEvent() error: %v", err)
	}

	if evt.Op != uint8(events.HostPodEviction) {
		t.Errorf("Op = %d, want %d (HostPodEviction)", evt.Op, events.HostPodEviction)
	}
	if evt.Args[1] != 7777 {
		t.Errorf("Args[1] (target_pid) = %d, want 7777", evt.Args[1])
	}
}

func TestParseEventPodOOMKill(t *testing.T) {
	raw := buildHostEventBytes(9000000000, 0, 0, uint8(events.SourceHost), uint8(events.HostPodOOMKill),
		0, 0,
		6666, // OOM-killed pod's PID
		700,
	)

	evt, err := parseEvent(raw)
	if err != nil {
		t.Fatalf("parseEvent() error: %v", err)
	}

	if evt.Op != uint8(events.HostPodOOMKill) {
		t.Errorf("Op = %d, want %d (HostPodOOMKill)", evt.Op, events.HostPodOOMKill)
	}
	if evt.Args[1] != 6666 {
		t.Errorf("Args[1] (target_pid) = %d, want 6666", evt.Args[1])
	}
	if evt.CGroupID != 700 {
		t.Errorf("CGroupID = %d, want 700", evt.CGroupID)
	}
}

func TestParseEventTooShort(t *testing.T) {
	_, err := parseEvent([]byte{1, 2, 3})
	if err == nil {
		t.Fatal("parseEvent() should fail on short buffer")
	}
}
