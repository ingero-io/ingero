package blockio

import (
	"encoding/binary"
	"testing"
	"time"
	"unsafe"

	"github.com/ingero-io/ingero/pkg/events"
)

// Compile-time size assertion: ensures bpf2go-generated struct matches 64 bytes.
//
//	struct ingero_io_event {
//	    struct ingero_event_hdr hdr;  // offset 0-31  (32 bytes)
//	    __u64 duration_ns;           // offset 32-39
//	    __u32 dev;                   // offset 40-43
//	    __u32 nr_sector;             // offset 44-47
//	    __u64 sector;                // offset 48-55
//	    __u8  rwbs;                  // offset 56
//	    __u8  _pad_io[7];           // offset 57-63
//	};                               // total: 64 bytes
var _ [64 - unsafe.Sizeof(ioTraceIngeroIoEvent{})]byte

// buildIOEventBytes constructs a raw byte buffer matching the C struct layout.
func buildIOEventBytes(tsNs uint64, pid, tid uint32, op uint8,
	durationNs uint64, dev, nrSector uint32, sector uint64, rwbs uint8, cgroupID uint64) []byte {
	buf := make([]byte, 64)
	binary.LittleEndian.PutUint64(buf[0:8], tsNs)
	binary.LittleEndian.PutUint32(buf[8:12], pid)
	binary.LittleEndian.PutUint32(buf[12:16], tid)
	buf[16] = uint8(events.SourceIO) // source
	buf[17] = op
	// buf[18:20] = _pad (zeros)
	// buf[20:24] = _pad2 (zeros)
	binary.LittleEndian.PutUint64(buf[24:32], cgroupID)
	binary.LittleEndian.PutUint64(buf[32:40], durationNs)
	binary.LittleEndian.PutUint32(buf[40:44], dev)
	binary.LittleEndian.PutUint32(buf[44:48], nrSector)
	binary.LittleEndian.PutUint64(buf[48:56], sector)
	buf[56] = rwbs
	return buf
}

func TestParseEventBlockRead(t *testing.T) {
	tsNs := uint64(1000000000)
	raw := buildIOEventBytes(tsNs, 1234, 1235, uint8(events.IORead),
		2000000, // 2ms duration
		0x0800,  // dev (/dev/sda)
		16,      // 16 sectors (8KB)
		1024,    // starting sector
		1,       // rwbs = read
		99,      // cgroup_id
	)

	tr := &Tracer{}
	evt, ok := tr.parseEvent(raw)
	if !ok {
		t.Fatal("parseEvent() returned false")
	}

	if evt.Source != events.SourceIO {
		t.Errorf("Source = %v, want SourceIO", evt.Source)
	}
	if evt.Op != uint8(events.IORead) {
		t.Errorf("Op = %d, want %d (IORead)", evt.Op, events.IORead)
	}
	if evt.PID != 1234 {
		t.Errorf("PID = %d, want 1234", evt.PID)
	}
	if evt.TID != 1235 {
		t.Errorf("TID = %d, want 1235", evt.TID)
	}
	if evt.Duration != 2*time.Millisecond {
		t.Errorf("Duration = %v, want 2ms", evt.Duration)
	}
	if evt.Args[0] != 16 { // nr_sector
		t.Errorf("Args[0] (nr_sector) = %d, want 16", evt.Args[0])
	}
	if evt.Args[1] != 1024 { // sector
		t.Errorf("Args[1] (sector) = %d, want 1024", evt.Args[1])
	}
	if evt.GPUID != 0x0800 { // dev
		t.Errorf("GPUID (dev) = %d, want 0x0800", evt.GPUID)
	}
	if evt.CGroupID != 99 {
		t.Errorf("CGroupID = %d, want 99", evt.CGroupID)
	}
}

func TestParseEventBlockWrite(t *testing.T) {
	raw := buildIOEventBytes(2000000000, 5678, 5679, uint8(events.IOWrite),
		10000000, // 10ms
		0x0800,
		256, // 256 sectors (128KB)
		8192,
		2, // rwbs = write
		0,
	)

	tr := &Tracer{}
	evt, ok := tr.parseEvent(raw)
	if !ok {
		t.Fatal("parseEvent() returned false")
	}

	if evt.Op != uint8(events.IOWrite) {
		t.Errorf("Op = %d, want %d (IOWrite)", evt.Op, events.IOWrite)
	}
	if evt.Duration != 10*time.Millisecond {
		t.Errorf("Duration = %v, want 10ms", evt.Duration)
	}
}

func TestParseEventTooShort(t *testing.T) {
	tr := &Tracer{}
	_, ok := tr.parseEvent([]byte{1, 2, 3})
	if ok {
		t.Fatal("parseEvent() should return false on short buffer")
	}
}
