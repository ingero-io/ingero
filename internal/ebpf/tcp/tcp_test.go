package tcp

import (
	"encoding/binary"
	"testing"
	"unsafe"

	"github.com/ingero-io/ingero/pkg/events"
)

// Compile-time size assertion: ensures bpf2go-generated struct matches 48 bytes.
//
//	struct ingero_tcp_event {
//	    struct ingero_event_hdr hdr;  // offset 0-31  (32 bytes)
//	    __u32 saddr;                 // offset 32-35
//	    __u32 daddr;                 // offset 36-39
//	    __u16 sport;                 // offset 40-41
//	    __u16 dport;                 // offset 42-43
//	    __u8  state;                 // offset 44
//	    __u8  _pad_tcp[3];          // offset 45-47
//	};                               // total: 48 bytes
var _ [48 - unsafe.Sizeof(tcpTraceIngeroTcpEvent{})]byte

// buildTCPEventBytes constructs a raw byte buffer matching the C struct layout.
func buildTCPEventBytes(tsNs uint64, pid, tid uint32, op uint8,
	saddr, daddr uint32, sport, dport uint16, state uint8, cgroupID uint64) []byte {
	buf := make([]byte, 48)
	binary.LittleEndian.PutUint64(buf[0:8], tsNs)
	binary.LittleEndian.PutUint32(buf[8:12], pid)
	binary.LittleEndian.PutUint32(buf[12:16], tid)
	buf[16] = uint8(events.SourceTCP) // source
	buf[17] = op
	// buf[18:20] = _pad (zeros)
	// buf[20:24] = _pad2 (zeros)
	binary.LittleEndian.PutUint64(buf[24:32], cgroupID)
	binary.LittleEndian.PutUint32(buf[32:36], saddr)
	binary.LittleEndian.PutUint32(buf[36:40], daddr)
	binary.LittleEndian.PutUint16(buf[40:42], sport)
	binary.LittleEndian.PutUint16(buf[42:44], dport)
	buf[44] = state
	return buf
}

func TestParseEventRetransmit(t *testing.T) {
	tsNs := uint64(1000000000)
	// 192.168.1.1 = 0xC0A80101, 10.0.0.1 = 0x0A000001
	saddr := uint32(0xC0A80101)
	daddr := uint32(0x0A000001)
	sport := uint16(12345)
	dport := uint16(443)
	state := uint8(1) // TCP_ESTABLISHED

	raw := buildTCPEventBytes(tsNs, 1234, 1235, uint8(events.TCPRetransmit),
		saddr, daddr, sport, dport, state, 42)

	tr := &Tracer{}
	evt, ok := tr.parseEvent(raw)
	if !ok {
		t.Fatal("parseEvent() returned false")
	}

	if evt.Source != events.SourceTCP {
		t.Errorf("Source = %v, want SourceTCP", evt.Source)
	}
	if evt.Op != uint8(events.TCPRetransmit) {
		t.Errorf("Op = %d, want %d (TCPRetransmit)", evt.Op, events.TCPRetransmit)
	}
	if evt.PID != 1234 {
		t.Errorf("PID = %d, want 1234", evt.PID)
	}
	if evt.TID != 1235 {
		t.Errorf("TID = %d, want 1235", evt.TID)
	}

	// Args[0] = (saddr << 32) | daddr
	wantArgs0 := uint64(saddr)<<32 | uint64(daddr)
	if evt.Args[0] != wantArgs0 {
		t.Errorf("Args[0] = 0x%x, want 0x%x", evt.Args[0], wantArgs0)
	}

	// Args[1] = (sport << 16) | dport
	wantArgs1 := uint64(sport)<<16 | uint64(dport)
	if evt.Args[1] != wantArgs1 {
		t.Errorf("Args[1] = 0x%x, want 0x%x", evt.Args[1], wantArgs1)
	}

	// GPUID stores TCP state
	if evt.GPUID != uint32(state) {
		t.Errorf("GPUID (state) = %d, want %d", evt.GPUID, state)
	}

	if evt.CGroupID != 42 {
		t.Errorf("CGroupID = %d, want 42", evt.CGroupID)
	}
}

func TestParseEventRetransmitZeroPorts(t *testing.T) {
	// Edge case: retransmit with zero ports (kernel may send this for
	// connections in TIME_WAIT or with incomplete socket state).
	raw := buildTCPEventBytes(3000000000, 100, 101, uint8(events.TCPRetransmit),
		0x7F000001, // 127.0.0.1
		0x7F000001, // 127.0.0.1
		0, 0,       // zero ports
		6, // TCP_TIME_WAIT
		0,
	)

	tr := &Tracer{}
	evt, ok := tr.parseEvent(raw)
	if !ok {
		t.Fatal("parseEvent() returned false")
	}

	if evt.Op != uint8(events.TCPRetransmit) {
		t.Errorf("Op = %d, want %d", evt.Op, events.TCPRetransmit)
	}
	// Args[1] should be 0 when both ports are zero
	if evt.Args[1] != 0 {
		t.Errorf("Args[1] = 0x%x, want 0x0 (zero ports)", evt.Args[1])
	}
	if evt.GPUID != 6 { // TCP_TIME_WAIT state
		t.Errorf("GPUID (state) = %d, want 6", evt.GPUID)
	}
}

func TestParseEventRetransmitAddressPacking(t *testing.T) {
	// Verify the (saddr << 32) | daddr packing for addresses near max uint32.
	saddr := uint32(0xFFFFFFFF) // 255.255.255.255
	daddr := uint32(0x01020304) // 1.2.3.4
	raw := buildTCPEventBytes(4000000000, 200, 201, uint8(events.TCPRetransmit),
		saddr, daddr, 8080, 80, 1, 0)

	tr := &Tracer{}
	evt, ok := tr.parseEvent(raw)
	if !ok {
		t.Fatal("parseEvent() returned false")
	}

	wantArgs0 := uint64(saddr)<<32 | uint64(daddr)
	if evt.Args[0] != wantArgs0 {
		t.Errorf("Args[0] = 0x%x, want 0x%x", evt.Args[0], wantArgs0)
	}
	wantArgs1 := uint64(8080)<<16 | uint64(80)
	if evt.Args[1] != wantArgs1 {
		t.Errorf("Args[1] = 0x%x, want 0x%x", evt.Args[1], wantArgs1)
	}
}

func TestParseEventTooShort(t *testing.T) {
	tr := &Tracer{}
	_, ok := tr.parseEvent([]byte{1, 2, 3})
	if ok {
		t.Fatal("parseEvent() should return false on short buffer")
	}
}
