package memfrag

import (
	"bytes"
	"encoding/binary"
	"testing"
	"unsafe"
)

// EventSize must match the C struct exactly so binary.Read parses
// without drift. v0.15 item K.
func TestEventSizeMatchesC(t *testing.T) {
	if got, want := unsafe.Sizeof(Event{}), uintptr(EventSize); got != want {
		t.Errorf("Event size = %d, want %d (must match struct memfrag_ioctl_event in bpf/memfrag_kprobe.bpf.c)", got, want)
	}
}

func TestNewReturnsTracer(t *testing.T) {
	tr := New()
	if tr == nil {
		t.Fatal("New() returned nil")
	}
	if tr.Events() == nil {
		t.Errorf("Events channel must be non-nil")
	}
}

func TestHandleRawSample_Forwards(t *testing.T) {
	tr := New()
	in := Event{TimestampNs: 1000, CgroupID: 42, TID: 101, PID: 100, Cmd: 0xC0184601}
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, &in)
	tr.handleRawSample(buf.Bytes())

	select {
	case got := <-tr.Events():
		if got.TimestampNs != in.TimestampNs || got.PID != in.PID || got.Cmd != in.Cmd {
			t.Errorf("forwarded event = %+v, want %+v", got, in)
		}
	default:
		t.Fatal("expected event forwarded")
	}
	if tr.ParseErrors() != 0 {
		t.Errorf("no parse errors expected; got %d", tr.ParseErrors())
	}
}

func TestHandleRawSample_TooShortIncrementsParseErr(t *testing.T) {
	tr := New()
	tr.handleRawSample([]byte{0x01, 0x02, 0x03}) // < EventSize
	if got := tr.ParseErrors(); got != 1 {
		t.Errorf("parseErr=%d, want 1", got)
	}
	select {
	case e := <-tr.Events():
		t.Errorf("no event should be forwarded on short sample; got %+v", e)
	default:
	}
}

func TestHandleRawSample_DropsWhenChannelFull(t *testing.T) {
	tr := New()
	in := Event{Cmd: 1}
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, &in)
	raw := buf.Bytes()

	for i := 0; i < 4096; i++ {
		tr.handleRawSample(raw)
	}
	if tr.Dropped() != 0 {
		t.Errorf("expected 0 drops at exactly cap; got %d", tr.Dropped())
	}
	tr.handleRawSample(raw)
	if tr.Dropped() != 1 {
		t.Errorf("expected 1 drop after cap+1; got %d", tr.Dropped())
	}
}

func TestClose_IdempotentBeforeAttach(t *testing.T) {
	tr := New()
	if err := tr.Close(); err != nil {
		t.Errorf("first Close: %v", err)
	}
	if err := tr.Close(); err != nil {
		t.Errorf("second Close (idempotent): %v", err)
	}
}

func TestEventBinaryReadDoesNotPanic(t *testing.T) {
	// Defensive: Go 1.26 reflect rules can panic on unexported padding
	// fields. Our Pad0 is exported; this test would catch a regression
	// where a future field is accidentally lowercased.
	var e Event
	defer func() {
		if r := recover(); r != nil {
			t.Errorf("binary.Read into Event panicked: %v", r)
		}
	}()
	buf := make([]byte, EventSize)
	_ = binary.Read(bytes.NewReader(buf), binary.LittleEndian, &e)
}
