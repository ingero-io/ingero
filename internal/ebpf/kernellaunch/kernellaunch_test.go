package kernellaunch

import (
	"bytes"
	"encoding/binary"
	"testing"
	"unsafe"
)

func TestEventSizeMatchesC(t *testing.T) {
	if got, want := unsafe.Sizeof(Event{}), uintptr(EventSize); got != want {
		t.Errorf("Event size = %d, want %d (must match struct kernel_launch_event in bpf/kernel_launch.bpf.c)", got, want)
	}
}

func TestThreadsPerBlock_Defaults(t *testing.T) {
	cases := []struct {
		name string
		ev   Event
		want uint64
	}{
		{"1D block", Event{BlockX: 256}, 256},
		{"2D block", Event{BlockX: 16, BlockY: 16}, 256},
		{"unknown (BlockX=0)", Event{}, 0},
		{"BlockY zero treated as 1", Event{BlockX: 128, BlockY: 0}, 128},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.ev.ThreadsPerBlock(); got != c.want {
				t.Errorf("got %d, want %d", got, c.want)
			}
		})
	}
}

func TestTotalGridBlocks(t *testing.T) {
	cases := []struct {
		name string
		ev   Event
		want uint64
	}{
		{"1D grid", Event{GridX: 100}, 100},
		{"2D grid", Event{GridX: 10, GridY: 10}, 100},
		{"3D grid", Event{GridX: 4, GridY: 4, GridZ: 4}, 64},
		{"missing dims default to 1", Event{GridX: 7}, 7},
	}
	for _, c := range cases {
		t.Run(c.name, func(t *testing.T) {
			if got := c.ev.TotalGridBlocks(); got != c.want {
				t.Errorf("got %d, want %d", got, c.want)
			}
		})
	}
}

func TestNew(t *testing.T) {
	tr := New("/usr/lib/x86_64-linux-gnu/libcuda.so.1")
	if tr == nil {
		t.Fatal("New returned nil")
	}
	if tr.Events() == nil {
		t.Errorf("Events channel must be non-nil")
	}
}

func TestAttach_EmptyPathRejected(t *testing.T) {
	tr := New("")
	if err := tr.Attach(); err == nil {
		t.Errorf("empty libcuda path should reject")
	}
}

func TestHandleRawSample_Forwards(t *testing.T) {
	tr := New("/dev/null")
	in := Event{
		TimestampNs: 1000,
		FuncHandle:  0xDEADBEEF,
		PID:         100,
		GridX:       10,
		GridY:       20,
		GridZ:       1,
		BlockX:      256,
		BlockY:      1,
	}
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, &in)
	tr.handleRawSample(buf.Bytes())

	select {
	case got := <-tr.Events():
		if got.GridX != 10 || got.BlockX != 256 || got.FuncHandle != 0xDEADBEEF {
			t.Errorf("forwarded event = %+v, want %+v", got, in)
		}
	default:
		t.Fatal("expected event forwarded")
	}
}

func TestHandleRawSample_TooShortIncrementsParseErr(t *testing.T) {
	tr := New("/dev/null")
	tr.handleRawSample([]byte{0x01, 0x02})
	if got := tr.ParseErrors(); got != 1 {
		t.Errorf("parseErr=%d, want 1", got)
	}
}

func TestHandleRawSample_DropsWhenChannelFull(t *testing.T) {
	tr := New("/dev/null")
	in := Event{BlockX: 1}
	var buf bytes.Buffer
	_ = binary.Write(&buf, binary.LittleEndian, &in)
	for i := 0; i < 4096; i++ {
		tr.handleRawSample(buf.Bytes())
	}
	if tr.Dropped() != 0 {
		t.Errorf("expected 0 drops at exactly cap; got %d", tr.Dropped())
	}
	tr.handleRawSample(buf.Bytes())
	if tr.Dropped() != 1 {
		t.Errorf("expected 1 drop after cap+1; got %d", tr.Dropped())
	}
}

func TestClose_IdempotentBeforeAttach(t *testing.T) {
	tr := New("/dev/null")
	if err := tr.Close(); err != nil {
		t.Errorf("first Close: %v", err)
	}
	if err := tr.Close(); err != nil {
		t.Errorf("second Close (idempotent): %v", err)
	}
}
