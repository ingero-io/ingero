package cudagraph

import (
	"encoding/binary"
	"testing"

	"github.com/ingero-io/ingero/pkg/events"
)

// buildGraphEventBytes constructs a raw 72-byte cuda_graph_event for testing.
func buildGraphEventBytes(source uint8, op uint8, durationNs uint64,
	stream, graph, exec uint64, captureMode uint32, retCode int32) []byte {

	buf := make([]byte, 72)

	// Header (32 bytes): timestamp(8), pid(4), tid(4), source(1), op(1), pad(2), pad2(4), cgroup(8)
	binary.LittleEndian.PutUint64(buf[0:8], 1000000) // timestamp_ns
	binary.LittleEndian.PutUint32(buf[8:12], 1234)    // pid
	binary.LittleEndian.PutUint32(buf[12:16], 5678)   // tid
	buf[16] = source                                   // source
	buf[17] = op                                       // op
	// pad[18:20], pad2[20:24] = 0
	binary.LittleEndian.PutUint64(buf[24:32], 42) // cgroup_id

	// Body
	binary.LittleEndian.PutUint64(buf[32:40], durationNs)
	binary.LittleEndian.PutUint64(buf[40:48], stream)
	binary.LittleEndian.PutUint64(buf[48:56], graph)
	binary.LittleEndian.PutUint64(buf[56:64], exec)
	binary.LittleEndian.PutUint32(buf[64:68], captureMode)
	binary.LittleEndian.PutUint32(buf[68:72], uint32(retCode))

	return buf
}

func TestParseGraphEvent_BeginCapture(t *testing.T) {
	raw := buildGraphEventBytes(
		uint8(events.SourceCUDAGraph), uint8(events.GraphBeginCapture),
		500000, // 500us duration
		0xABCD, // stream
		0, 0,   // no graph/exec
		1, // thread_local capture mode
		0, // success
	)

	evt, err := parseGraphEvent(raw)
	if err != nil {
		t.Fatalf("parseGraphEvent: %v", err)
	}

	if evt.Source != events.SourceCUDAGraph {
		t.Errorf("Source = %v, want SourceCUDAGraph", evt.Source)
	}
	if events.CUDAGraphOp(evt.Op) != events.GraphBeginCapture {
		t.Errorf("Op = %d, want GraphBeginCapture(%d)", evt.Op, events.GraphBeginCapture)
	}
	if evt.PID != 1234 {
		t.Errorf("PID = %d, want 1234", evt.PID)
	}
	if evt.TID != 5678 {
		t.Errorf("TID = %d, want 5678", evt.TID)
	}
	if evt.StreamHandle != 0xABCD {
		t.Errorf("StreamHandle = 0x%X, want 0xABCD", evt.StreamHandle)
	}
	if evt.CaptureMode != 1 {
		t.Errorf("CaptureMode = %d, want 1", evt.CaptureMode)
	}
	if evt.RetCode != 0 {
		t.Errorf("RetCode = %d, want 0", evt.RetCode)
	}
}

func TestParseGraphEvent_EndCapture(t *testing.T) {
	raw := buildGraphEventBytes(
		uint8(events.SourceCUDAGraph), uint8(events.GraphEndCapture),
		1000000, // 1ms
		0xABCD,  // stream
		0xBEEF,  // graph handle
		0, 0, 0,
	)

	evt, err := parseGraphEvent(raw)
	if err != nil {
		t.Fatalf("parseGraphEvent: %v", err)
	}

	if events.CUDAGraphOp(evt.Op) != events.GraphEndCapture {
		t.Errorf("Op = %d, want GraphEndCapture", evt.Op)
	}
	if evt.GraphHandle != 0xBEEF {
		t.Errorf("GraphHandle = 0x%X, want 0xBEEF", evt.GraphHandle)
	}
}

func TestParseGraphEvent_Instantiate(t *testing.T) {
	raw := buildGraphEventBytes(
		uint8(events.SourceCUDAGraph), uint8(events.GraphInstantiate),
		2000000, // 2ms
		0,       // no stream
		0xBEEF,  // graph
		0xCAFE,  // exec
		0, 0,
	)

	evt, err := parseGraphEvent(raw)
	if err != nil {
		t.Fatalf("parseGraphEvent: %v", err)
	}

	if events.CUDAGraphOp(evt.Op) != events.GraphInstantiate {
		t.Errorf("Op = %d, want GraphInstantiate", evt.Op)
	}
	if evt.GraphHandle != 0xBEEF {
		t.Errorf("GraphHandle = 0x%X, want 0xBEEF", evt.GraphHandle)
	}
	if evt.ExecHandle != 0xCAFE {
		t.Errorf("ExecHandle = 0x%X, want 0xCAFE", evt.ExecHandle)
	}
}

func TestParseGraphEvent_Launch(t *testing.T) {
	raw := buildGraphEventBytes(
		uint8(events.SourceCUDAGraph), uint8(events.GraphLaunch),
		100000,  // 100us
		0xABCD,  // stream
		0,       // no graph
		0xCAFE,  // exec
		0, 0,
	)

	evt, err := parseGraphEvent(raw)
	if err != nil {
		t.Fatalf("parseGraphEvent: %v", err)
	}

	if events.CUDAGraphOp(evt.Op) != events.GraphLaunch {
		t.Errorf("Op = %d, want GraphLaunch", evt.Op)
	}
	if evt.ExecHandle != 0xCAFE {
		t.Errorf("ExecHandle = 0x%X, want 0xCAFE", evt.ExecHandle)
	}
	if evt.StreamHandle != 0xABCD {
		t.Errorf("StreamHandle = 0x%X, want 0xABCD", evt.StreamHandle)
	}
}

func TestParseGraphEvent_AllOps(t *testing.T) {
	tests := []struct {
		name string
		op   events.CUDAGraphOp
	}{
		{"BeginCapture", events.GraphBeginCapture},
		{"EndCapture", events.GraphEndCapture},
		{"Instantiate", events.GraphInstantiate},
		{"Launch", events.GraphLaunch},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			raw := buildGraphEventBytes(
				uint8(events.SourceCUDAGraph), uint8(tt.op),
				1000, 0, 0, 0, 0, 0,
			)
			evt, err := parseGraphEvent(raw)
			if err != nil {
				t.Fatalf("parseGraphEvent: %v", err)
			}
			if events.CUDAGraphOp(evt.Op) != tt.op {
				t.Errorf("Op = %d, want %d", evt.Op, tt.op)
			}
			if evt.Source != events.SourceCUDAGraph {
				t.Errorf("Source = %v, want SourceCUDAGraph", evt.Source)
			}
		})
	}
}

func TestParseGraphEvent_TooShort(t *testing.T) {
	short := make([]byte, 40) // less than 72
	_, err := parseGraphEvent(short)
	if err == nil {
		t.Error("expected error for short event, got nil")
	}
}

func TestParseGraphEvent_ErrorCode(t *testing.T) {
	raw := buildGraphEventBytes(
		uint8(events.SourceCUDAGraph), uint8(events.GraphBeginCapture),
		500, 0xABCD, 0, 0, 0, 1, // retCode = 1 (error)
	)

	evt, err := parseGraphEvent(raw)
	if err != nil {
		t.Fatalf("parseGraphEvent: %v", err)
	}
	if evt.RetCode != 1 {
		t.Errorf("RetCode = %d, want 1", evt.RetCode)
	}
}

func TestNew(t *testing.T) {
	tr := New("/fake/libcudart.so")
	if tr == nil {
		t.Fatal("New returned nil")
	}
	if tr.libPath != "/fake/libcudart.so" {
		t.Errorf("libPath = %q, want /fake/libcudart.so", tr.libPath)
	}
	if tr.ProbeCount() != 0 {
		t.Errorf("ProbeCount = %d before Attach, want 0", tr.ProbeCount())
	}
	if tr.Dropped() != 0 {
		t.Errorf("Dropped = %d, want 0", tr.Dropped())
	}
}
