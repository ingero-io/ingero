package cli

import (
	"testing"

	"github.com/ingero-io/ingero/internal/ebpf/ncclprobe"
)

func TestNCCLDiscoveryDrainBeforeFirstScan(t *testing.T) {
	resetNCCLDiscoveryState()
	defer resetNCCLDiscoveryState()
	if got := drainNCCLDiscoveryBuf(); got != nil {
		t.Errorf("drain before first scan = %+v, want nil", got)
	}
}

func TestNCCLDiscoveryDrainAfterEmptyScan(t *testing.T) {
	resetNCCLDiscoveryState()
	defer resetNCCLDiscoveryState()
	setNCCLDiscoveryBatch(nil)
	got := drainNCCLDiscoveryBuf()
	if got == nil {
		t.Errorf("drain after empty scan = nil, want non-nil empty slice (so OTLP emits processes_total=0)")
	}
	if len(got) != 0 {
		t.Errorf("drain after empty scan len = %d, want 0", len(got))
	}
}

func TestNCCLDiscoveryDrainAfterPopulatedScan(t *testing.T) {
	resetNCCLDiscoveryState()
	defer resetNCCLDiscoveryState()
	in := []ncclprobe.NCCLProcess{
		{PID: 100, Comm: "python", LibPath: "/usr/lib/libnccl.so.2.21.5", LibVersion: "2.21.5"},
		{PID: 200, Comm: "torchrun", LibPath: "/opt/conda/lib/libnccl.so.2.18.3", LibVersion: "2.18.3"},
	}
	setNCCLDiscoveryBatch(in)
	got := drainNCCLDiscoveryBuf()
	if len(got) != 2 {
		t.Fatalf("len(got) = %d, want 2", len(got))
	}
	if got[0].PID != 100 || got[0].LibVersion != "2.21.5" || got[0].Comm != "python" {
		t.Errorf("got[0] = %+v", got[0])
	}
	if got[1].PID != 200 || got[1].LibVersion != "2.18.3" {
		t.Errorf("got[1] = %+v", got[1])
	}

	// Drain is non-destructive (last-batch-wins gauge semantics) - a
	// second drain returns the same data until the scanner pushes new.
	got2 := drainNCCLDiscoveryBuf()
	if len(got2) != 2 {
		t.Errorf("second drain len = %d, want 2 (gauge persistence)", len(got2))
	}
}
