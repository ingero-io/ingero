package cli

import (
	"context"
	"log/slog"
	"os"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/ingero-io/ingero/internal/nvml"
)

// Multi-line kmsg stream: two NVRM Xid messages interspersed with
// unrelated kernel chatter. Both Xids must surface; the chatter must
// be silently dropped. Mirrors what /dev/kmsg actually delivers.
func TestRunXidReader_EmitsForRecognisedLines(t *testing.T) {
	kmsg := `4,1,1,-;Linux version 5.15.0-foo
4,2,2,-;tcp: bbr: registered
4,3,3,-;NVRM: Xid (PCI:0000:1a:00): 79, pid='<unknown>', GPU has fallen off the bus.
4,4,4,-;some random message
4,5,5,-;NVRM: Xid (PCI:0000:3b:00): 13, pid=4321, Graphics SM Warp Exception.
`
	pciIndex := map[string]uint32{
		"00000000:1a:00.0": 0,
		"00000000:3b:00.0": 1,
	}
	var mu sync.Mutex
	var got []nvml.HardwareFault
	sink := func(f nvml.HardwareFault) {
		mu.Lock()
		defer mu.Unlock()
		got = append(got, f)
	}
	runXidReader(context.Background(), strings.NewReader(kmsg), pciIndex, sink, slog.Default())

	mu.Lock()
	defer mu.Unlock()
	if len(got) != 2 {
		t.Fatalf("emissions=%d want 2 (%+v)", len(got), got)
	}
	if got[0].XidNumber != 79 || got[0].GPUID != 0 {
		t.Errorf("first emission = %+v want Xid 79 GPUID 0", got[0])
	}
	if got[0].Severity != nvml.HardwareFaultCritical {
		t.Errorf("Xid 79 severity=%q want critical", got[0].Severity)
	}
	if got[1].XidNumber != 13 || got[1].GPUID != 1 {
		t.Errorf("second emission = %+v want Xid 13 GPUID 1", got[1])
	}
	if got[1].PID != 4321 {
		t.Errorf("second emission PID=%d want 4321", got[1].PID)
	}
}

// When the resolver map is empty (nvidia-smi unavailable at startup)
// or the kmsg bus_id is not in the map, emissions still fire with
// gpu_id=0. The orchestrator's VramTracker fallback handles target
// resolution downstream.
func TestRunXidReader_UnknownPciFallsBackToZero(t *testing.T) {
	kmsg := "4,1,1,-;NVRM: Xid (PCI:0000:99:00): 79, pid=1\n"
	var got []nvml.HardwareFault
	sink := func(f nvml.HardwareFault) { got = append(got, f) }
	pciIndex := map[string]uint32{"00000000:1a:00.0": 0} // no match for 99
	runXidReader(context.Background(), strings.NewReader(kmsg), pciIndex, sink, slog.Default())
	if len(got) != 1 {
		t.Fatalf("emissions=%d want 1", len(got))
	}
	if got[0].GPUID != 0 {
		t.Errorf("GPUID=%d want 0 (unknown PCI)", got[0].GPUID)
	}
}

// Non-critical Xids (e.g. transient GR exception code 8) must still
// surface as warning-severity hardware_fault emissions; the EE side
// counts them but does not dispatch.
func TestRunXidReader_NonCriticalXidEmitsWarning(t *testing.T) {
	kmsg := "4,1,1,-;NVRM: Xid (PCI:0000:1a:00): 8, pid=1\n"
	var got []nvml.HardwareFault
	sink := func(f nvml.HardwareFault) { got = append(got, f) }
	runXidReader(context.Background(), strings.NewReader(kmsg), nil, sink, slog.Default())
	if len(got) != 1 || got[0].Severity != nvml.HardwareFaultWarning {
		t.Fatalf("got=%+v want one warning-severity emission", got)
	}
}

// A blocked reader that never completes must exit cleanly when ctx
// is cancelled. Models the production /dev/kmsg path: blocking read,
// no EOF, only unblocks when the file is closed (which the watcher
// goroutine does on ctx.Done). This test uses a pipe whose write end
// stays open; the read end blocks forever. We close the read end
// from the cancel goroutine, mirroring production.
func TestStartXidReader_CtxCancelExits(t *testing.T) {
	pr, pw, err := os.Pipe()
	if err != nil {
		t.Fatal(err)
	}
	defer pw.Close()
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() {
		runXidReader(ctx, pr, nil, func(nvml.HardwareFault) {}, slog.Default())
		close(done)
	}()
	cancel()
	pr.Close() // unblock the read
	select {
	case <-done:
	case <-time.After(time.Second):
		t.Fatal("runXidReader did not exit within 1s of cancel")
	}
}

// A nil sink turns the wiring into a no-op without panicking. The
// production wiring constructs the sink conditionally; if the
// constructor returns nil (e.g. --remediate disabled), the reader
// must skip starting.
func TestStartXidReader_NilSinkIsNoOp(t *testing.T) {
	// Should not panic and should not start any goroutine. The lack
	// of any failure mode IS the test surface.
	startXidReader(context.Background(), "/nonexistent/kmsg", nil, nil, slog.Default())
}

// Opening a nonexistent kmsg path is the cross-platform-degrade case
// (no /dev/kmsg on macOS / WSL without /dev/kmsg). Must not panic.
func TestStartXidReader_MissingPathDegrades(t *testing.T) {
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	startXidReader(ctx, "/this/path/does/not/exist", nil,
		func(nvml.HardwareFault) {}, slog.Default())
}
