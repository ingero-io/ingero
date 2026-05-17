package cli

import (
	"context"
	"log/slog"
	"sync"
	"testing"
	"time"

	"github.com/ingero-io/ingero/internal/nvml"
)

// One-shot pollLinkOnce drives both trackers in lockstep. With a
// downtrained PCIe link and rising NVLink errors, by the third call
// both emissions should have fired (NVLink at poll 2, PCIe at poll 3
// per the constants in link_poller.go).
func TestPollLinkOnce_BothEmissionsFire(t *testing.T) {
	nvlinkCalls := 0
	nvlinkOutputs := []string{
		// poll 1: seeds at 0
		`GPU 0: x (UUID: GPU-a)
         Link 0: Replay Errors: 0
`,
		// poll 2: +5 errors, consecutive=1 (no emit yet, sustainPolls=2)
		`GPU 0: x (UUID: GPU-a)
         Link 0: Replay Errors: 5
`,
		// poll 3: +5 errors, consecutive=2 -> EMIT
		`GPU 0: x (UUID: GPU-a)
         Link 0: Replay Errors: 10
`,
	}
	nvlinkRun := nvml.Runner(func(ctx context.Context) ([]byte, error) {
		out := []byte(nvlinkOutputs[nvlinkCalls])
		nvlinkCalls++
		return out, nil
	})
	// PCIe stays downtrained across all three polls; emits at poll 3
	// (sustainPolls=3).
	pcieRun := nvml.Runner(func(ctx context.Context) ([]byte, error) {
		return []byte("0, GPU-a, 3, 4, 16, 16\n"), nil
	})

	tracker1 := nvml.NewNVLinkErrorTracker(linkNVLinkSustainPolls)
	tracker2 := nvml.NewPCIeDowntrainTracker(linkPCIeSustainPolls)
	var mu sync.Mutex
	var got []nvml.HardwareFault
	sink := func(f nvml.HardwareFault) {
		mu.Lock()
		defer mu.Unlock()
		got = append(got, f)
	}
	for i := 0; i < 3; i++ {
		pollLinkOnce(context.Background(), nvlinkRun, pcieRun, tracker1, tracker2, sink, slog.Default())
	}
	mu.Lock()
	defer mu.Unlock()
	kinds := map[nvml.HardwareFaultKind]int{}
	for _, f := range got {
		kinds[f.Kind]++
	}
	if kinds[nvml.FaultKindNVLink] != 1 {
		t.Errorf("NVLink emissions=%d want 1 (%+v)", kinds[nvml.FaultKindNVLink], got)
	}
	if kinds[nvml.FaultKindPCIeDowntrain] != 1 {
		t.Errorf("PCIe emissions=%d want 1 (%+v)", kinds[nvml.FaultKindPCIeDowntrain], got)
	}
}

// startLinkPoller must spawn its goroutine and exit when ctx is
// cancelled. The poller calls back into the runners on its ticker; we
// use a long interval and rely on the immediate first-tick poll to
// confirm wiring.
func TestStartLinkPoller_RunsThenCancels(t *testing.T) {
	var calls int
	var mu sync.Mutex
	pcieRun := nvml.Runner(func(ctx context.Context) ([]byte, error) {
		mu.Lock()
		calls++
		mu.Unlock()
		return []byte("0, GPU-a, 4, 4, 16, 16\n"), nil
	})

	ctx, cancel := context.WithCancel(context.Background())
	startLinkPoller(ctx, time.Hour, nil, pcieRun, func(nvml.HardwareFault) {}, slog.Default())
	// Give the first-tick poll time to land.
	time.Sleep(50 * time.Millisecond)
	cancel()
	time.Sleep(20 * time.Millisecond)

	mu.Lock()
	defer mu.Unlock()
	if calls < 1 {
		t.Fatalf("first-tick poll did not run (calls=%d)", calls)
	}
}

// Both runners nil and no faultSink mean the poller is a no-op: it
// must not panic and must not spawn anything. Mirrors the
// `--remediate` off + no nvidia-smi case.
func TestStartLinkPoller_NoopWhenDisabled(t *testing.T) {
	startLinkPoller(context.Background(), time.Second, nil, nil, nil, slog.Default())
	startLinkPoller(context.Background(), time.Second, nil, nil,
		func(nvml.HardwareFault) {}, slog.Default())
}
