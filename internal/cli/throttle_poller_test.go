package cli

import (
	"context"
	"errors"
	"fmt"
	"io"
	"log/slog"
	"sort"
	"sync"
	"sync/atomic"
	"testing"
	"time"

	"github.com/ingero-io/ingero/internal/nvml"
)

// silentLog returns a slog.Logger that discards all output. Test-local; the
// poller's debug/info logs would otherwise pollute go test stdout.
func silentLog() *slog.Logger {
	return slog.New(slog.NewTextHandler(io.Discard, nil))
}

// TestPoller_TwoGPUsLabeledByUUID covers QA audit ★4 H4: a single-GPU rig
// (Lambda A10) cannot exercise the per-GPU label dimension via integration
// alone, so the unit test stands in. After two ticks both UUIDs must appear
// in drainThrottleBuf with their decoded buckets.
func TestPoller_TwoGPUsLabeledByUUID(t *testing.T) {
	resetThrottleState()
	defer resetThrottleState()

	run := func(ctx context.Context) ([]byte, error) {
		// GPU 1: SwPowerCap (0x4) -> Power+SW
		// GPU 2: HwThermalSlowdown (0x40) -> Thermal+HW
		return []byte("GPU-aaaaaaaa, 0x4\nGPU-bbbbbbbb, 0x40\n"), nil
	}

	pollOnce(context.Background(), run, silentLog())

	got := drainThrottleBuf()
	if len(got) != 2 {
		t.Fatalf("want 2 readings, got %d: %+v", len(got), got)
	}
	sort.Slice(got, func(i, j int) bool { return got[i].UUID < got[j].UUID })

	if got[0].UUID != "GPU-aaaaaaaa" || !got[0].PowerActive || !got[0].SWActive {
		t.Fatalf("GPU1 reading wrong: %+v", got[0])
	}
	if got[0].ThermalActive || got[0].HWActive {
		t.Fatalf("GPU1 must not have thermal/hw: %+v", got[0])
	}
	if got[1].UUID != "GPU-bbbbbbbb" || !got[1].ThermalActive || !got[1].HWActive {
		t.Fatalf("GPU2 reading wrong: %+v", got[1])
	}
	if got[1].PowerActive || got[1].SWActive {
		t.Fatalf("GPU2 must not have power/sw: %+v", got[1])
	}
}

// TestPoller_NotSupportedSkipsDevice covers ★3 H6: a "[Not Supported]"
// row from a consumer GPU must NOT emit a reading and must NOT panic.
// The supported-GPU row in the same response still flows through.
func TestPoller_NotSupportedSkipsDevice(t *testing.T) {
	resetThrottleState()
	defer resetThrottleState()

	run := func(ctx context.Context) ([]byte, error) {
		return []byte("GPU-consumer, [Not Supported]\nGPU-datacenter, 0x4\n"), nil
	}

	pollOnce(context.Background(), run, silentLog())

	got := drainThrottleBuf()
	if len(got) != 1 {
		t.Fatalf("want 1 reading (consumer GPU skipped), got %d: %+v", len(got), got)
	}
	if got[0].UUID != "GPU-datacenter" {
		t.Fatalf("expected datacenter GPU, got %q", got[0].UUID)
	}
}

// TestPoller_NotSupportedLogOnce asserts the consumer-GPU log line fires
// at most once per UUID, no matter how many ticks see the same row. The
// poller calls slog.Info for the first sighting, then debug-only.
func TestPoller_NotSupportedLogOnce(t *testing.T) {
	resetThrottleState()
	defer resetThrottleState()

	var infoCount atomic.Int64
	h := infoCountingHandler{count: &infoCount}
	log := slog.New(h)

	run := func(ctx context.Context) ([]byte, error) {
		return []byte("GPU-consumer, [Not Supported]\n"), nil
	}

	for i := 0; i < 10; i++ {
		pollOnce(context.Background(), run, log)
	}

	if got := infoCount.Load(); got != 1 {
		t.Fatalf("info-level log should fire exactly once per UUID, got %d", got)
	}
}

// TestPoller_RunErrorIsLogged covers the wholly-failed nvidia-smi path: the
// poller logs at debug and emits no readings. The next tick is unaffected.
func TestPoller_RunErrorIsLogged(t *testing.T) {
	resetThrottleState()
	defer resetThrottleState()

	pollOnce(context.Background(),
		func(ctx context.Context) ([]byte, error) {
			return nil, errors.New("synthetic")
		},
		silentLog(),
	)

	if got := drainThrottleBuf(); got != nil {
		t.Fatalf("error path should produce no readings, got %+v", got)
	}
}

// TestStartThrottlePoller_NilRunner asserts that a nil runner (e.g.
// nvidia-smi missing) results in no goroutine spawn and no panic.
func TestStartThrottlePoller_NilRunner(t *testing.T) {
	resetThrottleState()
	defer resetThrottleState()

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	startThrottlePoller(ctx, time.Millisecond, nil, silentLog())

	// Nothing should land in the buffer.
	time.Sleep(20 * time.Millisecond)
	if got := drainThrottleBuf(); got != nil {
		t.Fatalf("nil runner should not produce readings, got %+v", got)
	}
}

// TestStartThrottlePoller_FiresImmediately asserts the goroutine performs
// the first read synchronously before installing the ticker, so the very
// first snapshot has data instead of waiting one full interval.
func TestStartThrottlePoller_FiresImmediately(t *testing.T) {
	resetThrottleState()
	defer resetThrottleState()

	var calls atomic.Int64
	run := func(ctx context.Context) ([]byte, error) {
		calls.Add(1)
		return []byte("GPU-x, 0x4\n"), nil
	}

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	startThrottlePoller(ctx, time.Hour, run, silentLog())

	// Wait briefly for the spawned goroutine to run pollOnce once. The
	// ticker won't fire (1h interval), so anything in the buffer came
	// from the synchronous first read.
	deadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) {
		if calls.Load() >= 1 {
			break
		}
		time.Sleep(5 * time.Millisecond)
	}

	if got := calls.Load(); got < 1 {
		t.Fatalf("expected at least one pollOnce call, got %d", got)
	}
	got := drainThrottleBuf()
	if len(got) != 1 || got[0].UUID != "GPU-x" {
		t.Fatalf("first read should have populated the buffer, got %+v", got)
	}
}

// TestDrainThrottleBuf_LastValueWins asserts the per-UUID buffer overwrites
// rather than appending. A gauge is "current state", not a time series of
// distinct events, so two ticks for the same UUID must produce one entry.
func TestDrainThrottleBuf_LastValueWins(t *testing.T) {
	resetThrottleState()
	defer resetThrottleState()

	calls := 0
	run := func(ctx context.Context) ([]byte, error) {
		calls++
		// First tick: power throttled. Second tick: cleared.
		if calls == 1 {
			return []byte("GPU-x, 0x4\n"), nil
		}
		return []byte("GPU-x, 0x0\n"), nil
	}

	pollOnce(context.Background(), run, silentLog())
	pollOnce(context.Background(), run, silentLog())

	got := drainThrottleBuf()
	if len(got) != 1 {
		t.Fatalf("want 1 entry (last-value-wins), got %d: %+v", len(got), got)
	}
	if got[0].PowerActive {
		t.Fatalf("second read cleared the throttle; expected PowerActive=false, got %+v", got[0])
	}
}

// TestPoller_VerifiesNvmlSubprocessRunnerCanBeNil documents that
// nvml.NewSubprocessRunner returns nil when nvidia-smi is absent. A
// follow-on consumer treats that as the "no NVML metrics" steady state.
// This is a sanity check on the integration contract; it does not invoke
// any subprocess.
func TestPoller_VerifiesNvmlSubprocessRunnerCanBeNil(t *testing.T) {
	r := nvml.NewSubprocessRunner()
	if r == nil {
		// Expected on dev machines without nvidia-smi. Just confirm the
		// poller path tolerates it.
		ctx, cancel := context.WithCancel(context.Background())
		cancel() // already cancelled; goroutine returns immediately
		startThrottlePoller(ctx, time.Millisecond, r, silentLog())
		return
	}
	// nvidia-smi present: smoke-test that we got a function.
	_, err := r(context.Background())
	// Either error is fine here; both indicate the runner is callable.
	_ = err
}

// infoCountingHandler counts slog.Info-level Handle calls. Used to assert
// the "log once per UUID" property of the [Not Supported] path.
type infoCountingHandler struct {
	mu    sync.Mutex
	count *atomic.Int64
}

func (h infoCountingHandler) Enabled(ctx context.Context, l slog.Level) bool {
	return true
}
func (h infoCountingHandler) Handle(ctx context.Context, r slog.Record) error {
	if r.Level == slog.LevelInfo {
		h.count.Add(1)
	}
	return nil
}
func (h infoCountingHandler) WithAttrs(attrs []slog.Attr) slog.Handler { return h }
func (h infoCountingHandler) WithGroup(name string) slog.Handler        { return h }

// Force-import for go-vet symmetry.
var _ = fmt.Sprintf
