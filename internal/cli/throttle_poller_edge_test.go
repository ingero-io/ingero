package cli

import (
	"context"
	"log/slog"
	"testing"
	"time"

	"github.com/ingero-io/ingero/internal/nvml"
)

// v0.15 item L wire test: pollOnce must feed every reading into
// the throttleEdgeDetector. Without this, the edge counter would
// silently stay at zero even on a host that's actively throttling.

func TestPollOnce_FeedsEdgeDetector(t *testing.T) {
	// Reset detector + ringbuf so this test is hermetic.
	throttleEdgeDetector.Reset()
	t.Cleanup(func() { throttleEdgeDetector.Reset() })

	// Synthesize an nvidia-smi runner that returns one healthy
	// throttling reading per call. Sequence: idle -> power-only ->
	// power+thermal. The detector should record 2 power events
	// (rising edge from idle, no event when sustained, but
	// power+thermal vs power-only is a thermal rising edge).
	calls := 0
	run := func(_ context.Context) ([]byte, error) {
		calls++
		switch calls {
		case 1:
			// idle: bitmask 0
			return []byte("GPU-A, 0x0\n"), nil
		case 2:
			// power-only: ReasonSwPowerCap
			return []byte("GPU-A, 0x4\n"), nil
		case 3:
			// power + thermal (HwThermalSlowdown)
			return []byte("GPU-A, 0x44\n"), nil
		}
		return nil, nil
	}

	// Drive 3 polls.
	pollOnce(context.Background(), run, slog.Default())
	pollOnce(context.Background(), run, slog.Default())
	pollOnce(context.Background(), run, slog.Default())

	got := throttleEdgeDetector.Snapshot()
	if got.PowerEvents != 1 {
		t.Errorf("PowerEvents=%d want 1 (rising edge idle -> power)", got.PowerEvents)
	}
	if got.ThermalEvents != 1 {
		t.Errorf("ThermalEvents=%d want 1 (rising edge from power-only -> power+thermal)", got.ThermalEvents)
	}
}

// pollOnce with no readings (e.g., consumer GPU returning [Not
// Supported]) must NOT inject anything into the detector.
func TestPollOnce_NotSupportedDoesNotIncrement(t *testing.T) {
	throttleEdgeDetector.Reset()
	t.Cleanup(func() { throttleEdgeDetector.Reset() })

	run := func(_ context.Context) ([]byte, error) {
		return []byte("GPU-A, [Not Supported]\n"), nil
	}
	pollOnce(context.Background(), run, slog.Default())
	got := throttleEdgeDetector.Snapshot()
	if got.PowerEvents != 0 || got.ThermalEvents != 0 || got.SWEvents != 0 || got.HWEvents != 0 {
		t.Errorf("not-supported reading should not increment any bucket; got %+v", got)
	}
}

// Process startup detector (no prior state) does NOT count the
// FIRST observation as a rising edge. Same as
// TestThrottleEdgeDetector_FirstObservationDoesNotIncrement at
// the unit level; this is the wire-level mirror.
func TestPollOnce_FirstReadingNotCountedAsEdge(t *testing.T) {
	throttleEdgeDetector.Reset()
	t.Cleanup(func() { throttleEdgeDetector.Reset() })

	run := func(_ context.Context) ([]byte, error) {
		// First poll already throttled.
		return []byte("GPU-A, 0x4\n"), nil
	}
	pollOnce(context.Background(), run, slog.Default())
	got := throttleEdgeDetector.Snapshot()
	if got.PowerEvents != 0 {
		t.Errorf("first observation should not register an edge; got %d", got.PowerEvents)
	}
}

// Defensive: verify the edge detector is per-process (not per-test)
// so a leak between tests would surface here. Run this AFTER the
// other detector tests by name ordering; the Reset cleanups should
// have left it empty.
func TestThrottleEdgeDetector_PristineAtTestBoundary(t *testing.T) {
	got := throttleEdgeDetector.Snapshot()
	if got != (nvml.ThrottleEventCounters{}) {
		t.Errorf("detector state leaked between tests: %+v", got)
	}
	_ = time.Now() // kill unused-import noise from a future copy/paste
}
