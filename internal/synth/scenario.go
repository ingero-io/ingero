// Package synth provides demo scenarios for CUDA workload patterns.
//
// Each scenario has two modes:
//   - Synthetic (--no-gpu): generates fake events in Go, no root/GPU needed
//   - GPU (default): runs a real Python+CUDA workload traced by eBPF
//
// Both modes feed through the same stats + display pipeline as watch.
package synth

import (
	"context"
	"math/rand"
	"time"

	"github.com/ingero-io/ingero/pkg/events"
)

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const (
	// SyntheticPID is the fake process ID used for all synthetic events.
	SyntheticPID uint32 = 12345

	// SyntheticTID is the fake thread ID used for synthetic events.
	SyntheticTID uint32 = 12345
)

// ---------------------------------------------------------------------------
// Scenario type and registry
// ---------------------------------------------------------------------------

// Scenario describes a CUDA workload pattern with both synthetic and GPU modes.
type Scenario struct {
	Name        string   // kebab-case name used on the command line
	Aliases     []string // alternative names (e.g., gpu-steal has aliases "gpu-contention", "contention")
	Title       string   // human-readable title for display
	Description string   // one-line description
	Insight     string   // the "WOW" takeaway shown after the demo

	// Generate produces synthetic events on ch until ctx is cancelled.
	// It runs ONE cycle of the scenario, then returns. The caller
	// is responsible for looping (calling Generate again) if desired.
	// speed is a multiplier: 2.0 = events generated 2x faster.
	// Used in --no-gpu mode.
	Generate func(ctx context.Context, ch chan<- events.Event, speed float64)

	// GPUScript is a Python script that produces the real CUDA workload
	// for this scenario. Written to a temp file and executed via python3.
	// Used in GPU mode (the default). Requires sudo + python3 + torch.
	GPUScript string
}

// Registry holds all available demo scenarios, ordered for display.
// Populated by init() in each scenario file.
var Registry []*Scenario

// register adds a scenario to the registry. Called from each scenario's init().
func register(s *Scenario) {
	Registry = append(Registry, s)
}

// Find looks up a scenario by name or alias. Returns nil if not found.
func Find(name string) *Scenario {
	for _, s := range Registry {
		if s.Name == name {
			return s
		}
		for _, alias := range s.Aliases {
			if alias == name {
				return s
			}
		}
	}
	return nil
}

// ---------------------------------------------------------------------------
// Event construction helpers
// ---------------------------------------------------------------------------

// makeEvent creates a synthetic CUDA event with standard fields populated.
//
// This is the single point where synthetic events are constructed, ensuring
// consistent PID, TID, Source, and Timestamp across all scenarios.
func makeEvent(op events.CUDAOp, dur time.Duration) events.Event {
	return events.Event{
		Timestamp: time.Now(),
		PID:       SyntheticPID,
		TID:       SyntheticTID,
		Source:    events.SourceCUDA,
		Op:        uint8(op),
		Duration:  dur,
	}
}

// emit sends an event on ch and then sleeps for the event's duration divided
// by speed, simulating real-time pacing. Returns false if ctx was cancelled.
//
// Paces at event duration / speed to simulate realistic event rates.
func emit(ctx context.Context, ch chan<- events.Event, evt events.Event, speed float64) bool {
	select {
	case ch <- evt:
	case <-ctx.Done():
		return false
	}

	// Pace: sleep for the event's duration adjusted by speed.
	sleepDur := time.Duration(float64(evt.Duration) / speed)
	if sleepDur < 100*time.Microsecond {
		sleepDur = 100 * time.Microsecond // minimum pacing to avoid busy-loop
	}

	select {
	case <-time.After(sleepDur):
		return true
	case <-ctx.Done():
		return false
	}
}

// makeHostEvent creates a synthetic host kernel event with standard fields.
func makeHostEvent(op events.HostOp, dur time.Duration) events.Event {
	return events.Event{
		Timestamp: time.Now(),
		PID:       SyntheticPID,
		TID:       SyntheticTID,
		Source:    events.SourceHost,
		Op:        uint8(op),
		Duration:  dur,
	}
}

// jitter adds ±pct random variation to a duration.
// jitter(100*time.Microsecond, 0.2) returns 80-120µs.
func jitter(base time.Duration, pct float64) time.Duration {
	factor := 1.0 + (rand.Float64()*2-1)*pct
	return time.Duration(float64(base) * factor)
}
