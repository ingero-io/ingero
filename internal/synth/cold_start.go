package synth

import (
	"context"
	"math/rand"
	"time"

	"github.com/ingero-io/ingero/pkg/events"
)

func init() {
	register(&Scenario{
		Name:  "cold-start",
		Title: "CUDA Cold Start Penalty",
		Description: "First CUDA calls take 50-200x longer than steady state " +
			"(context init, JIT compilation, first allocation)",
		Insight: "Your first inference takes 200ms+, not 5ms. " +
			"Every autoscaling cold start eats this penalty.",
		Generate: generateColdStart,
		GPUScript: `#!/usr/bin/env python3
"""Cold-start demo: first CUDA calls are 50-200x slower than steady state."""
import time, torch

print("Phase 1: Cold start (first allocations + kernel launches)...")
# First-ever CUDA calls trigger context init, JIT compilation, etc.
for size_mb in [512, 256, 128, 64, 32]:
    t = torch.empty(size_mb * 1024 * 1024 // 4, dtype=torch.float32, device="cuda")
    del t
    torch.cuda.synchronize()

# First kernel launches: PTX → device code JIT compilation.
a = torch.randn(1024, 1024, device="cuda")
for _ in range(3):
    torch.mm(a, a)
    torch.cuda.synchronize()

print("Phase 2: Steady state (200 iterations)...")
# Steady state: fast allocations, launches, memcpy, sync.
for i in range(200):
    t = torch.randn(256, 256, device="cuda")  # small alloc
    r = torch.mm(t, t)                        # kernel launch
    _ = r.cpu()                               # D→H memcpy
    torch.cuda.synchronize()                   # sync

print("Done.")
`,
	})
}

// generateColdStart produces one cycle of the cold-start scenario.
//
// Phase 1 (cold): 5 cudaMalloc at 100-340ms, 3 cudaLaunchKernel at 50-150ms.
// These simulate CUDA context initialization, JIT compilation, and first
// allocation — the hidden costs that make first-request latency 50-200x
// worse than steady state.
//
// Phase 2 (steady): ~200 events at normal latencies. This lets the user see
// the dramatic contrast between cold and warm in the stats table.
func generateColdStart(ctx context.Context, ch chan<- events.Event, speed float64) {
	// --- Phase 1: Cold start (first allocations + kernel launches) ---

	// Slow cudaMalloc calls: CUDA context creation + first memory allocation.
	coldMallocs := []time.Duration{
		340 * time.Millisecond, // first-ever: full context init
		210 * time.Millisecond, // JIT compilation for allocator
		150 * time.Millisecond,
		120 * time.Millisecond,
		100 * time.Millisecond, // still cold, but warming up
	}
	for _, dur := range coldMallocs {
		evt := makeEvent(events.CUDAMalloc, jitter(dur, 0.1))
		if !emit(ctx, ch, evt, speed) {
			return
		}
	}

	// Slow kernel launches: first-time JIT compilation of PTX to device code.
	coldLaunches := []time.Duration{
		150 * time.Millisecond,
		80 * time.Millisecond,
		50 * time.Millisecond,
	}
	for _, dur := range coldLaunches {
		evt := makeEvent(events.CUDALaunchKernel, jitter(dur, 0.1))
		if !emit(ctx, ch, evt, speed) {
			return
		}
	}

	// --- Phase 2: Steady state (~200 events at normal latencies) ---

	// Weighted operation distribution for a typical training iteration.
	type opSpec struct {
		op       events.CUDAOp
		baseDur  time.Duration
		weight   int // relative probability
	}
	ops := []opSpec{
		{events.CUDALaunchKernel, 14 * time.Microsecond, 45},
		{events.CUDAMemcpy, 200 * time.Microsecond, 25},
		{events.CUDADeviceSync, 1 * time.Millisecond, 10},
		{events.CUDAStreamSync, 1 * time.Millisecond, 5},
		{events.CUDAMalloc, 50 * time.Microsecond, 15},
	}

	// Build cumulative weight table for weighted random selection.
	totalWeight := 0
	for _, o := range ops {
		totalWeight += o.weight
	}

	for i := 0; i < 200; i++ {
		// Weighted random op selection.
		r := rand.Intn(totalWeight)
		var selected opSpec
		cumulative := 0
		for _, o := range ops {
			cumulative += o.weight
			if r < cumulative {
				selected = o
				break
			}
		}

		evt := makeEvent(selected.op, jitter(selected.baseDur, 0.2))
		if !emit(ctx, ch, evt, speed) {
			return
		}
	}
}
