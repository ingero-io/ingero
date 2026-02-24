package synth

import (
	"context"
	"math/rand"
	"time"

	"github.com/ingero-io/ingero/pkg/events"
)

func init() {
	register(&Scenario{
		Name:  "periodic-spike",
		Title: "Periodic Allocation Spikes",
		Description: "cudaMalloc spikes 50x every ~200 events — PyTorch's caching " +
			"allocator evicting and reallocating GPU memory",
		Insight: "cudaMalloc spikes 50x every ~200 batches. " +
			"PyTorch's caching allocator eviction cycle is invisible in loss curves.",
		Generate: generatePeriodicSpike,
		GPUScript: `#!/usr/bin/env python3
"""Periodic spike demo: allocation spikes every ~200 iterations."""
import torch

print("Running periodic-spike workload (1000 iterations)...")
print("Every ~200 iterations, a large allocation triggers caching allocator pressure.")

device = torch.device("cuda")

for i in range(1000):
    # Normal small allocation + compute.
    t = torch.randn(256, 256, device=device)
    r = torch.mm(t, t)
    torch.cuda.synchronize()

    # Every ~200 iterations: large allocation forces cache eviction.
    if i > 0 and i % 200 == 0:
        print(f"  Spike at iteration {i}: large allocation...")
        big = torch.empty(512 * 1024 * 1024 // 4, dtype=torch.float32, device=device)
        del big
        torch.cuda.synchronize()
        # Force cache flush to make the next regular allocs slower too.
        torch.cuda.empty_cache()

print("Done.")
`,
	})
}

// generatePeriodicSpike produces one cycle (~400 events) with periodic
// cudaMalloc spikes.
//
// Normal cudaMalloc takes ~50µs. Every ~200 events, one cudaMalloc spikes to
// ~2.5ms (50x normal) — simulating PyTorch's caching allocator eviction cycle.
// The stats engine's spike detector should identify the pattern.
func generatePeriodicSpike(ctx context.Context, ch chan<- events.Event, speed float64) {
	eventCount := 0
	spikeInterval := 200

	for i := 0; i < 400; i++ {
		eventCount++
		r := rand.Intn(100)
		var evt events.Event

		// Spike zone: force cudaMalloc spikes every ~200 events.
		// This is outside the random selection to guarantee spikes fire,
		// avoiding flaky test behavior where no cudaMalloc events land
		// in the spike window by random chance.
		if eventCount%spikeInterval > spikeInterval-5 {
			evt = makeEvent(events.CUDAMalloc, jitter(2500*time.Microsecond, 0.2))
		} else {
			switch {
			case r < 15:
				// cudaMalloc: normal ~50µs.
				evt = makeEvent(events.CUDAMalloc, jitter(50*time.Microsecond, 0.2))

			case r < 55:
				// cudaLaunchKernel: ~14µs (steady background).
				evt = makeEvent(events.CUDALaunchKernel, jitter(14*time.Microsecond, 0.15))

			case r < 80:
				// cudaMemcpy: ~200µs (normal transfers).
				evt = makeEvent(events.CUDAMemcpy, jitter(200*time.Microsecond, 0.2))

			case r < 90:
				// cudaDeviceSync: ~150µs (torch.cuda.synchronize()).
				evt = makeEvent(events.CUDADeviceSync, jitter(150*time.Microsecond, 0.2))

			default:
				// cudaStreamSync: ~150µs (per-stream sync).
				evt = makeEvent(events.CUDAStreamSync, jitter(150*time.Microsecond, 0.2))
			}
		}

		if !emit(ctx, ch, evt, speed) {
			return
		}
	}
}
