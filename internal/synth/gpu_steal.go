// gpu_steal.go — GPU time-slicing scenario (renamed from gpu-contention).
//
// Shows GPU context switch cost via CUDA API timing patterns.
// Two processes share the GPU via MPS/time-slicing. Process A gets 62%
// of GPU time, Process B gets 38%. ~0.8ms context switch overhead per swap.
package synth

import (
	"context"
	"time"

	"github.com/ingero-io/ingero/pkg/events"
)

func init() {
	register(gpuStealScenario)
}

var gpuStealScenario = &Scenario{
	Name:        "gpu-steal",
	Aliases:     []string{"gpu-contention", "contention"},
	Title:       "GPU Time Thief — Multi-Process GPU Sharing",
	Description: "Two processes share the GPU — quantify time-slicing overhead",
	Insight: `nvidia-smi shows both processes at "100% utilization" — but they're
time-slicing. Ingero traces CUDA API latency patterns to quantify exactly
how much GPU time each process gets, and infers the per-switch overhead
from timing gaps.`,

	Generate: generateGPUSteal,
	GPUScript: `#!/usr/bin/env python3
"""GPU steal demo: two processes fight for GPU time-slices.

Process A (traced): continuous matmul for 20s.
Process B (competitor): same workload, launched as subprocess.

nvidia-smi shows both at "100% utilization" but they're time-slicing.
Ingero sees the latency variance from context switch overhead.
"""
import subprocess, sys, time, torch

a = torch.randn(1024, 1024, device="cuda")

# Warm up
for _ in range(10):
    torch.mm(a, a)
    torch.cuda.synchronize()

# Launch competing process
competitor = subprocess.Popen(
    [sys.executable, "-c", """
import torch, time
a = torch.randn(1024, 1024, device='cuda')
start = time.time()
while time.time() - start < 20:
    torch.mm(a, a)
    torch.cuda.synchronize()
"""],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

print("Two processes sharing GPU (20s)...")
start = time.time()
while time.time() - start < 20:
    torch.mm(a, a)
    torch.cuda.synchronize()

competitor.wait()
print("Done.")
`,
}

func generateGPUSteal(ctx context.Context, ch chan<- events.Event, speed float64) {
	// Simulate two processes alternating on the GPU.
	// Process A (our traced process): 62% of GPU time.
	// Process B (competitor): 38% of GPU time.
	// Context switch adds ~0.8ms overhead visible in cudaLaunchKernel variance.

	for cycle := 0; cycle < 40; cycle++ {
		// Process A burst (our process, we see the events).
		for i := 0; i < 8; i++ {
			// Normal launch time + occasional context switch spike.
			launchDur := jitter(14*time.Microsecond, 0.15)
			if i == 0 && cycle > 0 {
				// First launch after context switch back: ~0.8ms overhead.
				launchDur = jitter(800*time.Microsecond, 0.2)
			}
			if !emit(ctx, ch, makeEvent(events.CUDALaunchKernel, launchDur), speed) {
				return
			}
			if !emit(ctx, ch, makeEvent(events.CUDAMemcpy, jitter(500*time.Microsecond, 0.2)), speed) {
				return
			}
		}

		// Sync at end of burst shows variable latency due to time-slicing.
		syncDur := jitter(12*time.Millisecond, 0.3) // higher variance than single-process
		if !emit(ctx, ch, makeEvent(events.CUDAStreamSync, syncDur), speed) {
			return
		}

		// Simulate "gap" while Process B runs on GPU.
		// We don't see Process B's events, but we see our process waiting.
		gapSync := jitter(8*time.Millisecond, 0.4) // waiting for GPU
		if !emit(ctx, ch, makeEvent(events.CUDAStreamSync, gapSync), speed) {
			return
		}
	}
}
