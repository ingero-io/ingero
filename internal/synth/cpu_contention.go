package synth

import (
	"context"

	"github.com/ingero-io/ingero/pkg/events"
)

func init() {
	register(&Scenario{
		Name:  "cpu-contention",
		Title: "CPU Contention → CUDA Latency Spike",
		Description: "Host sched_switch events preempt the CUDA driver thread, " +
			"causing cudaStreamSync p99 to spike 3x",
		Insight: "Your training slowed because another process stole CPU cores. " +
			"847 context switches during cudaStreamSync = 142ms tail latency.",
		Generate: generateCPUContention,
		GPUScript: `#!/usr/bin/env python3
"""CPU contention demo: stress-ng steals CPU, CUDA latency spikes."""
import os, time, signal, subprocess, torch

print("Phase 1: Baseline (clean system)...")
a = torch.randn(1024, 1024, device="cuda")
for i in range(100):
    torch.mm(a, a)
    torch.cuda.synchronize()

print("Phase 2: CPU contention (4 stress-ng workers)...")
stress = subprocess.Popen(["stress-ng", "--cpu", "4", "--timeout", "10s"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
for i in range(100):
    torch.mm(a, a)
    torch.cuda.synchronize()

stress.wait()

print("Phase 3: Recovery (clean system again)...")
for i in range(100):
    torch.mm(a, a)
    torch.cuda.synchronize()

print("Done.")
`,
	})
}

// generateCPUContention produces one cycle of the cpu-contention scenario.
//
// Phase 1 (baseline): Normal CUDA operations — fast, predictable latencies.
// Phase 2 (contention): Interleaved CUDA + sched_switch events. CUDA sync
//
//	latencies increase 3x due to simulated CPU preemption.
//
// Phase 3 (recovery): Back to normal after contention ends.
//
// This showcases the v0.2 correlation WOW: the stats table shows CUDA p99
// spiking, and the Host Context table shows sched_switch events happening
// at the same time. The correlation engine connects the two.
func generateCPUContention(ctx context.Context, ch chan<- events.Event, speed float64) {
	// --- Phase 1: Baseline (50 events, normal latencies) ---
	for i := 0; i < 50; i++ {
		if !emit(ctx, ch, makeEvent(events.CUDALaunchKernel, jitter(50*microsecond, 0.2)), speed) {
			return
		}
		if !emit(ctx, ch, makeEvent(events.CUDAStreamSync, jitter(200*microsecond, 0.2)), speed) {
			return
		}
	}

	// --- Phase 2: CPU contention (100 events with interleaved host events) ---
	// CUDA latencies spike because the driver thread gets preempted.
	for i := 0; i < 100; i++ {
		// Emit sched_switch event (process preempted off-CPU for 2-8ms).
		offCPU := jitter(5*millisecond, 0.5)
		if !emit(ctx, ch, makeHostEvent(events.HostSchedSwitch, offCPU), speed) {
			return
		}

		// CUDA launch is slightly slower due to CPU contention.
		if !emit(ctx, ch, makeEvent(events.CUDALaunchKernel, jitter(80*microsecond, 0.3)), speed) {
			return
		}

		// cudaStreamSync takes much longer — driver thread was preempted.
		syncDur := jitter(600*microsecond, 0.4) // 3x baseline
		if i%10 == 0 {
			// Every 10th event: extreme spike (simulating multiple preemptions).
			syncDur = jitter(2*millisecond, 0.3)
		}
		if !emit(ctx, ch, makeEvent(events.CUDAStreamSync, syncDur), speed) {
			return
		}

		// Occasional page alloc event (memory pressure).
		if i%20 == 0 {
			allocEvt := makeHostEvent(events.HostPageAlloc, 0)
			allocEvt.Args[0] = 4096 * 16 // 64KB allocation
			if !emit(ctx, ch, allocEvt, speed) {
				return
			}
		}
	}

	// --- Phase 3: Recovery (50 events, back to normal) ---
	for i := 0; i < 50; i++ {
		if !emit(ctx, ch, makeEvent(events.CUDALaunchKernel, jitter(50*microsecond, 0.2)), speed) {
			return
		}
		if !emit(ctx, ch, makeEvent(events.CUDAStreamSync, jitter(200*microsecond, 0.2)), speed) {
			return
		}
	}
}

// Time constant shortcuts for readability.
const (
	microsecond = 1000 // nanoseconds
	millisecond = 1000000
)
