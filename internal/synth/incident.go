// incident.go — The WOW demo scenario. Prepended as first scenario in Registry.
//
// Shows a normal CUDA training baseline suddenly hit by a storm:
// CPU spikes from ~47% to 94%, memory climbs to 97%, sched_switch storm,
// cudaStreamSync p99 jumps from 16ms to 142ms. Then recovery.
//
// This is the first demo customers see. Shows the full product in 30 seconds.
package synth

import (
	"context"
	"time"

	"github.com/ingero-io/ingero/pkg/events"
)

func init() {
	// Prepend incident as first scenario in Registry.
	Registry = append([]*Scenario{incidentScenario}, Registry...)
}

var incidentScenario = &Scenario{
	Name:        "incident",
	Title:       "Full Causal Chain Incident",
	Description: "Normal CUDA baseline → CPU storm → cudaStreamSync 8x spike → auto root cause",
	Insight: `This is the scenario every ML engineer hits but can't diagnose.
nvidia-smi says "GPU utilization 98%", but the GPU is actually stalling
because a cron job (logrotate) is stealing CPU cores. No existing tool
shows this cross-layer causal chain. Ingero catches it in real-time.`,

	Generate: generateIncident,
	GPUScript: `#!/usr/bin/env python3
"""Incident demo: normal baseline -> CPU storm -> CUDA latency spike -> recovery.

This is the WOW demo. Three phases:
  Phase 1 (10s): Clean training baseline — fast, consistent CUDA ops.
  Phase 2 (12s): stress-ng steals ALL CPU cores — cudaStreamSync spikes.
  Phase 3 (8s):  Storm ends, latencies recover.

Ingero traces all three layers (CUDA + host + driver) and shows the causal
chain: sched_switch storm -> driver thread preempted -> cudaStreamSync spike.
"""
import os, time, subprocess, torch

a = torch.randn(2048, 2048, device="cuda")

# Phase 1: Normal training baseline
print("Phase 1: Normal training baseline (10s)...")
start = time.time()
while time.time() - start < 10:
    torch.mm(a, a)
    torch.cuda.synchronize()

# Phase 2: INCIDENT — CPU storm (simulate logrotate / cron stealing cores)
ncpu = os.cpu_count() or 4
print(f"Phase 2: INCIDENT — CPU storm ({ncpu} stress-ng workers, 12s)...")
stress = subprocess.Popen(
    ["stress-ng", "--cpu", str(ncpu), "--timeout", "12s"],
    stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
start = time.time()
while time.time() - start < 12:
    torch.mm(a, a)
    torch.cuda.synchronize()
stress.wait()

# Phase 3: Recovery
print("Phase 3: Recovery (storm ended, 8s)...")
start = time.time()
while time.time() - start < 8:
    torch.mm(a, a)
    torch.cuda.synchronize()

print("Done.")
`,
}

func generateIncident(ctx context.Context, ch chan<- events.Event, speed float64) {
	// Phase 1: Normal steady-state (10s baseline).
	// Typical training loop: launch → memcpy → sync, repeat.
	for i := 0; i < 100; i++ {
		if !emit(ctx, ch, makeEvent(events.CUDALaunchKernel, jitter(14*time.Microsecond, 0.2)), speed) {
			return
		}
		if !emit(ctx, ch, makeEvent(events.CUDAMemcpy, jitter(800*time.Microsecond, 0.15)), speed) {
			return
		}
		if !emit(ctx, ch, makeEvent(events.CUDAStreamSync, jitter(16*time.Millisecond, 0.1)), speed) {
			return
		}
		// Occasional malloc.
		if i%20 == 0 {
			if !emit(ctx, ch, makeEvent(events.CUDAMalloc, jitter(120*time.Microsecond, 0.3)), speed) {
				return
			}
		}
	}

	// Phase 2: Incident — CPU storm begins.
	// Simulate stress-ng / logrotate stealing CPU cores.
	// sched_switch storm + cudaStreamSync latency spike.
	for i := 0; i < 50; i++ {
		// Host sched_switch events (preemption).
		for j := 0; j < 17; j++ { // ~847 total sched_switch events across incident
			evt := makeHostEvent(events.HostSchedSwitch, jitter(2100*time.Microsecond, 0.3))
			evt.Args = [2]uint64{8821, uint64(SyntheticPID)} // logrotate preempting our process
			if !emit(ctx, ch, evt, speed*5) {                // fast host events
				return
			}
		}

		// CUDA calls during contention — much slower.
		if !emit(ctx, ch, makeEvent(events.CUDALaunchKernel, jitter(45*time.Microsecond, 0.4)), speed) {
			return
		}
		if !emit(ctx, ch, makeEvent(events.CUDAMemcpy, jitter(3*time.Millisecond, 0.3)), speed) {
			return
		}
		// cudaStreamSync spike: 142ms instead of 16ms.
		if !emit(ctx, ch, makeEvent(events.CUDAStreamSync, jitter(142*time.Millisecond, 0.2)), speed) {
			return
		}

		// Some page allocations during memory pressure.
		if i%5 == 0 {
			evt := makeHostEvent(events.HostPageAlloc, jitter(50*time.Microsecond, 0.2))
			evt.Args = [2]uint64{4 * 1024 * 1024, 0} // 4MB allocation
			if !emit(ctx, ch, evt, speed*5) {
				return
			}
		}
	}

	// Phase 3: Recovery — storm ends.
	for i := 0; i < 50; i++ {
		if !emit(ctx, ch, makeEvent(events.CUDALaunchKernel, jitter(14*time.Microsecond, 0.2)), speed) {
			return
		}
		if !emit(ctx, ch, makeEvent(events.CUDAMemcpy, jitter(800*time.Microsecond, 0.15)), speed) {
			return
		}
		if !emit(ctx, ch, makeEvent(events.CUDAStreamSync, jitter(17*time.Millisecond, 0.1)), speed) {
			return
		}
	}
}
