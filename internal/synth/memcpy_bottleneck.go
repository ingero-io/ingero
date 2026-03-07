package synth

import (
	"context"
	"math/rand"
	"time"

	"github.com/ingero-io/ingero/pkg/events"
)

func init() {
	register(&Scenario{
		Name:  "memcpy-bottleneck",
		Title: "Memory Transfer Bottleneck",
		Description: "cudaMemcpy dominates wall-clock time despite similar event " +
			"counts — data movement is the bottleneck, not compute",
		Insight: "38%+ of wall-clock time is in cudaMemcpy, not compute. " +
			"nvidia-smi says 98% GPU utilization, but the compute cores are idle during transfers.",
		Generate: generateMemcpyBottleneck,
		GPUScript: `#!/usr/bin/env python3
"""Memcpy bottleneck demo: data transfers dominate wall-clock time."""
import torch

print("Running memcpy-heavy workload (200 iterations)...")
print("Watch the WALL% column — cudaMemcpy will dominate.")

device = torch.device("cuda")

for i in range(200):
    # Large H→D transfer: slow, dominates wall time.
    host_tensor = torch.randn(4096, 4096)
    gpu_tensor = host_tensor.to(device)        # H→D memcpy (~5ms)

    # Small kernel launch: fast, similar event count.
    result = torch.mm(gpu_tensor, gpu_tensor)   # kernel launch (~15µs)

    # D→H transfer: another slow memcpy.
    _ = result.cpu()                            # D→H memcpy (~5ms)

    torch.cuda.synchronize()

print("Done.")
`,
	})
}

// generateMemcpyBottleneck produces one cycle (~200 events) where cudaMemcpy
// dominates wall-clock time.
//
// The key insight: cudaMemcpy events have similar COUNT to cudaLaunchKernel,
// but each cudaMemcpy takes ~5ms while each launch takes ~15µs. The WALL%
// column in the stats table reveals that memory transfers consume 50%+ of
// wall time — invisible in nvidia-smi which shows "98% GPU utilization."
func generateMemcpyBottleneck(ctx context.Context, ch chan<- events.Event, speed float64) {
	for i := 0; i < 200; i++ {
		r := rand.Intn(100)
		var evt events.Event

		switch {
		case r < 35:
			// cudaMemcpy: ~5ms each — dominates wall time.
			// Simulates large H→D transfers for each training batch.
			evt = makeEvent(events.CUDAMemcpy, jitter(5*time.Millisecond, 0.3))

		case r < 70:
			// cudaLaunchKernel: ~15µs each — fast but many.
			// Similar event count as memcpy, but tiny duration.
			evt = makeEvent(events.CUDALaunchKernel, jitter(15*time.Microsecond, 0.2))

		case r < 80:
			// cudaDeviceSync: ~150µs (torch.cuda.synchronize()).
			evt = makeEvent(events.CUDADeviceSync, jitter(150*time.Microsecond, 0.2))

		case r < 90:
			// cudaStreamSync: ~150µs — per-stream sync.
			evt = makeEvent(events.CUDAStreamSync, jitter(150*time.Microsecond, 0.2))

		default:
			// cudaMalloc: ~50µs — infrequent, normal latency.
			evt = makeEvent(events.CUDAMalloc, jitter(50*time.Microsecond, 0.2))
		}

		if !emit(ctx, ch, evt, speed) {
			return
		}
	}
}
