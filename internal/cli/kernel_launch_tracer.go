package cli

import (
	"context"
	"log/slog"
	"sort"
	"sync"

	"github.com/ingero-io/ingero/internal/ebpf/kernellaunch"
	"github.com/ingero-io/ingero/internal/stats"
)

// v0.15 item M: per-process kernel-launch aggregates fed by the
// libcuda.so cuLaunchKernel uprobe.
//
// Producer: the BPF ringbuf reader started by startKernelLaunchTracer.
// Consumer: snapshotKernelLaunchCounters once per Prometheus / OTLP
// push. The exporter emits:
//   - gpu.kernel.launch.count{pid}     (cumulative counter)
//   - gpu.kernel.launch.threads_per_block{pid}  (histogram)
//   - gpu.kernel.launch.grid_blocks{pid}        (histogram)

var (
	kernelLaunchMu sync.Mutex
	kernelLaunchCount = map[uint32]int64{}
	kernelLaunchTPB   = map[uint32]*stats.Histogram{}
	kernelLaunchGB    = map[uint32]*stats.Histogram{}
)

// kernelLaunchTPBBoundaries: typical CUDA threads_per_block range
// from 32 (single warp) up to 1024 (CUDA hardware limit). Geometric
// progression to capture both tiny launches and full-tile launches.
var kernelLaunchTPBBoundaries = []float64{32, 64, 128, 256, 512, 1024}

// kernelLaunchGBBoundaries: typical grid_blocks range from 1 (a
// single block; rare) to ~1M (large parallel reduction over many
// rows). Geometric progression.
var kernelLaunchGBBoundaries = []float64{1, 4, 16, 64, 256, 1024, 4096, 16384, 65536, 262144, 1048576}

func recordKernelLaunchEvent(ev kernellaunch.Event) {
	kernelLaunchMu.Lock()
	kernelLaunchCount[ev.PID]++
	tpb, ok := kernelLaunchTPB[ev.PID]
	if !ok {
		tpb = stats.NewHistogram(kernelLaunchTPBBoundaries)
		kernelLaunchTPB[ev.PID] = tpb
	}
	gb, ok := kernelLaunchGB[ev.PID]
	if !ok {
		gb = stats.NewHistogram(kernelLaunchGBBoundaries)
		kernelLaunchGB[ev.PID] = gb
	}
	kernelLaunchMu.Unlock()
	if v := ev.ThreadsPerBlock(); v > 0 {
		tpb.Observe(float64(v))
	}
	if v := ev.TotalGridBlocks(); v > 0 {
		gb.Observe(float64(v))
	}
}

// snapshotKernelLaunchCounters returns one stats.KernelLaunchSnapshot
// per PID, sorted by PID ascending. nil when no events have been
// seen. Caller (onSnapshot) embeds the slice into stats.Snapshot.
func snapshotKernelLaunchCounters() []stats.KernelLaunchSnapshot {
	kernelLaunchMu.Lock()
	defer kernelLaunchMu.Unlock()
	if len(kernelLaunchCount) == 0 {
		return nil
	}
	pids := make([]uint32, 0, len(kernelLaunchCount))
	for p := range kernelLaunchCount {
		pids = append(pids, p)
	}
	sort.Slice(pids, func(i, j int) bool { return pids[i] < pids[j] })
	out := make([]stats.KernelLaunchSnapshot, 0, len(pids))
	for _, p := range pids {
		s := stats.KernelLaunchSnapshot{
			PID:   p,
			Count: kernelLaunchCount[p],
		}
		if h := kernelLaunchTPB[p]; h != nil {
			s.ThreadsPerBlockHist = h.Snapshot()
		}
		if h := kernelLaunchGB[p]; h != nil {
			s.GridBlocksHist = h.Snapshot()
		}
		out = append(out, s)
	}
	return out
}

func resetKernelLaunchCounters() {
	kernelLaunchMu.Lock()
	defer kernelLaunchMu.Unlock()
	kernelLaunchCount = map[uint32]int64{}
	kernelLaunchTPB = map[uint32]*stats.Histogram{}
	kernelLaunchGB = map[uint32]*stats.Histogram{}
}

// startKernelLaunchTracer attaches the v0.15 cuLaunchKernel uprobe
// at libcudaPath and spawns a drain goroutine.
func startKernelLaunchTracer(ctx context.Context, libcudaPath string, log *slog.Logger) {
	tr := kernellaunch.New(libcudaPath)
	if err := tr.Attach(); err != nil {
		log.Info("kernel-launch tracer: attach failed; counter will stay empty", "err", err)
		return
	}
	go func() {
		defer tr.Close()
		errCh := make(chan error, 1)
		go func() { errCh <- tr.Run(ctx) }()
		for {
			select {
			case <-ctx.Done():
				return
			case ev, ok := <-tr.Events():
				if !ok {
					return
				}
				recordKernelLaunchEvent(ev)
			case err := <-errCh:
				if err != nil {
					log.Debug("kernel-launch tracer: Run exited", "err", err)
				}
				return
			}
		}
	}()
	log.Info("kernel-launch tracer: attached to cuLaunchKernel", "lib", libcudaPath)
}
