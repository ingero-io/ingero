package cli

// libnccl process-discovery scanner wiring (v0.14 item A).
//
// Mirrors the throttle-poller pattern in throttle_poller.go: a
// background goroutine scans /proc on a configurable interval; the
// onSnapshot callback drains the latest result into the snapshot for
// OTLP emission as gpu.nccl.process_loaded + gpu.nccl.processes_total.

import (
	"context"
	"log/slog"
	"sync"
	"time"

	"github.com/ingero-io/ingero/internal/ebpf/ncclprobe"
	"github.com/ingero-io/ingero/internal/stats"
)

// nclDiscoveryBuf holds the most recent discovery scan result.
// Producer: scanner sink. Consumer: drainNCCLDiscoveryBuf at every
// snapshot tick. Last-batch-wins (a gauge is "current state").
//
// The non-nil empty-slice case is preserved (an explicit empty slice
// means "the scanner ran and found no NCCL processes") so the OTLP
// exporter still emits gpu.nccl.processes_total = 0 in that case;
// nil means "scanner has not run yet, do not emit".
var (
	ncclDiscoveryBufMu sync.Mutex
	ncclDiscoveryBuf   []stats.NCCLProcessReading
	ncclDiscoveryReady bool
)

// setNCCLDiscoveryBatch installs the latest batch from the scanner.
// Empty-but-non-nil batches are preserved so the OTLP exporter knows
// to emit a zero count.
func setNCCLDiscoveryBatch(batch []ncclprobe.NCCLProcess) {
	conv := make([]stats.NCCLProcessReading, len(batch))
	for i, p := range batch {
		conv[i] = stats.NCCLProcessReading{
			PID:        p.PID,
			Comm:       p.Comm,
			LibPath:    p.LibPath,
			LibVersion: p.LibVersion,
		}
	}
	ncclDiscoveryBufMu.Lock()
	ncclDiscoveryBuf = conv
	ncclDiscoveryReady = true
	ncclDiscoveryBufMu.Unlock()
}

// drainNCCLDiscoveryBuf returns the latest batch (or nil if no scan
// has completed yet). The buffer is NOT cleared; subsequent snapshot
// ticks will see the same batch until the scanner produces a new one.
func drainNCCLDiscoveryBuf() []stats.NCCLProcessReading {
	ncclDiscoveryBufMu.Lock()
	defer ncclDiscoveryBufMu.Unlock()
	if !ncclDiscoveryReady {
		return nil
	}
	out := make([]stats.NCCLProcessReading, len(ncclDiscoveryBuf))
	copy(out, ncclDiscoveryBuf)
	return out
}

// resetNCCLDiscoveryState clears module-level state. Test-only helper.
func resetNCCLDiscoveryState() {
	ncclDiscoveryBufMu.Lock()
	defer ncclDiscoveryBufMu.Unlock()
	ncclDiscoveryBuf = nil
	ncclDiscoveryReady = false
}

// startNCCLDiscoveryScanner spawns the goroutine that periodically
// enumerates /proc, looks up libnccl per PID, and forwards results
// to the snapshot drain. interval <= 0 disables the feature; the
// snapshot will see no NCCL discovery readings.
//
// v0.15 F1: when nt is non-nil, the sink also calls
// nt.AttachAt(libPath) for each unique libnccl path the scanner
// finds, so PyTorch+pip workloads (which ship libnccl in a venv that
// startup-time eager attach can never see) get NCCL collective
// uprobes attached as soon as the workload boots.
func startNCCLDiscoveryScanner(ctx context.Context, interval time.Duration, log *slog.Logger, nt *ncclprobe.Tracer) {
	if interval <= 0 {
		return
	}
	if log == nil {
		log = slog.Default()
	}
	scanner := ncclprobe.NewScanner(
		ncclprobe.ProcPIDLister(),
		func(batch []ncclprobe.NCCLProcess) {
			setNCCLDiscoveryBatch(batch)
			log.Debug("nccl-discovery: scan complete", "n_processes", len(batch))
			if nt == nil {
				return
			}
			seen := map[string]bool{}
			for _, p := range batch {
				if p.LibPath == "" || seen[p.LibPath] {
					continue
				}
				seen[p.LibPath] = true
				if err := nt.AttachAt(p.LibPath); err != nil {
					log.Debug("nccl-discovery: AttachAt failed",
						"path", p.LibPath, "err", err)
				}
			}
		},
		interval,
	)
	go scanner.Run(ctx)
}
