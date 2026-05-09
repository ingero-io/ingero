package cli

// Per-direction memcpy aggregator (v0.14 item C, v0.15 item C).
//
// CUDA memcpy events carry a direction byte in Args[1] (cudaMemcpyKind):
// 0=H2H, 1=H2D, 2=D2H, 3=D2D, 4=cudaMemcpyDefault. The OTLP exporter
// emits per-direction totals so dashboards can plot
// "host->device GB/s" separately from "device->device GB/s".
//
// v0.15: per-event duration histogram replaces per-window-average
// gauge. Bucket layout from stats.DefaultMemcpyDurationBoundsMs.

import (
	"sync"

	"github.com/ingero-io/ingero/internal/stats"
	"github.com/ingero-io/ingero/pkg/events"
)

var (
	memcpyStatsMu sync.Mutex

	// totals are monotonically increasing across the agent process
	// lifetime; the OTLP exporter emits them as
	// gpu.memcpy.bytes_total counters with delta=cumulative.
	memcpyBytesTotal = map[string]int64{}

	// memcpyDurationHist is per-direction; created lazily on first
	// observation. v0.15 cumulative: Prometheus histogram convention
	// is `_count`/`_sum`/`_bucket` are monotonically growing across
	// process lifetime (matches AggregationTemporality=cumulative on
	// the OTLP side). Resetting on drain produced near-empty rows in
	// /metrics whenever the scrape lagged the workload by more than
	// the snapshot interval (the v0.14.2 / v0.15 e2e regression
	// found 2026-05-07 on Lambda A10).
	memcpyDurationHist = map[string]*stats.Histogram{}
)

// recordMemcpyEvent ingests one memcpy event from the merged event
// stream. Cheap (one map write + one histogram observe); called from
// the event-loop's per-event dispatch path so the throughput cost
// matters.
//
// Op codes: CUDAMemcpy / Async / 2D / 2D_Async / Peer / PeerAsync.
// The Peer variants always carry direction=D2D; the 2D variants
// carry direction=unknown because the kind argument is the 7th
// parameter (BPF can't read it portably).
func recordMemcpyEvent(ev events.Event) {
	if ev.Source != events.SourceCUDA {
		return
	}
	switch events.CUDAOp(ev.Op) {
	case events.CUDAMemcpy, events.CUDAMemcpyAsync,
		events.CUDAMemcpy2D, events.CUDAMemcpy2DAsync,
		events.CUDAMemcpyPeer, events.CUDAMemcpyPeerAsync:
		// fall through
	default:
		return
	}

	dir := memcpyDirectionString(uint8(ev.Args[1]))
	bytes := int64(ev.Args[0])
	durMs := float64(ev.Duration.Nanoseconds()) / 1e6

	memcpyStatsMu.Lock()
	memcpyBytesTotal[dir] += bytes
	h, ok := memcpyDurationHist[dir]
	if !ok {
		h = stats.NewHistogram(stats.DefaultMemcpyDurationBoundsMs)
		memcpyDurationHist[dir] = h
	}
	memcpyStatsMu.Unlock()
	if stats.IsFinite(durMs) {
		h.Observe(durMs)
	}
}

// memcpyDirectionString translates a raw cudaMemcpyKind byte (Args[1])
// to an OTLP-friendly label. Mirrors internal/health.signal_collector
// memcpyDirectionLabel but lives in cli/ because the agent's main
// event loop does not import internal/health.
func memcpyDirectionString(b uint8) string {
	switch b {
	case 0:
		return "h2h"
	case 1:
		return "h2d"
	case 2:
		return "d2h"
	case 3:
		return "d2d"
	case 4:
		return "default"
	default:
		return "unknown"
	}
}

// drainMemcpyStats returns one stats.MemcpyDirStats per direction
// observed since process start. Both BytesTotal and the per-
// direction histogram are CUMULATIVE: snapshotted but not reset.
// The Prometheus histogram convention treats `_count`/`_sum` as
// monotonically growing series; OTLP cumulative-temporality (set
// by the histogramMetric helper) does the same. v0.15 item C.
//
// EventsInWindow is preserved as a snapshot of count for callers
// that want a "running total" without having to inspect the
// histogram.
func drainMemcpyStats() []stats.MemcpyDirStats {
	memcpyStatsMu.Lock()
	defer memcpyStatsMu.Unlock()

	if len(memcpyBytesTotal) == 0 {
		return nil
	}
	out := make([]stats.MemcpyDirStats, 0, len(memcpyBytesTotal))
	for dir, bytes := range memcpyBytesTotal {
		var snap stats.HistogramSnapshot
		var count int64
		if h, ok := memcpyDurationHist[dir]; ok {
			snap = h.Snapshot()
			count = int64(snap.Count)
			// Cumulative: do NOT reset.
		}
		out = append(out, stats.MemcpyDirStats{
			Direction:         dir,
			BytesTotal:        bytes,
			DurationHistogram: snap,
			EventsInWindow:    count,
		})
	}
	return out
}

// resetMemcpyStats wipes all module-level state. Test-only helper.
func resetMemcpyStats() {
	memcpyStatsMu.Lock()
	defer memcpyStatsMu.Unlock()
	memcpyBytesTotal = map[string]int64{}
	memcpyDurationHist = map[string]*stats.Histogram{}
}
