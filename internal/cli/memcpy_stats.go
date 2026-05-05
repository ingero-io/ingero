package cli

// Per-direction memcpy aggregator (v0.14 item C).
//
// CUDA memcpy events carry a direction byte in Args[1] (cudaMemcpyKind):
// 0=H2H, 1=H2D, 2=D2H, 3=D2D, 4=cudaMemcpyDefault. The OTLP exporter
// emits per-direction totals so dashboards can plot
// "host->device GB/s" separately from "device->device GB/s".
//
// The stats.Collector already groups by op-code, but op-code alone
// loses the direction dimension (every cudaMemcpy collapses onto a
// single counter regardless of direction). This aggregator carries
// the per-direction view in parallel.
//
// Last-window-wins: aggregator reset on each drain so each OTLP push
// reflects the events seen since the previous push. The cumulative
// counter shape is preserved by exporting a monotonically-growing
// totals map outside the per-window window aggregate; see drain
// helpers below.

import (
	"sync"

	"github.com/ingero-io/ingero/internal/stats"
	"github.com/ingero-io/ingero/pkg/events"
)

// memcpyStatsBuf is the per-direction aggregator state. Producer:
// the event-loop dispatcher. Consumer: drainMemcpyStats once per
// OTLP push.
var (
	memcpyStatsMu sync.Mutex

	// totals are monotonically increasing across the agent process
	// lifetime; the OTLP exporter emits them as
	// gpu.memcpy.bytes_total counters with delta=cumulative.
	memcpyBytesTotal = map[string]int64{}

	// durationSum / count let the exporter emit avg or precomputed
	// percentiles. We hold sum + count + last p50 / p95 / p99 from a
	// reservoir; full histogram serialization is deferred to v0.15
	// when we have an OTLP histogram encoder anyway.
	memcpyDurationNanosSum = map[string]int64{}
	memcpyDurationCount    = map[string]int64{}
)

// recordMemcpyEvent ingests one memcpy event from the merged event
// stream. Cheap (two map writes); called from the event-loop's
// per-event dispatch path so the throughput cost matters.
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
	durNs := ev.Duration.Nanoseconds()

	memcpyStatsMu.Lock()
	memcpyBytesTotal[dir] += bytes
	memcpyDurationNanosSum[dir] += durNs
	memcpyDurationCount[dir]++
	memcpyStatsMu.Unlock()
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
// observed since process start (totals) plus the per-direction
// duration sum/count for the OTLP exporter to translate into
// gauges or histograms.
//
// Last-window-wins is NOT applied to totals: the totals are
// monotonically growing for the OTLP "Sum" metric to be cumulative.
// The duration sum / count IS reset each drain so the exporter
// can emit a window average; long-window cumulative percentiles
// are out of scope for v0.14 (need a real histogram encoder).
func drainMemcpyStats() []stats.MemcpyDirStats {
	memcpyStatsMu.Lock()
	defer memcpyStatsMu.Unlock()

	if len(memcpyBytesTotal) == 0 {
		return nil
	}
	out := make([]stats.MemcpyDirStats, 0, len(memcpyBytesTotal))
	for dir, bytes := range memcpyBytesTotal {
		count := memcpyDurationCount[dir]
		sum := memcpyDurationNanosSum[dir]
		var avgMs float64
		if count > 0 {
			avgMs = float64(sum) / float64(count) / 1e6
		}
		out = append(out, stats.MemcpyDirStats{
			Direction:        dir,
			BytesTotal:       bytes,
			AverageDurationMs: avgMs,
			EventsInWindow:   count,
		})
	}
	// Reset window-window state; keep cumulative bytes.
	for k := range memcpyDurationCount {
		memcpyDurationCount[k] = 0
		memcpyDurationNanosSum[k] = 0
	}
	return out
}

// resetMemcpyStats wipes all module-level state. Test-only helper.
func resetMemcpyStats() {
	memcpyStatsMu.Lock()
	defer memcpyStatsMu.Unlock()
	memcpyBytesTotal = map[string]int64{}
	memcpyDurationNanosSum = map[string]int64{}
	memcpyDurationCount = map[string]int64{}
}

