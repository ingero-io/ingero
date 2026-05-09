// v0.15 F2: running counters for NCCL collectives so the Prometheus
// /metrics exporter has something to emit. The OTLP path emits one
// per-event gauge per collective (nccl.collective.duration_ms /
// .bytes / .barrier_wait_ms) which does not fit Prometheus pull
// semantics; the running counters here are the pull-friendly slice.
//
// Counters are monotonic across the agent process lifetime. Drained
// snapshot-style by the Prometheus emitter (via stats.Snapshot) but
// the underlying state never resets.
package cli

import (
	"sync"

	"github.com/ingero-io/ingero/internal/stats"
)

var (
	ncclCounterMu        sync.Mutex
	ncclCountByOp        = map[string]int64{}
	ncclBytesTotalByOp   = map[string]int64{}
	ncclBarrierTotalByOp = map[string]int64{} // barrier wait events per op
)

// recordNCCLCollective increments the running counters for one NCCL
// collective data point. Idempotent shape: every NCCLDataPoint that
// reaches the snapshot buffer also goes through this function.
//
// IsBarrier=true is treated as the same op_type but counted into a
// separate barrier-wait family so dashboards can plot waits without
// double-counting them as collectives.
func recordNCCLCollective(p stats.NCCLDataPoint) {
	if p.OpType == "" {
		return
	}
	ncclCounterMu.Lock()
	defer ncclCounterMu.Unlock()
	if p.IsBarrier {
		ncclBarrierTotalByOp[p.OpType]++
		return
	}
	ncclCountByOp[p.OpType]++
	ncclBytesTotalByOp[p.OpType] += int64(p.CountBytes)
}

// snapshotNCCLCollectiveCounters returns a copy of the running
// counters for snapshot consumers. Returns nil when no events have
// been recorded yet so the Prometheus emitter can stay silent.
func snapshotNCCLCollectiveCounters() []stats.NCCLCollectiveCounter {
	ncclCounterMu.Lock()
	defer ncclCounterMu.Unlock()
	if len(ncclCountByOp) == 0 && len(ncclBarrierTotalByOp) == 0 {
		return nil
	}
	out := make([]stats.NCCLCollectiveCounter, 0, len(ncclCountByOp)+len(ncclBarrierTotalByOp))
	for op, count := range ncclCountByOp {
		out = append(out, stats.NCCLCollectiveCounter{
			OpType:     op,
			Count:      count,
			BytesTotal: ncclBytesTotalByOp[op],
		})
	}
	for op, count := range ncclBarrierTotalByOp {
		out = append(out, stats.NCCLCollectiveCounter{
			OpType:        op,
			BarrierEvents: count,
		})
	}
	return out
}

// resetNCCLCollectiveCounters wipes module state. Test-only.
func resetNCCLCollectiveCounters() {
	ncclCounterMu.Lock()
	defer ncclCounterMu.Unlock()
	ncclCountByOp = map[string]int64{}
	ncclBytesTotalByOp = map[string]int64{}
	ncclBarrierTotalByOp = map[string]int64{}
}
