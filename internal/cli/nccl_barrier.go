package cli

// v0.12.1 (LHF #1 follow-on): agent-side correlation of NCCL collective
// uretprobe events with cudaStreamSynchronize events to derive
// `nccl.collective.barrier_wait_ms`. Both event streams flow through
// `internal/cli/trace.go`; correlation here keeps Fleet's
// ncclprocessor as a pass-through for this metric (the cross-rank
// peer-lag derivation already happens there from the duration_ms +
// barrier_wait_ms data points).
//
// Algorithm (per roadmap §4.3 v0.12.0 Workstream B):
//   - On every NCCL collective event with a non-zero stream_handle,
//     record (pid, stream_handle) -> {entry_ts, exit_ts, comm_id_hash,
//     op_type, rank, nranks, datatype, reduce_op, count_bytes}.
//   - On every cudaStreamSynchronize event for the same (pid, stream),
//     compute barrier_wait = (sync.exit_ts - sync.entry_ts) since the
//     NCCL collective is enqueued before the sync and the entire sync
//     duration is the barrier wait. Emit as a NCCLDataPoint flagged so
//     the OTLP encoder produces metric `nccl.collective.barrier_wait_ms`.
//   - Pending entries expire after barrierExpiry (5 min default) so a
//     missed sync doesn't leak memory.

import (
	"context"
	"sync"
	"time"

	"github.com/ingero-io/ingero/internal/ebpf/ncclprobe"
	"github.com/ingero-io/ingero/internal/stats"
	"github.com/ingero-io/ingero/pkg/events"
)

// forkCUDAForBarrier returns a new channel that forwards every CUDA
// event from `in`. As a side effect, every CUDAStreamSync event is
// also handed to onStreamSync for barrier-wait correlation. Closes the
// returned channel when `in` closes or `ctx` is cancelled.
//
// v0.12.1 (Sec audit ★4 #3): both the read from `in` and the write to
// `out` are guarded by ctx.Done so a stalled upstream OR a stalled
// downstream cannot leak this goroutine. Pre-fix the outer
// `for ev := range in` blocked uninterruptibly on a slow `in`.
func forkCUDAForBarrier(ctx context.Context, in <-chan events.Event) <-chan events.Event {
	out := make(chan events.Event, cap(in))
	go func() {
		defer close(out)
		for {
			select {
			case <-ctx.Done():
				return
			case ev, ok := <-in:
				if !ok {
					return
				}
				onStreamSync(ev)
				select {
				case <-ctx.Done():
					return
				case out <- ev:
				}
			}
		}
	}()
	return out
}

const barrierExpiry = 5 * time.Minute

type barrierKey struct {
	pid    uint32
	stream uint64
}

type barrierEntry struct {
	entryTs    int64
	commIDHash uint64
	opType     string
	rank       uint32
	nranks     uint32
	datatype   uint32
	reduceOp   uint32
	countBytes uint64
	created    time.Time
}

var (
	barrierStateMu sync.Mutex
	barrierState   = make(map[barrierKey]barrierEntry)
)

// recordNCCLForBarrier is called from the NCCL goroutine after each
// NCCL collective uretprobe event. Records the entry side so a later
// cudaStreamSynchronize on the same (pid, stream) can derive
// barrier_wait. NCCL CommInitRank / CommDestroy events (op codes 1, 2)
// are skipped: no stream involvement.
func recordNCCLForBarrier(ev ncclprobe.Event) {
	if ev.StreamHandle == 0 {
		return
	}
	if ev.Op == 1 || ev.Op == 2 {
		return
	}
	barrierStateMu.Lock()
	defer barrierStateMu.Unlock()
	gcBarrier(time.Now())
	barrierState[barrierKey{pid: ev.PID, stream: ev.StreamHandle}] = barrierEntry{
		entryTs:    int64(ev.TimestampNs),
		commIDHash: ev.CommIDHash,
		opType:     ev.OpName(),
		rank:       ev.Rank,
		nranks:     ev.NRanks,
		datatype:   ev.Datatype,
		reduceOp:   ev.ReduceOp,
		countBytes: ev.CountBytes,
		created:    time.Now(),
	}
}

// staleBarrierWindow bounds how long after the NCCL uretprobe a
// cudaStreamSynchronize is allowed to claim ownership. Beyond this the
// pending entry is treated as orphaned (the sync probably belongs to
// non-NCCL work on the same stream). Defends against false attribution
// when training scripts mix NCCL collectives with unrelated stream
// syncs, e.g. PyTorch DDP + checkpoint flush. v0.12.1 (Sec audit ★5).
const staleBarrierWindow = 1 * time.Second

// onStreamSync is called from the stream-sync tap goroutine for every
// CUDAStreamSync event. If a pending NCCL collective exists for the
// same (pid, stream) AND the sync entry is within staleBarrierWindow
// of the NCCL exit, emit a barrier_wait_ms data point and clear the
// entry. Older pending entries are dropped silently (correlation
// timed out). v0.12.1 fixes a phantom-attribution bug where a sync
// for unrelated work on the same stream would consume the pending NCCL
// entry and emit a misleading barrier_wait_ms.
func onStreamSync(ev events.Event) {
	if ev.Source != events.SourceCUDA {
		return
	}
	if ev.Op != uint8(events.CUDAStreamSync) {
		return
	}
	stream := ev.Args[0]
	if stream == 0 {
		return
	}
	syncEntryNs := ev.Timestamp.UnixNano() - ev.Duration.Nanoseconds()
	key := barrierKey{pid: ev.PID, stream: stream}
	barrierStateMu.Lock()
	entry, ok := barrierState[key]
	if ok {
		// Stale-entry check: only consume if the sync started within
		// staleBarrierWindow of the NCCL uretprobe. Older entries are
		// removed but NOT consumed (no barrier_wait emitted) so the
		// metric stays uncontaminated.
		if syncEntryNs-entry.entryTs > staleBarrierWindow.Nanoseconds() {
			delete(barrierState, key)
			barrierStateMu.Unlock()
			return
		}
		delete(barrierState, key)
	}
	barrierStateMu.Unlock()
	if !ok {
		return
	}
	// barrier_wait_ms = sync.exit_ts - allreduce.entry_ts - sync.entry_ts
	// (per roadmap §4.3 spec: time spent blocked AT the barrier).
	// Pre-fix used `ev.Duration` (= sync.exit - sync.entry) directly,
	// which under-reports when the user code does work between the
	// collective and the sync. Compute the spec-correct expression
	// here. For the typical PyTorch pattern (`dist.all_reduce(); torch.cuda.synchronize()`)
	// the difference is the sync's own duration only and matches.
	syncExitNs := ev.Timestamp.UnixNano()
	barrierNs := syncExitNs - entry.entryTs - ev.Duration.Nanoseconds()
	if barrierNs < 0 {
		barrierNs = 0
	}
	barrierMs := float64(barrierNs) / 1e6
	ncclBufferAdd(stats.NCCLDataPoint{
		TimestampUnixNano: ev.Timestamp.UnixNano(),
		OpType:            entry.opType,
		CommIDHash:        formatCommIDHash(entry.commIDHash),
		Rank:              entry.rank,
		NRanks:            entry.nranks,
		Datatype:           entry.datatype,
		ReduceOp:           entry.reduceOp,
		DurationMs:         barrierMs,
		CountBytes:         entry.countBytes,
		IsBarrier:          true,
	})
}

func gcBarrier(now time.Time) {
	cutoff := now.Add(-barrierExpiry)
	for k, v := range barrierState {
		if v.created.Before(cutoff) {
			delete(barrierState, k)
		}
	}
}

func formatCommIDHash(h uint64) string {
	const hexdigit = "0123456789abcdef"
	b := make([]byte, 16)
	for i := 15; i >= 0; i-- {
		b[i] = hexdigit[h&0xf]
		h >>= 4
	}
	return string(b)
}
