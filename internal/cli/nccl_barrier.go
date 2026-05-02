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
//
// v0.12.2 (Sec ★4 / Arch ★4): correlator state lives on a
// barrierCorrelator struct instead of package-scope vars, so test
// isolation is clean and a future long-running `ingero serve` mode can
// instantiate one per Tracer.

import (
	"context"
	"sync"
	"time"

	"github.com/ingero-io/ingero/internal/ebpf/ncclprobe"
	"github.com/ingero-io/ingero/internal/stats"
	"github.com/ingero-io/ingero/pkg/events"
)

const barrierExpiry = 5 * time.Minute

// staleBarrierWindow bounds how long after the NCCL uretprobe a
// cudaStreamSynchronize is allowed to claim ownership. Beyond this the
// pending entry is treated as orphaned (the sync probably belongs to
// non-NCCL work on the same stream). Defends against false attribution
// when training scripts mix NCCL collectives with unrelated stream
// syncs, e.g. PyTorch DDP + checkpoint flush. v0.12.1 (Sec audit ★5).
const staleBarrierWindow = 1 * time.Second

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

// barrierCorrelator owns the NCCL/CUDA-sync correlation map. One
// instance per `ingero trace` invocation; no package-scope state.
type barrierCorrelator struct {
	mu    sync.Mutex
	state map[barrierKey]barrierEntry
	out   func(stats.NCCLDataPoint) // typically ncclBufferAdd
}

// newBarrierCorrelator creates a correlator that pushes derived
// barrier_wait data points into out. out is required.
func newBarrierCorrelator(out func(stats.NCCLDataPoint)) *barrierCorrelator {
	return &barrierCorrelator{
		state: make(map[barrierKey]barrierEntry),
		out:   out,
	}
}

// defaultBarrier is the singleton correlator used by the v0.12.0/v0.12.1
// trace-command wiring. v0.12.2 keeps it as a default for the
// agent-CLI use case while the new `barrierCorrelator` type lets tests
// (and any future long-running mode) construct isolated correlators.
var defaultBarrier = newBarrierCorrelator(func(p stats.NCCLDataPoint) { ncclBufferAdd(p) })

// recordNCCLForBarrier records the NCCL entry side on the default
// correlator. Called from the NCCL goroutine after each NCCL collective
// uretprobe event. NCCL CommInitRank / CommDestroy events (op codes 1, 2)
// are skipped: no stream involvement.
func recordNCCLForBarrier(ev ncclprobe.Event) { defaultBarrier.recordNCCL(ev) }

// onStreamSync handles a CUDAStreamSync event on the default correlator.
func onStreamSync(ev events.Event) { defaultBarrier.onStreamSync(ev) }

func (c *barrierCorrelator) recordNCCL(ev ncclprobe.Event) {
	if ev.StreamHandle == 0 {
		return
	}
	if ev.Op == 1 || ev.Op == 2 {
		return
	}
	c.mu.Lock()
	defer c.mu.Unlock()
	c.gcLocked(time.Now())
	c.state[barrierKey{pid: ev.PID, stream: ev.StreamHandle}] = barrierEntry{
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

func (c *barrierCorrelator) onStreamSync(ev events.Event) {
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
	c.mu.Lock()
	entry, ok := c.state[key]
	if ok {
		if syncEntryNs-entry.entryTs > staleBarrierWindow.Nanoseconds() {
			delete(c.state, key)
			c.mu.Unlock()
			return
		}
		delete(c.state, key)
	}
	c.mu.Unlock()
	if !ok {
		return
	}
	syncExitNs := ev.Timestamp.UnixNano()
	barrierNs := syncExitNs - entry.entryTs - ev.Duration.Nanoseconds()
	if barrierNs < 0 {
		barrierNs = 0
	}
	c.out(stats.NCCLDataPoint{
		TimestampUnixNano: ev.Timestamp.UnixNano(),
		OpType:            entry.opType,
		CommIDHash:        formatCommIDHash(entry.commIDHash),
		Rank:              entry.rank,
		NRanks:            entry.nranks,
		Datatype:          entry.datatype,
		ReduceOp:          entry.reduceOp,
		DurationMs:        float64(barrierNs) / 1e6,
		CountBytes:        entry.countBytes,
		IsBarrier:         true,
	})
}

// gcLocked removes entries older than barrierExpiry. Caller must hold c.mu.
func (c *barrierCorrelator) gcLocked(now time.Time) {
	cutoff := now.Add(-barrierExpiry)
	for k, v := range c.state {
		if v.created.Before(cutoff) {
			delete(c.state, k)
		}
	}
}

// reset is for tests — empties the correlator state.
func (c *barrierCorrelator) reset() {
	c.mu.Lock()
	defer c.mu.Unlock()
	c.state = make(map[barrierKey]barrierEntry)
}

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

func formatCommIDHash(h uint64) string {
	const hexdigit = "0123456789abcdef"
	b := make([]byte, 16)
	for i := 15; i >= 0; i-- {
		b[i] = hexdigit[h&0xf]
		h >>= 4
	}
	return string(b)
}
