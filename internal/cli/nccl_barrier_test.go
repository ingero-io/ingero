package cli

import (
	"context"
	"testing"
	"time"

	"github.com/ingero-io/ingero/internal/ebpf/ncclprobe"
	"github.com/ingero-io/ingero/internal/stats"
	"github.com/ingero-io/ingero/pkg/events"
)

// resetBarrierState resets the default correlator's state for test
// isolation. v0.12.2 (Sec ★4 / Arch ★4): correlator is now an
// instance method; tests can also construct an isolated
// newBarrierCorrelator if they need parallel runs.
func resetBarrierState() {
	defaultBarrier.reset()
	ncclBufMu.Lock()
	ncclBuf = nil
	ncclBufMu.Unlock()
}

// TestBarrierWait_HappyPath asserts the v0.12.1 LHF #1 follow-on:
// when an NCCL collective is followed by a cudaStreamSynchronize on
// the same (pid, stream), a barrier_wait_ms data point lands in the
// ncclBuffer with IsBarrier=true and the spec-correct duration.
//
// Spec: barrier_wait = sync.exit_ts - allreduce.entry_ts - sync.entry_ts
// With: allreduce.entry_ts = 1ms, sync.exit_ts = 5ms, sync.duration = 3ms
//   -> sync.entry_ts = 5 - 3 = 2ms
//   -> barrier_wait = 5 - 1 - 3 = 1ms
func TestBarrierWait_HappyPath(t *testing.T) {
	resetBarrierState()
	defer resetBarrierState()

	recordNCCLForBarrier(ncclprobe.Event{
		TimestampNs:  1_000_000, // 1ms
		PID:          42,
		Op:           3, // ncclAllReduce
		StreamHandle: 0xdeadbeef,
		Rank:         2,
		NRanks:       8,
		CommIDHash:   0xabc123,
	})
	onStreamSync(events.Event{
		Source:    events.SourceCUDA,
		Op:        uint8(events.CUDAStreamSync),
		PID:       42,
		Args:      [2]uint64{0xdeadbeef, 0},
		Duration:  3 * time.Millisecond,
		Timestamp: time.Unix(0, 5_000_000), // sync.exit at 5ms
	})

	got := ncclBufferDrain()
	if len(got) != 1 {
		t.Fatalf("expected 1 data point, got %d", len(got))
	}
	if !got[0].IsBarrier {
		t.Errorf("IsBarrier=false, want true")
	}
	if got[0].OpType != "ncclAllReduce" {
		t.Errorf("OpType=%q, want 'ncclAllReduce' (no prefix scheme post-v0.12.1)", got[0].OpType)
	}
	if got[0].DurationMs != 1.0 {
		t.Errorf("DurationMs=%v, want 1.0 (sync.exit - allreduce.entry - sync.duration)", got[0].DurationMs)
	}
	if got[0].Rank != 2 || got[0].NRanks != 8 {
		t.Errorf("Rank/NRanks lost in correlation: got %d/%d, want 2/8", got[0].Rank, got[0].NRanks)
	}
}

// TestBarrierWait_NoMatchingNCCL: a cudaStreamSynchronize without a
// preceding NCCL on the same (pid, stream) is a no-op (no data point
// emitted). Defends against false-attribution of generic stream syncs
// to NCCL.
func TestBarrierWait_NoMatchingNCCL(t *testing.T) {
	resetBarrierState()
	defer resetBarrierState()

	onStreamSync(events.Event{
		Source:   events.SourceCUDA,
		Op:       uint8(events.CUDAStreamSync),
		PID:      42,
		Args:     [2]uint64{0xfeed, 0},
		Duration: 1 * time.Millisecond,
	})
	if got := ncclBufferDrain(); len(got) != 0 {
		t.Errorf("orphan stream sync emitted %d data points; want 0", len(got))
	}
}

// TestBarrierWait_NCCLEntriesExpire: defaultBarrier.state entries past
// barrierExpiry are gc'd; a stream sync arriving long after the NCCL
// op was recorded must NOT emit (the correlation has timed out).
func TestBarrierWait_NCCLEntriesExpire(t *testing.T) {
	resetBarrierState()
	defer resetBarrierState()

	// Record an entry, then forcibly age it.
	recordNCCLForBarrier(ncclprobe.Event{
		TimestampNs:  1,
		PID:          42,
		Op:           3,
		StreamHandle: 0xbeef,
		Rank:         0,
		NRanks:       4,
	})
	defaultBarrier.mu.Lock()
	for k, e := range defaultBarrier.state {
		e.created = time.Now().Add(-2 * barrierExpiry)
		defaultBarrier.state[k] = e
	}
	defaultBarrier.mu.Unlock()

	// Trigger the gc by recording another entry.
	recordNCCLForBarrier(ncclprobe.Event{
		TimestampNs:  10,
		PID:          43,
		Op:           3,
		StreamHandle: 0xcafe,
		Rank:         0,
		NRanks:       4,
	})

	defaultBarrier.mu.Lock()
	_, expiredStillThere := defaultBarrier.state[barrierKey{pid: 42, stream: 0xbeef}]
	defaultBarrier.mu.Unlock()
	if expiredStillThere {
		t.Error("expired defaultBarrier.state entry was not gc'd")
	}
}

// TestBarrierWait_ZeroStreamSkipped: NCCL events without a stream
// handle (e.g. ncclCommInitRank/Destroy) must not enter defaultBarrier.state.
func TestBarrierWait_ZeroStreamSkipped(t *testing.T) {
	resetBarrierState()
	defer resetBarrierState()

	recordNCCLForBarrier(ncclprobe.Event{
		TimestampNs:  1,
		PID:          42,
		Op:           1, // ncclCommInitRank
		StreamHandle: 0,
	})
	defaultBarrier.mu.Lock()
	defer defaultBarrier.mu.Unlock()
	if len(defaultBarrier.state) != 0 {
		t.Errorf("CommInitRank with stream=0 should not enter defaultBarrier.state, got %d entries", len(defaultBarrier.state))
	}
}

// TestBarrierWait_StaleSyncDropped covers v0.12.1 (Sec audit ★5 #1):
// a cudaStreamSynchronize arriving more than staleBarrierWindow after
// the NCCL uretprobe is treated as orphaned (likely belongs to non-
// NCCL work on the same stream); the pending entry is dropped without
// emitting a phantom barrier_wait_ms.
func TestBarrierWait_StaleSyncDropped(t *testing.T) {
	resetBarrierState()
	defer resetBarrierState()

	recordNCCLForBarrier(ncclprobe.Event{
		TimestampNs:  1_000_000,
		PID:          42,
		Op:           3,
		StreamHandle: 0xbeef,
		Rank:         0,
		NRanks:       4,
	})
	// sync.entry_ts = 1_000_000 + 2*staleBarrierWindow.Nanoseconds()
	syncEntry := int64(1_000_000) + 2*staleBarrierWindow.Nanoseconds()
	onStreamSync(events.Event{
		Source:    events.SourceCUDA,
		Op:        uint8(events.CUDAStreamSync),
		PID:       42,
		Args:      [2]uint64{0xbeef, 0},
		Duration:  1 * time.Millisecond,
		Timestamp: time.Unix(0, syncEntry+1_000_000),
	})

	if got := ncclBufferDrain(); len(got) != 0 {
		t.Errorf("stale sync emitted %d data points; want 0 (Sec audit ★5 #1)", len(got))
	}
	// Pending entry should be cleared either way.
	defaultBarrier.mu.Lock()
	defer defaultBarrier.mu.Unlock()
	if _, present := defaultBarrier.state[barrierKey{pid: 42, stream: 0xbeef}]; present {
		t.Error("stale entry not cleared from defaultBarrier.state after sync miss")
	}
}

// TestForkCUDAForBarrier_CtxCancelClosesOut covers v0.12.1 (Sec audit
// ★4 #3 + QA #6): the forked channel must close when ctx is cancelled,
// not leak a goroutine waiting on `in`.
func TestForkCUDAForBarrier_CtxCancelClosesOut(t *testing.T) {
	in := make(chan events.Event, 4)
	ctx, cancel := context.WithCancel(context.Background())
	out := forkCUDAForBarrier(ctx, in)
	in <- events.Event{Source: events.SourceCUDA}
	<-out // confirm forwarding
	cancel()
	// out should close shortly even though `in` is still open.
	select {
	case _, ok := <-out:
		if ok {
			// Drain any remaining buffered event then expect close.
			select {
			case _, ok2 := <-out:
				if ok2 {
					t.Fatal("out did not close after ctx cancel")
				}
			case <-time.After(500 * time.Millisecond):
				t.Fatal("out did not close after ctx cancel (timeout)")
			}
		}
	case <-time.After(500 * time.Millisecond):
		t.Fatal("ctx cancel did not propagate to out close")
	}
}

// TestBarrierWait_ConcurrentRecordAndSync covers v0.12.1 (QA audit
// ★3 #8): exercise the barrierStateMu under producer (record) +
// consumer (sync) contention. Run with -race.
func TestBarrierWait_ConcurrentRecordAndSync(t *testing.T) {
	resetBarrierState()
	defer resetBarrierState()

	const N = 100
	done := make(chan struct{})
	go func() {
		for i := 0; i < N; i++ {
			recordNCCLForBarrier(ncclprobe.Event{
				TimestampNs:  uint64(i * 1_000_000),
				PID:          uint32(i),
				Op:           3,
				StreamHandle: uint64(i + 1),
				Rank:         0,
				NRanks:       4,
			})
		}
		close(done)
	}()
	for i := 0; i < N; i++ {
		onStreamSync(events.Event{
			Source:    events.SourceCUDA,
			Op:        uint8(events.CUDAStreamSync),
			PID:       uint32(i),
			Args:      [2]uint64{uint64(i + 1), 0},
			Duration:  1 * time.Millisecond,
			Timestamp: time.Unix(0, int64(i*1_000_000)+1_500_000),
		})
	}
	<-done
}

// TestFormatCommIDHash: 16-hex-char left-padded.
func TestFormatCommIDHash(t *testing.T) {
	got := formatCommIDHash(0xabc)
	if got != "0000000000000abc" {
		t.Errorf("formatCommIDHash(0xabc)=%q, want 0000000000000abc", got)
	}
	if got := formatCommIDHash(0xdeadbeefcafebabe); got != "deadbeefcafebabe" {
		t.Errorf("formatCommIDHash(0xdeadbeefcafebabe)=%q, want deadbeefcafebabe", got)
	}
}

// _ keeps the stats import live.
var _ = stats.NCCLDataPoint{}

// TestBarrierCorrelator_Isolated covers v0.12.2 (Sec ★4 / Arch ★4):
// each correlator instance owns independent state, so two parallel
// `ingero trace` invocations (or unit-test calls running in parallel)
// can't cross-contaminate.
func TestBarrierCorrelator_Isolated(t *testing.T) {
	var got1, got2 []stats.NCCLDataPoint
	c1 := newBarrierCorrelator(func(p stats.NCCLDataPoint) { got1 = append(got1, p) })
	c2 := newBarrierCorrelator(func(p stats.NCCLDataPoint) { got2 = append(got2, p) })

	c1.recordNCCL(ncclprobe.Event{
		TimestampNs: 1_000_000, PID: 42, Op: 3, StreamHandle: 0xA, NRanks: 4,
	})
	c2.onStreamSync(events.Event{
		Source: events.SourceCUDA, Op: uint8(events.CUDAStreamSync),
		PID: 42, Args: [2]uint64{0xA, 0},
		Duration: 1 * time.Millisecond,
		Timestamp: time.Unix(0, 5_000_000),
	})

	// c2 has no record for (42, 0xA) so emits nothing.
	if len(got2) != 0 {
		t.Errorf("c2 leaked from c1: got %d points", len(got2))
	}
	if len(got1) != 0 {
		t.Errorf("c1 emitted without its own sync: got %d points", len(got1))
	}

	// Now drive c1 properly; it MUST emit, c2 still silent.
	c1.onStreamSync(events.Event{
		Source: events.SourceCUDA, Op: uint8(events.CUDAStreamSync),
		PID: 42, Args: [2]uint64{0xA, 0},
		Duration: 1 * time.Millisecond,
		Timestamp: time.Unix(0, 5_000_000),
	})
	if len(got1) != 1 || !got1[0].IsBarrier {
		t.Errorf("c1 missed its own emission: %+v", got1)
	}
	if len(got2) != 0 {
		t.Errorf("c2 cross-contaminated: %d points", len(got2))
	}
}
