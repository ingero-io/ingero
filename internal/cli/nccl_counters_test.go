package cli

import (
	"sync"
	"testing"

	"github.com/ingero-io/ingero/internal/stats"
)

func TestRecordNCCLCollective_BasicTallies(t *testing.T) {
	resetNCCLCollectiveCounters()
	defer resetNCCLCollectiveCounters()

	recordNCCLCollective(stats.NCCLDataPoint{OpType: "ncclAllReduce", CountBytes: 1024})
	recordNCCLCollective(stats.NCCLDataPoint{OpType: "ncclAllReduce", CountBytes: 2048})
	recordNCCLCollective(stats.NCCLDataPoint{OpType: "ncclBcast", CountBytes: 512})

	got := snapshotNCCLCollectiveCounters()
	if len(got) != 2 {
		t.Fatalf("expected 2 op-types, got %d (%+v)", len(got), got)
	}
	byOp := map[string]stats.NCCLCollectiveCounter{}
	for _, c := range got {
		byOp[c.OpType] = c
	}
	if byOp["ncclAllReduce"].Count != 2 || byOp["ncclAllReduce"].BytesTotal != 1024+2048 {
		t.Errorf("ncclAllReduce got %+v", byOp["ncclAllReduce"])
	}
	if byOp["ncclBcast"].Count != 1 || byOp["ncclBcast"].BytesTotal != 512 {
		t.Errorf("ncclBcast got %+v", byOp["ncclBcast"])
	}
}

func TestRecordNCCLCollective_BarrierSeparate(t *testing.T) {
	resetNCCLCollectiveCounters()
	defer resetNCCLCollectiveCounters()

	recordNCCLCollective(stats.NCCLDataPoint{OpType: "ncclAllReduce", CountBytes: 1024})
	recordNCCLCollective(stats.NCCLDataPoint{OpType: "ncclAllReduce", IsBarrier: true})
	recordNCCLCollective(stats.NCCLDataPoint{OpType: "ncclAllReduce", IsBarrier: true})

	got := snapshotNCCLCollectiveCounters()
	if len(got) != 2 {
		t.Fatalf("expected separate collective + barrier rows, got %d (%+v)", len(got), got)
	}
	var collective, barrier stats.NCCLCollectiveCounter
	for _, c := range got {
		if c.BarrierEvents > 0 {
			barrier = c
		} else {
			collective = c
		}
	}
	if collective.Count != 1 || collective.BytesTotal != 1024 {
		t.Errorf("collective row wrong: %+v", collective)
	}
	if barrier.BarrierEvents != 2 {
		t.Errorf("barrier row wrong: %+v", barrier)
	}
}

func TestRecordNCCLCollective_EmptyOpTypeIgnored(t *testing.T) {
	resetNCCLCollectiveCounters()
	defer resetNCCLCollectiveCounters()

	recordNCCLCollective(stats.NCCLDataPoint{OpType: "", CountBytes: 999})
	if got := snapshotNCCLCollectiveCounters(); got != nil {
		t.Errorf("expected nil snapshot after empty-op-type record, got %+v", got)
	}
}

func TestSnapshotNCCLCollectiveCounters_NilOnEmpty(t *testing.T) {
	resetNCCLCollectiveCounters()
	if got := snapshotNCCLCollectiveCounters(); got != nil {
		t.Errorf("expected nil on empty state, got %+v", got)
	}
}

// v0.15 F2 (HIGH): the running counter survives ringbuf back-pressure.
// When ncclBuf is full and ncclBufferAdd skips appending the data
// point, the counter MUST still increment so the total stays honest.
// Documented in the ncclBufferAdd comment; here we exercise it.
func TestNCCLBufferAdd_CounterSurvivesBufferFull(t *testing.T) {
	resetNCCLCollectiveCounters()
	defer resetNCCLCollectiveCounters()
	// Save + reset ncclBuf state.
	ncclBufMu.Lock()
	saved := ncclBuf
	ncclBuf = make([]stats.NCCLDataPoint, ncclBufMax) // pre-filled, at cap
	ncclBufMu.Unlock()
	defer func() {
		ncclBufMu.Lock()
		ncclBuf = saved
		ncclBufMu.Unlock()
	}()

	// One more push: ringbuf full, append is skipped, counter SHOULD
	// still tick.
	ncclBufferAdd(stats.NCCLDataPoint{OpType: "ncclAllReduce", CountBytes: 4096})

	got := snapshotNCCLCollectiveCounters()
	if len(got) != 1 || got[0].OpType != "ncclAllReduce" {
		t.Fatalf("expected ncclAllReduce row, got %+v", got)
	}
	if got[0].Count != 1 {
		t.Errorf("expected Count=1 even under buffer-full, got %d", got[0].Count)
	}
	if got[0].BytesTotal != 4096 {
		t.Errorf("expected BytesTotal=4096 even under buffer-full, got %d", got[0].BytesTotal)
	}
	// And the buffer was NOT extended (still at its prefilled cap).
	ncclBufMu.Lock()
	bufLen := len(ncclBuf)
	ncclBufMu.Unlock()
	if bufLen != ncclBufMax {
		t.Errorf("buffer should not have grown past cap, len=%d max=%d", bufLen, ncclBufMax)
	}
}

// v0.15 F2 (MED): integration shape: ncclBufferAdd produces data
// points that flow through to snapshotNCCLCollectiveCounters with
// correct totals across multiple ops.
func TestNCCLCollectiveCounters_EndToEndChain(t *testing.T) {
	resetNCCLCollectiveCounters()
	defer resetNCCLCollectiveCounters()
	// Reset the buffer too so this test is self-contained.
	ncclBufMu.Lock()
	saved := ncclBuf
	ncclBuf = nil
	ncclBufMu.Unlock()
	defer func() {
		ncclBufMu.Lock()
		ncclBuf = saved
		ncclBufMu.Unlock()
	}()

	ncclBufferAdd(stats.NCCLDataPoint{OpType: "ncclAllReduce", CountBytes: 1024})
	ncclBufferAdd(stats.NCCLDataPoint{OpType: "ncclAllReduce", CountBytes: 2048})
	ncclBufferAdd(stats.NCCLDataPoint{OpType: "ncclBcast", CountBytes: 512})
	ncclBufferAdd(stats.NCCLDataPoint{OpType: "ncclAllReduce", IsBarrier: true})
	ncclBufferAdd(stats.NCCLDataPoint{OpType: "ncclAllReduce", IsBarrier: true})

	rows := snapshotNCCLCollectiveCounters()
	byOp := map[string]stats.NCCLCollectiveCounter{}
	for _, r := range rows {
		// Same OpType can appear on two rows (collective + barrier);
		// merge for assertion convenience.
		merged := byOp[r.OpType]
		merged.OpType = r.OpType
		merged.Count += r.Count
		merged.BytesTotal += r.BytesTotal
		merged.BarrierEvents += r.BarrierEvents
		byOp[r.OpType] = merged
	}
	if byOp["ncclAllReduce"].Count != 2 {
		t.Errorf("ncclAllReduce.Count = %d, want 2", byOp["ncclAllReduce"].Count)
	}
	if byOp["ncclAllReduce"].BytesTotal != 1024+2048 {
		t.Errorf("ncclAllReduce.BytesTotal = %d, want %d", byOp["ncclAllReduce"].BytesTotal, 1024+2048)
	}
	if byOp["ncclAllReduce"].BarrierEvents != 2 {
		t.Errorf("ncclAllReduce.BarrierEvents = %d, want 2", byOp["ncclAllReduce"].BarrierEvents)
	}
	if byOp["ncclBcast"].Count != 1 {
		t.Errorf("ncclBcast.Count = %d, want 1", byOp["ncclBcast"].Count)
	}
	// Buffer should have 3 collective + 2 barrier = 5 entries.
	ncclBufMu.Lock()
	bufLen := len(ncclBuf)
	ncclBufMu.Unlock()
	if bufLen != 5 {
		t.Errorf("ncclBuf length = %d, want 5", bufLen)
	}
}

// v0.15 F2 (G6, MED): concurrent recordNCCLCollective vs
// snapshotNCCLCollectiveCounters. Both paths take ncclCounterMu;
// this test runs many writers + many readers in parallel under
// `go test -race` to prove the locking is honest. Without -race
// the test still asserts that final counts match the writes
// (no lost updates).
func TestRecordNCCLCollective_RaceWithSnapshot(t *testing.T) {
	resetNCCLCollectiveCounters()
	defer resetNCCLCollectiveCounters()

	const writers = 4
	const readers = 4
	const perWriter = 250

	var wWriters, wReaders sync.WaitGroup
	stop := make(chan struct{})
	for w := 0; w < writers; w++ {
		wWriters.Add(1)
		go func() {
			defer wWriters.Done()
			for i := 0; i < perWriter; i++ {
				recordNCCLCollective(stats.NCCLDataPoint{OpType: "ncclAllReduce", CountBytes: 100})
			}
		}()
	}
	for r := 0; r < readers; r++ {
		wReaders.Add(1)
		go func() {
			defer wReaders.Done()
			for {
				select {
				case <-stop:
					return
				default:
					_ = snapshotNCCLCollectiveCounters()
				}
			}
		}()
	}
	wWriters.Wait()
	close(stop)
	wReaders.Wait()

	final := snapshotNCCLCollectiveCounters()
	var totalCount int64
	var totalBytes int64
	for _, r := range final {
		totalCount += r.Count
		totalBytes += r.BytesTotal
	}
	if totalCount != int64(writers*perWriter) {
		t.Errorf("totalCount=%d, want %d (lost updates under contention)",
			totalCount, writers*perWriter)
	}
	if totalBytes != int64(writers*perWriter*100) {
		t.Errorf("totalBytes=%d, want %d", totalBytes, writers*perWriter*100)
	}
}

// v0.15 F2 (G7, MED): resetNCCLCollectiveCounters direct test.
// The reset path was implicit in every other test's t.Cleanup hook
// but never asserted on its own. Regression risk: a future refactor
// that drops the map clear (e.g. swapping to a struct field) could
// silently break test isolation across the package.
func TestResetNCCLCollectiveCounters_Clears(t *testing.T) {
	resetNCCLCollectiveCounters()
	defer resetNCCLCollectiveCounters()

	recordNCCLCollective(stats.NCCLDataPoint{OpType: "ncclAllReduce", CountBytes: 1024})
	recordNCCLCollective(stats.NCCLDataPoint{OpType: "ncclBcast", CountBytes: 512})
	recordNCCLCollective(stats.NCCLDataPoint{OpType: "ncclAllReduce", IsBarrier: true})

	if got := snapshotNCCLCollectiveCounters(); len(got) == 0 {
		t.Fatalf("pre-reset snapshot should be non-empty")
	}

	resetNCCLCollectiveCounters()

	if got := snapshotNCCLCollectiveCounters(); got != nil {
		t.Errorf("post-reset snapshot should be nil, got %+v", got)
	}
}
