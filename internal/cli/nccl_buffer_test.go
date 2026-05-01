package cli

import (
	"sync"
	"testing"

	"github.com/ingero-io/ingero/internal/stats"
)

// TestNCCLBufferDrain validates the v0.12.0 LHF #1 fix: the NCCL
// ringbuf goroutine pushes events into ncclBuf, and onSnapshot drains
// them into Snapshot.NCCLDataPoints for OTLP/Prometheus emission.
//
// Pre-fix the events were drained into a stderr counter and never
// reached the exporters, leaving Fleet's ncclprocessor with no input.
func TestNCCLBufferDrain(t *testing.T) {
	// Ensure a clean state across test runs.
	ncclBufMu.Lock()
	ncclBuf = nil
	ncclBufMu.Unlock()

	for i := 0; i < 5; i++ {
		ncclBufferAdd(stats.NCCLDataPoint{
			OpType:     "ncclAllReduce",
			CommIDHash: "abc",
			Rank:       uint32(i),
			NRanks:     5,
			DurationMs: float64(i + 1),
			CountBytes: 1024 * uint64(i+1),
		})
	}

	got := ncclBufferDrain()
	if len(got) != 5 {
		t.Fatalf("Drain returned %d points, want 5", len(got))
	}

	// Second drain should be empty.
	if again := ncclBufferDrain(); len(again) != 0 {
		t.Fatalf("second drain returned %d points, want 0 (buffer not cleared)", len(again))
	}
}

// TestNCCLBufferOverflowDrops asserts that pushes beyond ncclBufMax are
// dropped (silent in v0.12.0). Prevents producer from monopolizing
// memory if the consumer (snapshot tick) stalls or no exporter is
// configured.
func TestNCCLBufferOverflowDrops(t *testing.T) {
	ncclBufMu.Lock()
	ncclBuf = nil
	saved := ncclBufMax
	ncclBufMax = 4
	ncclBufMu.Unlock()
	defer func() {
		ncclBufMu.Lock()
		ncclBufMax = saved
		ncclBuf = nil
		ncclBufMu.Unlock()
	}()

	for i := 0; i < 100; i++ {
		ncclBufferAdd(stats.NCCLDataPoint{Rank: uint32(i)})
	}
	got := ncclBufferDrain()
	if len(got) != 4 {
		t.Fatalf("expected 4 (cap), got %d", len(got))
	}
	// First-N semantics: producer keeps the earliest events.
	for i, p := range got {
		if int(p.Rank) != i {
			t.Errorf("got[%d].Rank = %d, want %d (FIFO)", i, p.Rank, i)
		}
	}
}

// TestNCCLBufferConcurrent stress-tests the producer/consumer locks
// under contention. The buffer is mutex-guarded; reads and writes from
// many goroutines must not race or panic.
func TestNCCLBufferConcurrent(t *testing.T) {
	ncclBufMu.Lock()
	ncclBuf = nil
	saved := ncclBufMax
	ncclBufMax = 2048
	ncclBufMu.Unlock()
	defer func() {
		ncclBufMu.Lock()
		ncclBufMax = saved
		ncclBuf = nil
		ncclBufMu.Unlock()
	}()

	const producers = 8
	const perProducer = 200
	var wg sync.WaitGroup
	for p := 0; p < producers; p++ {
		wg.Add(1)
		go func(p int) {
			defer wg.Done()
			for i := 0; i < perProducer; i++ {
				ncclBufferAdd(stats.NCCLDataPoint{Rank: uint32(p*perProducer + i)})
			}
		}(p)
	}
	// Periodic drains while producers are working.
	wg.Add(1)
	go func() {
		defer wg.Done()
		for i := 0; i < 10; i++ {
			_ = ncclBufferDrain()
		}
	}()
	wg.Wait()
}
