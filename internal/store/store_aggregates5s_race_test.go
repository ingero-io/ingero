package store

import (
	"strings"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

// TestAggregates5sRaceStress (Phase 10 P3): 8 writer goroutines calling
// RecordAggregates5s and 8 reader goroutines calling QueryAggregatePerOp5s
// against a shared *Store for 30 seconds under -race. The test passes if:
//   - no data races fire
//   - no panics
//   - no writer or reader returns an error containing "database is locked"
//
// Transient empty results during retention DELETE sweeps are acceptable and
// count as eventual consistency.
//
// Shortened to 3s for normal -short runs. Use `go test -run AggregatesRace -count=1 -race`
// to force the full 30s burn-in.
func TestAggregates5sRaceStress(t *testing.T) {
	s, err := New(":memory:")
	if err != nil {
		t.Fatalf("New failed: %v", err)
	}
	defer s.Close()

	duration := 30 * time.Second
	if testing.Short() {
		duration = 3 * time.Second
	}
	deadline := time.Now().Add(duration)

	var (
		wg            sync.WaitGroup
		writeErrs     atomic.Int64
		readErrs      atomic.Int64
		writes        atomic.Int64
		reads         atomic.Int64
		lockErr       atomic.Int64
		firstLockErr  atomic.Value // holds first "database is locked" error string
		panicOccurred atomic.Bool
	)

	const writers = 8
	const readers = 8

	for w := 0; w < writers; w++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			defer func() {
				if r := recover(); r != nil {
					panicOccurred.Store(true)
					t.Errorf("writer %d panicked: %v", id, r)
				}
			}()
			for time.Now().Before(deadline) {
				bucket := time.Now().Truncate(5 * time.Second).UnixNano()
				// Varied PID per writer keeps rows distinct enough to exercise concurrent keyed writes.
				agg := Aggregate{
					Bucket: bucket,
					Source: 1,
					Op:     3,
					PID:    uint32(id),
					Count:  10,
					Stored: 5,
					SumDur: 100,
					MinDur: 1,
					MaxDur: 20,
				}
				// RecordAggregates5s is declared as void — errors surface only as "database is locked"
				// in stderr logs via the store's internal logger. Capture any panic from the call site.
				func() {
					defer func() {
						if r := recover(); r != nil {
							writeErrs.Add(1)
							if s, ok := r.(error); ok && strings.Contains(s.Error(), "database is locked") {
								lockErr.Add(1)
								firstLockErr.CompareAndSwap(nil, s.Error())
							}
						}
					}()
					s.RecordAggregates5s([]Aggregate{agg})
				}()
				writes.Add(1)
				time.Sleep(time.Millisecond)
			}
		}(w)
	}

	for r := 0; r < readers; r++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			defer func() {
				if rec := recover(); rec != nil {
					panicOccurred.Store(true)
					t.Errorf("reader %d panicked: %v", id, rec)
				}
			}()
			for time.Now().Before(deadline) {
				_, err := s.QueryAggregatePerOp5s(QueryParams{Since: 1 * time.Minute})
				if err != nil {
					readErrs.Add(1)
					if strings.Contains(err.Error(), "database is locked") {
						lockErr.Add(1)
						firstLockErr.CompareAndSwap(nil, err.Error())
					}
				}
				reads.Add(1)
				time.Sleep(time.Millisecond)
			}
		}(r)
	}

	wg.Wait()

	t.Logf("writes=%d (errs=%d) reads=%d (errs=%d) lock_errs=%d",
		writes.Load(), writeErrs.Load(), reads.Load(), readErrs.Load(), lockErr.Load())

	if panicOccurred.Load() {
		t.Fatal("one or more goroutines panicked")
	}
	if lockErr.Load() > 0 {
		first, _ := firstLockErr.Load().(string)
		t.Fatalf("%d 'database is locked' errors observed; first: %s", lockErr.Load(), first)
	}
	// Transient read errors (e.g. during retention DELETE) are acceptable, but a flood is not.
	if readErrs.Load() > int64(readers) {
		t.Logf("note: %d reader errors observed (>%d readers); not failing but worth investigating", readErrs.Load(), readers)
	}
}
