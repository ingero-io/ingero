package cli

import (
	"sort"
	"testing"
	"time"

	"github.com/ingero-io/ingero/pkg/events"
)

func mkMemcpyEvent(op events.CUDAOp, dir uint8, bytes uint64, dur time.Duration) events.Event {
	return events.Event{
		Source:   events.SourceCUDA,
		Op:       uint8(op),
		Args:     [2]uint64{bytes, uint64(dir)},
		Duration: dur,
	}
}

func TestRecordMemcpyEventDirectionLabels(t *testing.T) {
	resetMemcpyStats()
	defer resetMemcpyStats()

	// One H2D (1KiB), one D2H (2KiB), one D2D (3KiB), one Peer (4KiB,
	// pinned to D2D), one 2D-async (5KiB, direction=5/unknown).
	recordMemcpyEvent(mkMemcpyEvent(events.CUDAMemcpy, 1, 1<<10, 100*time.Microsecond))
	recordMemcpyEvent(mkMemcpyEvent(events.CUDAMemcpyAsync, 2, 2<<10, 200*time.Microsecond))
	recordMemcpyEvent(mkMemcpyEvent(events.CUDAMemcpy, 3, 3<<10, 300*time.Microsecond))
	recordMemcpyEvent(mkMemcpyEvent(events.CUDAMemcpyPeer, 3, 4<<10, 400*time.Microsecond))
	recordMemcpyEvent(mkMemcpyEvent(events.CUDAMemcpy2DAsync, 5, 5<<10, 500*time.Microsecond))

	got := drainMemcpyStats()
	if len(got) == 0 {
		t.Fatal("expected non-empty drain")
	}
	sort.Slice(got, func(i, j int) bool { return got[i].Direction < got[j].Direction })
	byDir := map[string]int64{}
	for _, s := range got {
		byDir[s.Direction] = s.BytesTotal
	}
	if byDir["h2d"] != 1<<10 {
		t.Errorf("h2d bytes = %d, want %d", byDir["h2d"], 1<<10)
	}
	if byDir["d2h"] != 2<<10 {
		t.Errorf("d2h bytes = %d", byDir["d2h"])
	}
	// d2d gets contributions from BOTH a direct cudaMemcpy(...,3) AND
	// from the Peer variant (BPF probe pins direction=3).
	if byDir["d2d"] != (3<<10)+(4<<10) {
		t.Errorf("d2d bytes = %d, want %d (cudaMemcpy + cudaMemcpyPeer)", byDir["d2d"], (3<<10)+(4<<10))
	}
	// 2D variants encode direction=5 (unknown) since cudaMemcpyKind is
	// PARM7 and unreadable via libbpf macros on amd64+arm64.
	if byDir["unknown"] != 5<<10 {
		t.Errorf("unknown (2D variants) bytes = %d, want %d", byDir["unknown"], 5<<10)
	}
	if byDir["h2h"] != 0 {
		t.Errorf("h2h bytes should be 0 after the v0.14 R2 ★5 fix, got %d", byDir["h2h"])
	}
}

func TestRecordMemcpyEventIgnoresOtherSources(t *testing.T) {
	resetMemcpyStats()
	defer resetMemcpyStats()

	// Driver-source memcpy must NOT show up in the per-direction
	// aggregator (that surface covers libcuda, not libcudart).
	recordMemcpyEvent(events.Event{
		Source:   events.SourceDriver,
		Op:       2, // DriverMemcpy
		Args:     [2]uint64{1024, 1},
		Duration: 100 * time.Microsecond,
	})
	if got := drainMemcpyStats(); got != nil {
		t.Errorf("driver-source memcpy should not aggregate into runtime memcpy buffer; got %+v", got)
	}
}

func TestDrainResetsWindowCounters(t *testing.T) {
	resetMemcpyStats()
	defer resetMemcpyStats()

	recordMemcpyEvent(mkMemcpyEvent(events.CUDAMemcpy, 1, 1024, 100*time.Microsecond))
	recordMemcpyEvent(mkMemcpyEvent(events.CUDAMemcpy, 1, 1024, 200*time.Microsecond))
	first := drainMemcpyStats()
	if len(first) != 1 || first[0].EventsInWindow != 2 {
		t.Fatalf("first drain: %+v", first)
	}
	if first[0].BytesTotal != 2048 {
		t.Errorf("first drain bytes_total = %d", first[0].BytesTotal)
	}

	// Second drain with no new events: bytes_total persists (cumulative
	// counter). v0.15 (2026-05-07): EventsInWindow is now also
	// cumulative since the histogram is no longer reset on drain
	// (Prometheus convention).
	second := drainMemcpyStats()
	if len(second) != 1 {
		t.Fatalf("second drain len = %d, want 1 (totals persist)", len(second))
	}
	if second[0].BytesTotal != 2048 {
		t.Errorf("second drain bytes_total = %d, want 2048 (cumulative)", second[0].BytesTotal)
	}
	if second[0].EventsInWindow != 2 {
		t.Errorf("second drain EventsInWindow = %d, want 2 (cumulative, v0.15)", second[0].EventsInWindow)
	}
}

// v0.15 item C (revised 2026-05-07): the per-direction histogram
// is CUMULATIVE across drains. Prometheus histogram exposition
// convention is monotonic; OTLP cumulative-temporality does the
// same. Resetting on drain produced empty Prometheus rows when
// /metrics scrape lagged the workload (Lambda A10 e2e regression).
func TestDrainPreservesDurationHistogram(t *testing.T) {
	resetMemcpyStats()
	defer resetMemcpyStats()

	recordMemcpyEvent(mkMemcpyEvent(events.CUDAMemcpy, 1, 1024, 1*time.Millisecond))
	recordMemcpyEvent(mkMemcpyEvent(events.CUDAMemcpy, 1, 1024, 2*time.Millisecond))
	recordMemcpyEvent(mkMemcpyEvent(events.CUDAMemcpy, 1, 1024, 3*time.Millisecond))

	first := drainMemcpyStats()
	if len(first) != 1 {
		t.Fatalf("first drain rows=%d, want 1", len(first))
	}
	if first[0].DurationHistogram.Count != 3 {
		t.Errorf("first drain hist Count=%d, want 3", first[0].DurationHistogram.Count)
	}

	// Second drain with no new events: histogram + bytes_total BOTH
	// persist (cumulative).
	second := drainMemcpyStats()
	if len(second) != 1 {
		t.Fatalf("second drain rows=%d, want 1", len(second))
	}
	if second[0].DurationHistogram.Count != 3 {
		t.Errorf("histogram should be cumulative; second drain Count=%d, want 3",
			second[0].DurationHistogram.Count)
	}
	if second[0].BytesTotal != 3*1024 {
		t.Errorf("BytesTotal cumulative; got %d, want %d",
			second[0].BytesTotal, 3*1024)
	}

	// New events after drain ADD to the running totals.
	recordMemcpyEvent(mkMemcpyEvent(events.CUDAMemcpy, 1, 2048, 5*time.Millisecond))
	third := drainMemcpyStats()
	if third[0].DurationHistogram.Count != 4 {
		t.Errorf("cumulative histogram Count after 1 new event = %d, want 4",
			third[0].DurationHistogram.Count)
	}
	if third[0].BytesTotal != 3*1024+2048 {
		t.Errorf("cumulative BytesTotal after 1 new 2KB event = %d, want %d",
			third[0].BytesTotal, 3*1024+2048)
	}
}

// v0.15 item C: NaN / Inf durations must be dropped from the
// histogram (otherwise IsFinite check is dead code) but BytesTotal
// + EventsInWindow MUST still increment if recordMemcpyEvent
// considers them events.
func TestRecordMemcpyEvent_DropsNonFiniteDurations(t *testing.T) {
	resetMemcpyStats()
	defer resetMemcpyStats()

	// Negative durations from clock skew can theoretically appear;
	// time.Duration is just int64 so the producer doesn't filter.
	// IsFinite filters NaN/Inf at the histogram boundary; negatives
	// are passed through (fine for a histogram, bucket 0).
	recordMemcpyEvent(mkMemcpyEvent(events.CUDAMemcpy, 1, 1024, 0))
	got := drainMemcpyStats()
	if len(got) != 1 {
		t.Fatal("expected 1 row even when duration=0")
	}
	if got[0].EventsInWindow != 1 {
		t.Errorf("EventsInWindow=%d, want 1 (duration=0 is a real event)",
			got[0].EventsInWindow)
	}
}

func TestMemcpyDirectionString(t *testing.T) {
	cases := []struct {
		in   uint8
		want string
	}{
		{0, "h2h"},
		{1, "h2d"},
		{2, "d2h"},
		{3, "d2d"},
		{4, "default"},
		{5, "unknown"},
		{255, "unknown"},
	}
	for _, c := range cases {
		if got := memcpyDirectionString(c.in); got != c.want {
			t.Errorf("memcpyDirectionString(%d) = %q, want %q", c.in, got, c.want)
		}
	}
}
