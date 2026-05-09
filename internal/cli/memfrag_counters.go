package cli

import (
	"sort"
	"sync"

	"github.com/ingero-io/ingero/internal/ebpf/memfrag"
	"github.com/ingero-io/ingero/internal/stats"
)

// v0.15 item K: per-cmd IOCTL invocation counters fed by the
// memfrag kprobe. Producer: the BPF ringbuf reader goroutine
// started by setupMemfragTracer. Consumer: snapshotMemfragCounters
// once per OTLP / Prometheus push.

var (
	memfragCounterMu sync.Mutex
	memfragCounter   = map[uint32]*stats.MemfragIOCTLCounter{}
)

func recordMemfragEvent(ev memfrag.Event) {
	memfragCounterMu.Lock()
	defer memfragCounterMu.Unlock()
	c, ok := memfragCounter[ev.Cmd]
	if !ok {
		c = &stats.MemfragIOCTLCounter{Cmd: ev.Cmd}
		memfragCounter[ev.Cmd] = c
	}
	c.Count++
}

// snapshotMemfragCounters returns a sorted-by-cmd copy of the
// running counters; nil when no events have been seen.
func snapshotMemfragCounters() []stats.MemfragIOCTLCounter {
	memfragCounterMu.Lock()
	defer memfragCounterMu.Unlock()
	if len(memfragCounter) == 0 {
		return nil
	}
	out := make([]stats.MemfragIOCTLCounter, 0, len(memfragCounter))
	for _, c := range memfragCounter {
		out = append(out, *c)
	}
	sort.Slice(out, func(i, j int) bool { return out[i].Cmd < out[j].Cmd })
	return out
}

func resetMemfragCounters() {
	memfragCounterMu.Lock()
	defer memfragCounterMu.Unlock()
	memfragCounter = map[uint32]*stats.MemfragIOCTLCounter{}
}
