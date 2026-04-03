package export

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/ingero-io/ingero/internal/store"
	"github.com/ingero-io/ingero/pkg/events"
)

func intPtr(v int) *int { return &v }

func makeTestEvent(node string, source events.Source, op uint8, dur time.Duration) events.Event {
	return events.Event{
		Timestamp: time.Now(),
		PID:       1234,
		TID:       1235,
		Source:    source,
		Op:        op,
		Duration:  dur,
		GPUID:     0,
		Node:      node,
	}
}

func TestWritePerfetto_ValidJSON(t *testing.T) {
	evts := []events.Event{
		makeTestEvent("node-a", events.SourceCUDA, uint8(events.CUDAMalloc), time.Millisecond),
		makeTestEvent("node-b", events.SourceHost, uint8(events.HostSchedSwitch), 10*time.Millisecond),
	}

	var buf bytes.Buffer
	err := WritePerfetto(evts, nil, &buf)
	if err != nil {
		t.Fatal(err)
	}

	// Parse as JSON array.
	var parsed []json.RawMessage
	if err := json.Unmarshal(buf.Bytes(), &parsed); err != nil {
		t.Fatalf("invalid JSON: %v\nOutput: %s", err, buf.String())
	}

	if len(parsed) < 2 {
		t.Errorf("expected at least 2 entries (metadata + events), got %d", len(parsed))
	}
}

func TestWritePerfetto_MultiTrack(t *testing.T) {
	evts := []events.Event{
		makeTestEvent("node-a", events.SourceCUDA, uint8(events.CUDAMalloc), time.Millisecond),
		makeTestEvent("node-b", events.SourceCUDA, uint8(events.CUDAFree), time.Millisecond),
	}

	var buf bytes.Buffer
	WritePerfetto(evts, nil, &buf)

	var parsed []traceEvent
	json.Unmarshal(buf.Bytes(), &parsed)

	// Find non-metadata events.
	pids := make(map[int]bool)
	for _, te := range parsed {
		if te.Ph != "M" {
			pids[te.PID] = true
		}
	}
	if len(pids) != 2 {
		t.Errorf("expected 2 different PIDs (2 nodes), got %d", len(pids))
	}
}

func TestWritePerfetto_DurationEvent(t *testing.T) {
	evts := []events.Event{
		makeTestEvent("node-a", events.SourceCUDA, uint8(events.CUDALaunchKernel), 42*time.Microsecond),
	}

	var buf bytes.Buffer
	WritePerfetto(evts, nil, &buf)

	var parsed []traceEvent
	json.Unmarshal(buf.Bytes(), &parsed)

	// Find the X event.
	var found bool
	for _, te := range parsed {
		if te.Ph == "X" {
			found = true
			if te.Name != "cudaLaunchKernel" {
				t.Errorf("name = %q, want %q", te.Name, "cudaLaunchKernel")
			}
			if te.Cat != "cuda" {
				t.Errorf("cat = %q, want %q", te.Cat, "cuda")
			}
			if te.Dur != 42 {
				t.Errorf("dur = %d, want 42", te.Dur)
			}
		}
	}
	if !found {
		t.Error("no X event found")
	}
}

func TestWritePerfetto_InstantEvent(t *testing.T) {
	evts := []events.Event{
		makeTestEvent("node-a", events.SourceHost, uint8(events.HostOOMKill), 0), // zero duration = instant
	}

	var buf bytes.Buffer
	WritePerfetto(evts, nil, &buf)

	var parsed []traceEvent
	json.Unmarshal(buf.Bytes(), &parsed)

	var found bool
	for _, te := range parsed {
		if te.Ph == "i" && te.Cat == "host" {
			found = true
			if te.Scope != "t" {
				t.Errorf("scope = %q, want %q", te.Scope, "t")
			}
		}
	}
	if !found {
		t.Error("no instant event found for zero-duration event")
	}
}

func TestWritePerfetto_CausalChain(t *testing.T) {
	chains := []store.StoredChain{
		{
			ID:              "node-a:tail-high",
			DetectedAt:      time.Now(),
			Severity:        "HIGH",
			Summary:         "tail latency spike",
			RootCause:       "CPU contention",
			Recommendations: []string{"pin cores"},
			Node:            "node-a",
		},
	}
	evts := []events.Event{
		makeTestEvent("node-a", events.SourceCUDA, uint8(events.CUDAMalloc), time.Millisecond),
	}

	var buf bytes.Buffer
	WritePerfetto(evts, chains, &buf)

	var parsed []traceEvent
	json.Unmarshal(buf.Bytes(), &parsed)

	var found bool
	for _, te := range parsed {
		if te.Cat == "causal_chain" {
			found = true
			if te.Ph != "i" {
				t.Errorf("chain ph = %q, want %q", te.Ph, "i")
			}
			if te.Scope != "g" {
				t.Errorf("scope = %q, want %q (global)", te.Scope, "g")
			}
			if te.CName != "bad" { // HIGH → bad (orange)
				t.Errorf("cname = %q, want %q", te.CName, "bad")
			}
			if te.Args["severity"] != "HIGH" {
				t.Errorf("args.severity = %v, want HIGH", te.Args["severity"])
			}
		}
	}
	if !found {
		t.Error("no causal_chain event found")
	}
}

func TestWritePerfetto_MetadataEvents(t *testing.T) {
	evts := []events.Event{
		makeTestEvent("node-a", events.SourceCUDA, uint8(events.CUDAMalloc), time.Millisecond),
		makeTestEvent("node-b", events.SourceCUDA, uint8(events.CUDAMalloc), time.Millisecond),
	}

	var buf bytes.Buffer
	WritePerfetto(evts, nil, &buf)

	var parsed []traceEvent
	json.Unmarshal(buf.Bytes(), &parsed)

	metaCount := 0
	for _, te := range parsed {
		if te.Ph == "M" && te.Name == "process_name" {
			metaCount++
		}
	}
	if metaCount != 2 {
		t.Errorf("expected 2 process_name metadata events, got %d", metaCount)
	}
}

func TestWritePerfetto_WithRank(t *testing.T) {
	evts := []events.Event{
		{
			Timestamp: time.Now(),
			PID:       1234,
			TID:       1235,
			Source:    events.SourceCUDA,
			Op:        uint8(events.CUDAMalloc),
			Duration:  time.Millisecond,
			Node:      "node-a",
			Rank:      intPtr(2),
			WorldSize: intPtr(4),
		},
	}

	var buf bytes.Buffer
	WritePerfetto(evts, nil, &buf)

	output := buf.String()
	if !strings.Contains(output, "rank 2") {
		t.Error("expected 'rank 2' in metadata, not found")
	}
}

func TestWritePerfetto_SingleNode(t *testing.T) {
	evts := []events.Event{
		makeTestEvent("", events.SourceCUDA, uint8(events.CUDAMalloc), time.Millisecond),
	}

	var buf bytes.Buffer
	err := WritePerfetto(evts, nil, &buf)
	if err != nil {
		t.Fatal(err)
	}

	var parsed []json.RawMessage
	if err := json.Unmarshal(buf.Bytes(), &parsed); err != nil {
		t.Fatalf("invalid JSON for single-node: %v", err)
	}
}

func TestWritePerfetto_FileSize(t *testing.T) {
	// Generate 10K events, verify output < 2MB.
	evts := make([]events.Event, 10000)
	for i := range evts {
		evts[i] = makeTestEvent("node-a", events.SourceCUDA, uint8(events.CUDALaunchKernel), time.Duration(i)*time.Microsecond)
	}

	var buf bytes.Buffer
	WritePerfetto(evts, nil, &buf)

	sizeMB := float64(buf.Len()) / (1024 * 1024)
	if sizeMB > 2.0 {
		t.Errorf("10K events produced %.1f MB, want < 2 MB", sizeMB)
	}
	t.Logf("10K events → %.2f MB (%.0f bytes/event)", sizeMB, float64(buf.Len())/10000)
}

func TestSourceCategory(t *testing.T) {
	tests := []struct {
		source events.Source
		want   string
	}{
		{events.SourceCUDA, "cuda"},
		{events.SourceDriver, "driver"},
		{events.SourceHost, "host"},
		{events.SourceCUDAGraph, "graph"},
		{events.SourceIO, "io"},
		{events.SourceTCP, "tcp"},
		{events.SourceNet, "net"},
	}
	for _, tt := range tests {
		if got := sourceCategory(tt.source); got != tt.want {
			t.Errorf("sourceCategory(%d) = %q, want %q", tt.source, got, tt.want)
		}
	}
}
