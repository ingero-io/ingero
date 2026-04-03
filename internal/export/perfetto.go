// Package export — perfetto.go implements Chrome Trace Event Format export
// for visualization in https://ui.perfetto.dev or chrome://tracing.
//
// Format spec: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
package export

import (
	"encoding/json"
	"fmt"
	"io"
	"strings"

	"github.com/ingero-io/ingero/internal/store"
	"github.com/ingero-io/ingero/pkg/events"
)

// traceEvent is a single Chrome Trace Event Format entry.
type traceEvent struct {
	Name  string         `json:"name"`
	Cat   string         `json:"cat,omitempty"`
	Ph    string         `json:"ph"`             // "X"=complete, "i"=instant, "M"=metadata
	Ts    int64          `json:"ts"`             // microseconds
	Dur   int64          `json:"dur,omitempty"`  // microseconds (X events only)
	PID   int            `json:"pid"`            // process track (node)
	TID   int            `json:"tid,omitempty"`  // thread (actual PID from event)
	Scope string         `json:"s,omitempty"`    // "g"=global (instant events)
	CName string         `json:"cname,omitempty"`// named color
	Args  map[string]any `json:"args,omitempty"`
}

// severityColor maps chain severity to Perfetto named colors.
var severityColor = map[string]string{
	"CRITICAL": "terrible", // red
	"HIGH":     "bad",      // orange
	"MEDIUM":   "yellow",
	"LOW":      "good",     // green
}

// sourceCategory maps event Source to a Perfetto category string.
func sourceCategory(s events.Source) string {
	switch s {
	case events.SourceCUDA:
		return "cuda"
	case events.SourceDriver:
		return "driver"
	case events.SourceHost:
		return "host"
	case events.SourceCUDAGraph:
		return "graph"
	case events.SourceIO:
		return "io"
	case events.SourceTCP:
		return "tcp"
	case events.SourceNet:
		return "net"
	default:
		return "other"
	}
}

// WritePerfetto writes events and chains as a Chrome Trace Event Format JSON array.
// nodeMap assigns node names to Perfetto process IDs (pid). If nil, it's built from events.
// Streaming: writes `[`, then events with commas, then `]`.
func WritePerfetto(evts []events.Event, chains []store.StoredChain, w io.Writer) error {
	// Build node → PID mapping from event data.
	nodeMap := buildNodeMap(evts, chains)

	// Build rank info per node for process name metadata.
	nodeRanks := buildNodeRanks(evts)

	enc := json.NewEncoder(w)
	enc.SetEscapeHTML(false)

	// Track first write error — once a write fails, all subsequent writes are no-ops.
	var writeErr error
	write := func(s string) {
		if writeErr != nil {
			return
		}
		_, writeErr = io.WriteString(w, s)
	}
	encode := func(v any) {
		if writeErr != nil {
			return
		}
		writeErr = enc.Encode(v)
	}

	write("[\n")
	first := true

	writeComma := func() {
		if !first {
			write(",\n")
		}
		first = false
	}

	// 1. Write metadata events (process names).
	for node, pid := range nodeMap {
		writeComma()
		name := node
		if ri, ok := nodeRanks[node]; ok && ri.rank != nil {
			name = fmt.Sprintf("%s (rank %d)", node, *ri.rank)
		}
		encode(traceEvent{
			Name: "process_name",
			Ph:   "M",
			PID:  pid,
			Args: map[string]any{"name": name},
		})
	}

	// 2. Write events as X (duration) or i (instant).
	for _, evt := range evts {
		writeComma()

		nodePID := nodeMap[evt.Node]
		if nodePID == 0 {
			nodePID = nodeMap["local"]
		}

		tsUs := evt.Timestamp.UnixMicro()
		durUs := int64(evt.Duration / 1000) // nanos → micros

		if durUs > 0 {
			// Complete duration event.
			te := traceEvent{
				Name: evt.OpName(),
				Cat:  sourceCategory(evt.Source),
				Ph:   "X",
				Ts:   tsUs,
				Dur:  durUs,
				PID:  nodePID,
				TID:  int(evt.PID),
			}
			// Only include non-zero args.
			args := make(map[string]any)
			if evt.GPUID > 0 {
				args["gpu_id"] = evt.GPUID
			}
			if evt.RetCode != 0 {
				args["ret_code"] = evt.RetCode
			}
			if len(args) > 0 {
				te.Args = args
			}
			enc.Encode(te)
		} else {
			// Instant event (e.g., OOM kill, page alloc).
			encode(traceEvent{
				Name:  evt.OpName(),
				Cat:   sourceCategory(evt.Source),
				Ph:    "i",
				Ts:    tsUs,
				PID:   nodePID,
				TID:   int(evt.PID),
				Scope: "t", // thread scope
			})
		}
	}

	// 3. Write causal chains as instant markers.
	for _, ch := range chains {
		writeComma()

		nodePID := nodeMap[ch.Node]
		if nodePID == 0 {
			nodePID = 1
		}

		tsUs := ch.DetectedAt.UnixMicro()
		color := severityColor[strings.ToUpper(ch.Severity)]

		te := traceEvent{
			Name:  fmt.Sprintf("%s: %s", ch.Severity, ch.Summary),
			Cat:   "causal_chain",
			Ph:    "i",
			Ts:    tsUs,
			PID:   nodePID,
			Scope: "g", // global scope — visible across all threads
			CName: color,
			Args: map[string]any{
				"severity":        ch.Severity,
				"root_cause":      ch.RootCause,
				"recommendations": strings.Join(ch.Recommendations, "; "),
			},
		}
		enc.Encode(te)
	}

	write("]\n")
	return writeErr
}

type rankInfo struct {
	rank      *int
	worldSize *int
}

func buildNodeMap(evts []events.Event, chains []store.StoredChain) map[string]int {
	seen := make(map[string]bool)
	for _, e := range evts {
		node := e.Node
		if node == "" {
			node = "local"
		}
		seen[node] = true
	}
	for _, ch := range chains {
		if ch.Node != "" {
			seen[ch.Node] = true
		}
	}

	nodeMap := make(map[string]int)
	pid := 1
	for node := range seen {
		nodeMap[node] = pid
		pid++
	}
	if len(nodeMap) == 0 {
		nodeMap["local"] = 1
	}
	return nodeMap
}

func buildNodeRanks(evts []events.Event) map[string]*rankInfo {
	ranks := make(map[string]*rankInfo)
	for _, e := range evts {
		if e.Node == "" {
			continue
		}
		if _, ok := ranks[e.Node]; ok {
			continue
		}
		ranks[e.Node] = &rankInfo{rank: e.Rank, worldSize: e.WorldSize}
	}
	return ranks
}
