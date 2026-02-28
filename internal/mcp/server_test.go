package mcp

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/ingero-io/ingero/internal/store"
	"github.com/ingero-io/ingero/pkg/events"
)

func TestFormatAggregateStatsTSC(t *testing.T) {
	ops := []store.AggregateOpStats{
		{Source: uint8(events.SourceCUDA), Op: 3, OpName: "cudaMemcpy", Count: 5000, SumDur: 50_000_000, MinDur: 1000, MaxDur: 100_000_000},
		{Source: uint8(events.SourceHost), Op: 0, OpName: "sched_switch", Count: 12000, SumDur: 120_000_000, MinDur: 500, MaxDur: 50_000_000},
	}
	descs := map[string]string{"cudaMemcpy": "Copy memory between host and device"}

	text := formatAggregateStats(ops, true, descs)

	var parsed map[string]interface{}
	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		t.Fatalf("TSC output not valid JSON: %v\ntext: %s", err, text)
	}

	if parsed["mode"] != "aggregate" {
		t.Errorf("expected mode=aggregate, got %v", parsed["mode"])
	}

	opsArr, ok := parsed["ops"].([]interface{})
	if !ok || len(opsArr) != 2 {
		t.Fatalf("expected 2 ops, got %v", parsed["ops"])
	}

	first := opsArr[0].(map[string]interface{})
	if first["op"] != "cudaMemcpy" {
		t.Errorf("first op = %v, want cudaMemcpy", first["op"])
	}
	if first["d"] != "Copy memory between host and device" {
		t.Errorf("description = %v", first["d"])
	}

	// Total events = 5000 + 12000 = 17000
	total := parsed["total_events"].(float64)
	if total != 17000 {
		t.Errorf("total_events = %v, want 17000", total)
	}
}

func TestFormatAggregateStatsVerbose(t *testing.T) {
	ops := []store.AggregateOpStats{
		{Source: uint8(events.SourceDriver), Op: 0, OpName: "cuLaunchKernel", Count: 1000, SumDur: 10_000_000_000, MinDur: 1_000_000, MaxDur: 500_000_000},
	}

	text := formatAggregateStats(ops, false, nil)

	if !strings.Contains(text, "Aggregate stats") {
		t.Errorf("expected 'Aggregate stats' header, got: %s", text)
	}
	if !strings.Contains(text, "[Driver] cuLaunchKernel") {
		t.Errorf("expected Driver source label, got: %s", text)
	}
	if !strings.Contains(text, "count=1000") {
		t.Errorf("expected count=1000, got: %s", text)
	}
	if !strings.Contains(text, "percentiles unavailable") {
		t.Errorf("expected percentiles note, got: %s", text)
	}
}

func TestFormatAggregateStatsEmpty(t *testing.T) {
	// Empty ops should produce valid JSON in TSC mode.
	text := formatAggregateStats(nil, true, nil)

	var parsed map[string]interface{}
	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		t.Fatalf("TSC output not valid JSON: %v\ntext: %s", err, text)
	}
	if parsed["mode"] != "aggregate" {
		t.Errorf("expected mode=aggregate, got %v", parsed["mode"])
	}
}
