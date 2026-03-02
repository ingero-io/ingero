package mcp

import (
	"encoding/json"
	"strings"
	"testing"
	"time"

	"github.com/ingero-io/ingero/internal/correlate"
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

func TestFormatCausalChainsEmpty(t *testing.T) {
	text := formatCausalChains(nil, false)
	if text != "No causal chains detected. System appears healthy." {
		t.Errorf("empty chains = %q, want healthy message", text)
	}
}

func TestFormatCausalChainsVerbose(t *testing.T) {
	chains := []correlate.CausalChain{
		{
			ID:       "test-chain-1",
			Severity: "HIGH",
			Summary:  "GPU stall due to CPU contention",
			RootCause: "sched_switch latency exceeded threshold",
			Recommendations: []string{"reduce CPU-bound work", "increase CPU quota"},
			Timeline: []correlate.ChainEvent{
				{Timestamp: time.Now(), Layer: "SYSTEM", Detail: "CPU 95%"},
				{Timestamp: time.Now(), Layer: "HOST", Detail: "sched_switch 15ms"},
				{Timestamp: time.Now(), Layer: "CUDA", Detail: "cudaLaunchKernel stalled"},
			},
		},
	}

	text := formatCausalChains(chains, false)

	if !strings.Contains(text, "1 causal chain(s) found") {
		t.Errorf("missing chain count header in: %s", text)
	}
	if !strings.Contains(text, "[HIGH]") {
		t.Errorf("missing severity in: %s", text)
	}
	if !strings.Contains(text, "GPU stall due to CPU contention") {
		t.Errorf("missing summary in: %s", text)
	}
	if !strings.Contains(text, "[SYSTEM]") || !strings.Contains(text, "[HOST]") || !strings.Contains(text, "[CUDA]") {
		t.Errorf("missing timeline layers in: %s", text)
	}
	if !strings.Contains(text, "reduce CPU-bound work") {
		t.Errorf("missing recommendation in: %s", text)
	}
}

func TestFormatCausalChainsTSC(t *testing.T) {
	chains := []correlate.CausalChain{
		{
			Severity:    "MEDIUM",
			Summary:     "Block I/O spike correlated with GPU stall",
			RootCause:   "disk latency >50ms",
			Recommendations: []string{"use NVMe SSD"},
			Timeline: []correlate.ChainEvent{
				{Timestamp: time.Now(), Layer: "IO", Detail: "120 block I/O ops"},
				{Timestamp: time.Now(), Layer: "CUDA", Detail: "cudaMemcpy slow"},
			},
		},
	}

	text := formatCausalChains(chains, true)

	// TSC mode must produce valid JSON.
	var parsed []interface{}
	if err := json.Unmarshal([]byte(text), &parsed); err != nil {
		t.Fatalf("TSC output not valid JSON: %v\ntext: %s", err, text)
	}
	if len(parsed) != 1 {
		t.Fatalf("expected 1 chain, got %d", len(parsed))
	}

	chain := parsed[0].(map[string]interface{})
	// TSC mode abbreviates keys: severity → sev, root_cause → rc.
	if chain["sev"] != "MEDIUM" {
		t.Errorf("sev = %v, want MEDIUM", chain["sev"])
	}
	if chain["rc"] != "disk latency >50ms" {
		t.Errorf("rc = %v", chain["rc"])
	}

	tl, ok := chain["tl"].([]interface{})
	if !ok || len(tl) != 2 {
		t.Fatalf("expected 2 timeline events, got %v", chain["tl"])
	}
}

func TestFormatCausalChainsMultiple(t *testing.T) {
	chains := []correlate.CausalChain{
		{Severity: "HIGH", Summary: "chain 1", RootCause: "cause 1"},
		{Severity: "LOW", Summary: "chain 2", RootCause: "cause 2"},
		{Severity: "MEDIUM", Summary: "chain 3", RootCause: "cause 3"},
	}

	text := formatCausalChains(chains, false)
	if !strings.Contains(text, "3 causal chain(s) found") {
		t.Errorf("wrong chain count in: %s", text)
	}
	// All three should appear.
	for _, s := range []string{"chain 1", "chain 2", "chain 3"} {
		if !strings.Contains(text, s) {
			t.Errorf("missing %q in output", s)
		}
	}
}

func TestTSCMapFromServerTest(t *testing.T) {
	// TSCMap with tsc=false should use full key names.
	m := TSCMap(false, "severity", "HIGH", "summary", "test")
	if m["severity"] != "HIGH" {
		t.Errorf("severity = %v, want HIGH", m["severity"])
	}
	if m["summary"] != "test" {
		t.Errorf("summary = %v, want test", m["summary"])
	}
}
