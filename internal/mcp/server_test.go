package mcp

import (
	"bufio"
	"context"
	"encoding/json"
	"fmt"
	"os"
	"os/exec"
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

func TestDeduplicateStoredChains(t *testing.T) {
	now := time.Now()

	tests := []struct {
		name      string
		chains    []store.StoredChain
		topN      int
		wantCount int
		wantFirst string // expected CUDAOp of first result
	}{
		{
			name:      "empty",
			chains:    nil,
			topN:      10,
			wantCount: 0,
		},
		{
			name: "dedup same op+severity keeps highest tail ratio",
			chains: []store.StoredChain{
				{ID: "c1", DetectedAt: now, CUDAOp: "cuLaunchKernel", Severity: "MEDIUM", TailRatio: 3.0},
				{ID: "c2", DetectedAt: now.Add(time.Second), CUDAOp: "cuLaunchKernel", Severity: "MEDIUM", TailRatio: 3.2},
				{ID: "c3", DetectedAt: now.Add(2 * time.Second), CUDAOp: "cuLaunchKernel", Severity: "MEDIUM", TailRatio: 2.8},
			},
			topN:      10,
			wantCount: 1,
		},
		{
			name: "different ops preserved",
			chains: []store.StoredChain{
				{ID: "c1", CUDAOp: "cuLaunchKernel", Severity: "MEDIUM", TailRatio: 3.0},
				{ID: "c2", CUDAOp: "cudaMemcpyAsync", Severity: "MEDIUM", TailRatio: 5.5},
				{ID: "c3", CUDAOp: "cudaLaunchKernel", Severity: "HIGH", TailRatio: 192.9},
			},
			topN:      10,
			wantCount: 3,
			wantFirst: "cudaLaunchKernel", // HIGH sorts first
		},
		{
			name: "HIGH before MEDIUM before LOW",
			chains: []store.StoredChain{
				{ID: "c1", CUDAOp: "opA", Severity: "LOW", TailRatio: 100.0},
				{ID: "c2", CUDAOp: "opB", Severity: "HIGH", TailRatio: 2.0},
				{ID: "c3", CUDAOp: "opC", Severity: "MEDIUM", TailRatio: 50.0},
			},
			topN:      10,
			wantCount: 3,
			wantFirst: "opB",
		},
		{
			name: "topN limits output",
			chains: []store.StoredChain{
				{ID: "c1", CUDAOp: "op1", Severity: "HIGH", TailRatio: 10.0},
				{ID: "c2", CUDAOp: "op2", Severity: "HIGH", TailRatio: 8.0},
				{ID: "c3", CUDAOp: "op3", Severity: "MEDIUM", TailRatio: 5.0},
				{ID: "c4", CUDAOp: "op4", Severity: "MEDIUM", TailRatio: 3.0},
				{ID: "c5", CUDAOp: "op5", Severity: "LOW", TailRatio: 1.0},
			},
			topN:      3,
			wantCount: 3,
			wantFirst: "op1",
		},
		{
			name: "topN=0 returns all",
			chains: []store.StoredChain{
				{ID: "c1", CUDAOp: "op1", Severity: "MEDIUM", TailRatio: 3.0},
				{ID: "c2", CUDAOp: "op2", Severity: "MEDIUM", TailRatio: 5.0},
			},
			topN:      0,
			wantCount: 2,
		},
		{
			name: "60 duplicate chains dedup to few",
			chains: func() []store.StoredChain {
				var chains []store.StoredChain
				for i := 0; i < 60; i++ {
					chains = append(chains, store.StoredChain{
						ID:        fmt.Sprintf("c%d", i),
						CUDAOp:    "cuLaunchKernel",
						Severity:  "MEDIUM",
						TailRatio: 3.0 + float64(i)*0.01,
					})
				}
				return chains
			}(),
			topN:      10,
			wantCount: 1, // all same op+severity = 1 unique
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := deduplicateStoredChains(tt.chains, tt.topN)
			if len(got) != tt.wantCount {
				t.Errorf("got %d chains, want %d", len(got), tt.wantCount)
			}
			if tt.wantFirst != "" && len(got) > 0 && got[0].CUDAOp != tt.wantFirst {
				t.Errorf("first chain op = %q, want %q", got[0].CUDAOp, tt.wantFirst)
			}
		})
	}
}

// TestMCPSubprocess is the MCP server subprocess entry point.
// When TEST_MCP_SUBPROCESS=1, it runs the MCP server on stdio and exits.
func TestMCPSubprocess(t *testing.T) {
	if os.Getenv("TEST_MCP_SUBPROCESS") != "1" {
		t.Skip("helper subprocess, not a real test")
	}
	srv := New(nil) // nil store — tools return "no database" messages
	_ = srv.Run(context.Background())
}

// TestMCPToolResponseNoStructuredContent verifies that MCP tool responses
// contain "content" but NOT "structuredContent". Claude Code reads
// structuredContent when present, so an empty structuredContent:{} causes
// Claude Code to show {} instead of the actual tool output.
//
// Root cause: using struct{} as the Out type in ToolHandlerFor causes the
// go-sdk to serialize structuredContent:{} in every response. The fix is
// to use any as the Out type and return nil, which makes the SDK skip
// structuredContent entirely.
//
// This test re-executes the test binary as a subprocess (like Claude Code
// does with the ingero binary) and communicates via stdio pipes.
func TestMCPToolResponseNoStructuredContent(t *testing.T) {
	if testing.Short() {
		t.Skip("skipping MCP integration test in short mode")
	}

	// Start self as subprocess running the MCP server.
	cmd := exec.Command(os.Args[0], "-test.run=^TestMCPSubprocess$")
	cmd.Env = append(os.Environ(), "TEST_MCP_SUBPROCESS=1")

	stdin, err := cmd.StdinPipe()
	if err != nil {
		t.Fatal(err)
	}
	stdout, err := cmd.StdoutPipe()
	if err != nil {
		t.Fatal(err)
	}
	cmd.Stderr = os.Stderr

	if err := cmd.Start(); err != nil {
		t.Fatal(err)
	}
	defer func() {
		stdin.Close()
		cmd.Process.Kill()
		cmd.Wait()
	}()

	// Single reader goroutine — all messages arrive on this channel.
	msgs := make(chan map[string]interface{}, 100)
	go func() {
		scanner := bufio.NewScanner(stdout)
		scanner.Buffer(make([]byte, 256*1024), 256*1024)
		for scanner.Scan() {
			var msg map[string]interface{}
			if json.Unmarshal(scanner.Bytes(), &msg) == nil {
				msgs <- msg
			}
		}
		close(msgs)
	}()

	send := func(msg map[string]interface{}) {
		b, _ := json.Marshal(msg)
		b = append(b, '\n')
		stdin.Write(b)
	}
	recvWithTimeout := func(d time.Duration) (map[string]interface{}, bool) {
		select {
		case msg, ok := <-msgs:
			return msg, ok && msg != nil
		case <-time.After(d):
			return nil, false
		}
	}

	// MCP initialize handshake.
	send(map[string]interface{}{
		"jsonrpc": "2.0", "id": 1, "method": "initialize",
		"params": map[string]interface{}{
			"protocolVersion": "2025-03-26",
			"capabilities":    map[string]interface{}{},
			"clientInfo":      map[string]interface{}{"name": "test", "version": "1.0"},
		},
	})
	initResp, ok := recvWithTimeout(5 * time.Second)
	if !ok || initResp["id"] != float64(1) {
		t.Fatalf("init failed: ok=%v resp=%v", ok, initResp)
	}

	// Drain notifications.
	recvWithTimeout(500 * time.Millisecond)

	send(map[string]interface{}{
		"jsonrpc": "2.0", "method": "notifications/initialized",
	})
	// Drain post-initialized notifications.
	for {
		if _, ok := recvWithTimeout(500 * time.Millisecond); !ok {
			break
		}
	}

	// Call each tool and verify the response format.
	tools := []struct {
		name string
		args map[string]interface{}
	}{
		{"get_check", nil},
		{"get_trace_stats", nil},
		{"get_causal_chains", nil},
		{"run_demo", map[string]interface{}{"scenario": "incident"}},
		{"run_sql", map[string]interface{}{"query": "SELECT 1 as ok"}},
		{"get_stacks", nil},
		{"get_test_report", nil},
	}

	for i, tool := range tools {
		reqID := float64(100 + i)
		args := tool.args
		if args == nil {
			args = map[string]interface{}{}
		}
		send(map[string]interface{}{
			"jsonrpc": "2.0", "id": reqID, "method": "tools/call",
			"params": map[string]interface{}{"name": tool.name, "arguments": args},
		})

		// Read responses, skip notifications.
		var resp map[string]interface{}
		for attempt := 0; attempt < 20; attempt++ {
			msg, ok := recvWithTimeout(5 * time.Second)
			if !ok {
				break
			}
			if msg["id"] == reqID {
				resp = msg
				break
			}
		}
		if resp == nil {
			t.Errorf("[%s] no response received", tool.name)
			continue
		}

		rawJSON, _ := json.Marshal(resp)
		rawStr := string(rawJSON)

		// CRITICAL: structuredContent must NOT appear in the response.
		// Claude Code reads structuredContent when present. If it's {} (empty),
		// Claude Code shows {} instead of the actual content.
		if strings.Contains(rawStr, "structuredContent") {
			t.Errorf("[%s] response contains structuredContent — this breaks Claude Code.\n"+
				"Fix: use 'any' (not struct{}) as the Out type in ToolHandlerFor, return nil.\n"+
				"Response: %s", tool.name, rawStr[:min(len(rawStr), 300)])
		}

		// content MUST be present with at least one entry.
		result, ok := resp["result"].(map[string]interface{})
		if !ok {
			t.Errorf("[%s] no result field in response", tool.name)
			continue
		}
		content, hasContent := result["content"]
		if !hasContent {
			t.Errorf("[%s] response missing content field", tool.name)
		} else if arr, ok := content.([]interface{}); !ok || len(arr) == 0 {
			t.Errorf("[%s] content is empty or not an array", tool.name)
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
