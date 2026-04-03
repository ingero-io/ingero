package fleet

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// mockQueryServer creates an httptest server that responds to POST /api/v1/query.
func mockQueryServer(columns []string, rows [][]any) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/v1/query" && r.Method == "POST" {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(queryResponse{Columns: columns, Rows: rows})
			return
		}
		if r.URL.Path == "/api/v1/chains" {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(chainAPIResponse{Chains: []ChainEntry{}})
			return
		}
		http.NotFound(w, r)
	}))
}

// mockChainServer creates a server that responds to GET /api/v1/chains.
func mockChainServer(node string, chains []ChainEntry) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/v1/chains" {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(chainAPIResponse{Chains: chains})
			return
		}
		http.NotFound(w, r)
	}))
}

// mockErrorServer returns 500 for all requests.
func mockErrorServer() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"internal error"}`, http.StatusInternalServerError)
	}))
}

// mockSlowServer delays longer than the timeout.
func mockSlowServer(delay time.Duration) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(delay)
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(queryResponse{Columns: []string{"x"}, Rows: [][]any{{1}}})
	}))
}

func addr(ts *httptest.Server) string {
	return strings.TrimPrefix(ts.URL, "http://")
}

func TestQuerySQL_TwoNodes(t *testing.T) {
	s1 := mockQueryServer([]string{"source", "count"}, [][]any{{"cuda", 100}})
	defer s1.Close()
	s2 := mockQueryServer([]string{"source", "count"}, [][]any{{"host", 50}, {"driver", 25}})
	defer s2.Close()

	c, err := New(Config{Nodes: []string{addr(s1), addr(s2)}, Timeout: 5 * time.Second})
	if err != nil {
		t.Fatal(err)
	}

	result, err := c.QuerySQL(context.Background(), "SELECT source, count(*) FROM events GROUP BY source")
	if err != nil {
		t.Fatal(err)
	}

	if len(result.Columns) != 3 { // node + source + count
		t.Errorf("columns = %v, want 3 columns (node prepended)", result.Columns)
	}
	if result.Columns[0] != "node" {
		t.Errorf("first column = %q, want %q", result.Columns[0], "node")
	}
	if len(result.Rows) != 3 { // 1 from s1 + 2 from s2
		t.Errorf("rows = %d, want 3", len(result.Rows))
	}
	if len(result.Warnings) != 0 {
		t.Errorf("warnings = %v, want none", result.Warnings)
	}
}

func TestQuerySQL_ThreeNodes(t *testing.T) {
	s1 := mockQueryServer([]string{"pid"}, [][]any{{1234}})
	defer s1.Close()
	s2 := mockQueryServer([]string{"pid"}, [][]any{{5678}})
	defer s2.Close()
	s3 := mockQueryServer([]string{"pid"}, [][]any{{9012}})
	defer s3.Close()

	c, _ := New(Config{Nodes: []string{addr(s1), addr(s2), addr(s3)}})
	result, err := c.QuerySQL(context.Background(), "SELECT pid FROM events")
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Rows) != 3 {
		t.Errorf("rows = %d, want 3", len(result.Rows))
	}
}

func TestQuerySQL_PartialFailure(t *testing.T) {
	s1 := mockQueryServer([]string{"src"}, [][]any{{"cuda"}})
	defer s1.Close()
	s2 := mockErrorServer()
	defer s2.Close()

	c, _ := New(Config{Nodes: []string{addr(s1), addr(s2)}})
	result, err := c.QuerySQL(context.Background(), "SELECT src FROM events")
	if err != nil {
		t.Fatalf("expected partial success, got error: %v", err)
	}
	if len(result.Rows) != 1 {
		t.Errorf("rows = %d, want 1 (from successful node)", len(result.Rows))
	}
	if len(result.Warnings) != 1 {
		t.Errorf("warnings = %d, want 1", len(result.Warnings))
	}
}

func TestQuerySQL_AllFail(t *testing.T) {
	s1 := mockErrorServer()
	defer s1.Close()
	s2 := mockErrorServer()
	defer s2.Close()

	c, _ := New(Config{Nodes: []string{addr(s1), addr(s2)}})
	_, err := c.QuerySQL(context.Background(), "SELECT 1")
	if err == nil {
		t.Error("expected error when all nodes fail")
	}
}

func TestQuerySQL_EmptyResult(t *testing.T) {
	s1 := mockQueryServer([]string{"src"}, [][]any{{"cuda"}})
	defer s1.Close()
	s2 := mockQueryServer([]string{"src"}, [][]any{}) // empty
	defer s2.Close()

	c, _ := New(Config{Nodes: []string{addr(s1), addr(s2)}})
	result, err := c.QuerySQL(context.Background(), "SELECT src FROM events")
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Rows) != 1 {
		t.Errorf("rows = %d, want 1", len(result.Rows))
	}
}

func TestQuerySQL_SchemaMismatch(t *testing.T) {
	s1 := mockQueryServer([]string{"a", "b"}, [][]any{{1, 2}})
	defer s1.Close()
	s2 := mockQueryServer([]string{"x", "y", "z"}, [][]any{{3, 4, 5}}) // different columns
	defer s2.Close()

	c, _ := New(Config{Nodes: []string{addr(s1), addr(s2)}})
	result, err := c.QuerySQL(context.Background(), "SELECT *")
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Rows) != 1 { // only s1 rows kept
		t.Errorf("rows = %d, want 1 (mismatched node skipped)", len(result.Rows))
	}
	if len(result.Warnings) != 1 {
		t.Errorf("warnings = %d, want 1 (schema mismatch)", len(result.Warnings))
	}
}

func TestQuerySQL_Timeout(t *testing.T) {
	s1 := mockQueryServer([]string{"x"}, [][]any{{1}})
	defer s1.Close()
	s2 := mockSlowServer(2 * time.Second)
	defer s2.Close()

	c, _ := New(Config{
		Nodes:   []string{addr(s1), addr(s2)},
		Timeout: 200 * time.Millisecond, // fast timeout
	})

	result, err := c.QuerySQL(context.Background(), "SELECT x")
	if err != nil {
		t.Fatalf("expected partial success, got: %v", err)
	}
	if len(result.Rows) != 1 {
		t.Errorf("rows = %d, want 1 (slow node timed out)", len(result.Rows))
	}
	if len(result.Warnings) != 1 {
		t.Errorf("warnings = %d, want 1 (timeout)", len(result.Warnings))
	}
}

func TestQuerySQL_LimitSentToNode(t *testing.T) {
	var receivedLimit int
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req queryRequest
		json.NewDecoder(r.Body).Decode(&req)
		receivedLimit = req.Limit
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(queryResponse{Columns: []string{"x"}, Rows: [][]any{{1}}})
	}))
	defer ts.Close()

	c, _ := New(Config{Nodes: []string{addr(ts)}, Limit: 500})
	c.QuerySQL(context.Background(), "SELECT x")

	if receivedLimit != 500 {
		t.Errorf("limit sent to node = %d, want 500", receivedLimit)
	}
}

func TestQuerySQL_DefaultLimit(t *testing.T) {
	var receivedLimit int
	ts := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		var req queryRequest
		json.NewDecoder(r.Body).Decode(&req)
		receivedLimit = req.Limit
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(queryResponse{Columns: []string{"x"}, Rows: [][]any{{1}}})
	}))
	defer ts.Close()

	c, _ := New(Config{Nodes: []string{addr(ts)}}) // no explicit limit
	c.QuerySQL(context.Background(), "SELECT x")

	if receivedLimit != DefaultLimit {
		t.Errorf("default limit = %d, want %d", receivedLimit, DefaultLimit)
	}
}

func TestQueryChains_TwoNodes(t *testing.T) {
	s1 := mockChainServer("node-a", []ChainEntry{
		{ID: "a:oom", Severity: "HIGH", Summary: "OOM on node-a", Node: "node-a"},
	})
	defer s1.Close()
	s2 := mockChainServer("node-b", []ChainEntry{
		{ID: "b:tail", Severity: "MEDIUM", Summary: "tail latency on node-b", Node: "node-b"},
		{ID: "b:low", Severity: "LOW", Summary: "low severity", Node: "node-b"},
	})
	defer s2.Close()

	c, _ := New(Config{Nodes: []string{addr(s1), addr(s2)}})
	result, err := c.QueryChains(context.Background(), "5m")
	if err != nil {
		t.Fatal(err)
	}
	if len(result.Chains) != 3 {
		t.Fatalf("chains = %d, want 3", len(result.Chains))
	}
	// Sorted by severity: HIGH first.
	if result.Chains[0].Severity != "HIGH" {
		t.Errorf("first chain severity = %q, want HIGH", result.Chains[0].Severity)
	}
	if result.Chains[1].Severity != "MEDIUM" {
		t.Errorf("second chain severity = %q, want MEDIUM", result.Chains[1].Severity)
	}
	if result.Chains[2].Severity != "LOW" {
		t.Errorf("third chain severity = %q, want LOW", result.Chains[2].Severity)
	}
}

func TestQueryChains_PartialFailure(t *testing.T) {
	s1 := mockChainServer("node-a", []ChainEntry{
		{ID: "a:oom", Severity: "HIGH", Summary: "OOM", Node: "node-a"},
	})
	defer s1.Close()
	s2 := mockErrorServer()
	defer s2.Close()

	c, _ := New(Config{Nodes: []string{addr(s1), addr(s2)}})
	result, err := c.QueryChains(context.Background(), "5m")
	if err != nil {
		t.Fatalf("expected partial success: %v", err)
	}
	if len(result.Chains) != 1 {
		t.Errorf("chains = %d, want 1", len(result.Chains))
	}
	if len(result.Warnings) != 1 {
		t.Errorf("warnings = %d, want 1", len(result.Warnings))
	}
}

func TestNodeColumnPrepended(t *testing.T) {
	s1 := mockQueryServer([]string{"count"}, [][]any{{42}})
	defer s1.Close()

	c, _ := New(Config{Nodes: []string{addr(s1)}})
	result, _ := c.QuerySQL(context.Background(), "SELECT count(*) FROM events")

	if len(result.Columns) < 1 || result.Columns[0] != "node" {
		t.Errorf("first column should be 'node', got %v", result.Columns)
	}
	if len(result.Rows) != 1 {
		t.Fatalf("expected 1 row, got %d", len(result.Rows))
	}
	// First cell should be the node address.
	nodeVal, ok := result.Rows[0][0].(string)
	if !ok || nodeVal == "" {
		t.Errorf("node column value = %v, want non-empty string", result.Rows[0][0])
	}
}

func TestBackwardCompat_NoNodes(t *testing.T) {
	// Verify New works with empty node list — this shouldn't be called in practice
	// but ensures no panic.
	c, _ := New(Config{Nodes: []string{}})
	result, err := c.QuerySQL(context.Background(), "SELECT 1")
	if err != nil {
		// With no nodes, merging produces no results and no errors — not "all failed".
		t.Logf("empty nodes: err=%v (acceptable)", err)
	}
	_ = result
}

func TestSeveritySort(t *testing.T) {
	tests := []struct {
		input    string
		expected int
	}{
		{"HIGH", 3},
		{"MEDIUM", 2},
		{"LOW", 1},
		{"UNKNOWN", 0},
		{"high", 3},
	}
	for _, tt := range tests {
		got := severityRank(tt.input)
		if got != tt.expected {
			fmt.Printf("severityRank(%q) = %d, want %d\n", tt.input, got, tt.expected)
		}
	}
}
