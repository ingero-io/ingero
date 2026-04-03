package mcp

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	gomcp "github.com/modelcontextprotocol/go-sdk/mcp"

	"github.com/ingero-io/ingero/internal/fleet"
)

// extractText gets the text from a CallToolResult's first content item.
func extractText(r *gomcp.CallToolResult) string {
	if len(r.Content) == 0 {
		return ""
	}
	if tc, ok := r.Content[0].(*gomcp.TextContent); ok {
		return tc.Text
	}
	return fmt.Sprintf("%v", r.Content[0])
}

// mockFleetQueryServer creates an httptest server for fleet endpoints.
func mockFleetQueryServer(columns []string, rows [][]any) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/v1/query" && r.Method == "POST" {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{"columns": columns, "rows": rows})
			return
		}
		if r.URL.Path == "/api/v1/chains" {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{
				"chains": []map[string]any{
					{"id": "test:oom", "severity": "HIGH", "summary": "OOM", "root_cause": "test", "explanation": "test", "node": r.Host},
				},
			})
			return
		}
		if r.URL.Path == "/api/v1/ops" {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{"mode": "aggregate", "ops": []any{}})
			return
		}
		if r.URL.Path == "/api/v1/overview" {
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(map[string]any{"event_count": 100, "chain_count": 1})
			return
		}
		http.NotFound(w, r)
	}))
}

func mockFleetErrorServer() *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		http.Error(w, `{"error":"down"}`, http.StatusInternalServerError)
	}))
}

func fleetAddr(ts *httptest.Server) string {
	return strings.TrimPrefix(ts.URL, "http://")
}

func TestFleetChainsAction(t *testing.T) {
	s1 := mockFleetQueryServer(nil, nil)
	defer s1.Close()
	s2 := mockFleetQueryServer(nil, nil)
	defer s2.Close()

	srv := New(nil)
	srv.SetFleetNodes([]string{fleetAddr(s1), fleetAddr(s2)})

	tsc := true
	result, _, err := srv.handleQueryFleet(context.Background(), queryFleetInput{
		Action: "chains",
		Since:  "5m",
		TSC:    &tsc,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", extractText(result))
	}
	if !strings.Contains(extractText(result), "Fleet Chains") {
		t.Errorf("expected 'Fleet Chains' in output, got: %s", extractText(result))
	}
}

func TestFleetSQLAction(t *testing.T) {
	s1 := mockFleetQueryServer([]string{"src", "cnt"}, [][]any{{"cuda", 100}})
	defer s1.Close()

	srv := New(nil)
	srv.SetFleetNodes([]string{fleetAddr(s1)})

	result, _, err := srv.handleQueryFleet(context.Background(), queryFleetInput{
		Action: "sql",
		SQL:    "SELECT source AS src, count(*) AS cnt FROM events GROUP BY source",
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", extractText(result))
	}
}

func TestFleetOpsAction(t *testing.T) {
	s1 := mockFleetQueryServer(nil, nil)
	defer s1.Close()

	srv := New(nil)
	srv.SetFleetNodes([]string{fleetAddr(s1)})

	result, _, err := srv.handleQueryFleet(context.Background(), queryFleetInput{
		Action: "ops",
		Since:  "5m",
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", extractText(result))
	}
}

func TestFleetOverviewAction(t *testing.T) {
	s1 := mockFleetQueryServer(nil, nil)
	defer s1.Close()

	srv := New(nil)
	srv.SetFleetNodes([]string{fleetAddr(s1)})

	result, _, err := srv.handleQueryFleet(context.Background(), queryFleetInput{
		Action: "overview",
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", extractText(result))
	}
}

func TestFleetUnconfigured(t *testing.T) {
	srv := New(nil)

	result, _, err := srv.handleQueryFleet(context.Background(), queryFleetInput{
		Action: "chains",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Error("expected error for unconfigured fleet")
	}
	if !strings.Contains(extractText(result), "No fleet nodes configured") {
		t.Errorf("expected 'No fleet nodes configured', got: %s", extractText(result))
	}
}

func TestFleetSQLMissingField(t *testing.T) {
	srv := New(nil)
	srv.SetFleetNodes([]string{"dummy:8080"})

	result, _, err := srv.handleQueryFleet(context.Background(), queryFleetInput{
		Action: "sql",
		SQL:    "",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Error("expected error for missing sql field")
	}
	if !strings.Contains(extractText(result), "sql field is required") {
		t.Errorf("expected 'sql field is required', got: %s", extractText(result))
	}
}

func TestFleetUnknownAction(t *testing.T) {
	srv := New(nil)
	srv.SetFleetNodes([]string{"dummy:8080"})

	result, _, err := srv.handleQueryFleet(context.Background(), queryFleetInput{
		Action: "invalid",
	})
	if err != nil {
		t.Fatal(err)
	}
	if !result.IsError {
		t.Error("expected error for unknown action")
	}
}

func TestFleetPartialFailure(t *testing.T) {
	s1 := mockFleetQueryServer(nil, nil)
	defer s1.Close()
	s2 := mockFleetErrorServer()
	defer s2.Close()

	srv := New(nil)
	srv.SetFleetNodes([]string{fleetAddr(s1), fleetAddr(s2)})

	tsc := false
	result, _, err := srv.handleQueryFleet(context.Background(), queryFleetInput{
		Action: "chains",
		Since:  "5m",
		TSC:    &tsc,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Errorf("expected partial success, got error: %s", extractText(result))
	}
	if !strings.Contains(extractText(result), "Warnings") {
		t.Error("expected warnings for failed node")
	}
}

func TestFleetTimeout(t *testing.T) {
	slow := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		time.Sleep(2 * time.Second)
	}))
	defer slow.Close()

	fast := mockFleetQueryServer(nil, nil)
	defer fast.Close()

	srv := New(nil)
	srv.SetFleetNodes([]string{fleetAddr(fast), fleetAddr(slow)})

	fc, _ := fleet.New(fleet.Config{
		Nodes:   []string{fleetAddr(fast), fleetAddr(slow)},
		Timeout: 200 * time.Millisecond,
	})
	srv.fleetClient = fc

	result, _, err := srv.handleQueryFleet(context.Background(), queryFleetInput{
		Action: "chains",
		Since:  "5m",
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Errorf("expected partial success, got error: %s", extractText(result))
	}
}

func TestFleetSQLWithTSC(t *testing.T) {
	s1 := mockFleetQueryServer([]string{"x", "y"}, [][]any{{1, 2}, {3, 4}})
	defer s1.Close()

	srv := New(nil)
	srv.SetFleetNodes([]string{fleetAddr(s1)})

	tsc := true
	result, _, err := srv.handleQueryFleet(context.Background(), queryFleetInput{
		Action: "sql",
		SQL:    "SELECT x, y FROM t",
		TSC:    &tsc,
	})
	if err != nil {
		t.Fatal(err)
	}
	if result.IsError {
		t.Fatalf("unexpected error: %s", extractText(result))
	}
	if !strings.Contains(extractText(result), "|") {
		t.Error("expected TSC pipe-separated format")
	}
}
