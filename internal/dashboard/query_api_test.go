package dashboard

import (
	"bytes"
	"context"
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/ingero-io/ingero/internal/store"
	"github.com/ingero-io/ingero/pkg/events"
)

func setupTestStore(t *testing.T) *store.Store {
	t.Helper()
	s, err := store.New(":memory:")
	if err != nil {
		t.Fatalf("New(:memory:) failed: %v", err)
	}

	s.SetNode("test-node")

	// Start background writer.
	ctx, cancel := context.WithCancel(context.Background())
	done := make(chan struct{})
	go func() { s.Run(ctx); close(done) }()

	// Insert test events.
	for i := 0; i < 5; i++ {
		s.Record(events.Event{
			Timestamp: time.Now(),
			PID:       1234,
			TID:       1235,
			Source:    events.SourceCUDA,
			Op:        uint8(events.CUDAMalloc),
			Duration:  time.Millisecond,
			Node:      "test-node",
		})
	}

	time.Sleep(300 * time.Millisecond)
	cancel()
	<-done

	return s
}

func TestHandleQuery_ValidSELECT(t *testing.T) {
	s := setupTestStore(t)
	defer s.Close()

	srv := New(s, ":0", "", "")

	body, _ := json.Marshal(map[string]any{
		"sql":   "SELECT source, count(*) as cnt FROM events GROUP BY source",
		"limit": 100,
	})
	req := httptest.NewRequest("POST", "/api/v1/query", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	srv.handleQuery(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200. Body: %s", rec.Code, rec.Body.String())
	}

	var resp struct {
		Columns []string `json:"columns"`
		Rows    [][]any  `json:"rows"`
	}
	if err := json.NewDecoder(rec.Body).Decode(&resp); err != nil {
		t.Fatalf("decoding response: %v", err)
	}
	if len(resp.Columns) < 1 {
		t.Error("expected at least 1 column")
	}
	if len(resp.Rows) < 1 {
		t.Error("expected at least 1 row")
	}
}

func TestHandleQuery_RejectINSERT(t *testing.T) {
	s := setupTestStore(t)
	defer s.Close()

	srv := New(s, ":0", "", "")

	body, _ := json.Marshal(map[string]any{
		"sql": "INSERT INTO events VALUES (1,2,3)",
	})
	req := httptest.NewRequest("POST", "/api/v1/query", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	srv.handleQuery(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400 for INSERT", rec.Code)
	}
}

func TestHandleQuery_RejectDELETE(t *testing.T) {
	s := setupTestStore(t)
	defer s.Close()

	srv := New(s, ":0", "", "")

	body, _ := json.Marshal(map[string]any{
		"sql": "DELETE FROM events",
	})
	req := httptest.NewRequest("POST", "/api/v1/query", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	srv.handleQuery(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400 for DELETE", rec.Code)
	}
}

func TestHandleQuery_RejectGET(t *testing.T) {
	srv := New(nil, ":0", "", "")

	req := httptest.NewRequest("GET", "/api/v1/query", nil)
	rec := httptest.NewRecorder()

	srv.handleQuery(rec, req)

	if rec.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405 for GET", rec.Code)
	}
}

func TestHandleQuery_EmptySQL(t *testing.T) {
	s := setupTestStore(t)
	defer s.Close()

	srv := New(s, ":0", "", "")

	body, _ := json.Marshal(map[string]any{
		"sql": "",
	})
	req := httptest.NewRequest("POST", "/api/v1/query", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	srv.handleQuery(rec, req)

	if rec.Code != http.StatusBadRequest {
		t.Errorf("status = %d, want 400 for empty SQL", rec.Code)
	}
}

func TestHandleQuery_LimitEnforced(t *testing.T) {
	s := setupTestStore(t)
	defer s.Close()

	srv := New(s, ":0", "", "")

	body, _ := json.Marshal(map[string]any{
		"sql":   "SELECT * FROM events",
		"limit": 2, // Only get 2 of the 5 events
	})
	req := httptest.NewRequest("POST", "/api/v1/query", bytes.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	rec := httptest.NewRecorder()

	srv.handleQuery(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200. Body: %s", rec.Code, rec.Body.String())
	}

	var resp struct {
		Rows [][]any `json:"rows"`
	}
	json.NewDecoder(rec.Body).Decode(&resp)
	if len(resp.Rows) > 2 {
		t.Errorf("rows = %d, want <= 2 (limit enforced)", len(resp.Rows))
	}
}

func TestNoTLSMode(t *testing.T) {
	srv := New(nil, "localhost:0", "", "")
	srv.SetNoTLS(true)

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	done := make(chan error, 1)
	go func() {
		done <- srv.Start(ctx)
	}()

	time.Sleep(100 * time.Millisecond)
	cancel()

	select {
	case err := <-done:
		if err != nil {
			t.Errorf("Start() returned error: %v", err)
		}
	case <-time.After(3 * time.Second):
		t.Fatal("server did not shut down within 3s")
	}
}

func TestHandleChains_IncludesNode(t *testing.T) {
	s := setupTestStore(t)
	defer s.Close()

	// Insert a chain with node field.
	s.RecordChains([]store.StoredChain{
		{
			ID:          "test-node:oom",
			DetectedAt:  time.Now(),
			Severity:    "HIGH",
			Summary:     "test",
			RootCause:   "test",
			Explanation: "test",
			Node:        "test-node",
		},
	})

	srv := New(s, ":0", "", "")
	req := httptest.NewRequest("GET", "/api/v1/chains", nil)
	req.Host = "localhost"
	rec := httptest.NewRecorder()

	srv.handleChains(rec, req)

	if rec.Code != http.StatusOK {
		t.Fatalf("status = %d, want 200", rec.Code)
	}

	var resp chainResponse
	json.NewDecoder(rec.Body).Decode(&resp)
	if len(resp.Chains) != 1 {
		t.Fatalf("chains = %d, want 1", len(resp.Chains))
	}
	if resp.Chains[0].Node != "test-node" {
		t.Errorf("chain node = %q, want %q", resp.Chains[0].Node, "test-node")
	}
}
