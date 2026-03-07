package dashboard

import (
	"context"
	"encoding/json"
	"net/http/httptest"
	"testing"
	"time"

	"github.com/ingero-io/ingero/internal/store"
	"github.com/ingero-io/ingero/pkg/events"
)

// newTestStore creates an in-memory SQLite store for testing.
func newTestStore(t *testing.T) *store.Store {
	t.Helper()
	s, err := store.New(":memory:")
	if err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { s.Close() })
	return s
}

// seedEvents inserts test events into the store.
func seedEvents(t *testing.T, s *store.Store) {
	t.Helper()
	// Start the store's Run() goroutine for async writes.
	ctx, cancel := context.WithCancel(context.Background())
	t.Cleanup(cancel)
	go s.Run(ctx)

	now := time.Now()
	for i := 0; i < 100; i++ {
		s.Record(events.Event{
			Timestamp: now.Add(-time.Duration(i) * time.Millisecond),
			PID:       1234,
			TID:       1234,
			Source:    events.SourceCUDA,
			Op:        uint8(events.CUDALaunchKernel),
			Duration:  time.Duration(100+i) * time.Microsecond,
		})
	}
	for i := 0; i < 10; i++ {
		s.Record(events.Event{
			Timestamp: now.Add(-time.Duration(i) * time.Millisecond),
			PID:       1234,
			TID:       1234,
			Source:    events.SourceHost,
			Op:        uint8(events.HostSchedSwitch),
			Duration:  time.Duration(50+i) * time.Microsecond,
		})
	}
	// Flush.
	time.Sleep(200 * time.Millisecond)
}

func TestHandleOverview(t *testing.T) {
	s := newTestStore(t)
	seedEvents(t, s)
	srv := &Server{store: s}

	req := httptest.NewRequest("GET", "/api/v1/overview", nil)
	req.Host = "localhost"
	rec := httptest.NewRecorder()

	srv.handleOverview(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status = %d, want 200", rec.Code)
	}

	var resp overviewResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if resp.EventCount == 0 {
		t.Error("expected non-zero event count")
	}
}

func TestHandleOverviewNoStore(t *testing.T) {
	srv := &Server{store: nil}

	req := httptest.NewRequest("GET", "/api/v1/overview", nil)
	req.Host = "localhost"
	rec := httptest.NewRecorder()

	srv.handleOverview(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status = %d, want 200", rec.Code)
	}

	var resp overviewResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if resp.EventCount != 0 {
		t.Errorf("expected 0 events, got %d", resp.EventCount)
	}
}

func TestHandleOps(t *testing.T) {
	s := newTestStore(t)
	seedEvents(t, s)
	srv := &Server{store: s}

	req := httptest.NewRequest("GET", "/api/v1/ops?since=5m", nil)
	req.Host = "localhost"
	rec := httptest.NewRecorder()

	srv.handleOps(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status = %d, want 200", rec.Code)
	}

	var resp opStatsResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if len(resp.Ops) == 0 {
		t.Error("expected non-empty ops list")
	}
	if resp.Mode != "percentile" && resp.Mode != "aggregate" {
		t.Errorf("unexpected mode: %q", resp.Mode)
	}
}

func TestHandleChains(t *testing.T) {
	s := newTestStore(t)
	srv := &Server{store: s}

	req := httptest.NewRequest("GET", "/api/v1/chains", nil)
	req.Host = "localhost"
	rec := httptest.NewRecorder()

	srv.handleChains(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status = %d, want 200", rec.Code)
	}

	var resp chainResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	// No chains in a fresh DB.
	if resp.Chains == nil {
		t.Error("expected non-nil chains array (even if empty)")
	}
}

func TestHandleSnapshots(t *testing.T) {
	s := newTestStore(t)
	srv := &Server{store: s}

	req := httptest.NewRequest("GET", "/api/v1/snapshots?since=60s", nil)
	req.Host = "localhost"
	rec := httptest.NewRecorder()

	srv.handleSnapshots(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status = %d, want 200", rec.Code)
	}

	var resp snapshotResponse
	if err := json.Unmarshal(rec.Body.Bytes(), &resp); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if resp.Snapshots == nil {
		t.Error("expected non-nil snapshots array")
	}
}

func TestHandleCapabilities(t *testing.T) {
	srv := &Server{store: nil}

	req := httptest.NewRequest("GET", "/api/v1/capabilities", nil)
	req.Host = "localhost"
	rec := httptest.NewRecorder()

	srv.handleCapabilities(rec, req)

	if rec.Code != 200 {
		t.Fatalf("status = %d, want 200", rec.Code)
	}

	var caps []Capability
	if err := json.Unmarshal(rec.Body.Bytes(), &caps); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}

	if len(caps) == 0 {
		t.Fatal("expected non-empty capabilities list")
	}

	// Verify structure.
	ids := make(map[string]bool)
	var availCount, unavailCount int
	for _, c := range caps {
		if c.ID == "" {
			t.Error("capability with empty ID")
		}
		if ids[c.ID] {
			t.Errorf("duplicate capability ID: %s", c.ID)
		}
		ids[c.ID] = true

		if c.Available {
			availCount++
			if c.Source == "" {
				t.Errorf("available capability %s has no source", c.ID)
			}
		} else {
			unavailCount++
			if c.Tooltip == "" {
				t.Errorf("unavailable capability %s has no tooltip", c.ID)
			}
		}
	}

	if availCount == 0 {
		t.Error("no available capabilities")
	}
	if unavailCount == 0 {
		t.Error("no unavailable capabilities")
	}
}

func TestCapabilitiesStructure(t *testing.T) {
	caps := Capabilities()
	if len(caps) == 0 {
		t.Fatal("empty capabilities list")
	}

	ids := make(map[string]bool)
	for _, c := range caps {
		if c.ID == "" {
			t.Error("capability with empty ID")
		}
		if ids[c.ID] {
			t.Errorf("duplicate ID: %s", c.ID)
		}
		ids[c.ID] = true

		if c.Label == "" {
			t.Errorf("capability %s has no label", c.ID)
		}
		if c.Available && c.Tooltip != "" {
			t.Errorf("available capability %s should not have tooltip", c.ID)
		}
		if !c.Available && c.Tooltip == "" {
			t.Errorf("unavailable capability %s missing tooltip", c.ID)
		}
	}
}

func TestHandleOpsBadSince(t *testing.T) {
	srv := &Server{store: nil}

	req := httptest.NewRequest("GET", "/api/v1/ops?since=invalid", nil)
	req.Host = "localhost"
	rec := httptest.NewRecorder()

	srv.handleOps(rec, req)

	if rec.Code != 400 {
		t.Errorf("status = %d, want 400", rec.Code)
	}
}
