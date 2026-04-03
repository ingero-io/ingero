package fleet

import (
	"context"
	"encoding/json"
	"math"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"
)

// mockTimeServer creates an httptest server that returns a fixed timestamp.
func mockTimeServer(offsetNs int64) *httptest.Server {
	return httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		if r.URL.Path == "/api/v1/time" {
			// Return current time + offset to simulate clock skew.
			ts := time.Now().UnixNano() + offsetNs
			w.Header().Set("Content-Type", "application/json")
			json.NewEncoder(w).Encode(timeResponse{TimestampNS: ts})
			return
		}
		http.NotFound(w, r)
	}))
}

func clockAddr(ts *httptest.Server) string {
	return strings.TrimPrefix(ts.URL, "http://")
}

func TestEstimateClockSkew_NoSkew(t *testing.T) {
	// Two servers with zero offset — should report near-zero skew.
	s1 := mockTimeServer(0)
	defer s1.Close()
	s2 := mockTimeServer(0)
	defer s2.Close()

	c, _ := New(Config{Nodes: []string{clockAddr(s1), clockAddr(s2)}, Timeout: 5 * time.Second})
	results, err := c.EstimateClockSkew(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	if len(results) != 1 {
		t.Fatalf("expected 1 pairwise result, got %d", len(results))
	}
	// With same-machine servers, offset should be very small (< 10ms).
	if math.Abs(results[0].OffsetMs) > 10 {
		t.Errorf("expected near-zero offset, got %.1fms", results[0].OffsetMs)
	}
}

func TestEstimateClockSkew_Skewed(t *testing.T) {
	// s1 has no offset, s2 is 50ms ahead.
	s1 := mockTimeServer(0)
	defer s1.Close()
	s2 := mockTimeServer(50 * int64(time.Millisecond))
	defer s2.Close()

	c, _ := New(Config{Nodes: []string{clockAddr(s1), clockAddr(s2)}, Timeout: 5 * time.Second})
	results, err := c.EstimateClockSkew(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	if len(results) != 1 {
		t.Fatalf("expected 1 pairwise result, got %d", len(results))
	}
	// s2 should be ~50ms ahead of s1.
	if math.Abs(results[0].OffsetMs-50) > 15 {
		t.Errorf("expected ~50ms offset, got %.1fms", results[0].OffsetMs)
	}
}

func TestEstimateClockSkew_ThreeNodes(t *testing.T) {
	s1 := mockTimeServer(0)
	defer s1.Close()
	s2 := mockTimeServer(100 * int64(time.Millisecond)) // 100ms ahead
	defer s2.Close()
	s3 := mockTimeServer(0)
	defer s3.Close()

	c, _ := New(Config{Nodes: []string{clockAddr(s1), clockAddr(s2), clockAddr(s3)}, Timeout: 5 * time.Second})
	results, err := c.EstimateClockSkew(context.Background())
	if err != nil {
		t.Fatal(err)
	}

	// 3 nodes → 3 pairwise comparisons: s1↔s2, s1↔s3, s2↔s3.
	if len(results) != 3 {
		t.Fatalf("expected 3 pairwise results, got %d", len(results))
	}
}

func TestFormatClockSkewWarnings_BelowThreshold(t *testing.T) {
	results := []ClockSkewResult{
		{NodeA: "a", NodeB: "b", OffsetMs: 5, RTTMs: 1},
	}
	warnings := FormatClockSkewWarnings(results, 10) // threshold 10ms
	if len(warnings) != 0 {
		t.Errorf("expected no warnings for 5ms offset with 10ms threshold, got %v", warnings)
	}
}

func TestFormatClockSkewWarnings_AboveThreshold(t *testing.T) {
	results := []ClockSkewResult{
		{NodeA: "a", NodeB: "b", OffsetMs: 47, RTTMs: 2},
	}
	warnings := FormatClockSkewWarnings(results, 10)
	if len(warnings) != 1 {
		t.Fatalf("expected 1 warning, got %d", len(warnings))
	}
	if !strings.Contains(warnings[0], "47ms") {
		t.Errorf("warning should mention 47ms, got: %s", warnings[0])
	}
	if !strings.Contains(warnings[0], "ahead of") {
		t.Errorf("warning should say 'ahead of', got: %s", warnings[0])
	}
}

func TestFormatClockSkewWarnings_NegativeOffset(t *testing.T) {
	results := []ClockSkewResult{
		{NodeA: "a", NodeB: "b", OffsetMs: -30, RTTMs: 1},
	}
	warnings := FormatClockSkewWarnings(results, 10)
	if len(warnings) != 1 {
		t.Fatalf("expected 1 warning, got %d", len(warnings))
	}
	if !strings.Contains(warnings[0], "behind") {
		t.Errorf("warning should say 'behind', got: %s", warnings[0])
	}
}

func TestPrintClockSkewWarnings_Empty(t *testing.T) {
	results := []ClockSkewResult{
		{NodeA: "a", NodeB: "b", OffsetMs: 2, RTTMs: 1},
	}
	out := PrintClockSkewWarnings(results, 10)
	if out != "" {
		t.Errorf("expected empty string for below-threshold, got: %q", out)
	}
}

func TestClockSkewResult_String(t *testing.T) {
	r := ClockSkewResult{NodeA: "node-1", NodeB: "node-2", OffsetMs: 47, RTTMs: 2}
	s := r.String()
	if !strings.Contains(s, "node-2") || !strings.Contains(s, "47ms") || !strings.Contains(s, "node-1") {
		t.Errorf("unexpected String(): %s", s)
	}
}
