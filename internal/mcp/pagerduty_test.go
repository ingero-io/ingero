package mcp

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ingero-io/ingero/internal/alerter"
)

func newTestPD(t *testing.T, handler http.HandlerFunc) (*alerter.PagerDuty, *httptest.Server) {
	t.Helper()
	srv := httptest.NewServer(handler)
	pd := alerter.NewPagerDuty(&alerter.PagerDutyConfig{RoutingKey: "rk-mcp-test"}, time.Second, nil)
	alerter.SetPagerDutyURLForTest(pd, srv.URL)
	return pd, srv
}

func TestPagerDutyTool_HappyPath(t *testing.T) {
	var bodySeen []byte
	pd, srv := newTestPD(t, func(w http.ResponseWriter, r *http.Request) {
		bodySeen, _ = io.ReadAll(r.Body)
		w.WriteHeader(http.StatusAccepted)
	})
	defer srv.Close()

	out, err := handlePagerDutyTrigger(context.Background(), pd, PagerDutyTriggerInput{
		Summary:       "investigation found GPU hang",
		Severity:      "error",
		Source:        "node-A",
		CustomDetails: map[string]any{"chain": "x", "score": 0.9},
	})
	if err != nil {
		t.Fatalf("handler: %v", err)
	}
	if out.Status != "accepted" {
		t.Errorf("status=%q, want accepted", out.Status)
	}
	if out.DedupKey == "" {
		t.Errorf("dedup_key empty")
	}

	var got map[string]any
	if err := json.Unmarshal(bodySeen, &got); err != nil {
		t.Fatalf("server received non-JSON: %v", err)
	}
	if got["event_action"] != "trigger" {
		t.Errorf("event_action=%v", got["event_action"])
	}
	pp := got["payload"].(map[string]any)
	if pp["summary"] != "investigation found GPU hang" {
		t.Errorf("summary=%v", pp["summary"])
	}
	if pp["severity"] != "error" {
		t.Errorf("severity=%v", pp["severity"])
	}
}

func TestPagerDutyTool_NotConfigured(t *testing.T) {
	out, err := handlePagerDutyTrigger(context.Background(), nil, PagerDutyTriggerInput{
		Summary:  "x",
		Severity: "warning",
	})
	if err == nil {
		t.Fatalf("expected error; got out=%+v", out)
	}
	if !strings.Contains(err.Error(), "not configured") {
		t.Errorf("error missing 'not configured': %v", err)
	}
	if !strings.Contains(err.Error(), "--pagerduty-routing-key") {
		t.Errorf("error missing flag hint: %v", err)
	}
}

func TestPagerDutyTool_InvalidSeverity(t *testing.T) {
	pd, srv := newTestPD(t, func(w http.ResponseWriter, r *http.Request) {
		t.Fatalf("network call should not happen on invalid severity")
	})
	defer srv.Close()

	_, err := handlePagerDutyTrigger(context.Background(), pd, PagerDutyTriggerInput{
		Summary:  "x",
		Severity: "urgent",
	})
	if err == nil {
		t.Fatal("expected error on invalid severity")
	}
	if !strings.Contains(err.Error(), "severity") {
		t.Errorf("error missing 'severity': %v", err)
	}
}

func TestPagerDutyTool_OversizedCustomDetails(t *testing.T) {
	pd, srv := newTestPD(t, func(w http.ResponseWriter, r *http.Request) {
		t.Fatalf("network call should not happen on oversized details")
	})
	defer srv.Close()

	big := strings.Repeat("a", 300*1024)
	_, err := handlePagerDutyTrigger(context.Background(), pd, PagerDutyTriggerInput{
		Summary:       "x",
		Severity:      "warning",
		CustomDetails: map[string]any{"blob": big},
	})
	if err == nil {
		t.Fatal("expected error on oversized custom_details")
	}
	if !strings.Contains(err.Error(), "256 KiB") {
		t.Errorf("error missing size limit: %v", err)
	}
}

func TestPagerDutyTool_NetworkFailure(t *testing.T) {
	pd, srv := newTestPD(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	})
	defer srv.Close()

	_, err := handlePagerDutyTrigger(context.Background(), pd, PagerDutyTriggerInput{
		Summary:  "x",
		Severity: "warning",
	})
	if err == nil {
		t.Fatal("expected error on 500")
	}
}

func TestPagerDutyTool_DedupKeyPassthrough(t *testing.T) {
	pd, srv := newTestPD(t, func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusAccepted)
	})
	defer srv.Close()

	const supplied = "investigation-2026-05-02-001"
	out, err := handlePagerDutyTrigger(context.Background(), pd, PagerDutyTriggerInput{
		Summary:  "x",
		Severity: "warning",
		DedupKey: supplied,
	})
	if err != nil {
		t.Fatalf("handler: %v", err)
	}
	if out.DedupKey != supplied {
		t.Errorf("dedup_key=%q, want %q (passthrough)", out.DedupKey, supplied)
	}
}
