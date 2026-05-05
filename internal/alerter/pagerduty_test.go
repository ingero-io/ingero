package alerter

import (
	"context"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"regexp"
	"strings"
	"testing"
	"time"
)

const testRoutingKey = "pd-secret-rk-test-12345"

func TestTrigger_HappyPath(t *testing.T) {
	var seenBody string
	var seenContentType string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		seenContentType = r.Header.Get("Content-Type")
		b, _ := io.ReadAll(r.Body)
		seenBody = string(b)
		w.WriteHeader(http.StatusAccepted)
	}))
	defer srv.Close()

	pd := NewPagerDuty(&PagerDutyConfig{RoutingKey: testRoutingKey}, time.Second, nil)
	pd.url = srv.URL

	dedup, err := pd.Trigger(context.Background(), TriggerParams{
		Summary:       "test incident",
		Severity:      "critical",
		Source:        "node-A",
		DedupKey:      "fixed-key-1",
		CustomDetails: map[string]any{"foo": "bar"},
	})
	if err != nil {
		t.Fatalf("Trigger: %v", err)
	}
	if dedup != "fixed-key-1" {
		t.Errorf("dedup=%q, want fixed-key-1", dedup)
	}
	if seenContentType != "application/json" {
		t.Errorf("content-type=%q", seenContentType)
	}

	var got map[string]any
	if err := json.Unmarshal([]byte(seenBody), &got); err != nil {
		t.Fatalf("unmarshal body: %v", err)
	}
	if got["routing_key"] != testRoutingKey {
		t.Errorf("routing_key=%v", got["routing_key"])
	}
	if got["event_action"] != "trigger" {
		t.Errorf("event_action=%v", got["event_action"])
	}
	if got["dedup_key"] != "fixed-key-1" {
		t.Errorf("dedup_key=%v", got["dedup_key"])
	}
	pp, ok := got["payload"].(map[string]any)
	if !ok {
		t.Fatalf("payload not an object: %v", got["payload"])
	}
	if pp["summary"] != "test incident" {
		t.Errorf("summary=%v", pp["summary"])
	}
	if pp["severity"] != "critical" {
		t.Errorf("severity=%v", pp["severity"])
	}
	if pp["source"] != "node-A" {
		t.Errorf("source=%v", pp["source"])
	}
	if pp["component"] != "ingero" {
		t.Errorf("component=%v", pp["component"])
	}
	if pp["class"] != "ai_investigation" {
		t.Errorf("class=%v", pp["class"])
	}
	cd, ok := pp["custom_details"].(map[string]any)
	if !ok || cd["foo"] != "bar" {
		t.Errorf("custom_details=%v", pp["custom_details"])
	}
}

func TestTrigger_AutoGeneratesDedupKey(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusAccepted)
	}))
	defer srv.Close()

	pd := NewPagerDuty(&PagerDutyConfig{RoutingKey: testRoutingKey}, time.Second, nil)
	pd.url = srv.URL

	dedup, err := pd.Trigger(context.Background(), TriggerParams{
		Summary:  "auto dedup",
		Severity: "warning",
	})
	if err != nil {
		t.Fatalf("Trigger: %v", err)
	}
	uuidPattern := regexp.MustCompile(`^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$`)
	if !uuidPattern.MatchString(dedup) {
		t.Errorf("dedup=%q does not match UUIDv4", dedup)
	}

	// A second call without dedup should yield a distinct UUID.
	dedup2, err := pd.Trigger(context.Background(), TriggerParams{
		Summary:  "auto dedup 2",
		Severity: "warning",
	})
	if err != nil {
		t.Fatalf("Trigger 2: %v", err)
	}
	if dedup == dedup2 {
		t.Errorf("expected distinct UUIDs across calls; got %q twice", dedup)
	}
}

func TestTrigger_InvalidSeverity(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatalf("network call should not happen on invalid severity")
	}))
	defer srv.Close()

	pd := NewPagerDuty(&PagerDutyConfig{RoutingKey: testRoutingKey}, time.Second, nil)
	pd.url = srv.URL

	_, err := pd.Trigger(context.Background(), TriggerParams{
		Summary:  "bad sev",
		Severity: "urgent",
	})
	if err == nil {
		t.Fatal("expected error on invalid severity")
	}
	if !strings.Contains(err.Error(), "severity") {
		t.Errorf("error did not mention severity: %v", err)
	}
}

func TestTrigger_EmptySummary(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		t.Fatalf("network call should not happen on empty summary")
	}))
	defer srv.Close()

	pd := NewPagerDuty(&PagerDutyConfig{RoutingKey: testRoutingKey}, time.Second, nil)
	pd.url = srv.URL

	_, err := pd.Trigger(context.Background(), TriggerParams{
		Summary:  "",
		Severity: "warning",
	})
	if err == nil {
		t.Fatal("expected error on empty summary")
	}
	if !strings.Contains(err.Error(), "summary") {
		t.Errorf("error did not mention summary: %v", err)
	}
}

func TestTrigger_RoutingKeyEmpty(t *testing.T) {
	pdNilCfg := NewPagerDuty(nil, time.Second, nil)
	_, err := pdNilCfg.Trigger(context.Background(), TriggerParams{
		Summary:  "x",
		Severity: "warning",
	})
	if err == nil {
		t.Fatal("expected error on nil cfg")
	}
	if !strings.Contains(err.Error(), "pagerduty not configured") {
		t.Errorf("error missing 'pagerduty not configured': %v", err)
	}

	pdEmpty := NewPagerDuty(&PagerDutyConfig{RoutingKey: ""}, time.Second, nil)
	_, err = pdEmpty.Trigger(context.Background(), TriggerParams{
		Summary:  "x",
		Severity: "warning",
	})
	if err == nil {
		t.Fatal("expected error on empty routing key")
	}
	if !strings.Contains(err.Error(), "pagerduty not configured") {
		t.Errorf("error missing 'pagerduty not configured': %v", err)
	}
}

func TestTrigger_StatusSurfaced(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
	}))
	defer srv.Close()

	pd := NewPagerDuty(&PagerDutyConfig{RoutingKey: testRoutingKey}, time.Second, nil)
	pd.url = srv.URL

	_, err := pd.Trigger(context.Background(), TriggerParams{
		Summary:  "ok",
		Severity: "warning",
	})
	if err == nil {
		t.Fatal("expected error on 400")
	}
	if !strings.Contains(err.Error(), "status 400") {
		t.Errorf("error missing 'status 400': %v", err)
	}
	if strings.Contains(err.Error(), testRoutingKey) {
		t.Errorf("error leaked routing_key: %v", err)
	}
}

func TestTrigger_RoutingKeyScrubbedFromErrors(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	pd := NewPagerDuty(&PagerDutyConfig{RoutingKey: testRoutingKey}, time.Second, nil)
	pd.url = srv.URL

	_, err := pd.Trigger(context.Background(), TriggerParams{
		Summary:  "ok",
		Severity: "warning",
	})
	if err == nil {
		t.Fatal("expected error on 500")
	}
	if strings.Contains(err.Error(), testRoutingKey) {
		t.Errorf("error leaked routing_key: %v", err)
	}
}
