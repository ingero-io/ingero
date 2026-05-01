package alerter

import (
	"bufio"
	"context"
	"encoding/json"
	"io"
	"log/slog"
	"net"
	"net/http"
	"net/http/httptest"
	"path/filepath"
	"strings"
	"sync"
	"testing"
	"time"
)

func TestConfig_Validate(t *testing.T) {
	cases := []struct {
		name    string
		cfg     Config
		wantErr bool
	}{
		{"empty UDS", Config{}, true},
		{"only UDS", Config{UDSPath: "/tmp/x.sock"}, false},
		{"slack missing webhook", Config{UDSPath: "/x", Slack: &SlackConfig{}}, true},
		{"slack ok", Config{UDSPath: "/x", Slack: &SlackConfig{WebhookURL: "https://hooks.slack/T"}}, false},
		{"pd missing routing", Config{UDSPath: "/x", PagerDuty: &PagerDutyConfig{}}, true},
		{"pd ok", Config{UDSPath: "/x", PagerDuty: &PagerDutyConfig{RoutingKey: "abc"}}, false},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			err := tc.cfg.Validate()
			if tc.wantErr && err == nil {
				t.Fatal("expected error")
			}
			if !tc.wantErr && err != nil {
				t.Fatalf("unexpected error: %v", err)
			}
		})
	}
}

func TestParseConfig(t *testing.T) {
	jsonData := []byte(`{
		"uds_path": "/tmp/foo.sock",
		"slack": {
			"webhook_url": "https://hooks.slack.com/T/B/X",
			"channel": "#alerts"
		},
		"pagerduty": {
			"routing_key": "pd-abc-123",
			"severity": "critical"
		},
		"http_client": {
			"request_timeout_seconds": 5
		}
	}`)
	cfg, err := ParseConfig(jsonData)
	if err != nil {
		t.Fatalf("ParseConfig: %v", err)
	}
	if cfg.UDSPath != "/tmp/foo.sock" {
		t.Errorf("UDSPath=%q", cfg.UDSPath)
	}
	if cfg.Slack == nil || cfg.Slack.WebhookURL != "https://hooks.slack.com/T/B/X" {
		t.Errorf("Slack: %+v", cfg.Slack)
	}
	if cfg.PagerDuty == nil || cfg.PagerDuty.RoutingKey != "pd-abc-123" {
		t.Errorf("PagerDuty: %+v", cfg.PagerDuty)
	}
	if cfg.HTTPClient == nil || cfg.HTTPClient.RequestTimeoutSeconds != 5 {
		t.Errorf("HTTPClient: %+v", cfg.HTTPClient)
	}
}

func TestSlackPayload_FleetEventShape(t *testing.T) {
	ev := StragglerEvent{
		Type: "straggler_state", NodeID: "gpu-01", ClusterID: "prod",
		Score: 0.42, Threshold: 0.6, DetectionMode: "fleet",
		DominantSignal: "throughput", Rank: 3, WorldSize: 8,
		EventID: "evt-abc",
	}
	p := slackPayload(ev, &SlackConfig{Channel: "#alerts"})
	text, _ := p["text"].(string)
	if !strings.Contains(text, "prod/gpu-01") {
		t.Errorf("text missing cluster/node: %q", text)
	}
	if !strings.Contains(text, "score=0.420") {
		t.Errorf("text missing score: %q", text)
	}
	if !strings.Contains(text, "rank=3/8") {
		t.Errorf("text missing rank/world_size: %q", text)
	}
	if p["channel"] != "#alerts" {
		t.Errorf("channel mismatch: %v", p["channel"])
	}
}

func TestPagerDutyPayload_DedupKeyAndSeverity(t *testing.T) {
	ev := StragglerEvent{
		Type: "straggler_state", NodeID: "gpu-01", ClusterID: "prod",
		Score: 0.42, Threshold: 0.6, EventID: "evt-uuid",
	}
	p := pagerdutyPayload(ev, &PagerDutyConfig{RoutingKey: "rk-1"})
	if p["routing_key"] != "rk-1" {
		t.Errorf("routing_key=%v", p["routing_key"])
	}
	if p["dedup_key"] != "evt-uuid" {
		t.Errorf("dedup_key=%v, want evt-uuid (event_id)", p["dedup_key"])
	}
	if p["event_action"] != "trigger" {
		t.Errorf("event_action=%v", p["event_action"])
	}
	pp := p["payload"].(map[string]any)
	if pp["severity"] != "warning" {
		t.Errorf("severity=%v, want default warning", pp["severity"])
	}

	// Override severity
	p2 := pagerdutyPayload(ev, &PagerDutyConfig{RoutingKey: "rk", Severity: "critical"})
	pp2 := p2["payload"].(map[string]any)
	if pp2["severity"] != "critical" {
		t.Errorf("override severity not respected: %v", pp2["severity"])
	}

	// Fallback dedup when EventID empty
	ev2 := StragglerEvent{NodeID: "gpu-X", TimestampNs: 1234567890}
	p3 := pagerdutyPayload(ev2, &PagerDutyConfig{RoutingKey: "rk"})
	if !strings.HasPrefix(p3["dedup_key"].(string), "gpu-X/") {
		t.Errorf("fallback dedup wrong: %v", p3["dedup_key"])
	}
}

func TestSlackBackend_PostsToWebhook(t *testing.T) {
	var seenBodies []string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		body, _ := io.ReadAll(r.Body)
		seenBodies = append(seenBodies, string(body))
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	s := NewSlack(&SlackConfig{WebhookURL: srv.URL}, time.Second, slog.New(slog.NewTextHandler(io.Discard, nil)))
	if err := s.Send(context.Background(), StragglerEvent{Type: "straggler_state", NodeID: "n1", EventID: "e1", Score: 0.5}); err != nil {
		t.Fatalf("Send: %v", err)
	}
	if len(seenBodies) != 1 {
		t.Fatalf("got %d bodies, want 1", len(seenBodies))
	}
	if !strings.Contains(seenBodies[0], `"text"`) {
		t.Errorf("body missing text key: %q", seenBodies[0])
	}
}

func TestSlackBackend_NonOKReturnsError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusBadRequest)
	}))
	defer srv.Close()
	s := NewSlack(&SlackConfig{WebhookURL: srv.URL}, time.Second, nil)
	if err := s.Send(context.Background(), StragglerEvent{Type: "straggler_state"}); err == nil {
		t.Fatal("expected error on 400")
	}
}

func TestPagerDutyBackend_Posts202OK(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusAccepted)
	}))
	defer srv.Close()
	pd := NewPagerDuty(&PagerDutyConfig{RoutingKey: "rk"}, time.Second, nil)
	pd.url = srv.URL
	if err := pd.Send(context.Background(), StragglerEvent{Type: "straggler_state", NodeID: "n", EventID: "e"}); err != nil {
		t.Fatalf("Send: %v", err)
	}
}

func TestRun_IntegrationHappyPath(t *testing.T) {
	// Stand up a fake UDS server that emits two NDJSON events,
	// then a fake Slack webhook, and assert both events arrive.
	tmpSock := filepath.Join(t.TempDir(), "test.sock")
	listener, err := net.Listen("unix", tmpSock)
	if err != nil {
		t.Fatalf("listen: %v", err)
	}
	defer listener.Close()

	var slackHits int
	var slackMu sync.Mutex
	slackSrv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		slackMu.Lock()
		slackHits++
		slackMu.Unlock()
		w.WriteHeader(http.StatusOK)
	}))
	defer slackSrv.Close()

	go func() {
		conn, err := listener.Accept()
		if err != nil {
			return
		}
		defer conn.Close()
		bw := bufio.NewWriter(conn)
		for _, ev := range []StragglerEvent{
			{Type: "straggler_state", NodeID: "n1", EventID: "e1", Score: 0.4},
			{Type: "straggler_state", NodeID: "n2", EventID: "e2", Score: 0.45},
		} {
			b, _ := json.Marshal(ev)
			bw.Write(b)
			bw.WriteByte('\n')
			bw.Flush()
		}
		// Hold the conn open briefly so the alerter has time to read.
		time.Sleep(100 * time.Millisecond)
	}()

	cfg := &Config{
		UDSPath: tmpSock,
		Slack:   &SlackConfig{WebhookURL: slackSrv.URL},
	}
	ctx, cancel := context.WithTimeout(context.Background(), 500*time.Millisecond)
	defer cancel()
	_ = Run(ctx, cfg, slog.New(slog.NewTextHandler(io.Discard, nil)))

	// Allow the parallel fanout goroutines to complete.
	deadline := time.Now().Add(500 * time.Millisecond)
	for time.Now().Before(deadline) {
		slackMu.Lock()
		hits := slackHits
		slackMu.Unlock()
		if hits >= 2 {
			break
		}
		time.Sleep(10 * time.Millisecond)
	}
	slackMu.Lock()
	defer slackMu.Unlock()
	if slackHits < 2 {
		t.Fatalf("slack hits=%d, want >= 2", slackHits)
	}
}
