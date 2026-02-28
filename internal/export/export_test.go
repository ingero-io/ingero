package export

import (
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/ingero-io/ingero/internal/stats"
	"github.com/ingero-io/ingero/pkg/events"
)

// ---------------------------------------------------------------------------
// Prometheus tests
// ---------------------------------------------------------------------------

func TestNewPrometheus_NilOnEmpty(t *testing.T) {
	p := NewPrometheus("")
	if p != nil {
		t.Error("NewPrometheus('') should return nil")
	}
}

func TestNewPrometheus_NonNil(t *testing.T) {
	p := NewPrometheus(":9090")
	if p == nil {
		t.Error("NewPrometheus(':9090') should return non-nil")
	}
}

func TestPrometheusUpdateSnapshot_NilSafe(t *testing.T) {
	var p *PrometheusServer
	// Should not panic.
	p.UpdateSnapshot(&stats.Snapshot{})
}

func TestPrometheusMetrics_NoData(t *testing.T) {
	p := NewPrometheus(":0")

	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	p.handleMetrics(w, req)

	resp := w.Result()
	body, _ := io.ReadAll(resp.Body)

	if resp.StatusCode != 200 {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}
	if !strings.Contains(string(body), "No data available") {
		t.Error("expected 'No data available' message")
	}
}

func TestPrometheusMetrics_WithSnapshot(t *testing.T) {
	p := NewPrometheus(":0")

	snap := &stats.Snapshot{
		TotalEvents:   1000,
		AnomalyEvents: 5,
		WallClock:     time.Minute,
		System: &stats.SystemSnapshot{
			CPUPercent: 47.2,
			MemUsedPct: 72.0,
			MemAvailMB: 11200,
			LoadAvg1:   3.2,
		},
		Ops: []stats.OpStats{
			{
				Op:           "cudaLaunchKernel",
				Source:       events.SourceCUDA,
				Count:        500,
				P50:          14 * time.Microsecond,
				P95:          45 * time.Microsecond,
				P99:          312 * time.Microsecond,
				TimeFraction: 0.24,
			},
			{
				Op:           "cuLaunchKernel",
				Source:       events.SourceDriver,
				Count:        300,
				P50:          10 * time.Microsecond,
				P95:          30 * time.Microsecond,
				P99:          200 * time.Microsecond,
				TimeFraction: 0.15,
			},
			{
				Op:           "sched_switch",
				Source:       events.SourceHost,
				Count:        200,
				P50:          2 * time.Millisecond,
				P95:          8 * time.Millisecond,
				P99:          15 * time.Millisecond,
				TimeFraction: 0.12,
			},
		},
	}
	p.UpdateSnapshot(snap)

	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	p.handleMetrics(w, req)

	resp := w.Result()
	body, _ := io.ReadAll(resp.Body)
	content := string(body)

	// Check content type.
	ct := resp.Header.Get("Content-Type")
	if !strings.Contains(ct, "text/plain") {
		t.Errorf("expected text/plain content type, got %q", ct)
	}

	// Check system metrics.
	checks := []string{
		"system_cpu_utilization",
		"system_memory_utilization",
		"system_memory_usage_available",
		"system_cpu_load_average_1m",
		"gpu_cuda_operation_duration_microseconds",
		"gpu_cuda_operation_count",
		"ingero_anomaly_count",
		`operation="cudaLaunchKernel"`,
		`operation="cuLaunchKernel"`,
		`operation="sched_switch"`,
		`source="cuda"`,
		`source="driver"`,
		`source="host"`,
		`percentile="p50"`,
		`percentile="p95"`,
		`percentile="p99"`,
	}

	for _, check := range checks {
		if !strings.Contains(content, check) {
			t.Errorf("metrics output missing %q", check)
		}
	}
}

func TestPrometheusMetrics_HTTPHandler(t *testing.T) {
	p := NewPrometheus(":0")
	snap := &stats.Snapshot{TotalEvents: 42, AnomalyEvents: 1}
	p.UpdateSnapshot(snap)

	// Create a real HTTP test server.
	mux := http.NewServeMux()
	mux.HandleFunc("/metrics", p.handleMetrics)
	srv := httptest.NewServer(mux)
	defer srv.Close()

	resp, err := http.Get(srv.URL + "/metrics")
	if err != nil {
		t.Fatalf("GET /metrics: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != 200 {
		t.Errorf("expected 200, got %d", resp.StatusCode)
	}

	body, _ := io.ReadAll(resp.Body)
	if !strings.Contains(string(body), "ingero_anomaly_count 1") {
		t.Error("expected anomaly count of 1")
	}
}

// ---------------------------------------------------------------------------
// OTLP tests
// ---------------------------------------------------------------------------

func TestNewOTLP_NilOnEmpty(t *testing.T) {
	e := NewOTLP(OTLPConfig{})
	if e != nil {
		t.Error("NewOTLP with empty endpoint should return nil")
	}
}

func TestNewOTLP_NonNil(t *testing.T) {
	e := NewOTLP(OTLPConfig{Endpoint: "localhost:4318"})
	if e == nil {
		t.Error("NewOTLP with endpoint should return non-nil")
	}
}

func TestNewOTLP_DefaultInterval(t *testing.T) {
	e := NewOTLP(OTLPConfig{Endpoint: "localhost:4318"})
	if e.config.ExportInterval != 10 {
		t.Errorf("expected default interval 10, got %d", e.config.ExportInterval)
	}
}

func TestNewOTLP_DefaultProtocol(t *testing.T) {
	e := NewOTLP(OTLPConfig{Endpoint: "localhost:4318"})
	if e.config.Protocol != "http" {
		t.Errorf("expected default protocol 'http', got %q", e.config.Protocol)
	}
}

func TestOTLP_Interval(t *testing.T) {
	e := NewOTLP(OTLPConfig{Endpoint: "localhost:4318", ExportInterval: 15})
	if e.Interval() != 15 {
		t.Errorf("expected interval 15, got %d", e.Interval())
	}

	var nilE *OTLPExporter
	if nilE.Interval() != 10 {
		t.Errorf("nil exporter should return default interval 10, got %d", nilE.Interval())
	}
}

func TestOTLPPush_NilSafe(t *testing.T) {
	var e *OTLPExporter
	if err := e.Push(&stats.Snapshot{}); err != nil {
		t.Errorf("nil exporter Push should return nil, got %v", err)
	}
}

func TestOTLPStats_NilSafe(t *testing.T) {
	var e *OTLPExporter
	pushes, errors := e.Stats()
	if pushes != 0 || errors != 0 {
		t.Errorf("nil exporter Stats should return 0,0, got %d,%d", pushes, errors)
	}
}

func TestOTLP_MetricsURL(t *testing.T) {
	tests := []struct {
		endpoint string
		insecure bool
		want     string
	}{
		{"localhost:4318", true, "http://localhost:4318/v1/metrics"},
		{"localhost:4318", false, "https://localhost:4318/v1/metrics"},
		{"http://localhost:4318", false, "http://localhost:4318/v1/metrics"},
		{"https://otel.example.com", false, "https://otel.example.com/v1/metrics"},
		{"https://otel.example.com/", false, "https://otel.example.com/v1/metrics"},
	}

	for _, tt := range tests {
		e := NewOTLP(OTLPConfig{Endpoint: tt.endpoint, Insecure: tt.insecure})
		got := e.metricsURL()
		if got != tt.want {
			t.Errorf("metricsURL(%q, insecure=%v) = %q, want %q", tt.endpoint, tt.insecure, got, tt.want)
		}
	}
}

func TestOTLP_BuildPayload(t *testing.T) {
	e := NewOTLP(OTLPConfig{Endpoint: "localhost:4318"})

	snap := &stats.Snapshot{
		TotalEvents:   1000,
		AnomalyEvents: 5,
		WallClock:     time.Minute,
		System: &stats.SystemSnapshot{
			CPUPercent: 47.2,
			MemUsedPct: 72.0,
			MemAvailMB: 11200,
			LoadAvg1:   3.2,
		},
		Ops: []stats.OpStats{
			{
				Op:     "cudaLaunchKernel",
				Source: events.SourceCUDA,
				Count:  500,
				P50:    14 * time.Microsecond,
				P95:    45 * time.Microsecond,
				P99:    312 * time.Microsecond,
			},
		},
	}

	payload := e.buildMetricsPayload(snap)

	// Verify structure.
	if len(payload.ResourceMetrics) != 1 {
		t.Fatalf("expected 1 ResourceMetrics, got %d", len(payload.ResourceMetrics))
	}

	rm := payload.ResourceMetrics[0]

	// Check resource attributes.
	foundService := false
	for _, attr := range rm.Resource.Attributes {
		if attr.Key == "service.name" && attr.Value.StringValue != nil && *attr.Value.StringValue == "ingero" {
			foundService = true
		}
	}
	if !foundService {
		t.Error("missing service.name=ingero resource attribute")
	}

	// Check scope.
	if len(rm.ScopeMetrics) != 1 {
		t.Fatalf("expected 1 ScopeMetrics, got %d", len(rm.ScopeMetrics))
	}
	if rm.ScopeMetrics[0].Scope.Name != "ingero" {
		t.Errorf("scope name = %q, want %q", rm.ScopeMetrics[0].Scope.Name, "ingero")
	}

	metrics := rm.ScopeMetrics[0].Metrics

	// Should have: 4 system + 3 percentiles + 1 count + 1 anomaly = 9 metrics.
	if len(metrics) != 9 {
		t.Errorf("expected 9 metrics, got %d", len(metrics))
		for _, m := range metrics {
			t.Logf("  metric: %s", m.Name)
		}
	}

	// Verify the payload serializes to valid JSON.
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("json.Marshal failed: %v", err)
	}
	if len(body) == 0 {
		t.Error("empty JSON body")
	}

	// Check key metric names present in JSON.
	jsonStr := string(body)
	for _, name := range []string{
		"system.cpu.utilization",
		"system.memory.utilization",
		"gpu.cuda.operation.duration",
		"gpu.cuda.operation.count",
		"ingero.anomaly.count",
	} {
		if !strings.Contains(jsonStr, name) {
			t.Errorf("JSON missing metric %q", name)
		}
	}
}

func TestOTLP_PushToServer(t *testing.T) {
	// Create a test OTLP receiver.
	var receivedBody []byte
	var receivedContentType string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedContentType = r.Header.Get("Content-Type")
		receivedBody, _ = io.ReadAll(r.Body)
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	// Track debug messages.
	var debugMsgs []string
	debugFn := func(format string, args ...any) {
		debugMsgs = append(debugMsgs, "called")
		_ = format
		_ = args
	}

	e := NewOTLP(OTLPConfig{
		Endpoint: srv.URL,
		DebugLog: debugFn,
	})

	snap := &stats.Snapshot{
		TotalEvents:   100,
		AnomalyEvents: 2,
		System: &stats.SystemSnapshot{
			CPUPercent: 50.0,
			MemUsedPct: 60.0,
			MemAvailMB: 8000,
			LoadAvg1:   1.5,
		},
	}

	err := e.Push(snap)
	if err != nil {
		t.Fatalf("Push returned error: %v", err)
	}

	// Verify the request was received.
	if receivedContentType != "application/json" {
		t.Errorf("Content-Type = %q, want application/json", receivedContentType)
	}
	if len(receivedBody) == 0 {
		t.Error("received empty body")
	}

	// Verify it's valid OTLP JSON.
	var payload otlpPayload
	if err := json.Unmarshal(receivedBody, &payload); err != nil {
		t.Fatalf("received body is not valid OTLP JSON: %v", err)
	}
	if len(payload.ResourceMetrics) == 0 {
		t.Error("no ResourceMetrics in payload")
	}

	// Verify stats.
	pushes, errors := e.Stats()
	if pushes != 1 {
		t.Errorf("pushes = %d, want 1", pushes)
	}
	if errors != 0 {
		t.Errorf("errors = %d, want 0", errors)
	}

	// Verify debug was called.
	if len(debugMsgs) == 0 {
		t.Error("debug log function was never called")
	}
}

func TestOTLP_PushServerError(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusInternalServerError)
	}))
	defer srv.Close()

	e := NewOTLP(OTLPConfig{Endpoint: srv.URL})
	err := e.Push(&stats.Snapshot{AnomalyEvents: 0})
	if err == nil {
		t.Error("expected error on 500 response")
	}

	_, errors := e.Stats()
	if errors != 1 {
		t.Errorf("errors = %d, want 1", errors)
	}
}

func TestOTLP_PushConnectionRefused(t *testing.T) {
	e := NewOTLP(OTLPConfig{
		Endpoint: "http://localhost:1", // nothing listening
		Insecure: true,
	})
	err := e.Push(&stats.Snapshot{AnomalyEvents: 0})
	if err == nil {
		t.Error("expected error on connection refused")
	}

	_, errors := e.Stats()
	if errors != 1 {
		t.Errorf("errors = %d, want 1", errors)
	}
}

func TestOTLP_PushWithHeaders(t *testing.T) {
	var receivedAuth string
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedAuth = r.Header.Get("Authorization")
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	e := NewOTLP(OTLPConfig{
		Endpoint: srv.URL,
		Headers:  map[string]string{"Authorization": "Bearer test-token"},
	})

	err := e.Push(&stats.Snapshot{})
	if err != nil {
		t.Fatalf("Push error: %v", err)
	}
	if receivedAuth != "Bearer test-token" {
		t.Errorf("Authorization header = %q, want 'Bearer test-token'", receivedAuth)
	}
}
