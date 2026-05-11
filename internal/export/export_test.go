package export

import (
	"context"
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

// TestPrometheusMetrics_V014 covers the v0.14 + v0.12.10 metric families
// added on v0.14.1: throttle, libnccl process discovery, NVML-poll memory
// + fragmentation, and per-direction memcpy aggregates. Pre-fix the
// Prometheus exporter only emitted system_*, gpu_cuda_operation_*, anomaly,
// ringbuf, and trace_db_*; all v0.14-vintage metrics were OTLP-only.
func TestPrometheusMetrics_V014(t *testing.T) {
	p := NewPrometheus(":0")

	snap := &stats.Snapshot{
		ThrottleReadings: []stats.ThrottleReading{
			{UUID: "GPU-AAAA", PowerActive: true, ThermalActive: false, SWActive: false, HWActive: true},
			{UUID: "GPU-BBBB", PowerActive: false, ThermalActive: true, SWActive: false, HWActive: false},
		},
		NCCLProcessReadings: []stats.NCCLProcessReading{
			{PID: 1234, Comm: "python3", LibPath: "/usr/lib/x86_64-linux-gnu/libnccl.so.2.26.2", LibVersion: "2.26.2"},
		},
		MemFragReadings: []stats.MemFragReading{
			{UUID: "GPU-AAAA", UsedBytes: 8 << 30, FreeBytes: 16 << 30, TotalBytes: 24 << 30, FragmentationEstimate: 0.31},
		},
		MemFragProcessReadings: []stats.MemFragProcessReading{
			{UUID: "GPU-AAAA", PID: 1234, UsedBytes: 4 << 30},
		},
		MemcpyDirReadings: []stats.MemcpyDirStats{
			{
				Direction:  "h2d",
				BytesTotal: 1 << 30,
				DurationHistogram: stats.HistogramSnapshot{
					Count: 10, Sum: 14.0, Min: 0.5, Max: 5.0, HasObservation: true,
					BucketCounts:   []uint64{0, 0, 5, 0, 5, 0, 0, 0, 0, 0, 0, 0, 0, 0},
					ExplicitBounds: stats.DefaultMemcpyDurationBoundsMs,
				},
				EventsInWindow: 10,
			},
			{
				Direction:  "d2h",
				BytesTotal: 512 << 20,
				DurationHistogram: stats.HistogramSnapshot{
					Count: 8, Sum: 9.6, Min: 0.4, Max: 3.0, HasObservation: true,
					BucketCounts:   []uint64{0, 0, 4, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
					ExplicitBounds: stats.DefaultMemcpyDurationBoundsMs,
				},
				EventsInWindow: 8,
			},
		},
	}
	p.UpdateSnapshot(snap)

	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	p.handleMetrics(w, req)

	body, _ := io.ReadAll(w.Result().Body)
	content := string(body)

	checks := []string{
		// throttle (v0.12.10)
		`gpu_throttle_power_active{gpu_uuid="GPU-AAAA"} 1`,
		`gpu_throttle_thermal_active{gpu_uuid="GPU-AAAA"} 0`,
		`gpu_throttle_hw_active{gpu_uuid="GPU-AAAA"} 1`,
		`gpu_throttle_thermal_active{gpu_uuid="GPU-BBBB"} 1`,
		// libnccl discovery (v0.14 item A)
		`gpu_nccl_process_loaded{pid="1234",comm="python3"`,
		`libnccl_version="2.26.2"`,
		`gpu_nccl_processes_total 1`,
		// memfrag (v0.14 item D)
		`gpu_memory_used_bytes{gpu_uuid="GPU-AAAA"}`,
		`gpu_memory_free_bytes{gpu_uuid="GPU-AAAA"}`,
		`gpu_memory_total_bytes{gpu_uuid="GPU-AAAA"}`,
		`gpu_memory_fragmentation_estimate{gpu_uuid="GPU-AAAA"}`,
		`gpu_memory_process_allocated_bytes{gpu_uuid="GPU-AAAA",pid="1234"}`,
		// memcpy (v0.14 item C, v0.15 item C: histogram replaces gauge)
		`gpu_memcpy_bytes_total{direction="h2d"}`,
		`gpu_memcpy_bytes_total{direction="d2h"}`,
		`# TYPE gpu_memcpy_duration_ms histogram`,
		`gpu_memcpy_duration_ms_bucket{direction="h2d",le="+Inf"} 10`,
		`gpu_memcpy_duration_ms_count{direction="h2d"} 10`,
		`gpu_memcpy_duration_ms_count{direction="d2h"} 8`,
	}
	for _, check := range checks {
		if !strings.Contains(content, check) {
			t.Errorf("metrics output missing %q\n--- body ---\n%s", check, content)
		}
	}
}

// TestPrometheusMetrics_V014_EmptySnapshot verifies the v0.14 sections
// are silent when the relevant fields are nil/empty (no spurious zero
// rows polluting scrapes when pollers are disabled).
func TestPrometheusMetrics_V014_EmptySnapshot(t *testing.T) {
	p := NewPrometheus(":0")
	p.UpdateSnapshot(&stats.Snapshot{})

	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	p.handleMetrics(w, req)

	body, _ := io.ReadAll(w.Result().Body)
	content := string(body)

	for _, name := range []string{
		"gpu_throttle_power_active",
		"gpu_nccl_process_loaded",
		"gpu_memory_fragmentation_estimate",
		"gpu_memcpy_bytes_total",
	} {
		if strings.Contains(content, name) {
			t.Errorf("expected %q to be absent from empty-snapshot scrape, but it was emitted", name)
		}
	}
}

// v0.15 F2: NCCL collective running counters appear as
// gpu_nccl_collective_count + gpu_nccl_collective_bytes_total +
// gpu_nccl_collective_barrier_events on /metrics.
func TestPrometheusMetrics_NCCLCollectiveCounters(t *testing.T) {
	p := NewPrometheus(":0")
	p.UpdateSnapshot(&stats.Snapshot{
		NCCLCollectiveCounters: []stats.NCCLCollectiveCounter{
			{OpType: "ncclAllReduce", Count: 42, BytesTotal: 1234567},
			{OpType: "ncclAllReduce", BarrierEvents: 7},
			{OpType: "ncclBcast", Count: 3, BytesTotal: 100},
		},
	})

	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	p.handleMetrics(w, req)
	body, _ := io.ReadAll(w.Result().Body)
	content := string(body)
	for _, want := range []string{
		`gpu_nccl_collective_count{op_type="ncclAllReduce"} 42`,
		`gpu_nccl_collective_bytes_total{op_type="ncclAllReduce"} 1234567`,
		`gpu_nccl_collective_count{op_type="ncclBcast"} 3`,
		`gpu_nccl_collective_barrier_events{op_type="ncclAllReduce"} 7`,
	} {
		if !strings.Contains(content, want) {
			t.Errorf("missing %q in /metrics output\n--- body ---\n%s", want, content)
		}
	}
}

// v0.15 F2: with no NCCL counter rows the section stays silent.
func TestPrometheusMetrics_NCCLCollectiveCounters_EmptyStaysSilent(t *testing.T) {
	p := NewPrometheus(":0")
	p.UpdateSnapshot(&stats.Snapshot{})

	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	p.handleMetrics(w, req)
	body, _ := io.ReadAll(w.Result().Body)
	if strings.Contains(string(body), "gpu_nccl_collective_") {
		t.Errorf("expected no nccl_collective_* lines on empty snapshot, got:\n%s", string(body))
	}
}

// v0.15 F2 (LOW): row ordering across scrapes is deterministic.
// snapshotNCCLCollectiveCounters returns rows in Go map iteration
// order; the exporter sorts by (op_type, kind) for stable output
// so downstream tools that diff /metrics get reproducible results.
func TestPrometheusMetrics_NCCLCollectiveCounters_DeterministicOrdering(t *testing.T) {
	p := NewPrometheus(":0")
	// Ten unsorted op_types; same Snapshot should produce identical
	// output across multiple scrapes regardless of input order.
	counters := []stats.NCCLCollectiveCounter{
		{OpType: "ncclSend", Count: 1},
		{OpType: "ncclAllReduce", Count: 100, BytesTotal: 1000},
		{OpType: "ncclAllGather", Count: 50},
		{OpType: "ncclBcast", Count: 25},
		{OpType: "ncclReduceScatter", Count: 10},
		{OpType: "ncclRecv", Count: 1},
		{OpType: "ncclAllReduce", BarrierEvents: 5},
	}
	p.UpdateSnapshot(&stats.Snapshot{NCCLCollectiveCounters: counters})

	doScrape := func() string {
		req := httptest.NewRequest("GET", "/metrics", nil)
		w := httptest.NewRecorder()
		p.handleMetrics(w, req)
		body, _ := io.ReadAll(w.Result().Body)
		return string(body)
	}
	first := doScrape()
	for i := 0; i < 5; i++ {
		if got := doScrape(); got != first {
			t.Fatalf("scrape %d differs from scrape 0\n--- scrape 0 ---\n%s\n--- scrape %d ---\n%s",
				i+1, first, i+1, got)
		}
	}

	// Also assert the count rows are alphabetically sorted by op_type.
	// Easier check: ncclAllGather appears before ncclAllReduce, both
	// before ncclBcast, etc.
	idx := func(s string) int { return strings.Index(first, s) }
	for _, pair := range [][2]string{
		{`gpu_nccl_collective_count{op_type="ncclAllGather"}`, `gpu_nccl_collective_count{op_type="ncclAllReduce"}`},
		{`gpu_nccl_collective_count{op_type="ncclAllReduce"}`, `gpu_nccl_collective_count{op_type="ncclBcast"}`},
		{`gpu_nccl_collective_count{op_type="ncclBcast"}`, `gpu_nccl_collective_count{op_type="ncclReduceScatter"}`},
	} {
		i, j := idx(pair[0]), idx(pair[1])
		if i < 0 || j < 0 {
			t.Errorf("missing row pair %q before %q in:\n%s", pair[0], pair[1], first)
			continue
		}
		if i >= j {
			t.Errorf("ordering wrong: %q at %d not before %q at %d", pair[0], i, pair[1], j)
		}
	}
}

// v0.15 F2 (LOW): op_type label values containing characters that
// would otherwise need Prometheus escaping are emitted via %q so
// they pass through correctly. Defensive coverage for the case where
// an upstream feed delivers an unexpected op_type string.
func TestPrometheusMetrics_NCCLCollectiveCounters_EscapesLabelValues(t *testing.T) {
	p := NewPrometheus(":0")
	p.UpdateSnapshot(&stats.Snapshot{
		NCCLCollectiveCounters: []stats.NCCLCollectiveCounter{
			{OpType: `weird"op\name`, Count: 1},
		},
	})
	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	p.handleMetrics(w, req)
	body, _ := io.ReadAll(w.Result().Body)
	content := string(body)
	// %q escapes both " and \ ; verify the emitted line is parseable.
	want := `gpu_nccl_collective_count{op_type="weird\"op\\name"} 1`
	if !strings.Contains(content, want) {
		t.Errorf("expected %q in output, got:\n%s", want, content)
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
	if err := e.Push(context.Background(), &stats.Snapshot{}); err != nil {
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

// TestOTLP_NCCLDataPoints asserts that buildMetricsPayload emits one
// nccl.collective.duration_ms gauge + one nccl.collective.bytes gauge
// per stats.NCCLDataPoint, with the v0.12.0 contract attribute set.
// LHF #1 from the v0.12.0 audit: without these data points, Fleet's
// ncclprocessor has nothing to consume.
func TestOTLP_NCCLDataPoints(t *testing.T) {
	e := NewOTLP(OTLPConfig{Endpoint: "localhost:4318"})

	snap := &stats.Snapshot{
		WallClock: time.Minute,
		NCCLDataPoints: []stats.NCCLDataPoint{
			{
				TimestampUnixNano: 1700000000000000000,
				OpType:            "ncclAllReduce",
				CommIDHash:        "deadbeefcafebabe",
				Rank:              2,
				NRanks:            8,
				Datatype:          7,
				ReduceOp:          0,
				DurationMs:        4.2,
				CountBytes:        1048576,
				ReturnCode:        0,
			},
			{
				TimestampUnixNano: 1700000000000000000,
				OpType:            "ncclAllGather",
				CommIDHash:        "1234567890abcdef",
				Rank:              0,
				NRanks:            4,
				Datatype:          0,
				ReduceOp:          0,
				DurationMs:        1.5,
				CountBytes:        524288,
				ReturnCode:        0,
			},
		},
	}

	payload := e.buildMetricsPayload(snap)
	body, err := json.Marshal(payload)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	jsonStr := string(body)

	// Two NCCL data points × 2 metrics each = 4 emissions.
	for _, want := range []string{
		"nccl.collective.duration_ms",
		"nccl.collective.bytes",
		"nccl.op_type", "ncclAllReduce", "ncclAllGather",
		"nccl.comm_id_hash", "deadbeefcafebabe", "1234567890abcdef",
		"nccl.rank", "nccl.nranks",
	} {
		if !strings.Contains(jsonStr, want) {
			t.Errorf("OTLP NCCL payload missing %q", want)
		}
	}

	// Count occurrences of each metric name (one per data point).
	if got := strings.Count(jsonStr, `"name":"nccl.collective.duration_ms"`); got != 2 {
		t.Errorf("expected 2 nccl.collective.duration_ms metrics, got %d", got)
	}
	if got := strings.Count(jsonStr, `"name":"nccl.collective.bytes"`); got != 2 {
		t.Errorf("expected 2 nccl.collective.bytes metrics, got %d", got)
	}
}

// TestOTLP_NCCLDataPoints_BarrierWaitMetricName covers v0.12.1
// (QA audit ★5 #4): a NCCLDataPoint with IsBarrier=true must emit
// metric name `nccl.collective.barrier_wait_ms` (NOT duration_ms or
// bytes), and the OpType attribute must be the bare op name without
// the legacy "barrier_wait:" prefix scheme.
func TestOTLP_NCCLDataPoints_BarrierWaitMetricName(t *testing.T) {
	e := NewOTLP(OTLPConfig{Endpoint: "localhost:4318"})
	snap := &stats.Snapshot{
		NCCLDataPoints: []stats.NCCLDataPoint{
			{
				TimestampUnixNano: 1700000000000000000,
				OpType:            "ncclAllReduce",
				CommIDHash:        "deadbeefcafebabe",
				Rank:              0,
				NRanks:             4,
				DurationMs:         2.5,
				IsBarrier:          true,
			},
		},
	}
	body, _ := json.Marshal(e.buildMetricsPayload(snap))
	jsonStr := string(body)

	if !strings.Contains(jsonStr, `"name":"nccl.collective.barrier_wait_ms"`) {
		t.Errorf("expected nccl.collective.barrier_wait_ms metric, got: %s", jsonStr)
	}
	if strings.Contains(jsonStr, `"name":"nccl.collective.duration_ms"`) {
		t.Errorf("barrier wait must NOT emit duration_ms; got: %s", jsonStr)
	}
	if strings.Contains(jsonStr, `"name":"nccl.collective.bytes"`) {
		t.Errorf("barrier wait must NOT emit bytes; got: %s", jsonStr)
	}
	// op_type attribute is the bare name (no prefix).
	if strings.Contains(jsonStr, "barrier_wait:ncclAllReduce") {
		t.Errorf("legacy stringly-typed prefix leaked: %s", jsonStr)
	}
	if !strings.Contains(jsonStr, `"stringValue":"ncclAllReduce"`) {
		t.Errorf("op_type attr expected 'ncclAllReduce', not found: %s", jsonStr)
	}
}

// TestOTLP_NCCLDataPoints_EmptyWhenNotProvided asserts that NCCL data
// points are not emitted when the snapshot's NCCLDataPoints field is
// nil or empty (e.g. agent ran without --nccl).
func TestOTLP_NCCLDataPoints_EmptyWhenNotProvided(t *testing.T) {
	e := NewOTLP(OTLPConfig{Endpoint: "localhost:4318"})
	for _, snap := range []*stats.Snapshot{
		{},
		{NCCLDataPoints: nil},
		{NCCLDataPoints: []stats.NCCLDataPoint{}},
	} {
		payload := e.buildMetricsPayload(snap)
		body, _ := json.Marshal(payload)
		if strings.Contains(string(body), "nccl.collective") {
			t.Errorf("nccl.collective.* present in payload with no NCCL data points")
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

	err := e.Push(context.Background(), snap)
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
	err := e.Push(context.Background(), &stats.Snapshot{AnomalyEvents: 0})
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
	err := e.Push(context.Background(), &stats.Snapshot{AnomalyEvents: 0})
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

	err := e.Push(context.Background(), &stats.Snapshot{})
	if err != nil {
		t.Fatalf("Push error: %v", err)
	}
	if receivedAuth != "Bearer test-token" {
		t.Errorf("Authorization header = %q, want 'Bearer test-token'", receivedAuth)
	}
}

// TestOTLP_ThrottleReadings_FourMetricsPerGPU asserts the v0.12.10
// W2-poller contract: each ThrottleReading produces exactly four gauge
// metric entries (power, thermal, sw, hw) labelled with gpu.uuid. The
// values are 1 when the bucket is active and 0 when it is not.
func TestOTLP_ThrottleReadings_FourMetricsPerGPU(t *testing.T) {
	e := NewOTLP(OTLPConfig{Endpoint: "localhost:4318"})

	snap := &stats.Snapshot{
		ThrottleReadings: []stats.ThrottleReading{
			{
				UUID:        "GPU-aaaaaaaa",
				Bitmask:     0x4,
				PowerActive: true,
				SWActive:    true,
			},
			{
				UUID:          "GPU-bbbbbbbb",
				Bitmask:       0x40,
				ThermalActive: true,
				HWActive:      true,
			},
		},
	}

	body, err := json.Marshal(e.buildMetricsPayload(snap))
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	jsonStr := string(body)

	// One entry per metric per GPU: 4 metrics x 2 GPUs = 8 series.
	for _, name := range []string{
		`"name":"gpu.throttle.power_active"`,
		`"name":"gpu.throttle.thermal_active"`,
		`"name":"gpu.throttle.sw_active"`,
		`"name":"gpu.throttle.hw_active"`,
	} {
		got := strings.Count(jsonStr, name)
		if got != 2 {
			t.Errorf("%s: expected 2 emissions (one per GPU), got %d", name, got)
		}
	}

	// Both UUIDs must appear in the payload as gpu.uuid attributes.
	for _, uuid := range []string{"GPU-aaaaaaaa", "GPU-bbbbbbbb"} {
		if !strings.Contains(jsonStr, uuid) {
			t.Errorf("expected gpu.uuid %q in payload, missing from %s", uuid, jsonStr)
		}
	}
	if !strings.Contains(jsonStr, `"key":"gpu.uuid"`) {
		t.Errorf("expected gpu.uuid attribute key in payload: %s", jsonStr)
	}
}

// TestOTLP_ThrottleReadings_EmptyWhenNotProvided asserts that no
// gpu.throttle.* series leak into the payload when the snapshot has no
// readings (e.g. no nvidia-smi on the host, or the poller is disabled).
func TestOTLP_ThrottleReadings_EmptyWhenNotProvided(t *testing.T) {
	e := NewOTLP(OTLPConfig{Endpoint: "localhost:4318"})
	for _, snap := range []*stats.Snapshot{
		{},
		{ThrottleReadings: nil},
		{ThrottleReadings: []stats.ThrottleReading{}},
	} {
		body, _ := json.Marshal(e.buildMetricsPayload(snap))
		if strings.Contains(string(body), "gpu.throttle.") {
			t.Errorf("gpu.throttle.* present in payload with no readings")
		}
	}
}

// TestOTLP_ThrottleReadings_BucketValuesAreOneOrZero asserts the gauge
// value semantics: an active bucket emits 1.0, an inactive bucket emits 0.0.
// Dashboards bind to this contract.
func TestOTLP_ThrottleReadings_BucketValuesAreOneOrZero(t *testing.T) {
	e := NewOTLP(OTLPConfig{Endpoint: "localhost:4318"})
	snap := &stats.Snapshot{
		ThrottleReadings: []stats.ThrottleReading{
			{
				UUID:          "GPU-x",
				PowerActive:   true,
				ThermalActive: false,
				SWActive:      true,
				HWActive:      false,
			},
		},
	}

	payload := e.buildMetricsPayload(snap)
	body, _ := json.Marshal(payload)
	jsonStr := string(body)

	// Walk the metrics slice and check each gauge data point's asDouble.
	// We can't easily index into the JSON, so verify by metric name.
	got := make(map[string]float64)
	for _, rm := range payload.ResourceMetrics {
		for _, sm := range rm.ScopeMetrics {
			for _, m := range sm.Metrics {
				if !strings.HasPrefix(m.Name, "gpu.throttle.") {
					continue
				}
				if m.Gauge == nil || len(m.Gauge.DataPoints) == 0 {
					t.Fatalf("metric %s missing gauge data points", m.Name)
				}
				if m.Gauge.DataPoints[0].AsDouble == nil {
					t.Fatalf("metric %s missing asDouble", m.Name)
				}
				got[m.Name] = *m.Gauge.DataPoints[0].AsDouble
			}
		}
	}

	want := map[string]float64{
		"gpu.throttle.power_active":   1,
		"gpu.throttle.thermal_active": 0,
		"gpu.throttle.sw_active":      1,
		"gpu.throttle.hw_active":      0,
	}
	for name, v := range want {
		if got[name] != v {
			t.Errorf("%s = %v, want %v (payload: %s)", name, got[name], v, jsonStr)
		}
	}
}

// ---------------------------------------------------------------------------
// v0.15 item B: OTLP Histogram + ExponentialHistogram encoder tests
// ---------------------------------------------------------------------------

func TestHistogramMetric_RoundTripsJSON(t *testing.T) {
	h := stats.NewHistogram([]float64{1, 10, 100})
	for _, v := range []float64{0.5, 5, 50, 500} {
		h.Observe(v)
	}
	snap := h.Snapshot()
	m := histogramMetric("gpu.memcpy.duration_ms", "memcpy duration", "ms", "1700000000000000000", "1690000000000000000", snap, nil)

	raw, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(raw), `"histogram"`) {
		t.Errorf("payload missing histogram key: %s", raw)
	}
	if !strings.Contains(string(raw), `"count":"4"`) {
		t.Errorf("payload missing count=4 (string-encoded): %s", raw)
	}
	if !strings.Contains(string(raw), `"bucketCounts":["1","1","1","1"]`) {
		t.Errorf("payload missing per-bucket counts: %s", raw)
	}
	if !strings.Contains(string(raw), `"explicitBounds":[1,10,100]`) {
		t.Errorf("payload missing explicit bounds: %s", raw)
	}
	// Round-trip back to confirm shape parses.
	var decoded otlpMetric
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if decoded.Histogram == nil || len(decoded.Histogram.DataPoints) != 1 {
		t.Errorf("decoded shape lost: %+v", decoded)
	}
	dp := decoded.Histogram.DataPoints[0]
	if dp.Count != "4" {
		t.Errorf("decoded count=%q want 4", dp.Count)
	}
	if dp.Sum == nil || *dp.Sum != 0.5+5+50+500 {
		t.Errorf("decoded sum=%v want 555.5", dp.Sum)
	}
	if dp.Min == nil || *dp.Min != 0.5 {
		t.Errorf("decoded min=%v want 0.5", dp.Min)
	}
	if dp.Max == nil || *dp.Max != 500 {
		t.Errorf("decoded max=%v want 500", dp.Max)
	}
	if decoded.Histogram.AggregationTemporality != 2 {
		t.Errorf("aggregationTemporality=%d want 2 (cumulative)", decoded.Histogram.AggregationTemporality)
	}
}

func TestHistogramMetric_EmptySnapshotEmitsZero(t *testing.T) {
	h := stats.NewHistogram([]float64{1, 10})
	snap := h.Snapshot()
	m := histogramMetric("gpu.memcpy.duration_ms", "", "ms", "1700000000000000000", "", snap, nil)
	raw, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	// count=0, no sum, no min, no max
	if !strings.Contains(string(raw), `"count":"0"`) {
		t.Errorf("empty: payload missing count=0: %s", raw)
	}
	if strings.Contains(string(raw), `"min"`) || strings.Contains(string(raw), `"max"`) {
		t.Errorf("empty: should not emit min/max: %s", raw)
	}
	if strings.Contains(string(raw), `"sum"`) {
		t.Errorf("empty: should not emit sum: %s", raw)
	}
}

func TestExponentialHistogramMetric_RoundTripsJSON(t *testing.T) {
	m := exponentialHistogramMetric(
		"gpu.memcpy.duration_ms",
		"memcpy duration",
		"ms",
		"1700000000000000000",
		"1690000000000000000",
		100,            // count
		1234.5,         // sum
		4,              // scale
		0,              // zero count
		0,              // pos offset
		[]uint64{10, 20, 30, 40}, // pos buckets
		0, nil,         // no negative
		0.5, 200.0, true, // min/max
		nil,            // no attrs
	)
	raw, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(raw), `"exponentialHistogram"`) {
		t.Errorf("payload missing exponentialHistogram key: %s", raw)
	}
	if !strings.Contains(string(raw), `"scale":4`) {
		t.Errorf("payload missing scale=4: %s", raw)
	}
	if !strings.Contains(string(raw), `"zeroCount":"0"`) {
		t.Errorf("payload missing zeroCount: %s", raw)
	}
	if !strings.Contains(string(raw), `"bucketCounts":["10","20","30","40"]`) {
		t.Errorf("payload missing pos bucketCounts: %s", raw)
	}
	if strings.Contains(string(raw), `"negative"`) {
		t.Errorf("payload should NOT include negative buckets when none provided: %s", raw)
	}
	var decoded otlpMetric
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	if decoded.ExponentialHistogram == nil {
		t.Fatalf("decoded ExponentialHistogram nil")
	}
	dp := decoded.ExponentialHistogram.DataPoints[0]
	if dp.Count != "100" || dp.Scale != 4 {
		t.Errorf("decoded count=%q scale=%d want 100/4", dp.Count, dp.Scale)
	}
	if dp.Positive == nil || len(dp.Positive.BucketCounts) != 4 {
		t.Errorf("decoded positive missing or wrong shape")
	}
}

// v0.15 item B: ExponentialHistogram with both positive AND
// negative buckets. Real-world data with both signs (e.g., a
// signed delta metric) needs the negative branch on the wire.
func TestExponentialHistogramMetric_WithNegativeBuckets(t *testing.T) {
	m := exponentialHistogramMetric(
		"signed.delta", "", "", "1700000000000000000", "",
		200,    // count
		-50.0,  // sum (negative)
		2,      // scale
		5,      // zero count
		0, []uint64{30, 40},  // pos buckets
		1, []uint64{60, 65},  // neg buckets
		-100, 100, true,      // min/max
		nil,
	)
	raw, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(raw), `"negative"`) {
		t.Errorf("payload should include negative bucket section: %s", raw)
	}
	if !strings.Contains(string(raw), `"positive"`) {
		t.Errorf("payload should include positive bucket section: %s", raw)
	}
	var decoded otlpMetric
	if err := json.Unmarshal(raw, &decoded); err != nil {
		t.Fatalf("unmarshal: %v", err)
	}
	dp := decoded.ExponentialHistogram.DataPoints[0]
	if dp.Negative == nil || len(dp.Negative.BucketCounts) != 2 {
		t.Errorf("decoded negative shape lost: %+v", dp.Negative)
	}
	if dp.Negative.Offset != 1 {
		t.Errorf("decoded negative offset = %d, want 1", dp.Negative.Offset)
	}
	if dp.Sum == nil || *dp.Sum != -50.0 {
		t.Errorf("decoded sum = %v, want -50.0", dp.Sum)
	}
}

// v0.15 item B: zeroCount > 0 with no observations elsewhere is a
// valid OTLP shape (a histogram of values that all rounded to zero).
func TestExponentialHistogramMetric_ZeroCountOnly(t *testing.T) {
	m := exponentialHistogramMetric(
		"only.zero", "", "", "1700000000000000000", "",
		7,      // count
		0,      // sum
		2,      // scale
		7,      // zero count = full count
		0, nil, // no positive
		0, nil, // no negative
		0, 0, false,
		nil,
	)
	raw, err := json.Marshal(m)
	if err != nil {
		t.Fatalf("marshal: %v", err)
	}
	if !strings.Contains(string(raw), `"zeroCount":"7"`) {
		t.Errorf("payload should carry zeroCount=7: %s", raw)
	}
	if strings.Contains(string(raw), `"positive"`) || strings.Contains(string(raw), `"negative"`) {
		t.Errorf("payload should NOT include positive/negative sections when both empty: %s", raw)
	}
}

func TestUintToStr(t *testing.T) {
	if got := uintToStr(0); got != "0" {
		t.Errorf("uintToStr(0)=%q want 0", got)
	}
	if got := uintToStr(18446744073709551615); got != "18446744073709551615" {
		t.Errorf("uintToStr(uint64-max)=%q (no precision loss)", got)
	}
}

// v0.15 item L: throttle event-edge counters render in /metrics.
func TestPrometheus_ThrottleEventCounters(t *testing.T) {
	p := NewPrometheus(":0")
	p.UpdateSnapshot(&stats.Snapshot{
		ThrottleEvents: stats.ThrottleEventCounters{
			PowerEvents:   3,
			ThermalEvents: 1,
			SWEvents:      0,
			HWEvents:      2,
		},
	})
	body := scrapeMetrics(t, p)
	for _, want := range []string{
		"# TYPE gpu_throttle_power_event_total counter",
		"gpu_throttle_power_event_total 3",
		"gpu_throttle_thermal_event_total 1",
		"gpu_throttle_sw_event_total 0",
		"gpu_throttle_hw_event_total 2",
	} {
		if !strings.Contains(body, want) {
			t.Errorf("missing %q in scrape body", want)
		}
	}
}

// All-zero counters must NOT emit (silence is the signal that the
// agent isn't running with the experimental flag, vs zero events).
func TestPrometheus_ThrottleEventCounters_SilentWhenAllZero(t *testing.T) {
	p := NewPrometheus(":0")
	p.UpdateSnapshot(&stats.Snapshot{ThrottleEvents: stats.ThrottleEventCounters{}})
	body := scrapeMetrics(t, p)
	if strings.Contains(body, "gpu_throttle_power_event_total") {
		t.Errorf("zero counters should stay silent; body included the metric")
	}
}

// v0.15 item K: per-cmd memfrag IOCTL counters render with hex cmd.
func TestPrometheus_MemfragIOCTLCounters(t *testing.T) {
	p := NewPrometheus(":0")
	p.UpdateSnapshot(&stats.Snapshot{
		MemfragIOCTLCounters: []stats.MemfragIOCTLCounter{
			{Cmd: 0xC0184601, Count: 100},
			{Cmd: 0xC0184602, Count: 50},
		},
	})
	body := scrapeMetrics(t, p)
	for _, want := range []string{
		"# TYPE gpu_memfrag_ioctl_event_total counter",
		`gpu_memfrag_ioctl_event_total{cmd="0xC0184601"} 100`,
		`gpu_memfrag_ioctl_event_total{cmd="0xC0184602"} 50`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("missing %q in scrape body\n--- body ---\n%s", want, body)
		}
	}
}

// v0.15 item M: per-PID kernel-launch count + histograms render.
func TestPrometheus_KernelLaunch(t *testing.T) {
	p := NewPrometheus(":0")
	tpb := stats.NewHistogram([]float64{32, 64, 128, 256, 512, 1024})
	tpb.Observe(256)
	tpb.Observe(512)
	gb := stats.NewHistogram([]float64{1, 4, 16, 64, 256, 1024})
	gb.Observe(64)
	gb.Observe(128)
	p.UpdateSnapshot(&stats.Snapshot{
		KernelLaunches: []stats.KernelLaunchSnapshot{
			{
				PID:                 100,
				Count:               2,
				ThreadsPerBlockHist: tpb.Snapshot(),
				GridBlocksHist:      gb.Snapshot(),
			},
		},
	})
	body := scrapeMetrics(t, p)
	for _, want := range []string{
		"# TYPE gpu_kernel_launch_count counter",
		`gpu_kernel_launch_count{pid="100"} 2`,
		"# TYPE gpu_kernel_launch_threads_per_block histogram",
		`gpu_kernel_launch_threads_per_block_count{pid="100"} 2`,
		`gpu_kernel_launch_threads_per_block_sum{pid="100"} 768`,
		"# TYPE gpu_kernel_launch_grid_blocks histogram",
		`gpu_kernel_launch_grid_blocks_count{pid="100"} 2`,
	} {
		if !strings.Contains(body, want) {
			t.Errorf("missing %q in scrape body\n--- body ---\n%s", want, body)
		}
	}
}

func TestPrometheus_KernelLaunch_SilentWhenEmpty(t *testing.T) {
	p := NewPrometheus(":0")
	p.UpdateSnapshot(&stats.Snapshot{})
	body := scrapeMetrics(t, p)
	if strings.Contains(body, "gpu_kernel_launch_") {
		t.Errorf("empty snapshot should not emit kernel_launch metrics: %s", body)
	}
}

func scrapeMetrics(t *testing.T, p *PrometheusServer) string {
	t.Helper()
	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()
	p.handleMetrics(w, req)
	body, _ := io.ReadAll(w.Result().Body)
	return string(body)
}

